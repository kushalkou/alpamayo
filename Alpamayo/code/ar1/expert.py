"""ar1/expert.py -- R2.3: flow-matching action expert, option A (AR1-style per-layer KV).

AR1 (paper): the expert "takes as input both the KV-cache from the sequence and the
embedded representation of the noisy control"; stop-gradient on the VLM KV-cache; same
number of attention heads and head dim as the VLM, smaller hidden / MLP width; Gaussian
conditional OT path, loss || v(a_t, o) - (a - eps) ||; Euler integration, dt = 0.1.
Ours:
  layers  28, one per VLM layer; layer l attends over [VLM K/V of layer l (frozen,
          stop-grad, post-RoPE as cached) | the expert's own 12 action-token K/V].
          Bidirectional among action tokens (AR1's mask: UNVERIFIED).
  width   hidden 512, SwiGLU MLP 1,536, RMSNorm pre-norm, dropout 0.1 on residuals.
  heads   28 query heads, 4 KV heads, head dim 128 (= Qwen2.5-7B / Cosmos-Reason1).
  RoPE    standard 1D RoPE, theta 1e6, rotate-half (Qwen2.5-VL's M-RoPE reduces to it
          for text-only positions); action token i sits at position ctx_len + i.
          Verified against the cached keys in kv_a3.py.
  input   a_t [B,12,2] (standardised accel, curvature) -> Linear(2, 512) + learned step
          embedding + time MLP(sinusoidal(t)) ; output RMSNorm -> Linear(512, 2).
  path    x_t = t a + (1 - t) eps, eps ~ N(0, I); target u = a - eps; t ~ U(0, 1)
          (AR1's t distribution: UNVERIFIED). fp32 throughout.
  sample  x_0 = eps; 10 Euler steps x += 0.1 v(x, t) at t = 0, 0.1, .., 0.9.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

NL, NQ, NKV, HD, THETA = 28, 28, 4, 128, 1e6


def rope(x, pos):
    """x [B,H,L,HD], pos [L] -> rotate-half RoPE (Qwen2 convention)."""
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=x.device, dtype=torch.float32) / HD))
    f = pos.float()[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class RMS(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Layer(nn.Module):
    def __init__(self, d, m, p):
        super().__init__()
        self.n1, self.n2 = RMS(d), RMS(d)
        self.q = nn.Linear(d, NQ * HD); self.k = nn.Linear(d, NKV * HD); self.v = nn.Linear(d, NKV * HD)
        self.o = nn.Linear(NQ * HD, d, bias=False)
        self.g = nn.Linear(d, m, bias=False); self.u = nn.Linear(d, m, bias=False)
        self.dn = nn.Linear(m, d, bias=False); self.drop = nn.Dropout(p)

    def forward(self, x, K, V, pos):
        B, L, _ = x.shape
        h = self.n1(x)
        q = rope(self.q(h).view(B, L, NQ, HD).transpose(1, 2), pos)
        k = rope(self.k(h).view(B, L, NKV, HD).transpose(1, 2), pos)
        v = self.v(h).view(B, L, NKV, HD).transpose(1, 2)
        k = torch.cat([K.to(x.dtype), k], 2)
        v = torch.cat([V.to(x.dtype), v], 2)
        # GQA without repeat_interleave: q head h uses kv head h // 7; with no mask this is
        # identical to repeating K/V (R3.3; saves 7x attention memory on long contexts)
        g = NQ // NKV
        a = F.scaled_dot_product_attention(q.reshape(B, NKV, g * L, HD), k, v).reshape(B, NQ, L, HD)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, NQ * HD)))
        h = self.n2(x)
        return x + self.drop(self.dn(F.silu(self.g(h)) * self.u(h)))


class FlowExpert(nn.Module):
    def __init__(self, n_steps=12, d=512, m=1536, p=0.1):
        super().__init__()
        self.n = n_steps; self.d = d
        self.inp = nn.Linear(2, d); self.step = nn.Parameter(torch.randn(n_steps, d) * 0.02)
        self.tmlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([Layer(d, m, p) for _ in range(NL)])
        self.nf = RMS(d); self.out = nn.Linear(d, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def temb(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * 1000.0 * f[None]
        return self.tmlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, xt, t, KV, ctx_len):
        """xt [B,12,2]; t [B]; KV [B,28,2,4,C,128] (fp16 ok; cast to fp32)."""
        x = self.inp(xt) + self.step[None] + self.temb(t)[:, None]
        pos = torch.arange(ctx_len, ctx_len + self.n, device=xt.device)
        for l, layer in enumerate(self.layers):
            x = layer(x, KV[:, l, 0], KV[:, l, 1], pos)
        return self.out(self.nf(x))

    def loss(self, a, KV, ctx_len):
        eps = torch.randn_like(a); t = torch.rand(a.shape[0], device=a.device)
        xt = t[:, None, None] * a + (1 - t[:, None, None]) * eps
        return F.mse_loss(self(xt, t, KV, ctx_len), a - eps)

    @torch.no_grad()
    def sample(self, KV, ctx_len, eps, steps=10):
        x = eps.clone(); dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + dt * self(x, t, KV, ctx_len)
        return x
