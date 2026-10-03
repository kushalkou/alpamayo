"""ar1/vc_patch.py -- R2.4: feed CACHED native-order visual tokens (ar1/vcache.py) through
the unchanged week-1 dataset / finetune / ar_eval / dump code paths.
mode 't0'   : 3 front cameras at t0                    -> 3 x 160 = 480 visual tokens
mode 'hist' : 3 front cameras x 4 keyframes (<= t0)    -> 12 x 160 = 1,920 visual tokens
Mechanism (same pattern as --no_vision): dataset.CAMERAS = ['VC'], each record's
cam_paths = {'VC': sample_token}, dataset.preprocess_image(sample_token) returns the
gathered [N, 3584] fp16 tokens, so a batch 'images' is [B, 1, N, 3584]; encode_live /
encode_live_one just move them to the GPU (the frozen tower is not run)."""
import sys, pickle, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import vcache

HIST = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl'
FRAMES = {'t0': (3,), 'hist': (0, 1, 2, 3)}
_H = None


def _h():
    global _H
    if _H is None:
        _H = pickle.load(open(HIST, 'rb'))
    return _H


def load_tokens(st, augment=False):
    return vcache.gather(_h()[st], FRAMES[MODE])


MODE = None


def install(mode, dataset_mod):
    global MODE
    MODE = mode
    _h()
    dataset_mod.CAMERAS = ['VC']
    dataset_mod.preprocess_image = load_tokens


def tag_records(R):
    for t in R:
        t['cam_paths'] = {'VC': t['sample_token']}
    return R


def encode_live(visual, images, device):
    return images[:, 0].to(device, torch.float16)


def encode_live_one(visual, images, device, dtype=torch.float16):
    return images[0:1].to(device, dtype)
