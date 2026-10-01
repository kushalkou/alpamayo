"""w1/q2_len.py -- G5 pre-registered length rule. Holdout selection metric = AR median
ADE@6s on the fixed 400-sample holdout subset (the number every run selects on). If the
best G5 (30 epochs, patience 5) value improves >= 3% over A3's best (10 epochs), LEN = 30
for all later VLA trajectory runs, else LEN = 10. Writes results/Q2_LEN."""
import re
L = '/home/dgx1user/Alpamayo-Kushal/Alpamayo'


def sel(log):
    v = [(int(e), float(m)) for e, m in re.findall(r'\[epoch (\d+)\] SELECT val_ADE@6s median=([\d.]+)',
                                                  open(log).read())]
    return v, min(v, key=lambda x: x[1])


va, a3 = sel(f'{L}/w1_stageA_A3.log')
vg, g5 = sel(f'{L}/w1_q2_G5.log')
imp = 1 - g5[1] / a3[1]
LEN = 30 if imp >= 0.03 else 10
print('A3 per epoch: ' + ' '.join(f'{e}:{m:.4f}' for e, m in va))
print('G5 per epoch: ' + ' '.join(f'{e}:{m:.4f}' for e, m in vg))
print(f'A3 best {a3[1]:.4f} (epoch {a3[0]}); G5 best {g5[1]:.4f} (epoch {g5[0]}, of {len(vg)} run)')
print(f'improvement {100*imp:+.2f}% (rule: >= 3% -> 30 epochs) => LEN = {LEN}')
open(f'{L}/results/Q2_LEN', 'w').write(f'{LEN}\n')
