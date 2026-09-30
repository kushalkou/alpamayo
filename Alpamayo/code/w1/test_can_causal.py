"""w1/test_can_causal.py -- CAN lookup never returns a message after t0."""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
from v0_sources import can_index


def test_synthetic():
    ut = np.array([0, 20, 40, 60])
    assert can_index(ut, 39) == 1 and can_index(ut, 40) == 2 and can_index(ut, -1) == -1
    assert can_index(np.array([]), 5) == -1


def test_built_table():
    V = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_v0.pkl', 'rb'))
    off = np.array([d['can_dt_ms'] for d in V.values() if d['can_ok']])
    assert len(off) and off.min() >= 0.0, off.min()      # t0 - utime >= 0: never after t0


if __name__ == '__main__':
    test_synthetic(); print('PASS test_synthetic')
    test_built_table(); print('PASS test_built_table')
