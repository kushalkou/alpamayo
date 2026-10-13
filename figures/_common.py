"""figures/_common.py -- shared style; every figure is saved 1600 px wide (16 in x 100 dpi)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})
C = {'base': '#8c8c8c', 'nocam': '#4c72b0', 'cam': '#dd8452', 'meta': '#55a868', 'coc': '#c44e52',
     'nr': '#8172b3', 'oracle': '#937860', 'expert': '#da8bc3'}


def save(fig, name):
    fig.set_size_inches(16, fig.get_size_inches()[1])
    p = f'{OUT}/{name}.png'
    fig.savefig(p, dpi=100, bbox_inches=None)
    plt.close(fig)
    print('wrote', p)
