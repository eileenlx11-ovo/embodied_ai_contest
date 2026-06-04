# -*- coding: utf-8 -*-
"""Generate crimson-theme figures for the technical proposal from real
per-epoch metrics.csv pulled off the AutoDL server. Output: assets/fig*.png."""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# theme (matches answer_deck.html + PPTX)
PAPER = '#fafaf8'
INK = '#0a0a0a'
ACCENT = '#9E1B1B'
BRIGHT = '#D94545'
GREY = '#737373'
GREY2 = '#c8c8c6'
BLUE = '#3a5a80'  # secondary series, muted

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.edgecolor': '#888',
    'axes.linewidth': 0.8,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#e8e8e6',
    'grid.linewidth': 0.7,
    'legend.frameon': False,
})
# CJK fallback for the few Chinese labels
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

D = os.path.join(os.path.dirname(__file__), 'data')
OUT = os.path.dirname(__file__)


def load(name):
    rows = list(csv.DictReader(open(os.path.join(D, name))))
    g = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, '', 'None')]
    out = {'epoch': [int(float(r['epoch'])) for r in rows],
           'train_loss': g('train_loss'), 'val_top1': g('val_top1')}
    # optional cols
    for k in ('val_raw_top1', 'val_ema_top1'):
        try:
            out[k] = [float(r[k]) if r.get(k) not in (None, '', 'None') else None
                      for r in rows]
        except Exception:
            out[k] = None
    return out


def fig1_training_curve():
    """full_mixcut: train loss (left) + val top-1 (right), dual axis."""
    d = load('full_mixcut.csv')
    fig, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax1.plot(d['epoch'], d['train_loss'], color=GREY, lw=1.8, label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss', color=GREY)
    ax1.tick_params(axis='y', labelcolor=GREY)
    ax1.set_xlim(0, 100)
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(d['epoch'], d['val_top1'], color=ACCENT, lw=2.2, label='Val Top-1')
    ax2.set_ylabel('Val Top-1 (%)', color=ACCENT)
    ax2.tick_params(axis='y', labelcolor=ACCENT)
    ax2.set_ylim(0, 85)
    # annotate final
    ax2.annotate('77.57%', xy=(100, 77.57), xytext=(78, 64),
                 color=ACCENT, fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_training_curve.png'))
    plt.close(fig)
    print('fig1 done')


def fig2_ablation_bars():
    """final ablation: 5 experiments, val top-1 bars."""
    labels = ['baseline_v1', 'cleaned_v1', 'full_strong\n_aug',
              'full_mixcut', 'full_mixcut\n+HFlip TTA']
    vals = [77.58, 76.84, 76.91, 77.57, 77.99]
    colors = [GREY, GREY2, GREY2, BRIGHT, ACCENT]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    ax.set_ylim(76.0, 78.3)
    ax.set_ylabel('Val Top-1 (%)')
    ax.grid(axis='x', visible=False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f'{v:.2f}',
                ha='center', va='bottom', fontsize=10.5, fontweight='bold',
                color=INK)
    ax.axhline(77.58, color=GREY, ls='--', lw=0.9, zorder=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_ablation_bars.png'))
    plt.close(fig)
    print('fig2 done')


def fig3_convergence():
    """baseline vs full_mixcut vs cleaned: val top-1 over 100 epochs."""
    runs = [('full_mixcut.csv', 'full_mixcut (final)', ACCENT, 2.3),
            ('cleaned_v1.csv', 'cleaned_v1', BLUE, 1.6),
            ('mixup_v2.csv', 'full_strong_aug (v2)', GREY, 1.6)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    for fn, lab, c, w in runs:
        d = load(fn)
        ax.plot(d['epoch'], d['val_top1'], color=c, lw=w, label=lab)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Top-1 (%)')
    ax.set_xlim(0, 100); ax.set_ylim(0, 82)
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_convergence.png'))
    plt.close(fig)
    print('fig3 done')


def fig4_raw_ema():
    """full_mixcut raw vs EMA val top-1, last 50 epochs (crossover)."""
    d = load('full_mixcut.csv')
    ep = d['epoch']; raw = d['val_raw_top1']; ema = d['val_ema_top1']
    # last 50 epochs window
    lo = 50
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(ep[lo:], raw[lo:], color=ACCENT, lw=2.0, label='raw weights')
    ax.plot(ep[lo:], ema[lo:], color=BLUE, lw=2.0, ls='--', label='EMA weights')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Top-1 (%)')
    ax.set_xlim(lo, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.annotate('raw 77.57 > EMA 77.438', xy=(100, 77.5), xytext=(72, 73.2),
                color=INK, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_raw_ema.png'))
    plt.close(fig)
    print('fig4 done')


if __name__ == '__main__':
    fig1_training_curve()
    fig2_ablation_bars()
    fig3_convergence()
    fig4_raw_ema()
    print('all figures generated')

