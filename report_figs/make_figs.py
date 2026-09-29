"""Generate figures for the MP1 report (data read from code/runs/*/metrics.json).
Output: report_figs/fig1_validation_curves.png, fig2_lrf_sweep.png, fig3_arch_flip.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / 'code'
OUT = Path(__file__).resolve().parent

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def hist(name):
    r = json.loads((ROOT / 'runs' / name / 'metrics.json').read_text())
    steps = [h['step'] for h in r['validation_history']]
    bpbs = [h['bpb'] for h in r['validation_history']]
    return steps, bpbs, r['validation']['bpb']


# ---------- Figure 1: validation curves (main figure) ----------
curves = [
    ('swiglu-w224h7d6-4800-cuda-ema995-lrf01',
     'SwiGLU annealed (LR floor 0.1, EMA 0.995)', '#d62728', '-', 'o', 1.6),
    ('swiglu-w224h7d6-4800-cuda-ema999-lrf099',
     'SwiGLU high-floor cosine (LR floor 0.99, EMA 0.999)', '#1f77b4', '-', 's', 1.6),
    ('student-gelu-w224h7d6-6000-cuda-ema999-lrf099',
     'GELU high-floor cosine (seed 17)', '#2ca02c', '-', '^', 1.6),
    ('gelu-w224h7d6-6000-cuda-ema999-lrf099-18',
     'GELU high-floor cosine (seed 18, final model)', '#7f00d6', '-', 'D', 2.2),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
ax.axhline(2.071084, color='gray', ls='--', lw=1.2)
ax.text(4180, 2.082, 'baseline (1,200 steps) 2.071', color='gray', fontsize=9)

for name, label, color, ls, marker, lw in curves:
    steps, bpbs, final = hist(name)
    ax.plot(steps, bpbs, color=color, ls=ls, marker=marker, ms=4, lw=lw, label=label)

# The 7,200-step control run shows the small late validation rise after
# 6,000 steps. It is a control trajectory, not the submitted checkpoint.
steps7200, bpbs7200, final7200 = hist('baseline-w224h7d6-7200-cuda-ema999-lrf099-seed18')
ax.plot(steps7200, bpbs7200, color='#555555', ls='--', marker='x', ms=3.5, lw=1.2,
        label='GELU control (7,200-step run, seed 18)')

# annotate only two key endpoints (annealed endpoint and final model)
for name, color, dy in [('swiglu-w224h7d6-4800-cuda-ema995-lrf01', '#d62728', 8),
                        ('gelu-w224h7d6-6000-cuda-ema999-lrf099-18', '#7f00d6', -12)]:
    steps, bpbs, final = hist(name)
    ax.annotate('%.4f' % final, xy=(steps[-1], bpbs[-1]), xytext=(6, dy),
                textcoords='offset points', fontsize=9, color=color, ha='left')
ax.annotate('%.4f' % final7200, xy=(steps7200[-1], bpbs7200[-1]), xytext=(6, 8),
            textcoords='offset points', fontsize=9, color='#555555', ha='left')
ax.set_xlim(0, 7600)

steps, bpbs, _ = hist('swiglu-w224h7d6-4800-cuda-ema995-lrf01')
k = bpbs.index(min(bpbs))
ax.annotate('later checkpoint is slightly worse\n(best=%.4f@%d)' % (min(bpbs), steps[k]),
            xy=(steps[k], min(bpbs)), xytext=(150, 1.548),
            fontsize=8.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.9))

ax.set_xlabel('Training steps (validation every 300 steps)')
ax.set_ylabel('Validation BPB (lower is better)')
ax.set_title('Validation curves under different training recipes (w224h7d6)')
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout()
fig.savefig(OUT / 'fig1_validation_curves.png')
plt.close(fig)

# ---------- Figure 2: lr floor dose-response ----------
sweep = [('lrf005', 0.05), ('lrf01', 0.1), ('lrf02', 0.2),
         ('lrf04', 0.4), ('lrf08', 0.8), ('lrf099', 0.99)]
xs, ys = [], []
for tag, x in sweep:
    _, _, final = hist('swiglu-w224h7d6-4800-cuda-ema999-' + tag)
    xs.append(x)
    ys.append(final)

# These two logged values are retained in the report table, but their
# original run directories are no longer present in the workspace.
xs[3:3] = [0.3]
ys[3:3] = [1.624230]
xs[5:5] = [0.5]
ys[5:5] = [1.614618]

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
ax.plot(xs, ys, marker='o', color='#1f77b4', lw=1.6)
for x, y in zip(xs, ys):
    ax.annotate('%.4f' % y, xy=(x, y), xytext=(0, 7), textcoords='offset points',
                fontsize=8, ha='center', color='#1f77b4')
ax.set_xlabel('Cosine LR floor (0.99 is near-constant learning rate)')
ax.set_ylabel('Validation BPB (4,800 steps, EMA 0.999)')
ax.set_title('Cosine learning-rate floor sweep (seed 17)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig2_lrf_sweep.png')
plt.close(fig)

# ---------- Figure 3: recipe-dependent MLP comparison (dot plot) ----------
groups = [
    ('3,600 steps\nannealed (LR floor 0, EMA 0.995)',
     'baseline-w224h7d6-3600-cuda-bf16-ema995',
     'swiglu-w224h7d6-3600-cuda-bf16-ema995'),
    ('6,000 steps\nhigh-floor cosine (LR floor 0.99, EMA 0.999)',
     'student-gelu-w224h7d6-6000-cuda-ema999-lrf099',
     'swiglu-w224h7d6-6000-cuda-ema999-lrf099'),
]
gelu = [hist(g)[2] for _, g, _ in groups]
swi = [hist(s)[2] for _, _, s in groups]

x = list(range(len(groups)))
offset = 0.09
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
for i, (g, s) in enumerate(zip(gelu, swi)):
    ax.plot([i - offset, i + offset], [g, s], color='#999999', lw=1.2, zorder=1)
    ax.scatter(i - offset, g, s=65, color='#1f77b4',
               label='GELU' if i == 0 else None, zorder=2)
    ax.scatter(i + offset, s, s=65, color='#ff7f0e',
               label='SwiGLU' if i == 0 else None, zorder=2)
    ax.annotate('%.4f' % g, xy=(i - offset, g), xytext=(0, 7),
                textcoords='offset points', fontsize=9, ha='center')
    ax.annotate('%.4f' % s, xy=(i + offset, s), xytext=(0, 7),
                textcoords='offset points', fontsize=9, ha='center')
ax.set_xticks(list(x))
ax.set_xticklabels([p[0] for p in groups])
ax.set_ylabel('Validation BPB (seed 17; lower is better)')
ax.set_ylim(1.55, 1.70)
ax.set_title('SwiGLU versus GELU under two training recipes (w224h7d6)')
ax.grid(alpha=0.3, axis='y')
ax.text(0.02, 0.03, 'Y-axis starts at 1.55; exact values are annotated.',
        transform=ax.transAxes, fontsize=8, color='#555555')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'fig3_arch_flip.png')
plt.close(fig)

print('done:', ', '.join(p.name for p in sorted(OUT.glob('*.png'))))
