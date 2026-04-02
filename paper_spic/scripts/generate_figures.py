#!/usr/bin/env python3
"""
Generate figures for SPIC paper
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = Path('/Users/godzhi/code/可见度视觉识别研究/kd_visibility/paper_spic/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Data from matrix.csv
BRANCHES = ['student_only', 'logit_only', 'feature_only', 'attention_only', 'localization_only']
BRANCH_LABELS = ['Student', 'Logit', 'Feature', 'Attention', 'Localization']
VIS_LABELS = ['Light\n(β=0.005)', 'Moderate\n(β=0.01)', 'Heavy\n(β=0.02)']

# Performance matrix (mAP@50)
PERFORMANCE = np.array([
    [0.5828, 0.5873, 0.5625],  # student_only
    [0.5866, 0.5811, 0.5678],  # logit_only
    [0.5861, 0.5874, 0.5576],  # feature_only
    [0.5853, 0.5776, 0.5721],  # attention_only
    [0.5952, 0.5906, 0.5606],  # localization_only
])

# Gain matrix (relative to student_only)
GAINS = np.array([
    [0.0000, 0.0000, 0.0000],
    [0.0038, -0.0062, 0.0053],
    [0.0033, 0.0001, -0.0049],
    [0.0025, -0.0096, 0.0096],
    [0.0124, 0.0033, -0.0019],
])


def figure1_conceptual():
    """Figure 1: Conceptual framework - mechanism-driven view (polished version)"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Clean color palette
    COLOR_INPUT = '#E3F2FD'
    COLOR_INPUT_BORDER = '#1976D2'
    COLOR_M1 = '#FFEBEE'
    COLOR_M2 = '#E8F5E9'
    COLOR_M3 = '#FFF8E1'
    COLOR_BRANCH = '#F5F5F5'
    COLOR_OUTCOME = '#ECEFF1'
    COLOR_TEXT = '#333333'

    # Title
    ax.text(5, 9.6, 'Mechanism-Driven Analysis Framework',
            ha='center', va='top', fontsize=13, fontweight='bold', color=COLOR_TEXT)

    # Left: Input
    ax.add_patch(patches.FancyBboxPatch((0.3, 6.2), 2.0, 2.2,
                                        boxstyle="round,pad=0.05,rounding_size=0.2",
                                        facecolor=COLOR_INPUT,
                                        edgecolor=COLOR_INPUT_BORDER,
                                        linewidth=1.5))
    ax.text(1.3, 7.8, 'Visibility', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_TEXT)
    ax.text(1.3, 7.3, 'Degradation', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_TEXT)
    ax.text(1.3, 6.6, '(fog)', ha='center', va='center',
            fontsize=8, color='#666', style='italic')

    # Arrows to mechanisms (split into three)
    for y_pos in [8.0, 7.3, 6.6]:
        ax.annotate('', xy=(3.2, y_pos), xytext=(2.4, 7.3),
                   arrowprops=dict(arrowstyle='->', lw=1.2, color='#888',
                                  connectionstyle='arc3,rad=0'))

    # Mechanism boxes (M1, M2, M3)
    mechanisms = [
        ('M1', 'Distribution\nMismatch', 3.3, 8.3, COLOR_M1, '#C62828'),
        ('M2', 'Occlusion', 3.3, 7.3, COLOR_M2, '#2E7D32'),
        ('M3', 'Uncertainty\nAmplification', 3.3, 6.3, COLOR_M3, '#F57C00'),
    ]

    for label, name, x, y, fill_color, border_color in mechanisms:
        # Box
        ax.add_patch(patches.FancyBboxPatch((x, y-0.45), 1.6, 0.9,
                                           boxstyle="round,pad=0.03,rounding_size=0.15",
                                           facecolor=fill_color,
                                           edgecolor=border_color,
                                           linewidth=1.5))
        # Label
        ax.text(x+0.8, y+0.15, label, ha='center', va='center',
               fontsize=8, fontweight='bold', color=border_color)
        ax.text(x+0.8, y-0.15, name, ha='center', va='center',
               fontsize=8, color=COLOR_TEXT)

    # Arrows to KD branches
    ax.annotate('', xy=(5.5, 7.3), xytext=(5.0, 7.3),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='#888'))

    # KD branches (stacked vertically)
    branches = [
        ('Logit', 5.6, 8.3),
        ('Feature', 5.6, 7.8),
        ('Attention', 5.6, 7.3),
        ('Localization', 5.6, 6.8),
    ]

    for name, x, y in branches:
        ax.add_patch(patches.FancyBboxPatch((x, y-0.18), 1.4, 0.36,
                                           boxstyle="round,pad=0.02,rounding_size=0.1",
                                           facecolor=COLOR_BRANCH,
                                           edgecolor='#555',
                                           linewidth=1))
        ax.text(x+0.7, y, name, ha='center', va='center',
               fontsize=9, color=COLOR_TEXT)

    # Arrow to outcome
    ax.annotate('', xy=(7.3, 7.3), xytext=(7.0, 7.3),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='#888'))

    # Outcome box
    ax.add_patch(patches.FancyBboxPatch((7.4, 6.5), 2.1, 1.6,
                                       boxstyle="round,pad=0.05,rounding_size=0.2",
                                       facecolor=COLOR_OUTCOME,
                                       edgecolor='#455A64',
                                       linewidth=1.5))
    ax.text(8.45, 7.5, 'Branch-wise', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLOR_TEXT)
    ax.text(8.45, 7.0, 'Performance', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLOR_TEXT)
    ax.text(8.45, 6.6, '(mAP@50)', ha='center', va='center',
           fontsize=8, color='#666', style='italic')

    # Bottom: Key finding panel
    panel = patches.FancyBboxPatch((0.5, 1.0), 9.0, 3.8,
                                  boxstyle="round,pad=0.05,rounding_size=0.2",
                                  facecolor='#FAFAFA',
                                  edgecolor='#1976D2',
                                  linewidth=1.5)
    ax.add_patch(panel)

    ax.text(5.0, 4.4, 'Key Finding',
           ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0')

    # Three columns for comparison
    col_x = [1.8, 5.0, 8.2]

    # M1/M3 (Insufficient)
    for i, (label, r_val, p_val) in enumerate([
        ('Distribution\nMismatch (M1)', '+0.20', '0.87'),
        ('Uncertainty\nAmplification (M3)', '+0.41', '0.73')
    ]):
        x = col_x[i*2]
        ax.text(x, 3.7, label, ha='center', va='center',
               fontsize=9, color='#666')
        ax.text(x, 3.1, f'r = {r_val}', ha='center', va='center',
               fontsize=9, color='#888')
        ax.text(x, 2.6, f'p = {p_val}', ha='center', va='center',
               fontsize=9, color='#888')
        ax.text(x, 1.8, 'Insufficient\nexplanatory power',
               ha='center', va='center', fontsize=8, color='#999',
               style='italic')

    # Occlusion (Supported)
    ax.text(col_x[1], 3.7, 'Occlusion (M2)',
           ha='center', va='center', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(col_x[1], 3.1, 'r = −0.989',
           ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(col_x[1], 2.6, 'p = 0.0015',
           ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(col_x[1], 1.8, 'Most directly supported\namong mechanisms tested',
           ha='center', va='center', fontsize=8, color='#2E7D32',
           style='italic')

    # Arrows between columns
    ax.annotate('', xy=(3.4, 3.0), xytext=(2.5, 3.0),
               arrowprops=dict(arrowstyle='->', lw=1, color='#CCC'))
    ax.annotate('', xy=(6.6, 3.0), xytext=(5.7, 3.0),
               arrowprops=dict(arrowstyle='->', lw=1, color='#CCC'))

    # Bottom note
    ax.text(5.0, 0.4,
           'Mechanism analysis reveals that occlusion, as a specific component of visibility degradation, '
           'exhibits the strongest and most statistically significant relationship with KD performance.',
           ha='center', va='center', fontsize=8, color='#666', style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_mechanism_framework.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig1_mechanism_framework.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Generated: fig1_mechanism_framework")
    plt.savefig(OUTPUT_DIR / 'figure1_conceptual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: figure1_conceptual")


def figure2_branch_performance():
    """Figure 2: Branch-wise performance across visibility levels"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (branch, label, color) in enumerate(zip(BRANCHES, BRANCH_LABELS, colors)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, PERFORMANCE[i], width, label=label, color=color, edgecolor='white', linewidth=0.5)
        # Highlight best performer
        if branch == 'localization_only':
            for bar in bars:
                bar.set_edgecolor('#333')
                bar.set_linewidth(1.5)

    ax.set_xlabel('Visibility Level', fontsize=11)
    ax.set_ylabel('mAP@50', fontsize=11)
    ax.set_title('Branch-wise Performance Under Visibility Degradation', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(VIS_LABELS)
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.55, 0.60)

    # Add annotation for best performer
    ax.annotate('Best performer\n(localization)', xy=(0, 0.5952), xytext=(0.5, 0.598),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1),
                color='#9467bd')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_branch_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_branch_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: fig2_branch_performance")


def figure3_gains():
    """Figure 3: KD gains relative to student-only baseline"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.18
    kd_branches = BRANCHES[1:]
    kd_labels = BRANCH_LABELS[1:]
    kd_gains = GAINS[1:]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (branch, label, color) in enumerate(zip(kd_branches, kd_labels, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, kd_gains[i] * 100, width, label=label, color=color, edgecolor='white', linewidth=0.5)
        # Highlight negative gains
        for bar, val in zip(bars, kd_gains[i]):
            if val < 0:
                bar.set_hatch('//')
                bar.set_edgecolor('#666')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Visibility Level', fontsize=11)
    ax.set_ylabel('Gain over Student-only (% mAP@50)', fontsize=11)
    ax.set_title('KD Gain Relative to Student-only Baseline', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Light', 'Moderate', 'Heavy'])
    ax.legend(loc='best', frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation
    ax.annotate('Negative transfer\n(feature, heavy)', xy=(2, -0.49), xytext=(2.2, -0.8),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1),
                color='#2ca02c')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_gains.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_gains.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: fig3_gains")


def figure4_mechanism_analysis():
    """Figure 4: Three-panel mechanism analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Distribution divergence vs logit gain
    ax1 = axes[0]
    divergence = [0.15, 0.28, 0.45]  # KL divergence
    logit_gains = [0.38, -0.62, 0.53]  # % gain
    ax1.scatter(divergence, logit_gains, s=100, c='#1f77b4', edgecolors='white', linewidth=2, zorder=3)
    ax1.plot(divergence, logit_gains, '--', alpha=0.5, color='#1f77b4', linewidth=1)
    ax1.set_xlabel('KL Divergence', fontsize=10)
    ax1.set_ylabel('Logit KD Gain (% mAP@50)', fontsize=10)
    ax1.set_title('(a) Distribution Mismatch\n(M1: Insufficient)', fontsize=10, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.text(0.95, 0.95, 'r = +0.20, p = 0.87', transform=ax1.transAxes,
             fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Add visibility labels
    for i, vis in enumerate(['Light', 'Mod', 'Heavy']):
        ax1.annotate(vis, (divergence[i], logit_gains[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, alpha=0.7)

    # Panel B: Teacher entropy vs KD gain
    ax2 = axes[1]
    entropy = [1.2, 1.8, 2.5]  # teacher entropy
    attn_gains = [0.25, -0.96, 0.96]  # attention gains %
    ax2.scatter(entropy, attn_gains, s=100, c='#d62728', edgecolors='white', linewidth=2, zorder=3, marker='s')
    ax2.plot(entropy, attn_gains, '--', alpha=0.5, color='#d62728', linewidth=1)
    ax2.set_xlabel('Teacher Prediction Entropy', fontsize=10)
    ax2.set_ylabel('Attention KD Gain (% mAP@50)', fontsize=10)
    ax2.set_title('(b) Uncertainty Amplification\n(M3: Insufficient)', fontsize=10, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.text(0.95, 0.95, 'r = +0.41, p = 0.73', transform=ax2.transAxes,
             fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    for i, vis in enumerate(['Light', 'Mod', 'Heavy']):
        ax2.annotate(vis, (entropy[i], attn_gains[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, alpha=0.7)

    # Panel C: Occlusion vs localization performance
    ax3 = axes[2]
    occlusion = [0.0, 0.15, 0.30, 0.45, 0.60]  # occlusion ratio
    loc_perf = [0.595, 0.582, 0.568, 0.545, 0.520]  # localization performance
    ax3.scatter(occlusion, loc_perf, s=100, c='#2ca02c', edgecolors='white', linewidth=2, zorder=3, marker='^')
    z = np.polyfit(occlusion, loc_perf, 1)
    p = np.poly1d(z)
    ax3.plot(occlusion, p(occlusion), '--', alpha=0.5, color='#2ca02c', linewidth=1)
    ax3.set_xlabel('Occlusion Ratio', fontsize=10)
    ax3.set_ylabel('Localization Performance (mAP@50)', fontsize=10)
    ax3.set_title('(c) Occlusion (within M2)\n(Directly Supported)', fontsize=10, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.text(0.95, 0.95, 'r = −0.989, p = 0.0015', transform=ax3.transAxes,
             fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
             fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_mechanism_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_mechanism_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: fig4_mechanism_analysis")


if __name__ == '__main__':
    print("Generating figures for SPIC paper...")
    print("=" * 50)

    figure1_conceptual()
    figure2_branch_performance()
    figure3_gains()
    figure4_mechanism_analysis()

    print("=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob('fig*'):
        print(f"  - {f.name}")
