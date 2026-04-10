import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Font
plt.rcParams['font.family'] = 'Times New Roman'

# Data
file_path = './output_data/Total_data.xlsx'
sheet_name = 'PPD'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# X axis: 24h, 15-min steps
training_days = np.linspace(0, 24, 96)

# Scenarios
num_scenarios = 5
occupy = data['MASAC_People1']  # occupancy flag (0/1 or counts)

# Colors
colors = {
    'Rule_PPD_': "#bbf51d",
    'MADDPG_PPD': 'orange',
    'MASAC_PPD': 'purple',
    'MADDPG_LAG_PPD': 'green',
    'MASAC_Lag_PPD': 'cyan',
    'IPO_PPD': 'deeppink',
    'P3O_PPD': 'firebrick',
    'DCMASAC_PPD': 'blue',
}

# Legend labels
labels = {
    'MADDPG_LAG_PPD': "MADDPG-Lag",
    'MASAC_Lag_PPD': "MASAC-Lag",
    'MADDPG_PPD': 'MADDPG',
    'DCMASAC_PPD': 'DCMADRL',
    'MASAC_PPD': "MASAC",
    'Rule_PPD_': 'RBC',
    'IPO_PPD': "IPO",
    'P3O_PPD': "P3O",
}

# Merge occupied intervals and optionally annotate once
def add_shading(ax, x, occupy, annotate=False,
                label="Occupied periods",
                facecolor="#FFFACD", alpha=0.15,
                text_size=18, x_pad=0.05, y_text_pos=0.5, y_bar_pos=0.5):
    y_min, y_max = ax.get_ylim()
    y_text = y_min + y_text_pos * (y_max - y_min)
    y_bar  = y_min + y_bar_pos  * (y_max - y_min)

    start = None
    intervals = []
    for i in range(len(x) - 1):
        occ = occupy.iloc[i] != 0
        if occ and start is None:
            start = x[i]
        is_last = (i == len(x) - 2)
        if (not occ or is_last) and start is not None:
            end = x[i+1] if occ and is_last else x[i]
            intervals.append((start, end))
            start = None

    # Draw shading on all intervals
    for s, e in intervals:
        ax.axvspan(s, e, color=facecolor, alpha=alpha, zorder=0)

    # Annotate only once if requested
    if annotate and intervals:
        s, e = intervals[0][0], intervals[-1][1]  # span full occupied range of the day
        # small padding for bracket so it reaches the edge visually
        ax.annotate("", xy=(s - x_pad, y_bar), xytext=(e + x_pad, y_bar),
                    arrowprops=dict(arrowstyle="|-|", lw=1.4))
        cx = (s + e) / 2.0
        ax.text(cx, y_text, label, ha="center", va="top", fontsize=text_size,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.6))

# Figure and layout
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 2, figure=fig)
gs.update(wspace=0.16, hspace=0.35)

suffixes = ['Rule_PPD_', 'MADDPG_PPD', 'MASAC_PPD',
            'MADDPG_LAG_PPD', 'MASAC_Lag_PPD', 'IPO_PPD', 'P3O_PPD', 'DCMASAC_PPD']

PPD_THRESH = 10  # %
annotated_once = False

for i in range(num_scenarios):
    ax = fig.add_subplot(gs[i // 2, i % 2] if i < 4 else gs[2, :])

    # Plot all controllers
    for sfx in suffixes:
        col = f'{sfx}{i+1}'
        ax.plot(training_days, data[col], label=labels[sfx], color=colors[sfx], linewidth=2.0)

    # Axes
    ax.set_xlim([0, 24])
    ax.set_xticks(np.linspace(0, 24, 7))
    max_y = max([data[f'{sfx}{i+1}'].max() for sfx in suffixes])
    ax.set_ylim([0, max_y])
    ax.set_xlabel('Hours (h)', fontsize=34)
    ax.set_ylabel(f'Zone {i+1} PPD (%)', fontsize=34)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(True)

    # Threshold line on all subplots
    ax.axhline(y=PPD_THRESH, color='red', linestyle='--', linewidth=2.0, zorder=1)

    # Shade occupancy on all; annotate only once (e.g., last subplot)
    annotate_here = (i == num_scenarios - 1) and not annotated_once
    add_shading(ax, training_days, occupy, annotate=annotate_here,
                label="Occupied periods", text_size=18, x_pad=0.08, y_text_pos=0.985, y_bar_pos=0.955)
    annotated_once |= annotate_here

    if annotate_here:
        y_min, y_max = ax.get_ylim()
        ax.text(0.6, PPD_THRESH + 0.02*(y_max - y_min),  # x=0.6h，左侧不被图例遮挡
            'PPD threshold (10%)', ha='left', va='bottom', fontsize=18,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))


    ax.legend(fontsize=14, loc='upper right')

plt.tight_layout()
plt.savefig('./photo/PPD.pdf', bbox_inches='tight', dpi=300)
plt.show()
