import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Config ----
plt.rcParams['font.family'] = 'Times New Roman'

file_path = './output_data/Total_data.xlsx'
sheet_name = 'Day_energy'
save_path = './photo/Energy_day.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ---- Load data ----
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 24h with 15-min steps -> 96 points
training_days = np.linspace(0, 24, 96)

# Series
RBC         = data['Rule_energy']
MADDPG      = data['MADDPG_energy']
MASAC       = data['MASAC_energy']
MADDPG_Lag  = data['MADDPG_LAG_energy']
MASAC_Lag   = data['MASAC_Lag_energy']
IPO         = data['IPO_energy']
P3O         = data['P3O_energy']
DCMASAC     = data['DCMASAC_energy']
occupy      = data['MADDPG_LAG_People1']  # occupancy flag

colors = {
    'RBC': "#bbf51d",
    'MADDPG': 'orange',
    'MASAC': 'purple',
    'MADDPG_Lag': 'green',
    'MASAC_Lag': 'cyan',
    'IPO': 'deeppink',
    'P3O': 'firebrick',
    'DCMASAC': 'blue',
}

# ---- Shade + annotate merged occupied intervals ----
def add_shading(ax, x, occupy, label="Occupied periods", facecolor="#FFFACD", alpha=0.15):
    y_min, y_max = ax.get_ylim()
    y_text = y_min + 0.97 * (y_max - y_min)
    y_bar  = y_min + 0.92 * (y_max - y_min)

    start = None
    for i in range(len(x) - 1):
        occ = occupy.iloc[i] != 0
        if occ and start is None:
            start = x[i]
        is_last = (i == len(x) - 2)
        if (not occ or is_last) and start is not None:
            end = x[i+1] if occ and is_last else x[i]
            ax.axvspan(start, end, color=facecolor, alpha=alpha, zorder=0)
            cx = (start + end) / 2.0
            ax.text(cx, y_text, label, ha="center", va="top", fontsize=24,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
            ax.annotate("", xy=(start - 0.10, y_bar), xytext=(end + 0.10, y_bar),
                        arrowprops=dict(arrowstyle="|-|", lw=1.2))
            start = None

# ---- Plot ----
fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(training_days, RBC,        label='RBC',         color=colors['RBC'],        linewidth=2.0)
ax.plot(training_days, MADDPG,     label='MADDPG',      color=colors['MADDPG'],     linewidth=2.0)
ax.plot(training_days, MASAC,      label='MASAC',       color=colors['MASAC'],      linewidth=2.0)
ax.plot(training_days, MADDPG_Lag, label='MADDPG-Lag',  color=colors['MADDPG_Lag'], linewidth=2.0)
ax.plot(training_days, MASAC_Lag,  label='MASAC-Lag',   color=colors['MASAC_Lag'],  linewidth=2.0)
ax.plot(training_days, IPO,        label='IPO',         color=colors['IPO'],        linewidth=2.0)
ax.plot(training_days, P3O,        label='P3O',         color=colors['P3O'],        linewidth=2.0)
ax.plot(training_days, DCMASAC,    label='DCMADRL',     color=colors['DCMASAC'],    linewidth=2.0)

ax.set_xlim([0, 24])
ax.set_xticks(np.linspace(0, 24, 7))

min_y = min(RBC.min(), MADDPG.min(), MASAC.min(), MADDPG_Lag.min(), MASAC_Lag.min(),
            IPO.min(), P3O.min(), DCMASAC.min(), 1)
max_y = max(RBC.max(), MADDPG.max(), MASAC.max(), MADDPG_Lag.max(), MASAC_Lag.max(),
            IPO.max(), P3O.max(), DCMASAC.max(), 4)
ax.set_ylim([min_y, max_y])

ax.set_xlabel('Hours (h)', fontsize=34)
ax.set_ylabel('Total HVAC Energy Consumption (kWh)', fontsize=34)
ax.tick_params(axis='both', labelsize=28)
ax.legend(loc='upper right', fontsize=28)
ax.grid(True)

# add shading after limits are set
add_shading(ax, training_days, occupy, label="Occupied periods")

plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print("Energy_day.pdf saved.")
plt.show()
