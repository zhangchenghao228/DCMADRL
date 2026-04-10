import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 字体
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 15

# 数据
file_path = "cost.xlsx"
excel_data = pd.ExcelFile(file_path)

# 颜色
colors = {
    'MASAC': 'purple',
    'CMADRL': 'green',
    'DCMADRL-REWARD': 'deeppink',
    'DCMADRL-COST': 'firebrick',
    'DCDRL': 'orange',
    'DCMADRL': 'blue',
}

# 主图
fig, ax = plt.subplots(figsize=(12, 6))

for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals  = np.nanstd(data,  axis=1)
    episodes  = np.arange(len(mean_vals))

    ax.plot(episodes, mean_vals, label=sheet,
            color=colors.get(sheet, None), linewidth=2.2, zorder=2)
    ax.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                    color=colors.get(sheet, None), alpha=0.20, zorder=1)

# 轴样式
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('Episodes', fontsize=26)
ax.set_ylabel('Cumulative Costs', fontsize=26)
ax.set_xlim(0, 300)
ax.grid(True, zorder=0)

# 图例：轴内顶部，两排
ax.legend(
    ncol=3,                      # 3列 -> 两排
    loc='upper center',
    bbox_to_anchor=(0.5, 1.025),  # 放在绘图区内顶端
    fontsize=18,
    frameon=True, fancybox=True,
    columnspacing=1.2, handlelength=2.0,
    borderaxespad=0.6
)

# 放大子图：右移，保持与原来相近的尺寸(65%)
axins = inset_axes(
    ax,
    width="65%", height="65%",
    loc='upper right',
    bbox_to_anchor=(0.40, 0.20, 0.55, 0.55),  # 调第1个数(0.60)可继续右移
    bbox_transform=ax.transAxes,
    borderpad=0.6
)
axins.set_facecolor('white')
axins.patch.set_alpha(0.95)  # 白底半透明以减少遮挡
axins.set_zorder(3)

# 子图内容
for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals  = np.nanstd(data,  axis=1)
    episodes  = np.arange(len(mean_vals))

    axins.plot(episodes, mean_vals, color=colors.get(sheet, None), linewidth=2.0)
    axins.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                       color=colors.get(sheet, None), alpha=0.20)
# 子图范围（收敛尾段）
axins.tick_params(axis='both', which='major', labelsize=12)
axins.set_xlim(250, 300)
axins.set_ylim(600, 1300)  # 视数据微调

# 连接线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.5)

plt.tight_layout()
plt.savefig("Cost_comparison.pdf", dpi=300, bbox_inches='tight')
plt.show()
