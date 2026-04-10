import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 字体与风格
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 15
sns.set(style="whitegrid")

# 数据
file_path = "reward.xlsx"  # 替换为实际路径
excel_data = pd.ExcelFile(file_path)

# 颜色
colors = {
    'MASAC': 'purple',
    'CMADRL': 'green',
    'DCMADRL-REWARD': 'deeppink',
    'DCMADRL-COST': 'firebrick',
    # 'DCDRL': 'orange',
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
            color=colors.get(sheet, None), linewidth=2.0, zorder=2)
    ax.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                    color=colors.get(sheet, None), alpha=0.2, zorder=1)

# 轴样式
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('Episodes', fontsize=26)
ax.set_ylabel('Cumulative Rewards', fontsize=26)
ax.set_xlim(0, 300)
ax.grid(True, zorder=0)

# 图例：两排，放轴内顶部，避免出界
ax.legend(
    ncol=3,                      # 两排 -> 3列 x 2行（6个方法）
    loc='upper center',
    bbox_to_anchor=(0.5, 1.025),  # 轴内顶部；要更高调成 1.00~1.05
    fontsize=18,
    frameon=True, fancybox=True,
    columnspacing=1.2, handlelength=2.0,
    borderaxespad=0.6
)

# 放大子图（inset）：与原来相近大小并右移
axins = inset_axes(
    ax,
    width="65%", height="65%",
    loc='upper right',
    bbox_to_anchor=(0.25, 0.10, 0.55, 0.55),  # (x,y,w,h) 相对主轴坐标；调 x/y 微调位置
    bbox_transform=ax.transAxes,
    borderpad=0.6
)
axins.set_facecolor('white')
axins.patch.set_alpha(0.95)
axins.set_zorder(3)

# 子图内容
for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals  = np.nanstd(data,  axis=1)
    episodes  = np.arange(len(mean_vals))

    axins.plot(episodes, mean_vals, color=colors.get(sheet, None), linewidth=2)
    axins.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                       color=colors.get(sheet, None), alpha=0.2)

# 子图范围（收敛尾段）
axins.tick_params(axis='both', which='major', labelsize=12)
axins.set_xlim(250, 300)
axins.set_ylim(-4350, -4220)  # 视数据微调

# 连接线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.5)

plt.tight_layout()
try:
    plt.savefig("Reward_comparison.pdf", dpi=300, bbox_inches='tight')
    print("PDF 已保存：Reward_comparison.pdf")
except Exception:
    plt.savefig("Reward_comparison.png", dpi=300, bbox_inches='tight')
    print("已改存 PNG：Reward_comparison.png")

plt.show()
