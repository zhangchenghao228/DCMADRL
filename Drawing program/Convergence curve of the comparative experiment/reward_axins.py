import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 设置字体与图形风格
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 15
sns.set(style="whitegrid")

# 加载 Excel 文件
file_path = "reward.xlsx"  # 替换为实际路径
excel_data = pd.ExcelFile(file_path)

# 设置方法对应颜色
colors = {
    'MADDPG': 'orange',
    'MASAC': 'purple',
    'MADDPG-Lag': 'green',
    'MASAC-Lag': 'cyan',
    'IPO': 'deeppink',
    'P3O': 'firebrick',   # 修改这里
    'DCMADRL': 'blue',
}

# 准备主图
fig, ax = plt.subplots(figsize=(12, 6))

# 主图绘制
for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals = np.nanstd(data, axis=1)
    episodes = np.arange(len(mean_vals))

    # 主图均值
    ax.plot(episodes, mean_vals, label=sheet, color=colors.get(sheet, None), linewidth=2.0)
    ax.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                    color=colors.get(sheet, None), alpha=0.2)

# 设置主图标签
# 设置刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('Episodes', fontsize=26)
ax.set_ylabel('Cumulative Rewards', fontsize=26)
ax.set_xlim(0, 300)
ax.legend(fontsize=18)
ax.grid(True)

# ==== 添加放大子图 ====
# 创建 inset axes

axins = inset_axes(
    ax,
    width="70%", height="70%",  # 放大小图尺寸
    loc='upper right',
    bbox_to_anchor=(0.18, 0.15, 0.55, 0.55),  # 调整位置，避免遮挡主图
    bbox_transform=ax.transAxes
)


# 绘制 inset 区域的图形（仅前20个 episodes）
for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals = np.nanstd(data, axis=1)
    episodes = np.arange(len(mean_vals))

    axins.plot(episodes, mean_vals, color=colors.get(sheet, None), linewidth=2)
    axins.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                       color=colors.get(sheet, None), alpha=0.2)

# 设置 inset 显示范围
# 放大收敛尾段（Episode 280-300），并且拉开 reward 尺度
# axins.tick_params(axis='both', which='major', labelsize=10)
axins.tick_params(axis='both', which='major', labelsize=12)
axins.set_xlim(250, 300)
axins.set_ylim(-4700, -4200)  # 原来是 -4500 到 -4000，现在缩紧以拉开 reward 间隔


# 添加连接线，标出放大区域
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# 添加连接线，标出放大区域
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.5)

# 输出图像
plt.tight_layout()

try:
    plt.savefig("Reward_comparison_inset.pdf", dpi=300, bbox_inches='tight')
    print("✅ 带放大图的 PDF 成功保存为 Reward_comparison_inset.pdf")
except ImportError:
    print("⚠️ PDF保存失败，改为PNG格式导出")
    plt.savefig("Reward_comparison_inset.png", dpi=300, bbox_inches='tight')

plt.show()
