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
file_path = "cost.xlsx"  # 替换为实际路径
excel_data = pd.ExcelFile(file_path)

# 设置方法对应颜色
colors = {
    'MADDPG': 'orange',
    'MASAC': 'purple',
    'MADDPG-Lag': 'green',
    'MASAC-Lag': 'cyan',
    'IPO': 'deeppink',
    'P3O': 'firebrick',
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

    ax.plot(episodes, mean_vals, label=sheet, color=colors.get(sheet, None), linewidth=2.0)
    ax.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                    color=colors.get(sheet, None), alpha=0.2)

# 设置主图格式
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('Episodes', fontsize=26)
ax.set_ylabel('Cumulative Costs', fontsize=26)
ax.set_xlim(0, 300)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=14)  # 图例放底部
# ax.set_xlim(0, 300)
ax.legend(fontsize=18)
ax.grid(True)

ax.grid(True)

# ==== 添加放大子图 ====
axins = inset_axes(
    ax,
    width="65%", height="65%",    # 子图尺寸
    loc='upper right',              # 相对于主图的位置
    bbox_to_anchor=(0.18, 0.25, 0.55, 0.55),  # 子图具体偏移位置
    bbox_transform=ax.transAxes
)

# 绘制子图内容（Episode 260–300）
for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals = np.nanstd(data, axis=1)
    episodes = np.arange(len(mean_vals))

    axins.plot(episodes, mean_vals, color=colors.get(sheet, None), linewidth=2)
    axins.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                       color=colors.get(sheet, None), alpha=0.2)

# 放大收敛尾段（Episode 280-300），并且拉开 reward 尺度
axins.tick_params(axis='both', which='major', labelsize=12)
axins.set_xlim(250, 300)
axins.set_ylim(600, 1700)  # ⚠️ 根据你实际 cost 数据手动调整这两个值
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.5)

# 输出图像
plt.tight_layout()
try:
    plt.savefig("Cost_comparison_with_inset.pdf", dpi=300, bbox_inches='tight')
    print("✅ PDF 成功保存为 Cost_comparison_with_inset.pdf")
except ImportError:
    print("⚠️ PDF保存失败，改为PNG格式导出")
    plt.savefig("Cost_comparison_with_inset.png", dpi=300, bbox_inches='tight')

plt.show()
