import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置字体与图形风格
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 15
sns.set(style="whitegrid")

# 加载 Excel 文件
file_path = "reward.xlsx"  # 替换为实际路径
excel_data = pd.ExcelFile(file_path)

# 设置方法对应颜色（可自行修改）
colors = {
    'MASAC-IPO-D': 'orange',
    'MASAC-IPO-DR-C': 'purple',
    'SAC-IPO-D': 'green',
    'MASAC-IPO': 'cyan'
}

# 创建图形
plt.figure(figsize=(10, 5))

# 遍历每个 sheet（每种方法）
for sheet in excel_data.sheet_names:
    df = excel_data.parse(sheet)
    data = df.to_numpy()
    mean_vals = np.nanmean(data, axis=1)
    std_vals = np.nanstd(data, axis=1)
    episodes = np.arange(len(mean_vals))

    # 绘制均值曲线
    plt.plot(episodes, mean_vals, label=sheet, color=colors.get(sheet, None), linewidth=2.2)

    # 绘制标准差阴影
    plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                     color=colors.get(sheet, None), alpha=0.2)

# 图形标签和格式设置
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Cumulative Reward', fontsize=18)
plt.xlim(0, 300)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

# 安全保存图像
try:
    plt.savefig("Reward_comparison.pdf", dpi=300, bbox_inches='tight')
    print("✅ PDF 成功保存为 Reward_comparison.pdf")
except ImportError:
    print("⚠️ PDF保存失败，改为PNG格式导出")
    plt.savefig("Reward_comparison.png", dpi=300, bbox_inches='tight')

plt.show()
