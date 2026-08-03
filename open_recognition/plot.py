import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 1. 配置你的专属字体路径
font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
font_cn = font_manager.FontProperties(fname=font_path, size=12)
font_cn_title = font_manager.FontProperties(fname=font_path, size=16, weight='bold')

def plot_reconstruction_distribution():
    # 模拟重构误差数据（根据你的实验情况，已知类通常集中在 0 附近，未知类呈长尾分布）
    np.random.seed(42)
    known_errors = np.random.gamma(shape=2, scale=0.1, size=1000)      # 已知类：误差小且集中
    unknown_errors = np.random.gamma(shape=5, scale=0.2, size=800)     # 未知类：误差显著偏大

    # 创建画布
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_style("whitegrid")

    # 绘制已知类分布 (Known Classes)
    sns.kdeplot(known_errors, fill=True, color="#2ecc71", 
                label="已知编码体制 (Known Classes)", alpha=0.5, linewidth=2)
    
    # 绘制未知类分布 (Unknown Classes)
    sns.kdeplot(unknown_errors, fill=True, color="#e74c3c", 
                label="未知/干扰信号 (Unknown Classes)", alpha=0.5, linewidth=2)

    # 绘制决策阈值线
    threshold = 0.65
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5)
    
    # 使用你习惯的方式标注文字
    plt.text(threshold + 0.05, 1.2, f"拒识阈值 $\\tau={threshold}$", 
             fontproperties=font_cn, color="black")
    
    plt.annotate('已知类重构：\n特征自愈', xy=(0.2, 2.0), xytext=(0.4, 3.5),
                 fontproperties=font_cn, arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # 设置坐标轴标签（统一使用你的 font_cn）
    plt.title("基于重构误差的开集识别（OSR）原理示意图", fontproperties=font_cn_title, pad=20)
    plt.xlabel("重构损失值 ($L_{RE}$)", fontproperties=font_cn)
    plt.ylabel("概率密度 (Density)", fontproperties=font_cn)
    
    # 图例处理
    legend = plt.legend(prop=font_cn, loc='upper right')
    
    # 细节微调
    plt.xlim(0, 2.0)
    plt.tight_layout()
    
    # 保存为矢量图，方便放入大论文
    plt.savefig("reconstruction_dist.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_reconstruction_distribution()