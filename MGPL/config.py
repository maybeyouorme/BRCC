import torch
import os


class Config:
    def __init__(self):
        # --- 硬件设置 ---
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.gpu_id = "1"

        # --- 核心任务参数 ---
        self.seq_len = 8192                     # LLR 序列长度
        self.num_coarse_classes = 6             # 已知粗类别数（会在 main.py 中动态更新）

        # =========================================================================
        # MGPL 模型参数
        # =========================================================================

        # --- VAE 编码器-解码器架构参数 ---
        self.base_channels = 64                 # ResNet 初始通道数
        self.cnn_layers = [2, 2, 2, 2]          # 各阶段残差块数量
        self.latent_dim = 128                   # VAE 潜变量维度

        # --- MGPL 原型参数 ---
        self.n_sub_prototypes = 3               # 每个类别的子原型数量（OSR 时可尝试 5~8）
        self.temp_inter = 0.1                   # 类间温度参数（LSE 温度，越小对最近原型越聚焦）
        self.temp_intra = 1.0                   # 类内温度参数（子原型分配 softmax 温度，越小分配越硬）

        # --- 损失平衡参数 ---
        # total_loss = lamda * (rec + kld + ent) + (1 - lamda) * dis
        #
        #   ⚠️  OSR 关键参数！当前 lamda=0.005 时生成性损失权重仅 0.5%，
        #      模型本质上是纯判别性训练，这是 OSCR 偏低的核心原因。
        #      推荐对 OSR 调大至 0.05 ~ 0.2。
        self.lamda = 0.05                      # 生成性约束 vs 判别性约束的平衡系数

        # --- 优化参数 ---
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.lr_milestones = [50, 80]           # 学习率衰减节点
        self.lr_gamma = 0.1                     # 学习率衰减因子
        self.clip_grad = True                   # 是否梯度裁剪
        self.max_norm = 10.0                    # 梯度裁剪最大范数
        self.prototype_reset_epochs = [40, 70]  # 原型重置的轮次

        # --- 早停 (Early Stopping) ---
        self.early_stopping_patience = 10        # 连续多少个epoch验证指标未改善即停止
        self.early_stopping_delta = 1e-4         # 视为改善的最小变化幅度
        self.early_stopping_metric = 'val_loss'  # 监控指标: 'val_loss' 或 'val_acc'

        # --- 路径 ---
        self.save_dir = "./checkpoints4"
        self.model_save_name = "best_model.pth"
        self.model_save_path = os.path.join(self.save_dir, self.model_save_name)

        self.results_dir = os.path.join(self.save_dir, 'results')
        self.cm_dir = os.path.join(self.results_dir, 'confusion_matrices')
        self.visualization_path = os.path.join(self.results_dir, 'tsne_feature_visualization')
        self.acc_dir = os.path.join(self.results_dir, 'accuracy_curves')
        self.eval_data_path = '/data/project_lyb/Open data/val_data_for_eval.pkl'

        # --- OSR 评估参数 ---
        self.osr_confidence_level = 0.95

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.acc_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)


# =========================================================================
# OSR 调优参考配置
# =========================================================================
#
# 当前 OSCR ≈ 0.78，偏低。核心原因是 lamda=0.005 使其几乎为纯判别性模型，
# 没有利用生成性约束来塑造潜空间结构，导致已知/未知样本的距离分布重叠严重。
#
# 按优先级推荐以下配置组合（替换上方对应值后重训即可）：
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 配置 A（推荐首选 — 温和改进）                                        │
# │   lamda            = 0.05     # 生成性权重 5%                       │
# │   n_sub_prototypes = 5        # 更多子原型捕获类内变化              │
# │   temp_intra       = 0.5      # 更硬的子原型分配                    │
# │   prototype_reset_epochs = [40, 70]  # 略早于 LR 衰减              │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 配置 B（激进 — 生成性与判别性并重）                                  │
# │   lamda            = 0.2      # 生成性权重 20%                      │
# │   n_sub_prototypes = 8        # 每个类 8 个子原型                   │
# │   temp_intra       = 0.3      # 更硬的子原型分配                    │
# │   temp_inter       = 0.05     # LSE 更聚焦最近原型                  │
# │   latent_dim       = 256      # 更大潜空间容量                     │
# │   prototype_reset_epochs = [40, 70]                                 │
# └─────────────────────────────────────────────────────────────────────┘
#
# 注意: 提高 lamda 后会增加生成性损失的权重，训练早期的 total loss 会升高
#       这是正常的。如果 validation loss 震荡大，可增加 warmup 或降低 lr。
# =========================================================================
