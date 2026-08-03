import torch
import os


class Config:
    def __init__(self):
        # --- 硬件设置 ---
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpu_id = "0"

        # --- 核心任务参数 ---
        self.seq_len = 8192                     # LLR 序列长度
        self.num_coarse_classes = 6             # 已知粗类别数（会在 main.py 中动态更新）

        # =========================================================================
        # 模型架构参数（与 Softmax / OpenMax 保持一致的多分支融合网络）
        # =========================================================================

        # --- 信号预处理参数 ---
        self.n_fft = 256
        self.hop_length = 128

        # --- CNN (Deep ResNet) 分支参数 ---
        self.cnn_base_channels = 64
        self.cnn_layers = [2, 2, 2]
        self.cnn_out = 128

        # --- LSTM 分支参数 ---
        self.lstm_frame_size = 128
        self.lstm_hidden_size = 128
        self.lstm_layers = 2
        self.lstm_out = 128

        self.spec_out = 64

        # --- 融合与共享特征参数 ---
        self.hidden_shared = 128
        self.dropout = 0.3

        # =========================================================================
        # ARPL / RPL 损失函数参数
        # =========================================================================
        self.loss_type = 'ARPLoss'              # 可选: 'ARPLoss', 'RPLoss', 'Softmax'
        self.weight_pl = 0.1                    # 原型损失权重 (center loss / margin loss)
        self.temp = 1.0                         # 温度参数 (用于 cross-entropy 缩放)
        self.num_centers = 1                    # 每个类别的原型/中心数 (RPLoss 用)

        # =========================================================================
        # 训练参数
        # =========================================================================
        self.epochs = 200
        self.batch_size = 64
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5
        self.lr_milestones = [80, 140]         # 学习率衰减节点
        self.lr_gamma = 0.1                    # 学习率衰减因子

        # --- 早停 (Early Stopping) ---
        self.early_stopping_patience = 30
        self.early_stopping_delta = 1e-4

        # --- 路径 ---
        self.save_dir = "./checkpoints4"        # 与实验编号对应
        self.model_save_name = "best_model.pth"
        self.model_save_path = os.path.join(self.save_dir, self.model_save_name)

        self.results_dir = os.path.join(self.save_dir, 'results')
        self.cm_dir = os.path.join(self.results_dir, 'confusion_matrices')
        self.acc_dir = os.path.join(self.results_dir, 'accuracy_curves')
        self.eval_data_path = '/data/project_lyb/Open data/val_data_for_eval.pkl'

        # --- OSR 评估参数 ---
        self.osr_confidence_level = 0.95

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.acc_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)
