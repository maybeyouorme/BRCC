import torch
import os

class Config:
    def __init__(self):
        # 硬件设置
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.gpu_id = "1"

        # --- 核心任务参数 ---
        self.seq_len = 8192
        self.num_coarse_classes = 5

        # --- 信号预处理参数 ---
        self.n_fft = 256
        self.hop_length = 128

        # --- CNN 分支参数 ---
        self.cnn_base_channels = 64
        self.cnn_layers = [2, 2, 2]
        self.cnn_out = 128

        # --- LSTM 分支参数 ---
        self.lstm_frame_size = 128
        self.lstm_hidden_size = 128
        self.lstm_layers = 2
        self.lstm_out = 128

        self.spec_out = 64

        # --- 融合特征参数 ---
        self.hidden_shared = 128
        self.dropout = 0.3

        # --- 训练参数 ---
        self.epochs = 200
        self.batch_size = 64
        self.learning_rate = 5e-4

        # --- 路径 ---
        self.save_dir = "./checkpoints1"

        self.model_save_name = "best_model.pth"
        self.model_save_path = os.path.join(self.save_dir, self.model_save_name)

        self.results_dir = os.path.join(self.save_dir, 'results')
        self.cm_dir = os.path.join(self.results_dir, 'confusion_matrices')
        self.visualization_path = os.path.join(self.results_dir, 'tsne_feature_visualization')
        self.acc_dir = os.path.join(self.results_dir, 'accuracy_curves')
        self.eval_data_path = '/data/project_lyb/Open data/val_data_for_eval.pkl'

        # OSR 评估参数
        self.osr_confidence_level = 0.99

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.acc_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)
