# config.py

import torch
import os

class Config:
    def __init__(self):
        # 硬件设置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpu_id = "0"
        
        # --- 核心任务参数 (MT-OSR) ---
        # self.embedding_size = 128  # <--- 已被新的详细参数取代，不再使用
        self.seq_len = 8192             # LLR序列长度 (需与数据集生成保持一致)
        
        # 已知类别数 (会在 main.py 中根据数据动态更新)
        self.num_coarse_classes = 5     # 粗类别数 (例如 Conv, Turbo, BCH, LDPC, Polar)
        # 细类别数 (所有已知码的参数组合总数，需手动/动态确定)
        self.num_fine_classes = 15      
        
        # =============================================================================
        # --- 新增：模型架构参数 (对应 MultiTaskOSRNet) ---
        # =============================================================================
        
        # --- 1. 信号预处理参数 (SignalPreprocessor) ---
        self.n_fft = 256           # FFT点数 (用于 SignalPreprocessor 的频域特征)
        self.hop_length = 128      # STFT步长 (用于 SignalPreprocessor 的频谱)
        
        # --- 2. CNN (Deep ResNet with MS-SE) 分支参数 ---
        self.cnn_base_channels = 64     # ResNet 初始通道数
        self.cnn_layers = [2, 2, 2]     # ResNet 各阶段残差块数量 [Stage1, Stage2, Stage3]
        self.cnn_out = 128              # CNN 最终输出特征维度
        
        # --- 3. LSTM 分支参数 ---
        self.lstm_frame_size = 128      # LSTM 将信号分帧的大小 (用于帧投影)
        self.lstm_hidden_size = 128     # Bi-LSTM 隐藏层维度
        self.lstm_layers = 2            # Bi-LSTM 层数
        self.lstm_out = 128             # LSTM 最终输出特征维度

        self.spec_out = 64
        
        # --- 4. 融合 (Fusion) 与共享特征参数 ---
        # 融合维度: self.cnn_out + self.lstm_out = 128 + 128 = 256 (输入到 fc_fuse)
        self.hidden_shared = 128        # 融合后的共享特征维度 (作为分类头和 Decoder 的输入)
        self.dropout = 0.3              # 融合层 Dropout 比率

        # --- 训练参数 ---
        self.epochs = 200 
        self.batch_size = 64
        self.learning_rate = 5e-4

        # --- 多任务损失权重 ---
        self.lambda_coarse = 0.5        # 粗分类损失权重
        self.lambda_fine = 2.0          # 细分类损失权重

        # 无监督正则化参数 (L_total = L_sup + gamma * L_unsup)
        self.use_autoencoder = True     # 是否使用 Autoencoder 进行正则化
        self.gamma_recon = 1.0         # 重构损失权重 (L_recon)
        self.lambda_center = 0.5        # 中心损失权重 (L_center)

        self.lambda_contrast = 2.5  #监督对比学习损失权重
        # --- 路径 ---
        self.save_dir = "./checkpoits4"
        #1:原始代码
        #2:加窗，去掉一个mean:OSR混淆矩阵有略微下降，AUC值也下降了0.03
        #3：加窗：
        self.model_save_name = "best_model.pth"
        self.model_save_path = os.path.join(self.save_dir, self.model_save_name)

        self.results_dir = os.path.join(self.save_dir, 'results')
        self.cm_dir = os.path.join(self.results_dir, 'confusion_matrices')
        self.visualization_path = os.path.join(self.results_dir, 'tsne_feature_visualization')
        self.acc_dir = os.path.join(self.results_dir, 'accuracy_curves')
        self.eval_data_path = '/data/project_lyb/Open data/val_data_for_eval.pkl'
        
        # OSR 评估参数
        self.osr_confidence_level = 0.90 # OSR 拒绝阈值选择的置信度 (例如：Known TPR=95%)