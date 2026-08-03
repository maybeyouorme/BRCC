"""
ARPLNet: 1D 信号开集识别网络 (适配 ARPL/RPL 损失)

模型架构与 Softmax / OpenMax 对比实验保持一致：
  SignalPreprocessor → MultiModalExtractor
      ├── DeepResNet1DEncoder (CNN 分支)
      ├── LSTMEncoder (时序分支)
      └── SpecEncoder (频谱分支)
          ↓ 特征融合
      Bottleneck (共享嵌入)
          ↓
      Classifier (不带偏置的 Linear, 用于 Softmax 基线)

在前向传播中返回 (embedding, logits) 元组，以供 ARPLoss / RPLoss 使用。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 信号预处理与多模态特征提取 (与 Softmax / OpenMax 一致)
# =========================================================================

class SignalPreprocessor(nn.Module):
    def __init__(self, use_smoothing=False, smoothing_kernel=3, n_fft=256, hop_length=128):
        super().__init__()
        self.use_smoothing = use_smoothing
        if use_smoothing:
            self.smooth = nn.Conv1d(1, 1, kernel_size=smoothing_kernel,
                                    padding=smoothing_kernel // 2, bias=False)
            kernel = torch.ones(1, 1, smoothing_kernel) / smoothing_kernel
            self.smooth.weight.data.copy_(kernel)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                      # [B, 1, L]
        t = x
        mean = t.mean(dim=2, keepdim=True)
        std = t.std(dim=2, keepdim=True) + 1e-9
        t = (t - mean) / std
        if self.use_smoothing:
            t = self.smooth(t)

        # 短时傅里叶变换 (STFT) → 频谱图
        window = torch.hann_window(self.n_fft).to(t.device)
        spec = torch.stft(t.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          return_complex=True, window=window)
        spec = torch.abs(spec)                       # [B, freq_bins, time_bins]
        return {"time": t, "spec": spec}


class MultiModalExtractor(nn.Module):
    def __init__(self, sample_len=8192, n_fft=256, hop_length=128):
        super().__init__()
        self.sample_len = sample_len
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, preproc_out):
        return preproc_out                           # 透传，保留扩展接口


# =========================================================================
# CNN 分支: 1D 深度残差网络编码器
# =========================================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)


class DeepResNet1DEncoder(nn.Module):
    """1D ResNet 编码器: [B, 1, L] → [B, out_dim]"""
    def __init__(self, in_ch=1, base_channels=64, layers=[2, 2, 2], out_dim=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),                              # [B, 64, L/2]
        )
        self.in_channels = base_channels
        self.stage1 = self._make_layer(ResidualBlock1D, base_channels, layers[0], stride=1)
        self.stage2 = self._make_layer(ResidualBlock1D, base_channels * 2, layers[1], stride=2)
        final_channels = base_channels * 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(final_channels, out_dim)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x_time):
        if x_time.dim() == 2:
            x_time = x_time.unsqueeze(1)
        out = self.in_conv(x_time)
        out = self.stage1(out)
        feature_map = self.stage2(out)
        out = self.pool(feature_map).squeeze(-1)
        embedding = self.fc(out)
        return embedding


# =========================================================================
# LSTM 分支: 帧级序列编码器
# =========================================================================

class LSTMEncoder(nn.Module):
    """LSTM 编码器: [B, 1, L] → [B, out_dim]"""
    def __init__(self, input_len=8192, frame_size=128, hidden_size=128,
                 num_layers=2, out_dim=128):
        super().__init__()
        self.frame_size = frame_size
        self.seq_len = math.ceil(input_len / frame_size)
        self.frame_proj = nn.Linear(frame_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.out_fc = nn.Linear(hidden_size * 2, out_dim)

    def forward(self, x_time):
        if x_time.dim() == 2:
            x_time = x_time.unsqueeze(1)
        B, C, L = x_time.shape
        seq = x_time[:, 0, :]
        pad_len = self.seq_len * self.frame_size - L
        if pad_len > 0:
            seq = F.pad(seq, (0, pad_len))
        frames = seq.view(B, self.seq_len, self.frame_size)     # [B, seq_len, frame_size]
        emb = self.frame_proj(frames)
        out, _ = self.lstm(emb)                                  # [B, seq_len, 2*hidden]
        feat = out.mean(dim=1)                                   # [B, 2*hidden]
        feat = self.out_fc(feat)                                 # [B, out_dim]
        return feat


# =========================================================================
# 频谱分支: 2D 卷积编码器 (STFT 频谱图)
# =========================================================================

class SpecEncoder(nn.Module):
    """频谱编码器: [B, freq_bins, time_bins] → [B, out_dim]"""
    def __init__(self, out_dim=64, base_channels=32):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_out = nn.Linear(base_channels * 4, out_dim)

    def forward(self, x_spec):
        x_spec = x_spec.unsqueeze(1)                             # [B, 1, freq, time]
        out = self.conv_in(x_spec)
        out = self.conv_blocks(out)
        out = out.squeeze(-1).squeeze(-1)                        # [B, base*4]
        return self.fc_out(out)


# =========================================================================
# ARPLNet: 主网络 (返回 embedding, logits 供 ARPLoss 使用)
# =========================================================================

class ARPLNet(nn.Module):
    """
    ARPL 开集识别网络。

    前向传播返回 (embedding, logits) 元组，与 ARPL-master 中 classifier32 的
    return_feature=True 行为一致。embedding 供 ARPLoss 做距离度量学习，
    logits 供 Softmax 基线使用（ARPLoss 内部会重新计算自己的 logits）。

    模型结构与 SoftmaxOSRNet / OpenMaxOSRNet 保持完全一致，仅修改输出格式。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_coarse = cfg.num_coarse_classes
        sample_len = cfg.seq_len

        # 预处理与特征提取
        self.preproc = SignalPreprocessor(use_smoothing=True,
                                          n_fft=cfg.n_fft,
                                          hop_length=cfg.hop_length)
        self.extractor = MultiModalExtractor(sample_len=sample_len,
                                             n_fft=cfg.n_fft,
                                             hop_length=cfg.hop_length)

        # 三分支编码器
        self.cnn = DeepResNet1DEncoder(in_ch=1,
                                       base_channels=cfg.cnn_base_channels,
                                       layers=cfg.cnn_layers,
                                       out_dim=cfg.cnn_out)
        self.lstm = LSTMEncoder(input_len=sample_len,
                                frame_size=cfg.lstm_frame_size,
                                hidden_size=cfg.lstm_hidden_size,
                                num_layers=cfg.lstm_layers,
                                out_dim=cfg.lstm_out)
        self.spec_encoder = SpecEncoder(out_dim=cfg.spec_out)

        # 特征融合
        fused_dim = cfg.cnn_out + cfg.lstm_out + cfg.spec_out
        self.bottleneck_base = nn.Sequential(
            nn.Linear(fused_dim, cfg.hidden_shared * 2),
            nn.BatchNorm1d(cfg.hidden_shared * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cfg.hidden_shared * 2, cfg.hidden_shared),
            nn.BatchNorm1d(cfg.hidden_shared),
            nn.ReLU(inplace=True),
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.shared_embedding_size = cfg.hidden_shared

        # 分类头（Softmax 基线用，ARPLoss 内部会重新计算自己的 logits）
        self.classifier = nn.Linear(self.shared_embedding_size, num_coarse, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: 输入信号 [B, L]
        Returns:
            embedding: 共享嵌入特征 [B, hidden_shared]，供 ARPLoss 使用
            logits:    分类 logits [B, num_classes]，供 Softmax 基线使用
        """
        # 1) 预处理
        pre = self.preproc(x)
        modal = self.extractor(pre)
        t_in = modal["time"]

        # 2) 分支特征提取
        feat_cnn = self.cnn(t_in)
        feat_lstm = self.lstm(t_in)
        feat_spec = self.spec_encoder(modal["spec"])

        # 3) 特征融合
        fused = torch.cat([feat_cnn, feat_lstm, feat_spec], dim=1)
        embedding = self.bottleneck_base(fused)

        # 4) 分类头
        embedding_dropped = self.drop(embedding)
        embedding_norm = F.normalize(embedding_dropped, p=2, dim=1)
        logits = self.classifier(embedding_norm)

        # 返回 (embedding, logits) 以兼容 ARPLoss 接口
        return embedding, logits
