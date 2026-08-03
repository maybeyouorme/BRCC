import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. 预处理模块
# =============================================================================
class SignalPreprocessor(nn.Module):
    def __init__(self, use_smoothing=False, smoothing_kernel=3, n_fft=256, hop_length=128):
        super().__init__()
        self.use_smoothing = use_smoothing
        if use_smoothing:
            self.smooth = nn.Conv1d(1, 1, kernel_size=smoothing_kernel, padding=smoothing_kernel // 2, bias=False)
            kernel = torch.ones(1, 1, smoothing_kernel) / smoothing_kernel
            self.smooth.weight.data.copy_(kernel)
            self.smooth.weight.requires_grad = False
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        t = x
        # 归一化
        mean = t.mean(dim=2, keepdim=True)
        std = t.std(dim=2, keepdim=True) + 1e-9
        t = (t - mean) / std

        if self.use_smoothing:
            t = self.smooth(t)

        # 频域特征
        freq_complex = torch.fft.rfft(t.squeeze(1), n=L)
        freq_mag = torch.abs(freq_complex)

        # 时频图特征
        spec = torch.stft(t.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        spec = torch.abs(spec)

        return {"time": t, "freq": freq_mag, "spec": spec}

class MultiModalExtractor(nn.Module):
    def __init__(self, sample_len=8192, n_fft=256, hop_length=128):
        super().__init__()
        self.sample_len = sample_len
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, preproc_out):
        return {
            "time": preproc_out["time"],
            "freq": preproc_out["freq"],
            "spec": preproc_out["spec"]
        }

# =============================================================================
# 2. 编码器组件 (ResNet + Bi-LSTM + SpecCNN)
# =============================================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)

class DeepResNet1DEncoder(nn.Module):
    def __init__(self, in_ch=1, base_channels=64, layers=[2, 2, 2], out_dim=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.in_channels = base_channels
        self.stage1 = self._make_layer(ResidualBlock1D, base_channels, layers[0], stride=1)
        self.stage2 = self._make_layer(ResidualBlock1D, base_channels * 2, layers[1], stride=2) 
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 2, out_dim)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )
        layers = [block(self.in_channels, channels, stride=stride, downsample=downsample)]
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x_time):
        out = self.in_conv(x_time)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

class LSTMEncoder(nn.Module):
    def __init__(self, input_len=8192, frame_size=128, hidden_size=128, num_layers=2, out_dim=128):
        super().__init__()
        self.frame_size = frame_size
        self.seq_len = math.ceil(input_len / frame_size)
        self.frame_proj = nn.Linear(frame_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_fc = nn.Linear(hidden_size * 2, out_dim) 

    def forward(self, x_time):
        B, _, L = x_time.shape
        seq = x_time[:, 0, :]
        pad_len = self.seq_len * self.frame_size - L
        if pad_len > 0:
            seq = F.pad(seq, (0, pad_len))
        frames = seq.view(B, self.seq_len, self.frame_size)
        emb = self.frame_proj(frames)
        out, _ = self.lstm(emb) 
        feat = out.mean(dim=1)
        return self.out_fc(feat)

class SpecEncoder(nn.Module):
    def __init__(self, out_dim=128, base_channels=32):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_out = nn.Linear(base_channels * 4, out_dim)

    def forward(self, x_spec):
        x_spec = x_spec.unsqueeze(1)
        out = self.conv_in(x_spec)
        out = self.conv_blocks(out)
        out = out.squeeze(-1).squeeze(-1)
        return self.fc_out(out)

# =============================================================================
# 4. 完整的多任务开集识别网络
# =============================================================================
class MultiTaskOSRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sample_len = cfg.seq_len
        self.hidden_shared = 256
        self.use_ae = getattr(cfg, "use_autoencoder", True)

        # 1. 基础组件
        self.preproc = SignalPreprocessor(use_smoothing=True, n_fft=256, hop_length=128)
        self.extractor = MultiModalExtractor(sample_len=self.sample_len)

        # 2. 编码分支
        self.cnn = DeepResNet1DEncoder(base_channels=64, out_dim=128)
        self.lstm = LSTMEncoder(input_len=self.sample_len, out_dim=128)
        self.spec_encoder = SpecEncoder(out_dim=64)

        # 3. 融合层
        self.fc_fuse = nn.Sequential(
            nn.Linear(128 + 128 + 64, self.hidden_shared),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 4. 分类头
        self.coarse_head = nn.Linear(self.hidden_shared, cfg.num_coarse_classes)
        self.fine_head = nn.Linear(self.hidden_shared, cfg.num_fine_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # 编码过程
        pre = self.preproc(x)
        modal = self.extractor(pre)
        
        f_cnn = self.cnn(modal["time"])
        f_lstm = self.lstm(modal["time"])
        f_spec = self.spec_encoder(modal["spec"])
        
        fused = torch.cat([f_cnn, f_lstm, f_spec], dim=1)
        shared = self.fc_fuse(fused)
        
        # 分类输出
        c_out = self.coarse_head(shared)
        f_out = self.fine_head(shared)
        
        
        return c_out, f_out

    def forward_features(self, x):
        """专用于可视化：返回融合后的特征向量"""
        if x.dim() == 2: x = x.unsqueeze(1)
        pre = self.preproc(x)
        modal = self.extractor(pre)
        f_cnn = self.cnn(modal["time"])
        f_lstm = self.lstm(modal["time"])
        f_spec = self.spec_encoder(modal["spec"])
        fused = torch.cat([f_cnn, f_lstm, f_spec], dim=1)
        return self.fc_fuse(fused)
