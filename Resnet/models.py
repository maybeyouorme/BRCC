import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 数据预处理（与 Softmax/OpenMax 共用）
# =============================================================================
class SignalPreprocessor(nn.Module):
    def __init__(self, use_smoothing=False, smoothing_kernel=3, n_fft=256, hop_length=128):
        super().__init__()
        self.use_smoothing = use_smoothing
        if use_smoothing:
            self.smooth = nn.Conv1d(1, 1, kernel_size=smoothing_kernel, padding=smoothing_kernel // 2, bias=False)
            kernel = torch.ones(1, 1, smoothing_kernel) / smoothing_kernel
            self.smooth.weight.data.copy_(kernel)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]
        raw_x = x.clone().detach()
        t = x
        mean = t.mean(dim=2, keepdim=True)
        std = t.std(dim=2, keepdim=True) + 1e-9
        t = (t - mean) / std
        if self.use_smoothing:
            t = self.smooth(t)
        freq_complex = torch.fft.rfft(t.squeeze(1), n=self.n_fft)
        freq_mag = torch.abs(freq_complex)  # [B, 129]
        window = torch.hann_window(256).to(t.device)
        spec = torch.stft(t.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          return_complex=True, window=window)
        spec = torch.abs(spec)  # [B, 频率轴(129), 时间轴(64)]
        return {"time": t, "freq": freq_mag, "spec": spec, "raw": raw_x}


class MultiModalExtractor(nn.Module):
    def __init__(self, sample_len=8192, n_fft=256, hop_length=128):
        super().__init__()
        self.sample_len = sample_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1
        self.time_bins = math.ceil(sample_len / hop_length)

    def forward(self, preproc_out):
        return preproc_out  # 透传


# =============================================================================
# ResNet1D 基础模块
# =============================================================================
class ResidualBlock1D(nn.Module):
    """Basic Residual Block for 1D (ResNet18/34 style)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
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


class Bottleneck1D(nn.Module):
    """Bottleneck Residual Block for 1D (ResNet50/101/152 style)"""
    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)


class ResNet1D(nn.Module):
    """
    标准 ResNet 的 1D 版本，用于时域信号特征提取。

    支持不同的深度配置：
        ResNet18:  layers=[2, 2, 2, 2], bottleneck=False
        ResNet34:  layers=[3, 4, 6, 3], bottleneck=False
        ResNet50:  layers=[3, 4, 6, 3], bottleneck=True
        ResNet101: layers=[3, 4, 23, 3], bottleneck=True
    """
    def __init__(self, in_channels=1, base_channels=64, layers=(2, 2, 2, 2),
                 use_bottleneck=False, out_dim=256):
        super().__init__()
        block_cls = Bottleneck1D if use_bottleneck else ResidualBlock1D
        self.base_channels = base_channels

        # 初始卷积 (适配 1D 信号)
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差阶段的输入通道数
        self.in_channels = base_channels

        # 四个残差阶段
        self.layer1 = self._make_layer(block_cls, base_channels,       layers[0], stride=1)
        self.layer2 = self._make_layer(block_cls, base_channels * 2,   layers[1], stride=2)
        self.layer3 = self._make_layer(block_cls, base_channels * 4,   layers[2], stride=2)
        self.layer4 = self._make_layer(block_cls, base_channels * 8,   layers[3], stride=2)

        # 输出投影
        final_channels = base_channels * 8
        if use_bottleneck:
            final_channels = final_channels * block_cls.expansion  # *4

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(final_channels, out_dim)

        self._init_weights()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        in_channels = self.in_channels

        if stride != 1 or in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )

        layers = []
        layers.append(block(in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, 1, L]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # [B, C, 1]
        x = x.squeeze(-1)            # [B, C]
        x = self.fc(x)               # [B, out_dim]
        return x


# =============================================================================
# OSR 主网络 (ResNetOSRNet)
# =============================================================================
class ResNetOSRNet(nn.Module):
    """
    基于 ResNet1D 的 OSR 网络。

    简化了原始多分支架构 (CNN + LSTM + SpecEncoder)，
    使用单一 ResNet1D 骨干网络处理时域信号，
    保持相同的 embedding + logits 输出接口。
    """
    def __init__(self, cfg):
        super().__init__()

        # 1. 配置参数
        self.cfg = cfg
        num_coarse = cfg.num_coarse_classes
        sample_len = cfg.seq_len

        # 2. 预处理
        self.preproc = SignalPreprocessor(use_smoothing=True, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        self.extractor = MultiModalExtractor(sample_len=sample_len, n_fft=cfg.n_fft, hop_length=cfg.hop_length)

        # 3. ResNet1D 骨干网络 (替代 CNN+LSTM+SpecEncoder)
        self.backbone = ResNet1D(
            in_channels=1,
            base_channels=cfg.resnet_base_channels,
            layers=cfg.resnet_layers,
            use_bottleneck=cfg.resnet_bottleneck,
            out_dim=cfg.resnet_out,
        )

        # 4. 瓶颈共享特征层
        self.bottleneck_base = nn.Sequential(
            nn.Linear(cfg.resnet_out, cfg.hidden_shared * 2),
            nn.BatchNorm1d(cfg.hidden_shared * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cfg.hidden_shared * 2, cfg.hidden_shared),
            nn.BatchNorm1d(cfg.hidden_shared),
            nn.ReLU(inplace=True),
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.shared_embedding_size = cfg.hidden_shared

        # 5. L2 归一化 + 不带偏置的分类头
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
        # 1) 预处理
        pre = self.preproc(x)
        modal = self.extractor(pre)
        t_in = modal["time"]  # [B, 1, L]

        # 2) ResNet1D 骨干提取特征
        feat = self.backbone(t_in)  # [B, resnet_out]

        # 3) 共享特征层
        embedding = self.bottleneck_base(feat)

        # 4) 分类：L2 归一化 → Linear(无bias)
        embedding_dropped = self.drop(embedding)
        embedding_norm = F.normalize(embedding_dropped, p=2, dim=1)
        logits = self.classifier(embedding_norm)

        # 5) 封装输出
        output = {
            'embedding': embedding,
            'logits': logits,
        }

        return output
