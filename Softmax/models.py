import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#-------------数据预处理
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
            x = x.unsqueeze(1)# [B, 1, L]
        raw_x = x.clone().detach()
        t = x
        mean = t.mean(dim=2, keepdim=True)
        std = t.std(dim=2, keepdim=True) + 1e-9
        t = (t - mean) / std
        if self.use_smoothing:
            t = self.smooth(t)
        freq_complex = torch.fft.rfft(t.squeeze(1), n=self.n_fft)
        freq_mag = torch.abs(freq_complex)#[B, 129]
        window = torch.hann_window(256).to(t.device)  # 汉宁窗
        spec = torch.stft(t.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)#[B, 频率轴(129), 时间轴(64)]
        spec = torch.abs(spec)#[B, 频率轴(129), 时间轴(64)]
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
        t = preproc_out["time"]
        freq = preproc_out["freq"]
        spec = preproc_out["spec"]
        raw = preproc_out["raw"]
        return {"time": t, "freq": freq, "spec": spec, "raw": raw}

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

# 编码器 (Encoder) - 保持不变 (只使用 time-domain x_time)
class DeepResNet1DEncoder(nn.Module):
    #x_time:[64,1,8192]-->[64,128]
    def __init__(self, in_ch=1, base_channels=64, layers=[2, 2, 2], out_dim=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_channels, kernel_size=7, stride=2, padding=3, bias=False), # L -> L/2
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)#[64,64,4096]
        )
        self.in_channels = base_channels
        self.stage1 = self._make_layer(ResidualBlock1D, base_channels, layers[0], stride=1)
        #1个stage2个残差块，共4层卷积，[64,64,4096]

        self.stage2 = self._make_layer(ResidualBlock1D, base_channels * 2, layers[1], stride=2) # L/2 -> L/4
        #2个残差块，共4层卷积，第1个块进行一次下采样
        #[64,128,2048]

        #self.stage3 = self._make_layer(ResidualBlock1D, base_channels * 4, layers[2], stride=2) # L/4 -> L/8
        final_channels = base_channels * 2 # 128（x4:256）
        self.pool = nn.AdaptiveAvgPool1d(1)# [64,128,1]-->[64,128]
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
        #out = self.stage2(out)
        feature_map = self.stage2(out)
        out = self.pool(feature_map).squeeze(-1)#[64,128]
        embedding = self.fc(out)#[64,128]
        return embedding

class LSTMEncoder(nn.Module):
    #x_time:[64,1,8192]
    def __init__(self, input_len=8192, frame_size=128, hidden_size=128, num_layers=2, out_dim=128):
        super().__init__()
        self.frame_size = frame_size
        self.seq_len = math.ceil(input_len / frame_size)#64
        self.frame_proj = nn.Linear(frame_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_fc = nn.Linear(hidden_size * 2, out_dim)
    def forward(self, x_time):
        if x_time.dim() == 2:
            x_time = x_time.unsqueeze(1)
        B, C, L = x_time.shape
        seq = x_time[:, 0, :]#[64,8192]
        pad_len = self.seq_len * self.frame_size - L
        if pad_len > 0:
            seq = F.pad(seq, (0, pad_len))
        frames = seq.view(B, self.seq_len, self.frame_size)#[64,64,128]
        emb = self.frame_proj(frames)
        out, _ = self.lstm(emb)#[64,64,256]
        feat = out.mean(dim=1)#[64,256]
        feat = self.out_fc(feat)#[64,128]
        return feat

class SpecEncoder(nn.Module):
    #x_spec:[64, 129, 64]
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
        x_spec = x_spec.unsqueeze(1)#[64, 1, 129, 64]
        out = self.conv_in(x_spec)#[64, 32, 64, 32]
        out = self.conv_blocks(out)#[64, 128, 1, 1]
        out = out.squeeze(-1).squeeze(-1)#[64,128]
        return self.fc_out(out)#[64,out_dim]

# 主网络 (SoftmaxOSRNet) - 只保留交叉熵损失
class SoftmaxOSRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 1. 配置参数
        self.cfg = cfg
        num_coarse = cfg.num_coarse_classes
        sample_len = cfg.seq_len

        # 2. 预处理和特征提取器
        self.preproc = SignalPreprocessor(use_smoothing=True, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        self.extractor = MultiModalExtractor(sample_len=sample_len, n_fft=cfg.n_fft, hop_length=cfg.hop_length)

        # 3. 分支编码器
        self.cnn = DeepResNet1DEncoder(in_ch=1, base_channels=cfg.cnn_base_channels,
                                       layers=cfg.cnn_layers, out_dim=cfg.cnn_out)
        self.lstm = LSTMEncoder(input_len=sample_len, frame_size=cfg.lstm_frame_size,
                                hidden_size=cfg.lstm_hidden_size, num_layers=cfg.lstm_layers, out_dim=cfg.lstm_out)
        self.spec_encoder = SpecEncoder(out_dim=64)

        # 4. 融合与共享特征层
        fused_dim = cfg.cnn_out + cfg.lstm_out + 64

        self.bottleneck_base = nn.Sequential(
            nn.Linear(fused_dim, cfg.hidden_shared * 2),
            nn.BatchNorm1d(cfg.hidden_shared * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cfg.hidden_shared * 2, cfg.hidden_shared),
            nn.BatchNorm1d(cfg.hidden_shared),
            nn.ReLU(inplace=True)
        )

        self.drop = nn.Dropout(0.2)
        self.shared_embedding_size = cfg.hidden_shared

        # 5. L2 归一化 + 不带偏置的 Linear 分类头（限制 logits 幅值）
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
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
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

        # 4) 分类：L2 归一化 → Linear(无bias) → 缩放
        # 限制 logits 幅值，避免 MSP 对所有样本都趋近 1
        embedding_dropped = self.drop(embedding)
        embedding_norm = F.normalize(embedding_dropped, p=2, dim=1)
        logits = self.classifier(embedding_norm)

        # 5) 封装输出
        output = {
            'embedding': embedding,
            'logits': logits,
        }

        return output
