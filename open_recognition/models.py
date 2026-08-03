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
# 2. 中心损失模块 (Center Loss)
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    def forward(self, x, labels):
        batch_size = x.size(0)
        # 计算欧氏距离平方: (x - centers)^2 = x^2 + centers^2 - 2*x*centers
        #[B,D], [C,D] -> [B,C]
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).t()#[B,1] + [C,1].T -> [B,C]
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)#[B,C]
        classes = torch.arange(self.num_classes).long().to(self.device)#[C]
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#[B, C]，每一行是当前样本的标签值
        mask = labels.eq(classes.expand(batch_size, self.num_classes))#[B, C]，括号里每一行是所有类别
        dist = distmat * mask.float()# 只保留样本到对应类中心的距离
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
# 2. 编码器 (Encoder) - 保持不变 (只使用 time-domain x_time)
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
# 3. 新增解码器 (Decoder) - 用于重构
class MultiTaskDecoder(nn.Module):
    #input:[64,128]-->[64,8192]
    def __init__(self, embedding_size, seq_len, initial_channels=32): # 通道从256降到128
        super().__init__()
        self.seq_len = seq_len
        downsample_factor = 8
        self.L_intermediate = seq_len // downsample_factor#1024   
        # 1. 核心瓶颈层：增加一个线性收缩，限制信息流量
        bottleneck_size = embedding_size // 2 #64
        self.bottleneck = nn.Sequential(
            nn.Linear(embedding_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU(),
        )#[64,64]

        # 1. 瓶颈层映射 (减少神经元数量，增加非线性)
        self.fc_expand = nn.Sequential(
            nn.Linear(bottleneck_size, initial_channels * self.L_intermediate),
            nn.BatchNorm1d(initial_channels * self.L_intermediate),
            nn.LeakyReLU(0.2, inplace=True), # 使用 LeakyReLU 增加神经元活跃度
        )#[64, 32*1024]=[64,32768]

        #输入：[64, 32, 1024]     
        # 2. 逐步上采样模块 (L/8 -> L/4 -> L/2 -> L)
        self.upsample_blocks = nn.Sequential(
            # Stage 1: L/8 -> L/4
            nn.ConvTranspose1d(initial_channels, initial_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(initial_channels // 2),
            nn.ELU(inplace=True),
            nn.Dropout1d(0.1), # <--- 新增：轻微扰动卷积特征图
            #[64, 32, 2048]

            # Stage 2: L/4 -> L/2
            nn.ConvTranspose1d(initial_channels // 2, initial_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(initial_channels // 4),
            nn.ReLU(inplace=True),
            #[64, 16, 4096]

          # Stage 3: L/2 -> L (Final Output)
            nn.ConvTranspose1d(initial_channels // 4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            #[64, 1, 8192]
        )
    def forward(self, embedding):
        # 映射并变形
        x = self.bottleneck(embedding)
        x = self.fc_expand(x)
        x = x.view(x.size(0), -1, self.L_intermediate)#[64, 32, 1024]
       # 上采样重构
        reconstructed_x = self.upsample_blocks(x)
        return reconstructed_x.squeeze(1) # [B, L]=[64, 8192]
# 基于角度的余弦距离 (用于粗分类)
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features #128
        self.out_features = out_features # 粗类数
        self.s = s  # 缩放因子，s值越大，分类越严格
        self.m = m  # 角度间隔，m越大，类内越紧凑
        # 这里的 weight 相当于你的“中心原型”
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # 1. 归一化特征和权重，计算余弦相似度
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) #[64,128] x [5,128].T -> [64,5]
        #cosin[i][j]:样本i与类j的中心原型的余弦相似度
        if label is None:
            return cosine * self.s
        
        # 计算 sin(theta) 用于后续判断
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        #cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        #角度加m,cos值减小，模型被强制学习得更好，必须让类别特征更加贴近其原型，使得phi恢复到足够大的值，才能保证分类正确。

        # 当 theta + m > pi 时，减弱惩罚，防止梯度爆炸
        phi = torch.where(cosine > 0, phi, cosine)

        # 3. 构造输出
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)# [64,15]，前一个1表示维度，横向操作；后一个1表示对应位置写入1
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output #[64,5]

# 4. 主网络 (MultiTaskOSRNet) - 核心改动
class MultiTaskOSRNet(nn.Module):
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
        self.spec_encoder = SpecEncoder(out_dim=64) # 对应图中 spec_out

        # 4. 融合与共享特征层 (特征拼接)
        fused_dim = cfg.cnn_out + cfg.lstm_out + 64 # 64 是 spec_out 的维度

        self.bottleneck_base = nn.Sequential(
            # 第一层：先做初步整合与筛选[64,320]-->[64,128]
            nn.Linear(fused_dim, cfg.hidden_shared * 2), 
            nn.BatchNorm1d(cfg.hidden_shared * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cfg.hidden_shared * 2, cfg.hidden_shared),
            nn.BatchNorm1d(cfg.hidden_shared),
            # 第二层：深度非线性映射
            nn.ReLU(inplace=True)
        )
#------引入对别学习
        self.contrastive_head = nn.Sequential(
            nn.Linear(cfg.hidden_shared, cfg.hidden_shared),
            nn.BatchNorm1d(cfg.hidden_shared),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_shared, 128)
        )

        self.drop = nn.Dropout(0.2)
        self.shared_embedding_size = cfg.hidden_shared
        # 5. 分类头 (原型学习)
        self.coarse_head = ArcMarginProduct(self.shared_embedding_size, num_coarse, s=32.0, m=0.8)
        self.center_loss_fn = CenterLoss(num_classes=num_coarse, feat_dim=self.shared_embedding_size, device=cfg.device)
        self.use_autoencoder = cfg.use_autoencoder
        if self.use_autoencoder:
            self.decoder = MultiTaskDecoder(self.shared_embedding_size, sample_len)
        else:
            self.decoder = None

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
        

    def forward(self, x, labels=None):
        # 1) 预处理
        pre = self.preproc(x) 
        modal = self.extractor(pre) 
        t_in = modal["time"] 

        # 2) 分支特征提取
        feat_cnn = self.cnn(t_in) 
        feat_lstm = self.lstm(t_in) 
        feat_spec = self.spec_encoder(modal["spec"])

        # 3) 特征融合
        fused = torch.cat([feat_cnn, feat_lstm, feat_spec], dim=1)#[64,320]
        z_raw = self.bottleneck_base(fused) # 这里的 z_raw 用于计算 Center Loss 和重构,[64,128]
        
       
        # 4) 分类决策 (使用 Dropout 提高泛化性)
         # 1. 对 z 进行 L2 归一化 (关键步骤)
        z_norm = F.normalize(z_raw, p=2, dim=1)
        #沿列维度（左右方向），对每个样本i，进行单位向量化

        z_dropped = self.drop(z_norm)
        
        # 粗分类 Logits
        coarse_logits = self.coarse_head(z_dropped, labels)
        

#----引入对比学习
        proj_feat = self.contrastive_head(z_raw)
        proj_feat = F.normalize(proj_feat, p=2, dim=1)
        
        # 5) 封装输出
        output = {
            'embedding': z_norm,        
            'coarse_logits': coarse_logits,
            'prototypes': self.coarse_head.weight, 
            'z_raw': z_raw,
            'proj_feat': proj_feat  # <--- 新增：供 SupConLoss 使用
        }

        # 6) 重构分支 (如果开启)
        if self.use_autoencoder and self.decoder is not None:
            # 训练时可以给重构增加一点难度 (Drop)
            #z_for_decoder = F.dropout(z_raw, p=0.2) if self.training else z_raw
            if self.training:
                z_for_decoder = z_norm
            else:
                z_for_decoder = z_norm
            reconstructed_x = self.decoder(z_for_decoder)
            
            output['reconstruction'] = reconstructed_x 
            output['target_signal'] = modal["raw"].squeeze(1) # [B, L]
        
        return output
    
    def get_center_loss(self, features, labels):
        """方便在 train.py 中一键调用"""
        return self.center_loss_fn(features, labels)
    