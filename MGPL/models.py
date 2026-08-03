"""
MGPL: Mixture of Gaussian Prototype Learning for Open-Set Recognition
Adapted from the original MGPL paper code (2D image) to 1D signal domain.

Key components:
  1. VAE-style 1D ResNet encoder → (mu, logvar, lateral features)
  2. 1D decoder with lateral connections → signal reconstruction
  3. Learnable Gaussian prototypes per class × sub-prototypes
  4. MGPL loss: Generative (recon + KLD + entropy) + Discriminative (LSE distance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================================
# 1D ResNet Encoder Components
# =========================================================================

class ResidualBlockEnc1D(nn.Module):
    """Residual block for encoder (with optional stride downsampling)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1DEnc(nn.Module):
    """
    1D ResNet VAE Encoder.
    Input:  [B, 1, L]
    Output: mu [B, latent_dim], logvar [B, latent_dim], lateral_z (dict of intermediate features)

    Architecture (with L=8192):
      in_conv:          [B,  64, L]
      layer1 (s=1):      [B,  64, L]     lateral_x1
      layer2 (s=2):      [B, 128, L/2]   lateral_x2
      layer3 (s=2):      [B, 256, L/4]   lateral_x3
      layer4 (s=2):      [B, 512, L/8]   lateral_x4
      pool → mu/logvar:  [B, latent_dim]
    """
    def __init__(self, in_ch=1, base_channels=64, layers=[2, 2, 2, 2], latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_planes = base_channels

        # Initial convolution (preserves length with stride=1)
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_channels, kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # ResNet stages with progressive downsampling
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)     # L
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2) # L/2
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2) # L/4
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2) # L/8

        # VAE output heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mu = nn.Linear(base_channels * 8, latent_dim)
        self.logvar = nn.Linear(base_channels * 8, latent_dim)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        layers.append(ResidualBlockEnc1D(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(ResidualBlockEnc1D(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        x = self.in_conv(x)               # [B, 64, L]
        x1 = self.layer1(x)               # [B, 64, L]
        x2 = self.layer2(x1)              # [B, 128, L/2]
        x3 = self.layer3(x2)              # [B, 256, L/4]
        x4 = self.layer4(x3)              # [B, 512, L/8]

        pooled = self.pool(x4).squeeze(-1)  # [B, 512]
        mu = self.mu(pooled)               # [B, latent_dim]
        logvar = self.logvar(pooled)        # [B, latent_dim]

        lateral_z = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}
        return mu, logvar, lateral_z


# =========================================================================
# 1D Decoder Components (with lateral connections)
# =========================================================================

class ResizeConv1d(nn.Module):
    """Upsample via interpolation + conv1d."""
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x


class LateralConv1d(nn.Module):
    """Lateral connection:  Dropout → Conv1d → ReLU."""
    def __init__(self, channels, drop_rate=0.5):
        super().__init__()
        self.lateral = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.lateral(x)


class ResidualBlockDec1D(nn.Module):
    """
    Residual block for decoder.
    - stride=1: preserves spatial dim, in_planes → planes (typically same)
    - stride>1: upsamples by stride×, in_planes → in_planes/stride
    """
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = int(in_planes // stride) if stride > 1 else in_planes

        self.conv2 = nn.Conv1d(in_planes, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv1d(in_planes, planes, kernel_size=3,
                                      scale_factor=stride)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1DDec(nn.Module):
    """
    1D ResNet VAE Decoder with lateral connections.
    Input:  z [B, latent_dim], lateral_z dict from encoder
    Output: reconstructed signal [B, 1, L]

    Architecture:
      linear:  [B, latent_dim] → [B, 512]
      reshape: [B, 512, 1]
      up to match lateral_x4 spatial dim → + lateral_conv(x4)
      layer4 (s=2): [512 → 256, 2× spatial] + lateral_conv(x3)
      layer3 (s=2): [256 → 128, 2× spatial] + lateral_conv(x2)
      layer2 (s=2): [128 → 64,  2× spatial] + lateral_conv(x1)
      layer1 (s=1): [64  → 64,  same spatial]
      conv_out:  [64 → 1, kernel_size=3]
    """
    def __init__(self, latent_dim=128, base_channels=64, layers=[2, 2, 2, 2], dropout=0.5):
        super().__init__()
        self.in_planes = base_channels * 8  # 512

        self.linear = nn.Linear(latent_dim, self.in_planes)

        # Decoder layers (reverse order of encoder)
        self.layer4 = self._make_layer(base_channels * 4, layers[3], stride=2)  # 512→256
        self.layer3 = self._make_layer(base_channels * 2, layers[2], stride=2)  # 256→128
        self.layer2 = self._make_layer(base_channels,     layers[1], stride=2)  # 128→64
        self.layer1 = self._make_layer(base_channels,     layers[0], stride=1)  # 64→64

        self.conv_out = nn.Conv1d(base_channels, 1, kernel_size=3,
                                  stride=1, padding=1)

        # Lateral connections matching encoder feature channels
        self.lateral4 = LateralConv1d(base_channels * 8, dropout)
        self.lateral3 = LateralConv1d(base_channels * 4, dropout)
        self.lateral2 = LateralConv1d(base_channels * 2, dropout)
        self.lateral1 = LateralConv1d(base_channels, dropout)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in reversed(strides):
            layers.append(ResidualBlockDec1D(self.in_planes, s))
            self.in_planes = int(self.in_planes // s) if s > 1 else self.in_planes
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z, lateral_z):
        x = self.linear(z)                          # [B, 512]
        x = x.unsqueeze(-1)                         # [B, 512, 1]

        # Upsample to match x4 spatial dim, add lateral connection
        x4_size = lateral_z['x4'].size(-1)
        x = F.interpolate(x, size=x4_size)          # [B, 512, L/8]
        x = x + self.lateral4(lateral_z['x4'])      # + lateral

        x = self.layer4(x)                          # [B, 256, L/4]
        x = x + self.lateral3(lateral_z['x3'])      # + lateral

        x = self.layer3(x)                          # [B, 128, L/2]
        x = x + self.lateral2(lateral_z['x2'])      # + lateral

        x = self.layer2(x)                          # [B, 64, L]
        x = x + self.lateral1(lateral_z['x1'])      # + lateral

        x = self.layer1(x)                          # [B, 64, L]
        x = self.conv_out(x)                        # [B, 1, L]
        return x


# =========================================================================
# MGPL Network
# =========================================================================

class MGPLNet(nn.Module):
    """
    Mixture of Gaussian Prototype Learning Network.

    Combines:
      - VAE encoder (1D ResNet) → (mu, logvar, lateral_z)
      - Reparameterization sampler
      - Learnable Gaussian prototypes
      - Decoder with lateral connections
      - MGPL loss: recon + KLD + entropy + discriminative
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.n_sub_prototypes = cfg.n_sub_prototypes
        self.n_classes = cfg.num_coarse_classes
        self.temp_inter = cfg.temp_inter
        self.temp_intra = cfg.temp_intra

        # VAE encoder-decoder
        self.encoder = ResNet1DEnc(
            in_ch=1,
            base_channels=cfg.base_channels,
            layers=cfg.cnn_layers,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = ResNet1DDec(
            latent_dim=cfg.latent_dim,
            base_channels=cfg.base_channels,
            layers=cfg.cnn_layers,
            dropout=0.5,
        )

        # Gaussian prototypes: [n_classes * n_sub_prototypes, latent_dim]
        self.prototypes = nn.Parameter(
            torch.randn(self.n_classes * self.n_sub_prototypes, self.latent_dim),
            requires_grad=True
        )
        nn.init.kaiming_normal_(self.prototypes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---------- Core MGPL functions ----------

    def sampler(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * exp(0.5 * logvar)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        return mu

    def distance(self, latent_z, prototypes):
        """
        Squared Euclidean distance between batch z and all prototypes.

        Args:
            latent_z:    [B, latent_dim]
            prototypes:  [n_prototypes, latent_dim]
        Returns:
            distance:    [B, n_prototypes]
        """
        B = latent_z.size(0)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
        a_sq = torch.sum(latent_z ** 2, dim=1, keepdim=True)           # [B, 1]
        b_sq = torch.sum(prototypes ** 2, dim=1).unsqueeze(0)          # [1, P]
        prod = torch.matmul(latent_z, prototypes.t())                  # [B, P]
        return a_sq + b_sq - 2 * prod

    def kl_div_to_prototypes(self, mean, logvar, prototypes):
        """
        KL divergence from z~N(mu, sigma^2) to prototypes w~N(mu_w, I).

        KL(N(mu,σ²) || N(mu_w,I)) = 0.5 * [||mu-mu_w||² + σ² - log(σ²) - 1]

        Args:
            mean:       [B, latent_dim]
            logvar:     [B, latent_dim]
            prototypes: [n_prototypes, latent_dim]
        Returns:
            kl_div:     [B, n_prototypes]
        """
        dist = self.distance(mean, prototypes)
        var_penalty = torch.sum(logvar.exp() - logvar - 1, dim=1, keepdim=True)
        return 0.5 * (dist + var_penalty)

    def compute_loss(self, x, y):
        """
        Compute the full MGPL loss.

        Args:
            x: input signal [B, L]
            y: class labels [B]
        Returns:
            latent_z:  [B, latent_dim]
            x_recon:   [B, 1, L]
            preds:     [B]  (predicted class labels by nearest prototype)
            loss_dict: dict of loss components
        """
        mu, logvar, lateral_z = self.encoder(x)
        latent_z = self.sampler(mu, logvar)

        # Distances and KL divergences to all prototypes
        dist = self.distance(latent_z, self.prototypes)                 # [B, P]
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes) # [B, P]

        # Reconstruction
        x_recon = self.decoder(latent_z, lateral_z)

        # ---- Prediction by nearest prototype ----
        dist_reshaped = dist.reshape(len(x), self.n_classes, self.n_sub_prototypes)
        dist_class_min, _ = torch.min(dist_reshaped, dim=2)            # [B, n_classes]
        _, preds = torch.min(dist_class_min, dim=1)                    # [B]

        # ---- MGPL Loss ----

        # Mask: distances/KLD to prototypes of the ground-truth class
        y_one_hot = F.one_hot(y, num_classes=self.n_classes)           # [B, n_classes]
        y_mask = y_one_hot.repeat_interleave(self.n_sub_prototypes, dim=1).bool()
        dist_y = dist[y_mask].reshape(len(dist), self.n_sub_prototypes)
        kl_div_y = kl_div[y_mask].reshape(len(kl_div), self.n_sub_prototypes)

        # q(w|z,y) — soft assignment of z to prototypes of the true class
        q = F.softmax(-dist_y / self.temp_intra, dim=1)                # [B, n_sub]

        # 1) Generative Constraint
        # Reconstruction loss (MSE)
        # x_recon: [B, 1, L], x: [B, L] — squeeze to match
        rec_loss = F.mse_loss(x_recon.squeeze(1), x)

        # KLD to prototypes of the true class (conditional prior)
        kld_loss = torch.mean(torch.sum(q * kl_div_y, dim=1))

        # Entropy prior (encourages using all sub-prototypes)
        ent_loss = torch.mean(torch.sum(
            q * torch.log(q * self.n_sub_prototypes + 1e-7), dim=1
        ))

        # 2) Discriminative Constraint
        # LogSumExp-based distance loss
        LSE_all = torch.logsumexp(-dist / self.temp_inter, dim=1)
        LSE_target = torch.logsumexp(-dist_y / self.temp_inter, dim=1)
        dis_loss = torch.mean(LSE_all - LSE_target)

        loss_dict = {
            'dis': dis_loss,
            'rec': rec_loss,
            'kld': kld_loss,
            'ent': ent_loss,
        }
        return latent_z, x_recon, preds, loss_dict

    def forward(self, x):
        """Simple forward pass for inference (returns embeddings + distances)."""
        mu, logvar, _ = self.encoder(x)
        latent_z = self.sampler(mu, logvar)
        dist = self.distance(latent_z, self.prototypes)
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes)
        return latent_z, dist, kl_div


# =========================================================================
# Utility functions for MGPL training
# =========================================================================

def weight_init(m):
    """Initialize weights for linear, conv, and batchnorm layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def extract_features(model, data_loader, device):
    """Extract VAE latent features from all samples in data_loader."""
    features = []
    targets = []
    model.eval()
    print('Extracting features for prototype reset...')
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            mu, logvar, _ = model.encoder(x)
            z = model.sampler(mu, logvar)
            features.append(z.cpu())
            targets.append(y)
    features = torch.cat(features, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return features, targets


def reset_prototypes(model, data_loader, device):
    """
    Reset prototypes using K-means clustering on encoder features.
    This is called at specified epochs (e.g., 50, 80) as in MGPL paper.
    """
    from sklearn.cluster import KMeans
    import numpy as np

    print('=' * 50)
    print('Resetting Prototypes via K-Means Clustering')
    model.eval()

    features, targets = extract_features(model, data_loader, device)
    n_classes = model.n_classes
    n_sub = model.n_sub_prototypes

    mu_list = []
    for i in range(n_classes):
        class_mask = targets == i
        class_feats = features[class_mask]
        if len(class_feats) < n_sub:
            print(f'  ⚠ Class {i}: only {len(class_feats)} samples, using mean + noise')
            # Fallback: use mean with small perturbations
            mean_feat = class_feats.mean(axis=0, keepdims=True)
            mu = np.tile(mean_feat, (n_sub, 1))
            mu += np.random.randn(n_sub, model.latent_dim) * 0.01
        else:
            kmeans = KMeans(n_clusters=n_sub, random_state=0, n_init='auto')
            kmeans.fit(class_feats)
            mu = kmeans.cluster_centers_
        mu_list.append(mu)

    new_protos = np.vstack(mu_list)
    new_protos_t = torch.from_numpy(new_protos).float().to(device)

    # Update in-place to keep optimizer parameter references valid
    with torch.no_grad():
        model.prototypes.data.copy_(new_protos_t)

    print(f'  ✅ Prototypes reset: {n_classes} classes × {n_sub} sub-prototypes')
    print('=' * 50)
    return model.prototypes
