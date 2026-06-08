import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from models import CenterLoss

# 辅助函数：保存模型权重
def save_checkpoint(model, path):
    """保存模型的 state_dict"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model': model.state_dict(),
        'center_loss': model.center_loss_fn.state_dict() if hasattr(model, 'center_loss_fn') else None
    }
    torch.save(state, path)

# ----------------------------------------------------------------------
# OSR 多任务训练核心函数 (SNR 加权版)
# ----------------------------------------------------------------------
class SupConLoss(nn.Module):
    """监督对比学习损失 (Supervised Contrastive Learning)"""
    def __init__(self, temperature=0.05):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )#[64,64]，每行是一个样本与所有样本的余弦相似度（经过温度缩放）
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)#[B, 1]即([64,1])
        logits = anchor_dot_contrast - logits_max.detach()#数值稳定性处理，减去每行的最大值

        labels = labels.contiguous().view(-1, 1)#[64,1]
        mask = torch.eq(labels, labels.T).float().to(device)#[64, 64]，每一行中同类为1，不同类为0
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )# 对角线为0，其他所有位置都为1的二维掩码矩阵，[64, 64]
        mask = mask * logits_mask# 保留同类样本的掩码，同时排除掉自身（对角线位置）
        exp_logits = torch.exp(logits) * logits_mask#对角线位置被置0，其他位置为exp(相似度)的矩阵，[64, 64]，每行是一个样本与所有样本的指数相似度（排除自身）
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)#[64, 64]，
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)#[64]，每行是一个样本与同类样本的平均对数概率
        loss = -mean_log_prob_pos.mean()
        return loss

def train_model(model, train_loader, val_loader, cfg):
    # 确保在训练前创建保存目录
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)
    print(f"DEBUG: Training with {cfg.num_fine_classes} fine classes (including Background)")
    
    # 1. 优化器
    optimizer_model = optim.Adam(model.parameters(), 
                           lr=cfg.learning_rate, 
                           weight_decay=1e-5) 
    #2 中心损失优化器
    optimizer_center = optim.SGD(
        model.center_loss_fn.parameters(),
        lr=0.05
    )
    
    # 3. 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer_model, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        verbose=True, 
        min_lr=1e-6
    )

# 1. 新增：对比损失函数初始化
    criterion_supcon = SupConLoss(temperature=0.05).to(cfg.device)

    # 4. 早停和最佳模型追踪
    best_loss = float('inf')
    best_epoch = 0
    patience = getattr(cfg, "patience", 40) 
    patience_counter = 0

    print(f"--- Start OSR Training with SNR-Weighted Loss (Dict-Output Mode) ---")
    print(f"Total Epochs: {cfg.epochs}, Device: {cfg.device}")


    for epoch in range(cfg.epochs):
        model.train()
        train_stats = {'loss': [], 'acc_coarse': [], 'acc_fine': [], 'loss_recon': [], 'loss_center': [], 'loss_contrast': []}

        # --- 训练循环 ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} Training")
        for data, coarse_labels, fine_labels, snrs in pbar:
            data = data.to(cfg.device)
            coarse_labels = coarse_labels.to(cfg.device)
            fine_labels = fine_labels.to(cfg.device)
            snrs = snrs.to(cfg.device)

            optimizer_model.zero_grad()
            optimizer_center.zero_grad()
            
            # 前向传播 (model 现在返回一个 dict)
            output = model(data, labels=coarse_labels)
            # 解析输出字典
            z = output['embedding']
            coarse_logits = output['coarse_logits']
            proj_feat = output['proj_feat'] # 对比学习专用特征
            
            # --- 动态计算 SNR 权重 ---
            snr_weights = 1.0 
            # --- 1. 监督分类损失 ---
            l_c_raw = F.cross_entropy(coarse_logits, coarse_labels, reduction='none')#[64]
            
            loss_coarse = (l_c_raw * snr_weights).mean()
            loss_sup = cfg.lambda_coarse * loss_coarse 
            
            # --- 2. 中心损失
            loss_center_raw = model.get_center_loss(z, coarse_labels) 

            loss_center = loss_center_raw * cfg.lambda_center
            
            # --- 3. 新增：对比学习损失 (SupCon) --
            lambda_contrast = getattr(cfg, "lambda_contrast", 2.5)
            loss_contrast = criterion_supcon(proj_feat, coarse_labels) * lambda_contrast

            # --- 4. 无监督重构损失 (L_RE) ---
            loss_recon = torch.tensor(0.0, device=cfg.device)
            if cfg.use_autoencoder and 'reconstruction' in output:
                reconstruction = output['reconstruction']
                target_data = output['target_signal']
                if target_data.dim() == 3 and reconstruction.dim() == 2:
                    target_data = target_data.squeeze(1)
                target_data = target_data.detach()

                #[64,8192]
                # 计算每个样本的 MSE: [B]
                mse_per_sample = F.mse_loss(reconstruction, target_data, reduction='none').mean(dim=1)

                # 类别特异性加权逻辑 (针对 Polar 类强化)
                class_weights = torch.ones_like(fine_labels).float()
                class_weights[fine_labels == 3] = 4.0
              
                # 应用 SNR 权重
                loss_recon = (mse_per_sample * snr_weights * class_weights).mean()
                
            # --- 3. 总损失 ---
            total_loss_batch = loss_sup + (cfg.gamma_recon * loss_recon) + loss_center + loss_contrast

            total_loss_batch.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#梯度裁剪
            
            # 中心损失梯度被放大了lambda_center倍，因此在参数更新前需要进行缩放以解耦学习率
            if cfg.lambda_center > 0:
                for param in model.center_loss_fn.parameters():
                    param.grad.data *= (1. / max(cfg.lambda_center, 1e-6))
            
            optimizer_model.step()
            optimizer_center.step()
            
            # 记录批次统计
            train_stats['loss'].append(total_loss_batch.item())
            train_stats['acc_coarse'].append((coarse_logits.argmax(1) == coarse_labels).float().mean().item())
            train_stats['loss_recon'].append(loss_recon.item())
            #train_stats['loss_center'].append(loss_center_raw.item())

            pbar.set_postfix({'Loss': f"{total_loss_batch.item():.4f}", 
                              'CoarseAcc': f"{train_stats['acc_coarse'][-1]:.3f}",
                              'Ctrst': f"{loss_contrast.item():.4f}"
                              })
        
        # --- 验证循环 ---
        model.eval()
        val_losses = []
        val_acc_f = []

        with torch.no_grad():
            for data, coarse_labels, fine_labels, snrs in val_loader:
                data, coarse_labels, fine_labels, snrs = data.to(cfg.device), coarse_labels.to(cfg.device), fine_labels.to(cfg.device), snrs.to(cfg.device)

                output = model(data, labels=None)
                z = output['embedding']
                c_out = output['coarse_logits']
                f_feat = output['proj_feat']
                
                # 验证集 SNR 权重逻辑保持一致
                w = 1
                
                l_c = (F.cross_entropy(c_out, coarse_labels, reduction='none') * w).mean()

        
                v_loss = cfg.lambda_coarse * l_c
                # --- [新增] 对比学习损失 ---
                l_contrast = criterion_supcon(f_feat, coarse_labels)
                v_loss += getattr(cfg, "lambda_contrast", 2.5) * l_contrast

                # --- 中心损失 ---
                v_loss += cfg.lambda_center * model.get_center_loss(z, coarse_labels)

                # --- 重构损失 ---
                if cfg.use_autoencoder and 'reconstruction' in output:
                    recon = output['reconstruction']
                    #t_data = data.squeeze(1) if data.dim() == 3 else data
                    t_data = output['target_signal']
                    if t_data.dim() == 3 and recon.dim() == 2:
                        t_data = t_data.squeeze(1)
                    
                    #l_r = (F.mse_loss(recon, t_data, reduction='none').mean(dim=1) * w).mean()#[]
                    l_r = F.mse_loss(recon, t_data, reduction='none').mean(dim=1) * w
                    #TODO:我把师兄原来的第二个mean去掉了,结果有细微浮动

                    # 保持权重逻辑与训练集一致
                    c_w = torch.ones_like(fine_labels).float()
                    c_w[fine_labels == 3] = 4.0
                    l_r = (l_r * c_w).mean()
                    v_loss += cfg.gamma_recon * l_r
                
                val_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_losses)

        
        scheduler.step(avg_val_loss)

        print(f"\n[Epoch {epoch + 1}/{cfg.epochs}]")
        print(f"  Train Loss: {np.mean(train_stats['loss']):.4f} (Recon: {np.mean(train_stats['loss_recon']):.6f})")
        print(f"  Val Loss  : {avg_val_loss:.4f}")

        # 早停与保存
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            save_checkpoint(model, cfg.model_save_path) 
            print(f"✅ Best Model Saved! (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"ℹ️ No improvement ({patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"Early Stopping! Best Epoch: {best_epoch}")
            break
            
    print(f"\nTraining Finished. Best Val Loss: {best_loss:.4f} at Epoch {best_epoch}")