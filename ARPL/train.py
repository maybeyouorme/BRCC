"""
ARPL 训练与验证函数。

实现:
  - train_one_epoch:  单轮训练 (使用 ARPLoss / RPLoss / Softmax)
  - validate:         验证集评估
  - EarlyStopping:    早停机制
  - train_model:      完整训练流程 (含模型保存、学习率调度、早停)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# =========================================================================
# 早停 (Early Stopping)
# =========================================================================

class EarlyStopping:
    """
    早停机制：监控验证集指标，连续 patience 个 epoch 未改善则停止训练。

    用法:
        early_stopping = EarlyStopping(patience=10, delta=1e-4, mode='min')
        for epoch in range(epochs):
            val_loss = validate(...)
            if early_stopping.step(val_loss, epoch):
                break
    """
    def __init__(self, patience=10, delta=1e-4, mode='min', verbose=True):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        if mode == 'max':
            self.delta *= -1

    def step(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - abs(self.delta)
        else:
            improved = score > self.best_score + abs(self.delta)

        if improved:
            if self.verbose:
                print(f'  ✨ Validation improved ({self.best_score:.6f} → {score:.6f})')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'  ⏳ EarlyStopping: {self.counter}/{self.patience} '
                      f'(best={self.best_score:.6f} @ epoch {self.best_epoch+1})')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


# =========================================================================
# 训练单轮
# =========================================================================

def train_one_epoch(model, criterion, optimizer, loader, cfg, epoch):
    """
    执行一轮训练。

    Args:
        model:     ARPLNet
        criterion: ARPLoss / RPLoss / Softmax
        optimizer: PyTorch 优化器
        loader:    训练 DataLoader
        cfg:       配置
        epoch:     当前 epoch (0-based, 仅用于日志)
    Returns:
        avg_loss:  平均损失
        acc:       训练准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg.epochs} [Train]')
    for data, labels in pbar:
        data = data.to(cfg.device)
        labels = labels.to(cfg.device)

        optimizer.zero_grad()

        # 前向: ARPLNet 返回 (embedding, logits)
        embedding, logits_from_model = model(data)

        # 损失: criterion 接收 (embedding, logits, labels)
        #       ARPLoss 内部用 embedding 重新计算 logits
        _, loss = criterion(embedding, logits_from_model, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # 准确率计算 (使用 ARPLoss 时，用 criterion 内部的 logits；Softmax 时用模型 logits)
        with torch.no_grad():
            logits_for_pred, _ = criterion(embedding, logits_from_model)
            pred = logits_for_pred.max(1)[1]
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct/total:.3f}',
        })

    avg_loss = running_loss / len(loader)
    acc = correct / total
    print(f'  Train Loss: {avg_loss:.4f}, Acc: {acc:.4f}')
    return avg_loss, acc


# =========================================================================
# 验证
# =========================================================================

@torch.no_grad()
def validate(model, criterion, loader, cfg):
    """
    在验证集上评估模型。

    Returns:
        avg_loss:  平均损失
        acc:       验证准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in loader:
        data = data.to(cfg.device)
        labels = labels.to(cfg.device)

        embedding, logits_from_model = model(data)
        logits, loss = criterion(embedding, logits_from_model, labels)

        running_loss += loss.item()
        pred = logits.max(1)[1]
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    print(f'  Val Loss: {avg_loss:.4f}, Acc: {acc:.4f}')
    return avg_loss, acc


# =========================================================================
# 模型保存
# =========================================================================

def save_checkpoint(model, criterion, path):
    """保存模型和 criterion 的 state_dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model': model.state_dict(),
        'criterion': criterion.state_dict(),
    }
    torch.save(state, path)


# =========================================================================
# 完整训练流程
# =========================================================================

def train_model(model, criterion, train_loader, val_loader, cfg):
    """
    完整训练流程：
      1. 优化器 (Adam)
      2. 学习率调度 (MultiStepLR)
      3. 早停
      4. 逐 epoch 训练 + 验证 + 保存最优模型
    """
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)

    # 优化器: criterion 中也包含可学习参数 (Dist.centers, radius)
    params_list = [
        {'params': model.parameters()},
        {'params': criterion.parameters()},
    ]
    optimizer = optim.Adam(params_list, lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma
    )

    # 早停
    early_stopping = EarlyStopping(
        patience=cfg.early_stopping_patience,
        delta=cfg.early_stopping_delta,
        mode='min',
    )

    best_val_loss = float('inf')
    best_epoch = 0

    # ── 训练历史记录 (用于绘图) ──
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'\n{"=" * 50}')
    print(f'ARPL Training Started (loss: {cfg.loss_type})')
    print(f'  Epochs: {cfg.epochs}, Batch: {cfg.batch_size}, LR: {cfg.learning_rate}')
    print(f'  weight_pl: {cfg.weight_pl}, temp: {cfg.temp}')
    print(f'  LR milestones: {cfg.lr_milestones}, gamma: {cfg.lr_gamma}')
    print(f'  Early stopping patience: {cfg.early_stopping_patience}')
    print(f'{"=" * 50}\n')

    for epoch in range(cfg.epochs):
        print(f'\n--- Epoch {epoch+1}/{cfg.epochs} ---')

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, cfg, epoch
        )
        val_loss, val_acc = validate(model, criterion, val_loader, cfg)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  LR: {current_lr:.6f}')

        # 保存最优模型 (按验证损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_checkpoint(model, criterion, cfg.model_save_path)
            print(f'  ✅ Best model saved (epoch {best_epoch}, val_loss={best_val_loss:.4f})')

        # 早停检查
        if early_stopping.step(val_loss, epoch):
            print(f'\n  🛑 Early stopping triggered after {epoch+1} epochs')
            break

    print(f'\n{"=" * 50}')
    print(f'Training Complete.')
    print(f'  Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}')
    print(f'  Model saved: {cfg.model_save_path}')
    print(f'{"=" * 50}')

    return history
