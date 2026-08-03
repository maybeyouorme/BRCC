import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def save_checkpoint(model, path):
    """保存模型权重"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'model': model.state_dict()}
    torch.save(state, path)


def train_model(model, train_loader, val_loader, cfg):
    """使用交叉熵损失训练 ResNetOSRNet"""
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)

    # 1. 优化器
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=1e-5)

    # 2. 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )

    # 3. 早停和最佳模型追踪
    best_loss = float('inf')
    best_epoch = 0
    patience = getattr(cfg, "patience", 40)
    patience_counter = 0

    print(f"--- Start ResNet Training (CrossEntropyLoss only) ---")
    print(f"Total Epochs: {cfg.epochs}, Device: {cfg.device}")

    for epoch in range(cfg.epochs):
        model.train()
        train_stats = {'loss': [], 'acc': []}

        # --- 训练循环 ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} Training")
        for data, coarse_labels in pbar:
            data = data.to(cfg.device)
            coarse_labels = coarse_labels.to(cfg.device)

            optimizer.zero_grad()

            # 前向传播
            output = model(data)
            logits = output['logits']

            # 交叉熵损失
            loss = F.cross_entropy(logits, coarse_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录统计
            train_stats['loss'].append(loss.item())
            acc = (logits.argmax(1) == coarse_labels).float().mean().item()
            train_stats['acc'].append(acc)

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{acc:.3f}"})

        # --- 验证循环 ---
        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for data, coarse_labels in val_loader:
                data = data.to(cfg.device)
                coarse_labels = coarse_labels.to(cfg.device)

                output = model(data)
                logits = output['logits']

                v_loss = F.cross_entropy(logits, coarse_labels)
                val_losses.append(v_loss.item())
                val_accs.append((logits.argmax(1) == coarse_labels).float().mean().item())

        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)

        scheduler.step(avg_val_loss)

        print(f"\n[Epoch {epoch + 1}/{cfg.epochs}]")
        print(f"  Train Loss: {np.mean(train_stats['loss']):.4f}, Train Acc: {np.mean(train_stats['acc']):.4f}")
        print(f"  Val Loss  : {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 早停与保存
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            save_checkpoint(model, cfg.model_save_path)
            print(f"  Best Model Saved! (Epoch {best_epoch}, Val Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"Early Stopping! Best Epoch: {best_epoch}")
            break

    print(f"\nTraining Finished. Best Val Loss: {best_loss:.4f} at Epoch {best_epoch}")
