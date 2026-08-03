"""
MGPL Training Script.

Handles:
  1. Data loading and splitting (following project conventions)
  2. MGPL model training with combined generative + discriminative loss
  3. Prototype reset at specified epochs
  4. Model checkpointing
  5. Saving evaluation pickle files for OSR evaluation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from open_recognition.dataset import ChannelCodeDataset

from config import Config
from models import MGPLNet, reset_prototypes

import pickle
import collections


# =========================================================================
# Data Loading Utilities
# =========================================================================

def map_labels_to_continuous(labels):
    """Map original labels to 0..C-1 continuous integers."""
    unique_labels = np.unique(labels)
    label_map = {val: i for i, val in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    return mapped_labels, label_map


def load_and_combine_data(data_dir, is_known, exclude_labels=None):
    """Load all .mat files from a directory and combine."""
    all_features = []
    all_coarse = []
    all_snrs = []

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    file_list.sort()
    print(f"Files in {'Known' if is_known else 'Unknown'} dir: {file_list}")

    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        ds = ChannelCodeDataset(filepath, exclude_labels=exclude_labels)
        all_features.append(ds.features)
        all_coarse.append(ds.coarse_labels)
        all_snrs.append(ds.snrs)

    if not all_features:
        if is_known:
            raise FileNotFoundError(f'No .mat files found in {data_dir}')
        return np.array([]), np.array([]), np.array([])

    features = np.concatenate(all_features, axis=0)
    coarse = np.concatenate(all_coarse, axis=0)
    snrs = np.concatenate(all_snrs, axis=0)
    print(f'Total {"known" if is_known else "unknown"} samples: {len(features)}')
    return features, coarse, snrs


def save_checkpoint(model, path, extra_info=None):
    """Save model state dict with optional extra info."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'model': model.state_dict()}
    if extra_info is not None:
        state.update(extra_info)
    torch.save(state, path)


# =========================================================================
# Early Stopping
# =========================================================================

class EarlyStopping:
    """
    早停机制：监控验证集指标，连续 patience 个 epoch 未改善则停止训练。

    用法:
        early_stopping = EarlyStopping(patience=10, delta=1e-4, mode='min')
        for epoch in range(epochs):
            val_loss = validate(...)
            if early_stopping.step(val_loss, epoch):
                print("早停触发！")
                break
    """

    def __init__(self, patience=10, delta=1e-4, mode='min', verbose=True):
        """
        Args:
            patience: 连续多少个 epoch 没有改善则停止
            delta:    视为改善的最小变化幅度
            mode:     'min' 监控指标越小越好（如 loss），'max' 越大越好（如 accuracy）
            verbose:  是否打印改善信息
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

        if mode == 'max':
            self.delta *= -1  # max 模式：改善 = score > best_score + delta

    def step(self, score, epoch):
        """
        每 epoch 调用一次，返回 True 表示应停止训练。

        Args:
            score: 当前 epoch 的验证指标值
            epoch: 当前 epoch (0-based)
        Returns:
            stop: True 表示应停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - abs(self.delta)
        else:
            improved = score > self.best_score + abs(self.delta)

        if improved:
            improvement = self.best_score - score if self.mode == 'min' else score - self.best_score
            if self.verbose:
                print(f'  ✨ Validation improved by {improvement:.6f} '
                      f'({self.best_score:.6f} → {score:.6f})')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'  ⏳ EarlyStopping: {self.counter}/{self.patience} '
                      f'(best={self.best_score:.6f} @ epoch {self.best_epoch + 1})')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


# =========================================================================
# Training Loop
# =========================================================================

def train_one_epoch(model, loader, optimizer, cfg, epoch):
    """Run one training epoch."""
    model.train()
    running_loss = {}
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{cfg.epochs} [Train]')
    for data, labels in pbar:
        data = data.to(cfg.device)
        labels = labels.to(cfg.device)

        optimizer.zero_grad()

        _, _, preds, loss_dict = model.compute_loss(data, labels)
        total_loss = cfg.lamda * (loss_dict['rec'] + loss_dict['kld'] + loss_dict['ent']) \
                     + (1 - cfg.lamda) * loss_dict['dis']
        loss_dict['total'] = total_loss

        total_loss.backward()
        if cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
        optimizer.step()

        # Accumulate stats
        for k, v in loss_dict.items():
            running_loss[k] = v.item() + running_loss.get(k, 0)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'Loss': f'{total_loss.item():.3f}',
            'Acc': f'{correct / total:.3f}',
        })

    acc = correct / total
    avg_loss = {k: v / len(loader) for k, v in running_loss.items()}

    print(f'  Train Acc: {acc:.4f} | Total Loss: {avg_loss["total"]:.3f} '
          f'(rec={avg_loss["rec"]:.3f}, kld={avg_loss["kld"]:.3f}, '
          f'ent={avg_loss["ent"]:.3f}, dis={avg_loss["dis"]:.3f})')
    return acc


@torch.no_grad()
def validate(model, loader, cfg):
    """Run validation."""
    model.eval()
    running_loss = {}
    correct = 0
    total = 0

    for data, labels in loader:
        data = data.to(cfg.device)
        labels = labels.to(cfg.device)

        _, _, preds, loss_dict = model.compute_loss(data, labels)
        total_loss = cfg.lamda * (loss_dict['rec'] + loss_dict['kld'] + loss_dict['ent']) \
                     + (1 - cfg.lamda) * loss_dict['dis']
        loss_dict['total'] = total_loss

        for k, v in loss_dict.items():
            running_loss[k] = v.item() + running_loss.get(k, 0)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = {k: v / len(loader) for k, v in running_loss.items()}

    print(f'  Val Acc: {acc:.4f} | Total Loss: {avg_loss["total"]:.3f} '
          f'(rec={avg_loss["rec"]:.3f}, kld={avg_loss["kld"]:.3f}, '
          f'ent={avg_loss["ent"]:.3f}, dis={avg_loss["dis"]:.3f})')
    return acc, avg_loss['total']


# =========================================================================
# Main Training Pipeline
# =========================================================================

def train_model(model, train_loader, val_loader, cfg):
    """Full MGPL training pipeline with prototype reset, checkpointing, and early stopping."""
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma
    )

    # Early stopping setup
    if cfg.early_stopping_metric == 'val_acc':
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience,
            delta=cfg.early_stopping_delta,
            mode='max',
        )
    else:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience,
            delta=cfg.early_stopping_delta,
            mode='min',
        )

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0

    print(f'\n{"=" * 50}')
    print('MGPL Training Started')
    print(f'  Epochs: {cfg.epochs}, Batch: {cfg.batch_size}, LR: {cfg.learning_rate}')
    print(f'  lamda: {cfg.lamda} (gen weight)')
    print(f'  Prototype reset at epochs: {cfg.prototype_reset_epochs}')
    print(f'  Early stopping: patience={cfg.early_stopping_patience}, '
          f'metric={cfg.early_stopping_metric}')
    print(f'{"=" * 50}\n')

    for epoch in range(cfg.epochs):
        print(f'\n--- Epoch {epoch + 1}/{cfg.epochs} ---')

        # Prototype reset at specified epochs
        if epoch in cfg.prototype_reset_epochs:
            model.prototypes = reset_prototypes(model, train_loader, cfg.device)

        train_acc = train_one_epoch(model, train_loader, optimizer, cfg, epoch)
        val_acc, val_loss = validate(model, val_loader, cfg)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  LR: {current_lr:.6f}')

        # Save best model (by val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(model, cfg.model_save_path,
                            extra_info={'epoch': best_epoch, 'val_acc': best_val_acc})
            print(f'  ✅ Best model saved (epoch {best_epoch}, val_acc={best_val_acc:.4f})')

        # Also track best loss for early stopping reference
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Early stopping check
        monitor_score = val_loss if cfg.early_stopping_metric == 'val_loss' else val_acc
        if early_stopping.step(monitor_score, epoch):
            print(f'\n  🛑 Early stopping triggered after {epoch + 1} epochs '
                  f'(no improvement for {cfg.early_stopping_patience} consecutive epochs)')
            break

    print(f'\n{"=" * 50}')
    print(f'Training Complete. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}')
    print(f'Best Val Loss: {best_val_loss:.6f}')
    print(f'Total epochs run: {epoch + 1}/{cfg.epochs}')
    print(f'Model saved: {cfg.model_save_path}')
    print(f'{"=" * 50}')


# =========================================================================
# Entry Point
# =========================================================================

def main():
    cfg = Config()

    # Ensure data directories exist
    os.makedirs('/data/project_lyb/Open data/matfile/12dBknown_codes', exist_ok=True)
    os.makedirs('/data/project_lyb/Open data/matfile/1unknown_codes', exist_ok=True)
    os.makedirs(os.path.dirname(cfg.eval_data_path), exist_ok=True)

    # --- 1. Define data paths ---
    base_data_dir = '/data/project_lyb/Open data/matfile'
    known_data_dir = os.path.join(base_data_dir, '12dBknown_codes')
    unknown_data_dir = os.path.join(base_data_dir, '4unknown_codes')

    # 1.1 Load known and unknown code data
    print('\n--- Loading Known Codes ---')
    features_known, coarse_known, snrs_known = load_and_combine_data(
        known_data_dir, is_known=True
    )

    print('\n--- Loading Unknown Codes ---')
    features_unknown, coarse_unknown, snrs_unknown = load_and_combine_data(
        unknown_data_dir, is_known=False
    )

    # 1.2 Map labels to continuous integers
    mapped_coarse, coarse_map = map_labels_to_continuous(coarse_known)
    cfg.num_coarse_classes = len(coarse_map)
    cfg.seq_len = features_known.shape[1]

    print(f'\nCoarse Label Map: {coarse_map}, Classes: {cfg.num_coarse_classes}')

    # --- 2. Split known: 80% train, 10% val, 10% test ---
    x_train, x_rem, y_train, y_rem, snrs_train, snrs_rem = train_test_split(
        features_known, mapped_coarse, snrs_known,
        test_size=0.2, random_state=42, shuffle=True, stratify=mapped_coarse
    )
    x_val, x_test, y_val, y_test, snrs_val, snrs_test = train_test_split(
        x_rem, y_rem, snrs_rem,
        test_size=0.5, random_state=42, shuffle=True, stratify=y_rem
    )

    # --- 3. Split unknown ---
    x_val_unk, x_test_unk, y_val_unk_orig, y_test_unk_orig, \
        snrs_val_unk, snrs_test_unk = train_test_split(
            features_unknown, coarse_unknown, snrs_unknown,
            test_size=0.5, random_state=42, shuffle=True
        )

    print(f'\n--- Data Split ---')
    print(f'Known: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}')
    print(f'Unknown: Val={len(x_val_unk)}, Test={len(x_test_unk)}')
    print('------------------\n')

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long()),
        batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).long()),
        batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    )

    # --- 5. Initialize and train ---
    device = torch.device(cfg.device)
    model = MGPLNet(cfg).to(device)
    print(f'Model initialized. latent_dim={cfg.latent_dim}, '
          f'n_sub_prototypes={cfg.n_sub_prototypes}, '
          f'n_classes={cfg.num_coarse_classes}')

    train_model(model, train_loader, val_loader, cfg)

    # --- 6. Save evaluation pickle files ---
    print('\n--- Saving evaluation data ---')

    # Validation set (known + unknown)
    x_val_eval = np.concatenate([x_val, x_val_unk], axis=0)
    y_val_eval = np.concatenate([y_val, np.full(len(y_val_unk_orig), -1)])
    snrs_val_eval = np.concatenate([snrs_val, snrs_val_unk], axis=0)

    # Test set (known + unknown)
    x_test_eval = np.concatenate([x_test, x_test_unk], axis=0)
    y_test_eval = np.concatenate([y_test, np.full(len(y_test_unk_orig), -1)])
    snrs_test_eval = np.concatenate([snrs_test, snrs_test_unk], axis=0)

    eval_data_val = {
        'features': x_val_eval,
        'coarse_labels': y_val_eval,
        'snrs': snrs_val_eval,
        'coarse_map': coarse_map,
        'num_known_val': len(x_val),
    }
    eval_data_test = {
        'features': x_test_eval,
        'coarse_labels': y_test_eval,
        'snrs': snrs_test_eval,
        'coarse_map': coarse_map,
        'num_known_test': len(x_test),
    }

    val_path = cfg.eval_data_path.replace('.pkl', '_val.pkl')
    test_path = cfg.eval_data_path.replace('.pkl', '_test.pkl')

    with open(val_path, 'wb') as f:
        pickle.dump(eval_data_val, f)
    print(f'Validation data saved: {val_path}')

    with open(test_path, 'wb') as f:
        pickle.dump(eval_data_test, f)
    print(f'Test data saved: {test_path}')

    print(f'\n🎯 Training + data prep complete!')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main()
