import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import collections

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from open_recognition.dataset import ChannelCodeDataset

from config import Config
from models import SoftmaxOSRNet


# 设置随机种子
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(42)
np.random.seed(42)

EVAL_DATA_PATH = '/data/project_lyb/Open data/val_data_for_eval.pkl'


def map_labels_to_continuous(labels):
    """将标签映射到 0 到 C-1 的连续整数"""
    unique_labels = np.unique(labels)
    label_map = {val: i for i, val in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    return mapped_labels, label_map


def load_and_combine_data(data_dir, is_known, exclude_labels=None):
    """加载指定目录下所有 .mat 文件中的数据"""
    all_features = []
    all_coarse = []
    all_snrs = []

    file_list = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    file_list.sort()
    print(f"Files found in {'Known' if is_known else 'Unknown'} dir: {file_list}")

    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        current_ds = ChannelCodeDataset(filepath, exclude_labels=exclude_labels)

        all_features.append(current_ds.features)
        all_coarse.append(current_ds.coarse_labels)
        all_snrs.append(current_ds.snrs)

    if not all_features:
        if is_known:
            raise FileNotFoundError(f"No .mat files found in {data_dir}.")
        else:
            print(f"Warning: No unknown .mat files found in {data_dir}. Skipping.")
            return np.array([]), np.array([]), np.array([])

    features_full = np.concatenate(all_features, axis=0)
    coarse_full = np.concatenate(all_coarse, axis=0)
    snrs_full = np.concatenate(all_snrs, axis=0)

    print(f"Total merged {'known' if is_known else 'unknown'} samples: {len(features_full)}")
    return features_full, coarse_full, snrs_full


def save_checkpoint(model, path):
    """保存模型权重"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model': model.state_dict(),
    }
    torch.save(state, path)


def train_model(model, train_loader, val_loader, cfg):
    """使用仅交叉熵损失训练模型"""
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
        min_lr=1e-6
    )

    # 3. 早停和最佳模型追踪
    best_loss = float('inf')
    best_epoch = 0
    patience = getattr(cfg, "patience", 40)
    patience_counter = 0

    print(f"--- Start Softmax Training (CrossEntropyLoss only) ---")
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

            # 唯一使用的损失：交叉熵
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
            print(f"✅ Best Model Saved! (Epoch {best_epoch}, Val Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"ℹ️ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"Early Stopping! Best Epoch: {best_epoch}")
            break

    print(f"\nTraining Finished. Best Val Loss: {best_loss:.4f} at Epoch {best_epoch}")


def main():
    cfg = Config()

    # --- 1. 定义数据路径 ---
    base_data_dir = "/data/Project_lc/Open data/matfile"
    known_data_dir = os.path.join(base_data_dir, "12dBknown_codes")
    unknown_data_dir = os.path.join(base_data_dir, "1unknown_codes")

    # 1.1 加载所有已知码和未知码数据
    print("\n--- Loading Known Codes ---")
    features_known, coarse_known, snrs_known = load_and_combine_data(known_data_dir, is_known=True)

    print("\n--- Loading Unknown Codes ---")
    features_unknown, coarse_unknown, snrs_unknown = load_and_combine_data(unknown_data_dir, is_known=False)

    # 1.2 映射已知码标签到连续整数
    mapped_coarse_labels, coarse_map = map_labels_to_continuous(coarse_known)

    cfg.num_coarse_classes = len(coarse_map)
    cfg.seq_len = features_known.shape[1]

    print(f"\nCoarse Label Map: {coarse_map}, Total Coarse Classes: {cfg.num_coarse_classes}")

    # --- 2. 划分已知码：80% 训练, 10% 验证, 10% 测试 ---
    known_test_size = 0.2
    x_train, x_rem_known, y_coarse_train, y_coarse_rem_known, snrs_train, snrs_rem_known = train_test_split(
        features_known, mapped_coarse_labels, snrs_known,
        test_size=known_test_size, random_state=42, shuffle=True, stratify=mapped_coarse_labels
    )

    known_val_test_size = 0.5
    x_val_known, x_test_known, y_coarse_val_known, y_coarse_test_known, snrs_val_known, snrs_test_known = train_test_split(
        x_rem_known, y_coarse_rem_known, snrs_rem_known,
        test_size=known_val_test_size, random_state=42, shuffle=True, stratify=y_coarse_rem_known
    )

    # --- 3. 划分未知码 ---
    unknown_test_size = 0.5
    x_val_unknown, x_test_unknown, y_coarse_val_unknown_orig, y_coarse_test_unknown_orig, snrs_val_unknown, snrs_test_unknown = train_test_split(
        features_unknown, coarse_unknown, snrs_unknown,
        test_size=unknown_test_size, random_state=42, shuffle=True,
    )

    print(f"\n--- Data Split Results ---")
    print(f"Known Samples: Train={len(x_train)}, Val={len(x_val_known)}, Test={len(x_test_known)}")
    print(f"Unknown Samples: Val={len(x_val_unknown)}, Test={len(x_test_unknown)}")
    print("----------------------------\n")

    # --- 4. 训练集 DataLoader ---
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_coarse_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 5. 验证集 DataLoader
    val_known_dataset = TensorDataset(
        torch.tensor(x_val_known, dtype=torch.float32),
        torch.tensor(y_coarse_val_known, dtype=torch.long),
    )
    val_known_loader = DataLoader(val_known_dataset, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)

    # 6. 初始化模型并开始训练
    model = SoftmaxOSRNet(cfg).to(cfg.device)
    print(f"Model Initialized: Coarse classes={cfg.num_coarse_classes}")

    train_model(model, train_loader, val_known_loader, cfg)

    # --- 7. 保存 OSR 评估数据 ---
    # 构建验证集评估数据
    x_val_eval = np.concatenate([x_val_known, x_val_unknown], axis=0)
    y_coarse_val_eval = np.concatenate([y_coarse_val_known, np.full(len(y_coarse_val_unknown_orig), -1)])
    snrs_val_eval = np.concatenate([snrs_val_known, snrs_val_unknown], axis=0)
    val_data_for_eval = {
        'features': x_val_eval,
        'coarse_labels': y_coarse_val_eval,
        'snrs': snrs_val_eval,
        'coarse_map': coarse_map,
        'num_known_val': len(x_val_known)
    }

    # 构建测试集评估数据
    x_test_eval = np.concatenate([x_test_known, x_test_unknown], axis=0)
    y_coarse_test_eval = np.concatenate([y_coarse_test_known, np.full(len(y_coarse_test_unknown_orig), -1)])
    snrs_test_eval = np.concatenate([snrs_test_known, snrs_test_unknown], axis=0)
    test_data_for_eval = {
        'features': x_test_eval,
        'coarse_labels': y_coarse_test_eval,
        'snrs': snrs_test_eval,
        'coarse_map': coarse_map,
        'num_known_test': len(x_test_known)
    }

    val_path = EVAL_DATA_PATH.replace('.pkl', '_val.pkl')
    test_path = EVAL_DATA_PATH.replace('.pkl', '_test.pkl')
    print(f"Unique labels in validation eval: {np.unique(y_coarse_val_eval)}")
    with open(val_path, 'wb') as f:
        pickle.dump(val_data_for_eval, f)
    print(f"Validation data saved to {val_path}")

    with open(test_path, 'wb') as f:
        pickle.dump(test_data_for_eval, f)
    print(f"Test data saved to {test_path}")


if __name__ == "__main__":
    os.makedirs('/data/Project_lc/Open data/matfile/12dBknown_codes', exist_ok=True)
    os.makedirs('/data/Project_lc/Open data/matfile/3unknown_codes', exist_ok=True)
    os.makedirs(os.path.dirname(EVAL_DATA_PATH), exist_ok=True)
    main()
