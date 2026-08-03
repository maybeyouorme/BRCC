"""
ARPL 对比实验入口脚本。

流程:
  1. 加载已知码/未知码数据并划分 (train/val/test)
  2. 初始化 ARPLNet 和 ARPLoss
  3. 训练模型
  4. 在验证集上进行 OSR 评估 (AUROC, OSCR, ACC)
  5. 保存评估数据以供 evaluator 绘图

用法:
    cd ARPL
    CUDA_VISIBLE_DEVICES=0 python main.py

参考:
  - ARPL-master/osr.py:   ARPL 原文的入口逻辑
  - MGPL/train.py:        1D 信号实验的 main() 函数
  - Softmax/main.py:      对比实验的 main() 函数
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from open_recognition.dataset import ChannelCodeDataset

from config import Config
from models import ARPLNet
from loss import ARPLoss, RPLoss, Softmax
from train import train_model
from evaluator import evaluate_and_plot

import collections
import warnings
warnings.filterwarnings('ignore')


# =========================================================================
# 数据加载工具函数
# =========================================================================

def map_labels_to_continuous(labels):
    """将原始标签映射到 0..C-1 的连续整数。"""
    unique_labels = np.unique(labels)
    label_map = {val: i for i, val in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    return mapped_labels, label_map


def load_and_combine_data(data_dir, is_known, exclude_labels=None):
    """加载指定目录下所有 .mat 文件并合并。"""
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


# =========================================================================
# 主函数
# =========================================================================

def main():
    cfg = Config()

    torch.manual_seed(42)
    np.random.seed(42)

    # 确保数据目录存在
    os.makedirs('/data/project_lyb/Open data/matfile/12dBknown_codes', exist_ok=True)
    os.makedirs('/data/project_lyb/Open data/matfile/1unknown_codes', exist_ok=True)
    os.makedirs(os.path.dirname(cfg.eval_data_path), exist_ok=True)

    # --- 1. 加载数据 ---
    base_data_dir = '/data/project_lyb/Open data/matfile'
    known_data_dir = os.path.join(base_data_dir, '12dBknown_codes')
    unknown_data_dir = os.path.join(base_data_dir, '4unknown_codes')

    print('\n--- Loading Known Codes ---')
    features_known, coarse_known, snrs_known = load_and_combine_data(
        known_data_dir, is_known=True
    )

    print('\n--- Loading Unknown Codes ---')
    features_unknown, coarse_unknown, snrs_unknown = load_and_combine_data(
        unknown_data_dir, is_known=False
    )

    # --- 2. 标签映射 ---
    mapped_coarse, coarse_map = map_labels_to_continuous(coarse_known)
    cfg.num_coarse_classes = len(coarse_map)
    cfg.seq_len = features_known.shape[1]

    print(f'\nCoarse Label Map: {coarse_map}, Classes: {cfg.num_coarse_classes}')

    # --- 3. 数据划分 ---
    # 已知类: 80% train, 10% val, 10% test
    x_train, x_rem, y_train, y_rem, snrs_train, snrs_rem = train_test_split(
        features_known, mapped_coarse, snrs_known,
        test_size=0.2, random_state=42, shuffle=True, stratify=mapped_coarse
    )
    x_val, x_test, y_val, y_test, snrs_val, snrs_test = train_test_split(
        x_rem, y_rem, snrs_rem,
        test_size=0.5, random_state=42, shuffle=True, stratify=y_rem
    )

    # 未知类: 50% val, 50% test
    x_val_unk, x_test_unk, y_val_unk_orig, y_test_unk_orig, \
        snrs_val_unk, snrs_test_unk = train_test_split(
            features_unknown, coarse_unknown, snrs_unknown,
            test_size=0.5, random_state=42, shuffle=True
        )

    print(f'\n--- Data Split ---')
    print(f'Known: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}')
    print(f'Unknown: Val={len(x_val_unk)}, Test={len(x_test_unk)}')
    print('------------------\n')

    # --- 4. 创建 DataLoader ---
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long()),
        batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).long()),
        batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).long()),
        batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    )
    out_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_unk).float(),
                      torch.tensor(np.zeros(len(x_test_unk))).long()),  # 占位标签
        batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    )

    # --- 5. 初始化模型和损失函数 ---
    device = torch.device(cfg.device)
    model = ARPLNet(cfg).to(device)

    # 根据配置选择损失函数
    if cfg.loss_type == 'ARPLoss':
        criterion = ARPLoss(
            num_classes=cfg.num_coarse_classes,
            feat_dim=cfg.hidden_shared,
            weight_pl=cfg.weight_pl,
            temp=cfg.temp,
        ).to(device)
    elif cfg.loss_type == 'RPLoss':
        criterion = RPLoss(
            num_classes=cfg.num_coarse_classes,
            feat_dim=cfg.hidden_shared,
            weight_pl=cfg.weight_pl,
            temp=cfg.temp,
            num_centers=cfg.num_centers,
        ).to(device)
    elif cfg.loss_type == 'Softmax':
        criterion = Softmax(temp=cfg.temp).to(device)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    print(f'\nModel: ARPLNet ({sum(p.numel() for p in model.parameters()):,} params)')
    print(f'Loss: {cfg.loss_type}')
    print(f'Device: {device}')

    # --- 6. 训练 ---
    history = train_model(model, criterion, train_loader, val_loader, cfg)

    # --- 7. 加载最优模型进行评估 ---
    checkpoint = torch.load(cfg.model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if 'criterion' in checkpoint:
        criterion.load_state_dict(checkpoint['criterion'])
    print(f'\nLoaded best model from {cfg.model_save_path}')

    # --- 8. OSR 评估 + 可视化 (调用 evaluator) ---
    results = evaluate_and_plot(
        model, criterion, test_loader, out_loader, cfg,
        coarse_map=coarse_map, train_history=history
    )

    # --- 9. 保存评估数据 (用于后续对比分析) ---
    print('\n--- Saving evaluation data ---')

    # 验证集 (已知 + 未知)
    x_val_eval = np.concatenate([x_val, x_val_unk], axis=0)
    y_val_eval = np.concatenate([y_val, np.full(len(y_val_unk_orig), -1)])
    snrs_val_eval = np.concatenate([snrs_val, snrs_val_unk], axis=0)

    # 测试集 (已知 + 未知)
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

    # 保存时注明 loss_type 以区分不同实验
    suffix = f'_{cfg.loss_type}'
    val_path = cfg.eval_data_path.replace('.pkl', f'{suffix}_val.pkl')
    test_path = cfg.eval_data_path.replace('.pkl', f'{suffix}_test.pkl')

    with open(val_path, 'wb') as f:
        pickle.dump(eval_data_val, f)
    print(f'Validation data saved: {val_path}')

    with open(test_path, 'wb') as f:
        pickle.dump(eval_data_test, f)
    print(f'Test data saved: {test_path}')

    # --- 10. 打印汇总 ---
    print(f'\n{"=" * 50}')
    print(f'ARPL ({cfg.loss_type}) Results Summary:')
    print(f'  ACC:   {results["ACC"]:.2f}%')
    print(f'  AUROC: {results["AUROC"]:.2f}%')
    print(f'  OSCR:  {results["OSCR"]:.2f}%')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
