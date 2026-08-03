import os
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from config import Config
from models import OpenMaxOSRNet
from dataset import ChannelCodeDataset
from train import train_model

# 设置环境和随机种子
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(42)
np.random.seed(42)

# 定义验证集数据保存路径，供 evaluate.py 使用
EVAL_DATA_PATH = '/data/project_lyb/Open data/val_data_for_eval.pkl'


def map_labels_to_continuous(labels):
    """将标签映射到 0 到 C-1 的连续整数，用于 PyTorch 训练"""
    unique_labels = np.unique(labels)
    label_map = {val: i for i, val in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    return mapped_labels, label_map


def load_and_combine_data(data_dir, is_known, exclude_labels=None):
    """加载指定目录下的所有 .mat 文件中的数据，返回特征、粗标签、SNR"""
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
            print(f"Warning: No unknown .mat files found in {data_dir}. Skipping unknown data loading.")
            return np.array([]), np.array([]), np.array([])

    features_full = np.concatenate(all_features, axis=0)
    coarse_full = np.concatenate(all_coarse, axis=0)
    snrs_full = np.concatenate(all_snrs, axis=0)

    print(f"Total merged {'known' if is_known else 'unknown'} samples: {len(features_full)}")
    return features_full, coarse_full, snrs_full


def main():
    cfg = Config()

    # --- 1. 定义数据路径 ---
    base_data_dir = "/data/project_lyb/Open data/matfile"
    known_data_dir = os.path.join(base_data_dir, "12dBknown_codes")
    unknown_data_dir = os.path.join(base_data_dir, "4unknown_codes")

    # 1.1 加载所有已知码和未知码数据
    print("\n--- Loading Known Codes ---")
    features_known, coarse_known, snrs_known = load_and_combine_data(known_data_dir, is_known=True)

    print("\n--- Loading Unknown Codes ---")
    features_unknown, coarse_unknown, snrs_unknown = load_and_combine_data(unknown_data_dir, is_known=False)

    # 1.2 映射已知码标签到连续整数
    mapped_coarse_labels, coarse_map = map_labels_to_continuous(coarse_known)

    cfg.num_coarse_classes = len(coarse_map)
    cfg.seq_len = features_known.shape[1]

    print(f"\nOriginal Coarse Label Map: {coarse_map}, Total Coarse Classes: {cfg.num_coarse_classes}")

    # --- 2. 划分已知码 ---
    known_test_size = 0.2
    (
        x_train, x_rem_known,
        y_coarse_train, y_coarse_rem_known,
        snrs_train, snrs_rem_known
    ) = train_test_split(
        features_known, mapped_coarse_labels, snrs_known,
        test_size=known_test_size,
        random_state=42,
        shuffle=True,
        stratify=mapped_coarse_labels
    )

    # 第二次划分：验证 (10%) vs 测试 (10%)
    known_val_test_size = 0.5
    (
        x_val_known, x_test_known,
        y_coarse_val_known, y_coarse_test_known,
        snrs_val_known, snrs_test_known
    ) = train_test_split(
        x_rem_known, y_coarse_rem_known, snrs_rem_known,
        test_size=known_val_test_size,
        random_state=42,
        shuffle=True,
        stratify=y_coarse_rem_known
    )

    # --- 3. 划分未知码 ---
    unknown_test_size = 0.5
    (
        x_val_unknown, x_test_unknown,
        y_coarse_val_unknown_orig, y_coarse_test_unknown_orig,
        snrs_val_unknown, snrs_test_unknown
    ) = train_test_split(
        features_unknown, coarse_unknown, snrs_unknown,
        test_size=unknown_test_size,
        random_state=42,
        shuffle=True,
    )

    print(f"\n--- Data Split Results ---")
    print(f"Known Samples: Train={len(x_train)}, Val={len(x_val_known)}, Test={len(x_test_known)}")
    print(f"Unknown Samples: Val={len(x_val_unknown)}, Test={len(x_test_unknown)}")
    print("----------------------------\n")

    # --- 4. 保存训练集数据 (用于 OpenMax MAV 校准) ---
    train_data_for_eval = {
        'features': x_train,
        'coarse_labels': y_coarse_train,
        'snrs': snrs_train,
        'coarse_map': coarse_map,
        'num_known_train': len(x_train)
    }
    train_path = EVAL_DATA_PATH.replace('.pkl', '_train.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train_data_for_eval, f)
    print(f"Training data saved to {train_path} for OpenMax MAV calibration.")

    # --- 5. 训练集的 DataLoader ---
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_coarse_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 6. 已知验证集 DataLoader
    val_known_dataset = TensorDataset(
        torch.tensor(x_val_known, dtype=torch.float32),
        torch.tensor(y_coarse_val_known, dtype=torch.long)
    )
    val_known_loader = DataLoader(val_known_dataset, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)

    # 7. 初始化模型并开始训练
    model = OpenMaxOSRNet(cfg).to(cfg.device)

    print(f"Model Initialized: Coarse={cfg.num_coarse_classes}")

    train_model(model, train_loader, val_known_loader, cfg)

    # --- 8. 保存 OSR 评估数据 ---
    # 构建验证集评估数据 (Known Val + Unknown Val)
    x_val_eval = np.concatenate([x_val_known, x_val_unknown], axis=0)
    y_coarse_val_eval = np.concatenate([y_coarse_val_known, np.full(len(y_coarse_val_unknown_orig), -1)])
    import collections
    print(f"Final Val Labels Distribution: {collections.Counter(y_coarse_val_eval)}")
    snrs_val_eval = np.concatenate([snrs_val_known, snrs_val_unknown], axis=0)
    val_data_for_eval = {
        'features': x_val_eval,
        'coarse_labels': y_coarse_val_eval,
        'snrs': snrs_val_eval,
        'coarse_map': coarse_map,
        'num_known_val': len(x_val_known)
    }

    # 构建测试集评估数据 (Known Test + Unknown Test)
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
    print(f"Validation data (Mixed) saved to {val_path} for OSR evaluation.")

    with open(test_path, 'wb') as f:
        pickle.dump(test_data_for_eval, f)
    print(f"Test data (Mixed) saved to {test_path} for OSR evaluation.")


if __name__ == "__main__":
    os.makedirs('/data/Project_lc/Open data/matfile/12dBknown_codes', exist_ok=True)
    os.makedirs('/data/Project_lc/Open data/matfile/3unknown_codes', exist_ok=True)
    os.makedirs(os.path.dirname(EVAL_DATA_PATH), exist_ok=True)

    main()
