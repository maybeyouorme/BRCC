# main.py
import os
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from config import Config
from models import MultiTaskOSRNet # <-- 更改模型名称
from dataset import ChannelCodeDataset, map_labels_to_continuous
from train import train_model


#这个是fix/known-class-number分支


# 设置环境和随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
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
    """加载指定目录下的所有 .mat 文件中的数据，返回特征、粗标签、细标签"""
    all_features = []
    all_coarse = []
    all_fine = [] # <-- 新增细标签
    all_snrs = []

    file_list = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    #遍历 data_dir 文件夹里的所有文件，只把后缀是 .mat 的文件名挑出来，放进列表 file_list 里
    file_list.sort()
    print(f"Files found in {'Known' if is_known else 'Unknown'} dir: {file_list}")
    
    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        current_ds = ChannelCodeDataset(filepath, exclude_labels=exclude_labels)
        
        # 收集特征、粗标签和细标签
        all_features.append(current_ds.features)
        all_coarse.append(current_ds.coarse_labels) 
        all_fine.append(current_ds.fine_labels) # <-- 收集细标签
        all_snrs.append(current_ds.snrs)

    if not all_features:
        # ... (错误处理保持不变)
        if is_known:
            raise FileNotFoundError(f"No .mat files found in {data_dir}.")
        else:
            print(f"Warning: No unknown .mat files found in {data_dir}. Skipping unknown data loading.")
            return np.array([]), np.array([]), np.array([])
        
    features_full = np.concatenate(all_features, axis=0)
    coarse_full = np.concatenate(all_coarse, axis=0)
    fine_full = np.concatenate(all_fine, axis=0) # <-- 合并细标签
    snrs_full = np.concatenate(all_snrs, axis=0)
    
    print(f"Total merged {'known' if is_known else 'unknown'} samples: {len(features_full)}")
    return features_full, coarse_full, fine_full, snrs_full # <-- 返回细标签

def main():
    cfg = Config()
    
    # --- 1. 定义数据路径 ---
    base_data_dir = "/data/Project_lc/Open data/matfile"#TODO
    known_data_dir = os.path.join(base_data_dir, "12dBknown_codes")
    unknown_data_dir = os.path.join(base_data_dir, "1unknown_codes")
    
    # 1.1 加载所有已知码和未知码数据 (新增 fine_full)
    print("\n--- Loading Known Codes ---")
    features_known, coarse_known, fine_known, snrs_known = load_and_combine_data(known_data_dir, is_known=True)
    #[15000, 8192] [15000,] [15000,] [15000,]

    print("\n--- Loading Unknown Codes ---")
    features_unknown, coarse_unknown, fine_unknown, snrs_unknown = load_and_combine_data(unknown_data_dir, is_known=False)
    #1unknown:[3000, 8192] [3000,] [3000,] [3000,]
    #2unknown:[3000, 8192] [3000,] [3000,] [3000,] 12dBunknown
    #3unknown:[3006, 8192] [3006,] [3006,] [3006,]

    # 1.2 映射已知码标签到连续整数
    mapped_coarse_labels, coarse_map = map_labels_to_continuous(coarse_known)
    mapped_fine_labels, fine_map = map_labels_to_continuous(fine_known) # <-- 映射细标签

    cfg.num_coarse_classes = len(coarse_map) # 更新粗类别数
    cfg.num_fine_classes = len(fine_map)   # <-- 更新细类别数
    cfg.seq_len = features_known.shape[1]    # <-- 更新序列长度
    
    print(f"\nOriginal Coarse Label Map: {coarse_map}, Total Coarse Classes: {cfg.num_coarse_classes}")
    #[1,2,3,4,5]-->[0,1,2,3,4]
    print(f"Original Fine Label Map: {fine_map}, Total Fine Classes: {cfg.num_fine_classes}")
    #[1,2,...,14,15]-->[0,1,...,13,14]


    # --- 2. 划分已知码：80% 训练, 10% 验证, 10% 测试 (使用细标签进行分层抽样，确保参数均衡) ---
    known_test_size = 0.2
    # 使用细标签进行分层抽样，确保训练集中不同参数组合的均衡性
    (
        x_train, x_rem_known,
        y_coarse_train, y_coarse_rem_known,
        y_fine_train, y_fine_rem_known ,# <-- 细标签也加入划分
        snrs_train, snrs_rem_known
    ) = train_test_split(
        features_known, mapped_coarse_labels, mapped_fine_labels, snrs_known,# <-- 传入细标签
        test_size=known_test_size,
        random_state=42,
        shuffle=True,
        stratify=mapped_fine_labels # <-- 关键：按细标签分层，保证所有参数组合都有样本
    )
    
    # 第二次划分：验证 (10%) vs 测试 (10%)
    known_val_test_size = 0.5
    (
        x_val_known, x_test_known,
        y_coarse_val_known, y_coarse_test_known,
        y_fine_val_known, y_fine_test_known,
        snrs_val_known, snrs_test_known
    ) = train_test_split(
        x_rem_known, y_coarse_rem_known, y_fine_rem_known, snrs_rem_known,# <-- 细标签也加入划分
        test_size=known_val_test_size,
        random_state=42,
        shuffle=True,
        stratify=y_fine_rem_known # <-- 按细标签分层
    )

    # --- 3. 划分未知码 (保持不变) ---
    unknown_test_size = 0.5
    (
        x_val_unknown, x_test_unknown,
        y_coarse_val_unknown_orig, y_coarse_test_unknown_orig,
        y_fine_val_unknown_orig, y_fine_test_unknown_orig, # <-- 细标签也加入划分
        snrs_val_unknown, snrs_test_unknown
    ) = train_test_split(
        features_unknown, coarse_unknown, fine_unknown, snrs_unknown,
        test_size=unknown_test_size,
        random_state=42,
        shuffle=True,
    )
    
    # ... (打印结果保持不变) ...
    print(f"\n--- Data Split Results ---")
    print(f"Known Samples: Train={len(x_train)}, Val={len(x_val_known)}, Test={len(x_test_known)}")
    print(f"Unknown Samples: Val={len(x_val_unknown)}, Test={len(x_test_unknown)}")
    print("----------------------------\n")

    # --- 4. 训练集的 DataLoader (仅已知码) ---
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_coarse_train, dtype=torch.long), # 粗标签
        torch.tensor(y_fine_train, dtype=torch.long),    # <-- 细标签
        torch.tensor(snrs_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # 5. 已知验证集 DataLoader
    val_known_dataset = TensorDataset(
        torch.tensor(x_val_known, dtype=torch.float32),
        torch.tensor(y_coarse_val_known, dtype=torch.long),
        torch.tensor(y_fine_val_known, dtype=torch.long), # <-- 细标签
        torch.tensor(snrs_val_known, dtype=torch.float32)
    )
    val_known_loader = DataLoader(val_known_dataset, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)
    
    # 6. 初始化模型并开始训练
    model = MultiTaskOSRNet(cfg).to(cfg.device) # <-- 实例化新模型
    
    # 打印模型结构，确认多头和AE存在
    print(f"Model Initialized: Coarse={cfg.num_coarse_classes}, Fine={cfg.num_fine_classes}, AE={cfg.use_autoencoder}")

    train_model(model, train_loader, val_known_loader, cfg)

    # --- 7. 保存 OSR 评估数据 (混合集) ---
    # 评估数据只需要特征和粗标签 (细标签不用于 OSR 判别，但可以用于闭集精度评估)
    
    # 构建验证集评估数据 (Known Val + Unknown Val)
    x_val_eval = np.concatenate([x_val_known, x_val_unknown], axis=0)
    # 未知码粗标签统一设置为 -1
    y_coarse_val_eval = np.concatenate([y_coarse_val_known, np.full(len(y_coarse_val_unknown_orig), -1)]) 
    import collections
    print(f"Final Val Labels Distribution: {collections.Counter(y_coarse_val_eval)}")
    # 细标签也保存，方便 evaluate.py 评估闭集细分类精度
    y_fine_val_eval = np.concatenate([y_fine_val_known, np.full(len(y_fine_val_unknown_orig), -1)]) 
    snrs_val_eval = np.concatenate([snrs_val_known, snrs_val_unknown], axis=0)
    val_data_for_eval = {
        'features': x_val_eval,
        'coarse_labels': y_coarse_val_eval, 
        'fine_labels': y_fine_val_eval, # <-- 保存细标签
        'snrs':snrs_val_eval,
        'coarse_map': coarse_map, 
        'fine_map': fine_map,
        'num_known_val': len(x_val_known) 
    }
    
    # 构建测试集评估数据 (Known Test + Unknown Test)
    x_test_eval = np.concatenate([x_test_known, x_test_unknown], axis=0)
    y_coarse_test_eval = np.concatenate([y_coarse_test_known, np.full(len(y_coarse_test_unknown_orig), -1)])
    y_fine_test_eval = np.concatenate([y_fine_test_known, np.full(len(y_fine_test_unknown_orig), -1)]) 
    snrs_test_eval = np.concatenate([snrs_test_known, snrs_test_unknown], axis=0)

    test_data_for_eval = {
        'features': x_test_eval,
        'coarse_labels': y_coarse_test_eval, 
        'fine_labels': y_fine_test_eval, # <-- 保存细标签
        'snrs':snrs_test_eval,
        'coarse_map': coarse_map,
        'fine_map': fine_map,
        'num_known_test': len(x_test_known)
    }
    
    # ... (保存文件逻辑保持不变) ...
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
    # 确保数据目录存在
    os.makedirs('/data/Project_lc/Open data/matfile/12dBknown_codes', exist_ok=True)
    os.makedirs('/data/Project_lc/Open data/matfile/3unknown_codes', exist_ok=True)
    os.makedirs(os.path.dirname(EVAL_DATA_PATH), exist_ok=True)
    
    main()