# dataset.py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import h5py
import os
import pickle
from sklearn.model_selection import train_test_split


class ChannelCodeDataset(Dataset):
    def __init__(self, filepath, normalize=True, exclude_labels=None):
        """
        filepath: 单个 .mat 文件路径
        normalize: 是否对 LLR 序列归一化
        """
        self.normalize = normalize
        self.exclude_labels = exclude_labels if exclude_labels is not None else []#排除标签
        # 属性初始化
        self.features = None
        self.coarse_labels = None
        self.fine_labels = None
        self.snrs = None

        self._load_file(filepath)

    def _load_file(self, filepath):
        """
        鲁棒地加载 .mat 文件，支持 v5 和 v7.3 格式。
        约定：倒数第三列是 SNR，倒数第二列是 Coarse_Label，倒数第一列是 Fine_Label。
        """
        filename = os.path.basename(filepath)
        print(f"Loading {filename} ...")

        try:
            # 尝试加载 MATLAB v5 文件
            mat = sio.loadmat(filepath)
            key = [k for k in mat.keys() if not k.startswith("__")][0]#这里每个.mat文件只有一个变量
            data = mat[key]
        except NotImplementedError:
            # 可能是 MATLAB v7.3 文件，使用 h5py
            print(f"MATLAB v7.3 detected, using h5py...")
            with h5py.File(filepath, 'r') as f:
                # 假设数据是根目录下唯一的数组，并进行转置以匹配行样本/列特征的结构
                key = list(f.keys())[0]
                data = np.array(f[key]).T
                #或data = f[key][:].T
        except Exception as e:
            raise IOError(f"Error loading {filepath}: {e}")
        

        coarse_all = data[:, -2].astype(np.int64)
        mask = ~np.isin(coarse_all, self.exclude_labels)
        filtered_data = data[mask]
        print(f"Original samples: {len(data)}, After filtering: {len(filtered_data)}")
        # 划分过滤后的数据列
        X = filtered_data[:, :-3].astype(np.float32)
        snr = filtered_data[:, -3].astype(np.float32)
        coarse = filtered_data[:, -2].astype(np.int64)
        fine = filtered_data[:, -1].astype(np.int64)

        # 归一化 (如果需要)
        if self.normalize:
            # 假设您的 LLR 数据最大值为 32767.0，归一化到 [-1, 1]
            X /= 32767.0

        self.features = X
        self.snrs = snr
        self.coarse_labels = coarse
        self.fine_labels = fine

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 仅返回特征和粗标签（用于 OSR 任务的已知分类）
        # main.py 的 DataLoader 只需要这两个
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        coarse = torch.tensor(self.coarse_labels[idx], dtype=torch.long)
        return x, coarse


# 辅助函数：将训练集的原始标签映射到 PyTorch 连续标签 0, 1, 2, 3
# 此函数应在 main.py 中被调用，用于处理聚合后的已知码标签
def map_labels_to_continuous(coarse_labels):
    unique_labels = np.unique(coarse_labels)#提取不重复的值并排序
    label_map = {val: i for i, val in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in coarse_labels])
    return mapped_labels, label_map#返回映射后的标签和映射字典（如果需要反向映射）