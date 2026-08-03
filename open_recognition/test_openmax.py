import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from models import MultiTaskOSRNet
from openmax import (compute_train_score_and_mavs_and_dists, 
                     fit_weibull, openmax)
from sklearn.metrics import classification_report, confusion_matrix

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def run_test():
    cfg = Config()
    
    # 1. 加载模型
    print("--- 正在加载模型权重 ---")
    model = MultiTaskOSRNet(cfg).to(device)
    # 注意：确保你的路径正确
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # 2. 准备校准数据（Calibration）
    # OpenMax 需要用【训练集】中分类正确的样本来拟合 Weibull 分布
    # 这里加载你 main.py 中保存的训练数据或重新加载已知码
    # 假设你已经准备好了 train_loader (只有已知类)
    print("--- 正在计算训练集的 MAV (平均激活向量) ---")
    # 注意：这里需要你传入 train_loader。如果没有，可以临时构建一个。
    # 这里的 num_coarse_classes 对应已知类的数量
    
    # 假设我们重新加载训练特征
    # train_data = load_data_from_pickle('path_to_train_data.pkl')
    # train_loader = DataLoader(TensorDataset(torch.tensor(train_data['features']), 
    #                                       torch.tensor(train_data['labels'])), batch_size=64)
    
    # 调用 openmax.py 里的函数
    # 注意：MultiTaskOSRNet 返回 (feat, logits_coarse, logits_fine)
    # 我们对 logits_coarse 进行 OpenMax
    _, mavs, dists = compute_train_score_and_mavs_and_dists(
        cfg.num_coarse_classes, train_loader, device, model
    )

    # 3. 拟合 Weibull 模型
    print("--- 正在拟合 Weibull 分布 (EVT理论) ---")
    categories = list(range(cfg.num_coarse_classes))
    weibull_model = fit_weibull(mavs, dists, categories, tailsize=20, distance_type='eucos')

    # 4. 加载混合测试集 (包含已知类和未知类)
    print("--- 正在加载混合测试集 ---")
    test_data = load_data_from_pickle('/data/Project_lc/Open data/val_data_for_eval_test.pkl')
    test_features = torch.tensor(test_data['features'], dtype=torch.float32)
    test_labels_coarse = test_data['coarse_labels'] # 已知: 0~N-1, 未知: -1
    
    # 5. 执行 OpenMax 推理
    print("--- 正在执行 OpenMax 推理与拒识 ---")
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for i in range(len(test_features)):
            input_tensor = test_features[i:i+1].to(device)
            _, logits_c, _ = model(input_tensor)
            
            # 将 logits 转为 numpy 并调整维度符合 openmax.py 预期 (channel, C)
            # 信号处理中通常 channel=1
            input_score = logits_c.cpu().numpy()
            input_score = input_score[:, np.newaxis, :] 

            # 调用 OpenMax
            # alpha 是考虑的前几个最高得分的类别，threshold 是拒识阈值
            openmax_prob, _ = openmax(weibull_model, categories, input_score[0], 
                                     eu_weight=0.5, alpha=10, distance_type='eucos')
            
            # openmax_prob 的最后一个元素是“未知类”的概率
            pred_class = np.argmax(openmax_prob)
            
            # 如果 pred_class == cfg.num_coarse_classes，说明被判定为未知类
            # 映射回 -1 方便对比
            final_pred = pred_class if pred_class < cfg.num_coarse_classes else -1
            
            y_pred.append(final_pred)
            y_true.append(test_labels_coarse[i])

    # 6. 输出评估结果
    print("\n--- 开放集识别结果汇总 ---")
    print(classification_report(y_true, y_pred, digits=4))
    
    # 计算拒识成功率
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:")
    print(cm)

if __name__ == "__main__":
    # 注意：运行前请确保环境中已安装 libmr
    # pip install libmr
    run_test()