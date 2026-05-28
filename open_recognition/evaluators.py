import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from thop import profile, clever_format
# 核心：导入你之前准备好的 openmax 函数
from openmax import compute_train_score_and_mavs_and_dists, fit_weibull, openmax

# --- 配置与常量 ---
COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH"]

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = "./checkpoits9/best_model.pth"
        # 注意：这里需要训练集和测试集两个路径
        self.train_data_path = '/data/Project_lc/Open data/val_data_for_eval_val.pkl' # 假设这是训练/验证集
        self.test_data_path = '/data/Project_lc/Open data/val_data_for_eval_test.pkl'
        self.results_dir = "./checkpoints/openmax_results"
        self.batch_size = 64
        self.seq_len = 8192
        self.num_coarse_classes = 5
        self.num_fine_classes = 15
        self.n_fft = 256
        self.hop_length = 128
        self.cnn_base_channels = 64
        self.cnn_layers = [2, 2, 2]
        self.cnn_out = 128
        self.lstm_frame_size = 128 
        self.lstm_hidden_size = 128
        self.lstm_layers = 2
        self.lstm_out = 128
        self.spec_out = 64
        self.hidden_shared = 128
        self.dropout = 0.3 
        self.use_autoencoder = True 
        os.makedirs(self.results_dir, exist_ok=True)

# --- OpenMax 特有逻辑 ---

def calibrate_openmax(model, train_loader, cfg):
    print("📏 正在进行 OpenMax 校准 (计算训练集特征分布)...")
    model.eval()
    
    scores = [[] for _ in range(cfg.num_coarse_classes)]
    
    with torch.no_grad():
        for x, y_c, _, _ in train_loader:
            x, y_c = x.to(cfg.device), y_c.to(cfg.device)
            output = model(x)
            c_logits = output['coarse_logits']
            
            for i in range(c_logits.size(0)):
                logit = c_logits[i]
                label = y_c[i].item()
                
                # 只有预测正确的样本才用于拟合 Weibull 分布
                if torch.argmax(logit) == label:
                    # 适配 OpenMax 库需要的 (1, 1, C) 维度
                    scores[label].append(logit.unsqueeze(0).unsqueeze(0).cpu().numpy())

    # --- 增加防御性代码和调试信息 ---
    scores_compact = []
    active_categories = []
    
    for idx, s_list in enumerate(scores):
        if len(s_list) > 0:
            scores_compact.append(np.concatenate(s_list, axis=0))
            active_categories.append(idx)
        else:
            print(f"⚠️ 警告: 类别索引 {idx} ({COARSE_LABEL_NAMES[idx]}) 没有分类正确的样本，将被跳过！")

    if not scores_compact:
        raise ValueError("❌ 错误：所有类别都没有正确分类的样本，请检查模型权重或数据集标签！")

    # 计算 MAVs
    mavs = np.array([np.mean(x, axis=0) for x in scores_compact])
    
    # 计算距离
    from openmax import compute_channel_distances
    dists = [compute_channel_distances(mcv, s) for mcv, s in zip(mavs, scores_compact)]
    
    # 拟合 Weibull (注意传入的是 active_categories)
    weibull_model = fit_weibull(mavs, dists, active_categories, tailsize=20, distance_type='eucos')
    
    return weibull_model, active_categories

@torch.no_grad()
def get_predictions_openmax(model, loader, weibull_model, categories, cfg):
    """第二步：执行 OpenMax 推理"""
    model.eval()
    results = {'c_true': [], 'c_pred': [], 'features': []}
    
    for x, y_c, _, _ in loader:
        x = x.to(cfg.device)
        output = model(x)
        c_logits = output['coarse_logits']
        feat = output['embedding']
        
        # 对 Batch 中的每个样本运行 OpenMax
        for i in range(c_logits.size(0)):
            # 准备单样本 score (channel=1, classes=C)
            score = c_logits[i].unsqueeze(0).cpu().numpy()
            
            # 核心调用
            openmax_prob, _ = openmax(weibull_model, categories, score, 
                                     eu_weight=0.5, alpha=min(len(categories), 5), distance_type='eucos')
            
            # 结果判定：如果最大概率索引等于类数，则是未知类 (-1)
            pred = np.argmax(openmax_prob)
            final_pred = pred if pred < cfg.num_coarse_classes else -1
            
            results['c_pred'].append(final_pred)
            results['c_true'].append(y_c[i].item())
            results['features'].append(feat[i].cpu().numpy())

    for k in results:
        results[k] = np.array(results[k])
    return results

# --- 绘图函数直接沿用你的 MSP 版本 (略作适配) ---
# plot_oscr, plot_tsne, plot_confusion_matrix_custom 保持不变

# --- Main 函数 ---

def main():
    cfg = Config()
    print(f"DEBUG: 正在检查权重路径: {os.path.abspath(cfg.model_save_path)}") # 打印绝对路径
    print(f"DEBUG: 路径是否存在: {os.path.exists(cfg.model_save_path)}")
    from models import MultiTaskOSRNet 
    model = MultiTaskOSRNet(cfg).to(cfg.device)
    
    # 加载权重
    if os.path.exists(cfg.model_save_path):
        checkpoint = torch.load(cfg.model_save_path, map_location=cfg.device)
        # 兼容你的两种保存格式（直接存dict或存{'model':...}）
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"✅ 加载权重成功")
    else:
        print(f"❌ 严重错误: 找不到权重文件！当前尝试路径为: {cfg.model_save_path}")
        return
    # 统计复杂度
    dummy_in = torch.randn(1, 1, cfg.seq_len).to(cfg.device)
    flops, params = profile(model, inputs=(dummy_in,), verbose=False)
    print(f"📊 Params: {clever_format([params], '%.3f')}, GFLOPS: {clever_format([flops], '%.3f')}")

    # --- 1. 加载校准集 (训练集中的已知类) ---
    with open(cfg.train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_data['features']).float(),
        torch.tensor(train_data['coarse_labels']).long(),
        torch.tensor(train_data['fine_labels']).long(),
        torch.tensor(train_data['snrs']).float()
    ), batch_size=cfg.batch_size, shuffle=False)

    # 执行校准
    weibull_model, categories = calibrate_openmax(model, train_loader, cfg)

    # --- 2. 加载测试集 (包含未知类) ---
    with open(cfg.test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(test_data['features']).float(),
        torch.tensor(test_data['coarse_labels']).long(),
        torch.tensor(test_data['fine_labels']).long(),
        torch.tensor(test_data['snrs']).float()
    ), batch_size=cfg.batch_size, shuffle=False)

    # --- 3. 执行 OpenMax 推理 ---
    print(f"🚀 正在执行 OpenMax 开集评估...")
    test_res = get_predictions_openmax(model, test_loader, weibull_model, categories, cfg)

    # --- 4. 计算指标 ---
    y_true = test_res['c_true']
    y_pred = test_res['c_pred']
    
    # 闭集准确率 (仅看已知类且预测不为-1的部分)
    known_mask = y_true != -1
    closed_oa = accuracy_score(y_true[known_mask], y_pred[known_mask])
    # 开集准确率 (整体)
    open_oa = accuracy_score(y_true, y_pred)

    print("\n" + "="*30)
    print(f"📈 [OpenMax 结果报告]")
    print(f"闭集分类准确率 (Closed Acc): {closed_oa*100:.2f}%")
    print(f"开集识别准确率 (Total Acc):  {open_oa*100:.2f}%")
    print("="*30)

    # --- 5. 可视化 ---
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # 定义标签名称 (0-4 是已知类，-1 是未知类)
    labels = [0, 1, 2, 3, 4, -1]
    target_names = ['Conv', 'LDPC', 'Turbo', 'Polar', 'BCH', 'Unknown']

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title('OpenMax Recognition Confusion Matrix')
    
    # 保存图片
    save_path = os.path.join(cfg.results_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"📊 混淆矩阵已保存至: {save_path}")
    
    # 如果你想跑 t-SNE，确保代码里有 plot_tsne 的定义，或者先注释掉
    # plot_tsne(test_res, cfg.results_dir) 

    print(f"\n✨ 评估完成！结果保存在: {cfg.results_dir}")

if __name__ == "__main__":
    main()