import matplotlib
matplotlib.use('Agg')  # 核心指令：强制使用 Agg 后端，不弹窗
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
import os
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, recall_score
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F 
from tqdm import tqdm
from thop import profile, clever_format
from scipy.stats import weibull_max
# 导入你的模型和配置
from models import MultiTaskOSRNet 
from config import Config
import gc
from matplotlib import font_manager
font_cn = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    size=12
)
# =========================================================================
# 1. 标签映射与名称定义
# =========================================================================

COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH"]
class EVTCalibrator:
    def __init__(self, tail_size=20):
        """
        tail_size: 选取每个类重构误差最大的前 N 个样本来拟合尾部分布
        """
        self.tail_size = tail_size
        self.models = {}

    def fit(self, class_recon_errors, class_dist_scores):
        all_classes = set(class_recon_errors.keys()).union(set(class_dist_scores.keys()))#获取所有已知类别ID集合
        print("\n⚙️ 正在执行 EVT 校准 (Weibull 拟合)...")
        for cls_id in all_classes:
            self.models[cls_id] = {}
            
            #-------拟合重构误差尾部
            re_errors = np.array(class_recon_errors.get(cls_id, []))
            if len(re_errors) >= self.tail_size:
                tail_re = np.sort(re_errors)[-self.tail_size:]#升序
                if np.std(tail_re) < 1e-9:
                    tail_re = tail_re + np.random.normal(0, 1e-7, len(tail_re))
                    print(f"数据全相等拟合失败")
                self.models[cls_id]['recon'] = weibull_max.fit(tail_re + 1e-6)
            
            # --- 2. 拟合距离得分尾部 ---
            dist_scores = np.array(class_dist_scores.get(cls_id, []))
            if len(dist_scores) >= self.tail_size:
                tail_dist = np.sort(dist_scores)[-self.tail_size:]
                # 避免极小方差导致拟合崩溃
                if np.std(tail_dist) < 1e-9:
                    tail_dist = tail_dist + np.random.normal(0, 1e-7, len(tail_dist))
                self.models[cls_id]['dist'] = weibull_max.fit(tail_dist + 1e-6)

            if 'recon' in self.models[cls_id] and 'dist' in self.models[cls_id]:
                print(f"  ✅ 类别 {cls_id} 双维度建模成功")
            else:
                print(f"  ⚠️ 类别 {cls_id} 样本不足，当前样本数:")
                #print(f"  ⚠️ 类别 {cls_id} 样本不足，当前样本数: {len(re_errors)}")

    def predict_outlier_prob(self, cls_id, recon_error, dist_score):
        """计算离群概率"""
        p_recon = 1.0
        p_dist = 1.0
        if cls_id in self.models:
        
            # 计算重构维度的离群概率
            if 'recon' in self.models[cls_id]:
                params = self.models[cls_id]['recon']
                p_recon = weibull_max.cdf(recon_error, *params)
  
            # 计算距离维度的离群概率
            if 'dist' in self.models[cls_id]:
                params = self.models[cls_id]['dist']
                p_dist = weibull_max.cdf(dist_score, *params)

        return np.clip(p_recon, 0, 1), np.clip(p_dist, 0, 1)
    
def get_predictions(model, loader, device, centers=None):
    model.eval()
    res = {
        'embs': [], 'c_true': [], 'snr': [],
        'c_pred': [], 'max_sim': []
    }
    # 记录用于 OSR 判定的三个原始维度
    raw_metrics = {
        'dist': [],    # 距离得分
        'recon': [],   # 重构误差
        'msp': [],      # 最大分类概率 (可选辅助)
        'entropy': []
    }

    with torch.no_grad():
        for x, y_c, snr in tqdm(loader, desc="Inference"):
            x_dev = x.to(device)
            input_data = x_dev.unsqueeze(1) if x_dev.dim() == 2 else x_dev#[B, 1, SeqLen]=[64,1,8192]
            output = model(input_data)
            # --- 1. 获取粗预测结果 ---
            c_logits = output['coarse_logits']

            res['c_true'].append(y_c.numpy())#[64],作为一个元素
            res['snr'].append(snr.numpy())#[64]
            res['c_pred'].append(c_logits.argmax(1).cpu().numpy())#取最大值索引，[64]
            res['embs'].append(output['embedding'].cpu().numpy())#z_norm,[64,128]

            # --- 2. 原始维度提取：MSP (Maximum Softmax Probability) ---
            probs = torch.softmax(c_logits, dim=1)#概率：[64,5]
            msp, _ = torch.max(probs, dim=1)#置信度:[64]
            raw_metrics['msp'].append(msp.cpu().numpy())

            #----熵越高，代表越可能是未知类
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)#[64]
            raw_metrics['entropy'].append((ent / np.log(5)).cpu().numpy())#熵归一化

            # --- 3. 原始维度提取：Reconstruction Error (MAE/MSE) ---
            recon = output['reconstruction']
            target = output['target_signal']
            # 统一形状计算每个样本的平均误差 [B]
            if target.dim() == 3: target = target.squeeze(1)
            if recon.dim() == 3: recon = recon.squeeze(1)
            #[64,8192]

            # 计算互相关系数 (Normalized Cross-Correlation)
            # 接近 1 表示波形结构一致，接近 0 表示结构完全不同
            cos_sim = F.cosine_similarity(recon, target, dim=1)#[64]
            r_err = torch.abs(torch.mean(torch.abs(recon - target), dim=1) / (cos_sim + 1e-8))#[64]
            raw_metrics['recon'].append(r_err.cpu().numpy())

            # 4. ArcMargin 核心：计算角度/余弦相似度
            z = F.normalize(output['embedding'], p=2, dim=1) # 必须 L2 归一化，[64, 128]
            prototypes = F.normalize(model.coarse_head.weight, p=2, dim=1)#[15, 128]
            sim_matrix = torch.matmul(z, prototypes.t())#[64, 15]
            max_sim, _ = torch.max(sim_matrix, dim=1)#每个样本与所有原型的最大相似度，[64]
            res['max_sim'].append(max_sim.cpu().numpy())
            # 将“不相似度”作为距离得分 (1 - sim)，值越大越像未知
            raw_metrics['dist'].append((1.0 - max_sim).cpu().numpy())
    # 合并所有 batch 结果
    for k in res:
        res[k] = np.concatenate(res[k])#每个元素值沿行方向拼接
    for k in raw_metrics:
        raw_metrics[k] = np.concatenate(raw_metrics[k])

    all_true_labels = res['c_true']
    raw_metrics['recon'] = raw_metrics['recon']
    print(f"Total Background samples in results: {np.sum(all_true_labels == -1)}")
    # 将原始指标并入结果字典
    res.update(raw_metrics)
    res['re_score_raw'] = res['recon']#为重构误差创建别名
    return res
# =========================================================================
# 类特异性阈值搜索
# =========================================================================
def find_class_specific_thresholds(fusion_scores, c_true, c_pred, target_recall=0.95):
    """
    为每一个已知类别单独计算阈值。
    解决 Polar 类由于底噪大导致全局阈值失效的问题。
    """
    class_thresholds = {}
    unique_classes = np.unique(c_true[c_true != -1])#已知类别ID
    global_median = np.median(fusion_scores[c_true != -1])#全局已知类得分中位数

    special_targets = {3: 1,}

    for cls_id in unique_classes:
        # 找出验证集中：真实是该类，且预测也是该类的样本（确保建模的是“正确特征”的边界）
        if cls_id == -1: continue
        #current_recall = target_recall
        current_recall = special_targets.get(cls_id, target_recall)
        
        mask = (c_true == cls_id) & (c_pred == cls_id)
        
        if mask.sum() > 10:
            scores = np.sort(fusion_scores[mask])#升序
            idx = int(len(scores) * current_recall)
            class_thresholds[cls_id] = scores[min(idx, len(scores)-1)]
        else:
            class_thresholds[cls_id] = np.percentile(fusion_scores[c_true != -1], target_recall*100)
            
    return class_thresholds

# =========================================================================
# 3. 绘图与可视化
# =========================================================================
def diagnose_osr_scores(res, cfg):
    print("\n" + "="*50)
    print("🔍 OSR 深度诊断报告 (Known vs Unknown)")
    print("="*50)
    
    # 建立已知类和未知类的掩码 (假设标签 -1 为未知类)
    known_mask = (res['c_true'] != -1)
    unknown_mask = ~known_mask
    
    metrics = {
        'Distance Score (EVT)': res.get('dist'),
        'Reconstruction MSE': res.get('re_score_raw'),
        'Normalized Entropy': res.get('entropy'),
        'Final Score': res.get('final_score')
    }
    
    print(f"{'Metric':<25} | {'Known Avg':<12} | {'Unknown Avg':<12} | {'Diff (%)'}")
    print("-" * 70)
    
    for name, data in metrics.items():
        if data is not None:
            # 去除可能存在的 nan
            k_val = np.mean(data[known_mask])
            u_val = np.mean(data[unknown_mask])
            diff = ((u_val - k_val) / (k_val + 1e-9)) * 100
            print(f"{name:<25} | {k_val:<12.6f} | {u_val:<12.6f} | {diff:>+8.2f}%")
        else:
            print(f"{name:<25} | 数据缺失")
            
    print("="*50 + "\n")

def plot_results(res, cfg, threshold, coarse_map, closed_oa, open_oa):
    """
    集成了雷达图、OSR分布及T-SNE的综合绘图函数
    closed_oa: 传入计算好的闭集准确率 (0-1)
    open_oa: 传入计算好的开集准确率 (0-1)
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.acc_dir, exist_ok=True)
    
    # 基础准备
    known_mask = res['c_true'] != -1
    snr_vals = sorted(np.unique(res['snr'][known_mask]))
    num_snr = len(snr_vals)
    num_coarse = len(coarse_map)
    coarse_names = [COARSE_LABEL_NAMES[i] for i in range(num_coarse)]
    # =========================================================================
    # 新增 A: OSCR 曲线计算与绘制 (Open Set Classification Rate)
    # =========================================================================
    x1 = res['final_score'][known_mask]   # 已知类分数
    x2 = res['final_score'][~known_mask]  # 未知类分数
    pred = res['c_pred'][known_mask]
    labels = res['c_true'][known_mask]
    correct = (pred == labels)

    scores_all = np.concatenate([x1, x2])
    # 标记：已知且正确为1，其他为0
    gt_correct = np.concatenate([correct.astype(int), np.zeros(len(x2))])
    # 标记：真正未知为1，已知为0
    gt_unknown = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

    # 3. 按分数从小到大排序 
    indices = np.argsort(scores_all)
    gt_correct = gt_correct[indices]
    gt_unknown = gt_unknown[indices]

    # 4. 随着阈值逐渐增大（放行更多样本为已知），累计计算
    num_k = len(x1)
    num_u = len(x2)

    # 这里的阈值移动是从“只接受最确定的样本”到“接受所有样本”
    ccr = np.cumsum(gt_correct) / num_k
    fpr = np.cumsum(gt_unknown) / num_u

    # 确保曲线从 (0,0) 开始
    ccr = np.concatenate([[0], ccr])
    fpr = np.concatenate([[0], fpr])
    oscr_score = np.trapz(ccr, fpr) # 计算面积

    plt.figure(figsize=(7, 6))
    #plt.plot(fpr, ccr, lw=2, color='darkorange', label=f'OSCR Curve (Area = {oscr_score:.4f})')
    plt.plot(fpr, ccr, label=f'OSCR (AUC={np.trapz(ccr, fpr):.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate (Unknowns accepted)')
    #plt.ylabel('CCR (Knowns correctly classified)')
    plt.xlabel("FPR（未知类被接受）", fontproperties=font_cn)
    plt.ylabel("CCR (已知类正确识别率）", fontproperties=font_cn)
    plt.title('OSCR 曲线', fontproperties=font_cn)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.results_dir, "oscr_curve.png"), dpi=300); plt.close(); gc.collect()

    # =========================================================================
    # 新增 B: 分 SNR 的 ROC 曲线绘制
    # =========================================================================
    plt.figure(figsize=(10, 7))
    # 选取代表性 SNR 以免画面太乱
    plot_snrs = snr_vals[::2] if len(snr_vals) > 5 else snr_vals
    for s in plot_snrs:
        m_k = (res['snr'][known_mask] == s)
        m_u = (res['snr'][~known_mask] == s)
        if m_k.any() and m_u.any():
            s_score = np.concatenate([x1[m_k], x2[m_u]])
            s_true = np.concatenate([np.zeros(m_k.sum()), np.ones(m_u.sum())]) # 0已知, 1未知
            fpr_s, tpr_s, _ = roc_curve(s_true, s_score)
            auc_s = auc(fpr_s, tpr_s)
            plt.plot(fpr_s, tpr_s, label=f'SNR {s}dB (AUC = {auc_s:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Detection ROC Curves at Different SNRs'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.results_dir, "snr_roc_curves.png"), dpi=300); plt.close(); gc.collect()
    # =========================================================================
    # 1. Coarse-level 准确率矩阵计算
    # =========================================================================
    coarse_acc_matrix = np.zeros((num_coarse, num_snr))
    
    for j, s in enumerate(snr_vals):
        mask_s = (res['snr'] == s) & known_mask
        for i in range(num_coarse):
            mask = mask_s & (res['c_true'] == i)
            coarse_acc_matrix[i, j] = np.mean(res['c_pred'][mask] == i) if mask.sum() > 0 else np.nan

    # 保存 Coarse CSV
    coarse_names = [COARSE_LABEL_NAMES[i] for i in range(num_coarse)]
    df_coarse = pd.DataFrame(coarse_acc_matrix, index=coarse_names, columns=[f"{s}dB" for s in snr_vals])
    df_coarse['AvgAcc'] = df_coarse.mean(axis=1)
    coarse_csv_path = os.path.join(cfg.acc_dir, "coarse_accuracy_data.csv")
    df_coarse.to_csv(coarse_csv_path, float_format="%.4f")
    print(f"Coarse accuracy CSV saved to: {coarse_csv_path}")

    # --- 1a. Coarse 级雷达图 (Radar Chart) ---
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2*np.pi, num_snr, endpoint=False).tolist()
    angles += angles[:1]  # 闭合曲线

    for i in range(num_coarse):
        values = coarse_acc_matrix[i].tolist()
        values += values[:1]
        name = COARSE_LABEL_NAMES[i] if i < len(COARSE_LABEL_NAMES) else f"C-{i}"
        ax.plot(angles, values, label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{s:.0f}dB" for s in snr_vals], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Coarse Accuracy Radar Chart\n(Closed OA: {closed_oa*100:.2f}%)", pad=30, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right',bbox_to_anchor=(1.3, 1.1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.acc_dir, "coarse_accuracy_radar.png"), dpi=300)
    plt.close(); gc.collect()

    # --- 1b. Coarse 级线性图 (Accuracy vs SNR) ---
    plt.figure(figsize=(10, 6))
    for i in range(num_coarse):
        avg_acc = np.nanmean(coarse_acc_matrix[i]) * 100
        name = COARSE_LABEL_NAMES[i]
        plt.plot(snr_vals, coarse_acc_matrix[i], marker='o', linewidth=2, label=f"{name} (Avg:{avg_acc:.1f}%)")
    plt.title(f"Coarse-level Recognition Accuracy vs SNR\n(Open OA: {open_oa*100:.2f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(os.path.join(cfg.acc_dir, "coarse_accuracy_linear.png"), dpi=300)
    plt.close(); gc.collect()

    # =========================================================================
    # 3. OSR 分数分布图 (MSE 判别)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    sns.histplot(res['final_score'][known_mask], label="Known Codes", color="blue", kde=True, stat="density", alpha=0.4)
    sns.histplot(res['final_score'][~known_mask], label="Unknown Codes", color="orange", kde=True, stat="density", alpha=0.4)
    # 计算平均阈值用于可视化参考
    avg_thr = np.mean(list(threshold.values()))
    min_thr = min(threshold.values())
    max_thr = max(threshold.values())
    plt.axvline(avg_thr, color='red', linestyle='--', linewidth=2, label=f'Avg Threshold: {avg_thr:.6f}')
    # 可选：画一个淡红色区域表示不同类别的阈值波动范围
    plt.axvspan(min_thr, max_thr, color='red', alpha=0.1, label='Threshold Range')
    plt.title(f"OSR Decision Score Distribution ({'final_score'})", fontsize=13, fontweight='bold')
    plt.xlabel("Fusion Score (0=Known, 1=Unknown)"); plt.ylabel("Density"); plt.legend()
    plt.savefig(os.path.join(cfg.results_dir, "osr_score_distribution.png"), dpi=300)
    plt.close(); gc.collect()

    # =========================================================================
    # 4. T-SNE 特征可视化
    # =========================================================================
    print("Computing T-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    max_pts = 3000
    indices = np.random.choice(len(res['embs']), min(max_pts, len(res['embs'])), replace=False)
    embs_norm = res['embs'][indices] / (np.linalg.norm(res['embs'][indices], axis=1, keepdims=True) + 1e-9)
    low_dim = tsne.fit_transform(embs_norm)
    subset_labels = res['c_true'][indices]

    plt.figure(figsize=(10, 8))
    for i in range(num_coarse):
        m = subset_labels == i
        name = COARSE_LABEL_NAMES[i]
        plt.scatter(low_dim[m, 0], low_dim[m, 1], label=name, s=25, alpha=0.7)
    
    # 标记未知类
    m_un = subset_labels == -1
    if m_un.any():
        plt.scatter(low_dim[m_un, 0], low_dim[m_un, 1], c='black', marker='x', label='Unknown', s=35, alpha=0.5)
    
    #plt.title("T-SNE Visualization of Feature Space", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "tsne_visualization.png"), dpi=300)
    plt.close(); gc.collect()

def plot_all_confusion_matrices(res, cfg, threshold, coarse_names):
    """
    生成粗标签闭集和开集识别的混淆矩阵 (图片+CSV)
    """
    cm_dir = os.path.join(cfg.results_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # 准备 OSR 预测结果
    # 逻辑：误差 > 阈值判为 -1 (Unknown)，否则保留模型分类结果
    open_pred = res['c_pred'].copy()
    for i in range(len(open_pred)):
        assigned_cls = open_pred[i]
        # 从字典中获取该类阈值，若无则用所有阈值的平均值兜底
        thr = threshold.get(assigned_cls, np.mean(list(threshold.values())))
        if res['final_score'][i] > thr:
            open_pred[i] = -1
    
    # -------------------------------------------------------------------------
    # 1. 开集识别混淆矩阵 (OSR Confusion Matrix: 6x6)
    # -------------------------------------------------------------------------
    osr_labels = list(range(len(coarse_names))) + [-1]
    osr_display_names = coarse_names + ["Unknown"]
    
    cm_osr = confusion_matrix(res['c_true'], open_pred, labels=osr_labels)
    # 转换为百分比（按行归一化）
    cm_osr_perc = cm_osr.astype('float') / cm_osr.sum(axis=1)[:, np.newaxis]
    cm_osr_perc = np.nan_to_num(cm_osr_perc)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_osr_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=osr_display_names, yticklabels=osr_display_names)
    plt.title("OSR 混淆矩阵", fontproperties=font_cn)
    plt.xlabel("预测标签", fontproperties=font_cn); plt.ylabel("真实标签", fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, "osr_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存 CSV
    pd.DataFrame(cm_osr, index=osr_display_names, columns=osr_display_names).to_csv(
        os.path.join(cm_dir, "osr_confusion_matrix.csv"))

    # -------------------------------------------------------------------------
    # 2. 闭集粗标签混淆矩阵 (Coarse CM: 5x5 - 仅针对已知样本)
    # -------------------------------------------------------------------------
    known_mask = res['c_true'] != -1
    true_labels_closed = res['c_true'][known_mask]
    pred_labels_closed = res['c_pred'][known_mask]
    closed_labels = list(range(len(coarse_names)))
    closed_display_names = coarse_names
    cm_closed = confusion_matrix(true_labels_closed, pred_labels_closed, labels=closed_labels)
    cm_closed_perc = cm_closed.astype('float') / cm_closed.sum(axis=1)[:, np.newaxis]
    cm_closed_perc = np.nan_to_num(cm_closed_perc)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_closed_perc, annot=True, fmt='.2f', cmap='Greens', # 换个颜色区分
                xticklabels=closed_display_names, yticklabels=closed_display_names)
    plt.title("闭集混淆矩阵", fontproperties=font_cn)
    plt.xlabel("预测标签", fontproperties=font_cn); plt.ylabel("真实标签", fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, "closed_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All confusion matrices (images & CSVs) saved to: {cm_dir}")
# =========================================================================
# 1. 绘图增强：专门的 EVT 概率直方图
# =========================================================================
def plot_evt_distribution(test_res, threshold, cfg):
    plt.figure(figsize=(10, 6))
    known_mask = test_res['c_true'] != -1
    
    # 画出已知类和未知类的概率分布
    # 注意：这里的 test_res['re_score'] 已经被我们替换成了 EVT 概率
    sns.histplot(test_res['final_score'][known_mask], label="Known Classes", 
                 color="royalblue", kde=True, stat="density", alpha=0.4, bins=50)
    
    if (~known_mask).any():
        sns.histplot(test_res['final_score'][~known_mask], label="Unknown Classes", 
                     color="crimson", kde=True, stat="density", alpha=0.4, bins=50)
    
    # 为每个类别画出它专属的阈值线
    colors = ['r', 'g', 'b', 'c', 'm']
    for idx, (cls_idx, thr) in enumerate(threshold.items()):
        plt.axvline(thr, color=colors[idx % len(colors)], linestyle=':', 
                    alpha=0.7, label=f'Class {cls_idx} Thr: {thr:.3f}')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title("EVT Outlier Probability Distribution\n(0=Likely Known, 1=Likely Unknown)", fontsize=13, fontweight='bold')
    plt.xlabel("Weibull Outlier Probability"); plt.ylabel("Density"); plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(cfg.results_dir, "evt_probability_distribution.png")
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"📊 EVT 概率分布图已保存至: {save_path}")

def find_best_threshold(scores, labels):
    """
    寻找等错误率点 (EER)
    scores: 融合后的离群分数 (0~1)
    labels: 0 为已知类, 1 为未知类
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # 寻找 FPR 和 FNR 最接近的点
    idx = np.nanargmin(np.absolute(fpr - fnr))
    return thresholds[idx], fpr[idx]

def find_recall_threshold(fusion_scores, true_labels, target_recall=0.80):
    """
    根据目标召回率在验证集的已知类中寻找阈值
    """
    # 只提取已知类样本的分数 (true_labels != -1)
    known_scores = fusion_scores[true_labels != -1]
    
    if len(known_scores) == 0:
        return 0.5 # 兜底值       
    # 将分数从小到大排序
    sorted_scores = np.sort(known_scores)
    
    # 找到对应召回率的索引位置
    idx = int(len(sorted_scores) * target_recall)
    idx = min(idx, len(sorted_scores) - 1)
    
    threshold = sorted_scores[idx]
    return threshold

# =========================================================================
# 4. 主程序
# =========================================================================
def main():
    cfg = Config()
    
    # --- 0. 环境与模型初始化 ---
    model = MultiTaskOSRNet(cfg).to(cfg.device)
    if os.path.exists(cfg.model_save_path):
        checkpoint = torch.load(cfg.model_save_path, map_location=cfg.device,weights_only=True)
        model.load_state_dict(checkpoint['model'])
        if hasattr(model, 'center_loss_fn'):
            centers = model.center_loss_fn.centers.detach()
            print("✅ 成功从模型中提取 Center Loss 聚类中心")
        else:
            centers = None
            print("⚠️ 警告: 未发现 center_loss_fn，centers 设为 None")
        #centers = checkpoint['center_loss']['centers'].to(cfg.device)
        print(f"✅ 成功加载模型权重: {cfg.model_save_path}")
    else:
        print(f"❌ 错误: 未找到模型文件 {cfg.model_save_path}")
        return

    # 打印模型复杂度统计
    dummy_in = torch.randn(1, 1, cfg.seq_len).to(cfg.device)
    flops, params = profile(model, inputs=(dummy_in,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"📊 模型统计: 参数量={params}, 计算量={flops}")

   # =========================================================================
    # PHASE 1: 在验证集 (Validation Set) 上寻找拒绝阈值
    # =========================================================================
    val_pkl_path = cfg.eval_data_path.replace('.pkl', '_val.pkl')
    print(f"\n🚀 [PHASE 1] 正在执行OSR校准与EER阈值搜索...")
    if not os.path.exists(val_pkl_path):
        print(f"❌ 错误: 未找到验证集文件 {val_pkl_path}")
    else:
        with open(val_pkl_path, 'rb') as f:
            val_data = pickle.load(f)
        # 核心诊断代码
        import collections
        print(f"DEBUG: 原始 val_pkl 中的标签分布: {collections.Counter(val_data['coarse_labels'])}")
        val_loader = DataLoader(TensorDataset(
            torch.tensor(val_data['features']).float(),
            torch.tensor(val_data['coarse_labels']).long(),
            torch.tensor(val_data['snrs']).float()
        ), batch_size=cfg.batch_size, shuffle=False)

        # 1. 提取验证集原始指标 (dist, recon, msp 等)
        val_res = get_predictions(model, val_loader, cfg.device)

        print(f"🔍 [验证集] Recon 均值 (放大后): {val_res['recon'].mean():.6f}")

        # 2. 拟合 EVT 模型
        val_known_correct_mask = (val_res['c_true'] != -1) & (val_res['c_pred'] == val_res['c_true'])#只保留预测正确的已知类样本的掩码

        class_recon_errors = {i: val_res['recon'][val_known_correct_mask & (val_res['c_true'] == i)] 
                          for i in range(len(COARSE_LABEL_NAMES))}
        #val_res['recon']:list,长度为符合条件的样本数
        #为每个粗分类分别收集预测正确样本的重构误差
        '''
        {
            0: [recon_err_1, recon_err_2, ...],# Conv 类的重构误差
            1: [recon_err_3, recon_err_4, ...],# LDPC 类的重构误差
            2: [...],                         # Turbo 类的重构误差
            3: [...],                         # Polar 类的重构误差
            4: [...]                           # BCH 类的重构误差
        }
        '''

        class_dist_scores = {i: val_res['dist'][val_known_correct_mask & (val_res['c_true'] == i)] 
                          for i in range(len(COARSE_LABEL_NAMES))}
        #val_res['dist']:list,长度为符合条件的样本数
        #为每个粗分类分别收集预测正确样本的距离分数

        calibrator = EVTCalibrator(tail_size=110) 
        calibrator.fit(class_recon_errors, class_dist_scores)#为每个类用预测正确的数据构建EVT模型

        # 3. 计算验证集的融合离群分数 (双阈值逻辑)
        p_recons_val, p_dists_val = [], []
        for i in range(len(val_res['c_true'])):
            pr, pd = calibrator.predict_outlier_prob(val_res['c_pred'][i], val_res['recon'][i], val_res['dist'][i])
            p_recons_val.append(pr)#每个样本的重构误差 EVT 概率
            p_dists_val.append(pd)#每个样本的距离分数 EVT 概率
        # 融合 EVT 概率与信息熵
        p_recons_val = np.array(p_recons_val)
        p_dists_val = np.array(p_dists_val)
        entropy_val = np.array(val_res['entropy'])
        fusion_scores_val = (p_dists_val**2.1) * (p_recons_val**0.5) / ((1 + entropy_val)**1.0)#TODO
        # fusion_scores_val = (  
        #     (p_dists_val ** 2.1)   
        #     * (p_recons_val ** 0.5)
        #     * (1 + entropy_val)
        # )

        # 4. 锁定类特异性阈值 (针对 Recall-0.95)
        class_thresholds = find_class_specific_thresholds(
            fusion_scores_val, val_res['c_true'], val_res['c_pred'], target_recall=1
        )
        avg_threshold = np.mean(list(class_thresholds.values()))
        hard_upper_bound = np.percentile(fusion_scores_val, 98)
        for cls_id in class_thresholds:
            original_thr = class_thresholds[cls_id]
            if original_thr > hard_upper_bound:
               print(f"⚠️ 警告: 类别 {cls_id} 阈值过高 ({original_thr:.4f})，已重置为硬顶 {hard_upper_bound:.4f}")
               class_thresholds[cls_id] = hard_upper_bound
        #class_thresholds[3] = 0.0220
        print(f"🎯 类特异性阈值已锁定: {class_thresholds}")
        print(f"🛡️ 全局兜底阈值 (Fallback): {avg_threshold:.4f}")
        #print(f"📈 验证集参考: 全局阈值={best_threshold:.4f}, EER={eer_val:.4f}")
    # =========================================================================
    # PHASE 2: 在测试集 (Test Set) 上进行最终评估
    # =========================================================================
    test_pkl_path = cfg.eval_data_path.replace('.pkl', '_test.pkl')
    print(f"\n🚀 [PHASE 2] 正在测试集上执行EVT最终评估 ...")
    
    if not os.path.exists(test_pkl_path):
        print(f"❌ 错误: 未找到测试集文件 {test_pkl_path}")
        return

    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(test_data['features']).float(),
        torch.tensor(test_data['coarse_labels']).long(),
        torch.tensor(test_data['snrs']).float()
    ), batch_size=cfg.batch_size, shuffle=False)

    # 在测试集上推理
    test_res = get_predictions(model, test_loader, cfg.device, centers=centers)

    known_mask_test = (test_res['c_true'] != -1)
    unknown_mask_test = (test_res['c_true'] == -1)

    print(f"🔍 [测试集-已知类] Recon 均值: {test_res['recon'][known_mask_test].mean():.6f}")
    if unknown_mask_test.any():
        print(f"🔍 [测试集-未知类] Recon 均值: {test_res['recon'][unknown_mask_test].mean():.6f}")  
 

    # 2. 应用校准好的 EVT 模型计算测试集分数
    p_recons_test, p_dists_test = [], []
    for i in range(len(test_res['c_true'])):
        pr, pd = calibrator.predict_outlier_prob(test_res['c_pred'][i], test_res['recon'][i], test_res['dist'][i])
        p_recons_test.append(pr)
        p_dists_test.append(pd)
    
    p_recons_test = np.array(p_recons_test)
    p_dists_test = np.array(p_dists_test)
    entropy_test = np.array(test_res['entropy'])
    test_res['final_score'] = (p_dists_test**2.1) * (p_recons_test**0.5) / ((1 + entropy_test)**1.0)
    # test_res['final_score'] = (
    #     (p_dists_test ** 2.1)
    #     * (p_recons_test ** 0.5)
    #     * (1 + entropy_test)
    # )

    known_scores = test_res['final_score'][known_mask_test]
    unknown_scores = test_res['final_score'][unknown_mask_test]
    print(f"DEBUG: 已知类分位数 [50, 90, 95, 99]: {np.percentile(known_scores, [50, 90, 95, 99])}")
    print(f"DEBUG: 未知类分位数 [50, 90, 95, 99]: {np.percentile(unknown_scores, [50, 90, 95, 99])}")

    # 应用判定

    open_pred = test_res['c_pred'].copy()
    for i in range(len(open_pred)):
        assigned_cls = open_pred[i]
        
        thr = class_thresholds.get(assigned_cls, avg_threshold) 
        if test_res['final_score'][i] > thr:
            open_pred[i] = -1   #如果分数超过该类的特异性阈值，则判定为未知类 (-1)

    # --- 计算指标 ---
    known_mask = test_res['c_true'] != -1
    closed_oa = accuracy_score(test_res['c_true'][known_mask], test_res['c_pred'][known_mask])
    open_oa = accuracy_score(test_res['c_true'], open_pred)
    # ======= 新增：计算真正的召回率 (Recall Score) =======

    known_mask = test_res['c_true'] != -1
    true_labels = test_res['c_true'][known_mask] #已知类的真实标签
    pred_labels = open_pred[known_mask] #已知类的预测标签
    class_indices = list(range(len(COARSE_LABEL_NAMES)))
    per_class_recall = recall_score(true_labels, pred_labels, labels=class_indices, average=None)   #计算每个类别的召回率
    macro_recall = np.mean(per_class_recall)    #平均召回率
    print(f"\n📈 [真实召回率报告 (Measured Recall)]")
    print(f"   - 总体平均召回率 (Known Recall): {macro_recall*100:.2f}%")
    for i in range(len(per_class_recall)):
        label_name = COARSE_LABEL_NAMES[i]
        recall_val = per_class_recall[i]
        print(f"   - 类别 {i} ({label_name}) 实际召回率: {recall_val*100:.2f}%")

    print(f"\n📊 [EVT 评测结果报告]")
    #print(f"   - 采用 EVT 判定阈值: {best_threshold:.4f}")
    print(f"   - 闭集分类准确率 (Known): {closed_oa*100:.2f}%")
    print(f"   - 开集识别准确率 (Total): {open_oa*100:.2f}%")
# 1. 定义掩码：真正标签是未知(-1)，但预测标签是 Polar(3)

    polar_fp_mask = (test_res['c_true'] == -1) & (test_res['c_pred'] == 3)
    known_polar_mask = (test_res['c_true'] == 3)
    print(f" - [误判样本] MSP 均值: {np.mean(test_res['msp'][polar_fp_mask]):.4f}")
    print(f" - [误判样本] MSP 中位数: {np.median(test_res['msp'][polar_fp_mask]):.4f}")
    print(f" - [真正 Polar] MSP 均值: {np.mean(test_res['msp'][known_polar_mask]):.4f}")
    print(f" - [真正 Polar] MSP 95分位数: {np.percentile(test_res['msp'][known_polar_mask], 5):.4f} (即95%的Polar都高于此值)")
# 注意：请根据你的 coarse_map 确认 Polar 的索引是否为 3
    polar_idx = 3 
    misclassified_unknown_mask = (test_res['c_true'] == -1) & (open_pred == 3)
    true_polar_mask = (test_res['c_true'] == polar_idx) & (test_res['c_pred'] == polar_idx)
# 2. 提取这两类样本的重构误差
    recon_misclassified = test_res['recon'][misclassified_unknown_mask]
    recon_true_polar = test_res['recon'][true_polar_mask]
# 3. 打印对比统计分析
    print(f"\n🔍 [深度诊断: Polar 误判分析]")
    print(f"   - 误判为 Polar 的未知类数量: {misclassified_unknown_mask.sum()}")
    print(f"   - 误判样本的 Recon 均值: {recon_misclassified.mean():.4f}")
    print(f"   - 误判样本的 Recon 中位数: {np.median(recon_misclassified):.4f}")
    print(f"   - 真正 Polar 的 Recon 均值: {recon_true_polar.mean():.4f}")
    print(f"   - 真正 Polar 的 Recon 95分位数: {np.percentile(recon_true_polar, 95):.4f}")
# 4. 可视化对比分布
    plt.figure(figsize=(8, 5))
    plt.hist(recon_true_polar, bins=100, alpha=0.5, label='True Polar Recon', color='blue', density=True)
    plt.hist(recon_misclassified, bins=100, alpha=0.5, label='Misclassified Unknown Recon', color='red', density=True)
    plt.axvline(10.0, color='green', linestyle='-', label='Current Gate (10.0)')
    plt.axvline(np.percentile(recon_true_polar, 95), color='k', linestyle='--', label='Polar 95th Per.')
    plt.yscale('log')
    plt.title("Critical Diagnosis: Recon Error Distribution (Log Scale)")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density (Log)")
    plt.xlim(0, 10)
    plt.legend()
    plt.savefig('polar_recon_diagnosis.png', dpi=300)
    plt.close()

    test_res['c_pred_original'] = test_res['c_pred'].copy()
    test_res['c_pred'] = open_pred
    diagnose_osr_scores(test_res, cfg)
    # =========================================================================
    # 5. 绘图与结果持久化 (使用测试集结果)
    # =========================================================================
    # 这里我们把测试集推理出的 test_res 传给绘图函数
    
    plot_all_confusion_matrices(
        test_res, cfg, class_thresholds, COARSE_LABEL_NAMES
    )

    plot_evt_distribution(test_res, class_thresholds, cfg)

    plot_results(test_res, cfg, class_thresholds, test_data['coarse_map'], closed_oa, open_oa)

    print(f"\n✨ 评估任务全部完成！所有图表和数据已保存至: {cfg.results_dir}")

if __name__ == "__main__":
    main()