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
import matplotlib.pyplot as plt
font_cn = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    size=12
)
# ===================================================================
# 1. 标签映射与名称定义
# ===================================================================

COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH"]
FINE_LABEL_NAMES = [
    "Conv-A", "Conv-B", "Conv-C", 
    "LDPC-A", "LDPC-B", "LDPC-C",
    "Turbo-A", "Turbo-B", "Turbo-C",
    "Polar-A", "Polar-B", "Polar-C",#TODO
    "BCH-A", "BCH-B", "BCH-C"
]


class OpenMaxCalibrator:
    """OpenMax 开集识别校准器

    为每个已知类别计算 MAV (Mean Activation Vector)，并用样本到 MAV 的
    欧氏距离的尾部分布拟合 Weibull 模型。测试时通过 Weibull CDF 估计
    样本属于各类的"离群概率"，从而重新校准 softmax 分数并添加 unknown 类。

    Reference: Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016.
    """
    def __init__(self, tail_size=20):
        """
        tail_size: 每个类选取距离最大的前 N 个样本来拟合 Weibull 尾部分布
        """
        self.tail_size = tail_size
        self.mavs = {}           # Mean Activation Vectors: {class_id: np.array [D]}
        self.weibull_params = {} # Weibull 参数: {class_id: (c, loc, scale)}
        self.num_classes = 0
        self.activation_dim = None

    def fit(self, class_activations, num_classes):
        """
        为每个已知类别计算 MAV 并拟合 Weibull 分布。

        Parameters:
            class_activations: dict {class_id: np.array of shape [N, D]}
                               每个类中正确分类样本的激活向量 z_raw
            num_classes: 已知类别总数
        """
        self.num_classes = num_classes
        print("\n⚙️ 正在执行 OpenMax 校准 (MAV + Weibull 距离拟合)...")

        for cls_id in range(num_classes):
            acts = class_activations.get(cls_id, None)
            if acts is None or len(acts) == 0:
                print(f"  ⚠️ 类别 {cls_id} 没有激活向量，使用零向量占位")
                if self.activation_dim is not None:
                    self.mavs[cls_id] = np.zeros(self.activation_dim)
                continue

            # 记录激活向量维度
            if self.activation_dim is None:
                self.activation_dim = acts.shape[1]

            # 计算 MAV (Mean Activation Vector)
            self.mavs[cls_id] = np.mean(acts, axis=0)

            # 计算该类所有正确样本到其 MAV 的欧氏距离
            dists = np.linalg.norm(acts - self.mavs[cls_id], axis=1)

            # 用距离的尾部 (最大的 tail_size 个) 拟合 Weibull 分布
            if len(dists) >= self.tail_size:
                tail = np.sort(dists)[-self.tail_size:]
                if np.std(tail) < 1e-9:
                    tail = tail + np.random.normal(0, 1e-7, len(tail))
                    print(f"  ⚠️ 类别 {cls_id} 距离方差极小，已添加噪声")
                self.weibull_params[cls_id] = weibull_max.fit(tail + 1e-6)
                c, loc, scale = self.weibull_params[cls_id]
                print(f"  ✅ 类别 {cls_id}: 样本数={len(acts)}, MAV dim={self.activation_dim}, "
                      f"Weibull(c={c:.4f}, loc={loc:.4f}, scale={scale:.4f})")
            else:
                print(f"  ⚠️ 类别 {cls_id} 样本不足 ({len(dists)} < {self.tail_size})，跳过 Weibull 拟合")

    def predict_openmax(self, z_raw, coarse_logits):
        """
        对单个样本执行 OpenMax 推理，返回重新校准的概率分布。

        Parameters:
            z_raw: 激活向量 [D] (output['z_raw']，未经过 L2 归一化)
            coarse_logits: 粗分类 logits [num_classes]

        Returns:
            openmax_probs: [num_classes + 1] 概率分布 (最后一维为 unknown)
            distances: [num_classes] 到各类 MAV 的欧氏距离
            w: [num_classes] 各类的 Weibull 离群权重
        """
        num_classes = self.num_classes

        # --- 1. 计算到每个 MAV 的欧氏距离 ---
        distances = np.zeros(num_classes)
        for cls_id in range(num_classes):
            if cls_id in self.mavs:
                distances[cls_id] = np.linalg.norm(z_raw - self.mavs[cls_id])
            else:
                distances[cls_id] = np.inf

        # --- 2. 计算 softmax 基础分数 ---
        logits = coarse_logits - np.max(coarse_logits)  # 数值稳定
        exp_logits = np.exp(logits)
        scores = exp_logits / exp_logits.sum()

        # --- 3. 对每个类计算 Weibull 离群概率 ---
        # w[i] 越高 → 样本越不可能是类别 i 的成员
        w = np.zeros(num_classes)
        for cls_id in range(num_classes):
            if cls_id in self.weibull_params:
                params = self.weibull_params[cls_id]
                # weibull_max.cdf(d) 给出距离 d 在已知类尾部分布中的累积概率
                # 距离越大 → CDF 越接近 1 → 越可能是离群/未知样本
                w[cls_id] = np.clip(weibull_max.cdf(distances[cls_id], *params), 0, 1)
            else:
                w[cls_id] = 0.5  # 没有 Weibull 模型时的不确定默认值

        # --- 4. 重新校准分数 (OpenMax 核心公式) ---
        s_hat = np.zeros(num_classes + 1)  # +1 为 unknown 类
        for i in range(num_classes):
            s_hat[i] = scores[i] * (1.0 - w[i])
        s_hat[num_classes] = np.sum(scores * w)  # unknown 类的分数

        # --- 5. 应用 softmax 得到最终概率 ---
        s_hat = s_hat - np.max(s_hat)  # 数值稳定
        s_hat = np.exp(s_hat) / np.exp(s_hat).sum()

        return s_hat, distances, w
    
def get_AV(model, loader, device, num_classes):
    """计算训练集中每个类的均值激活向量 (MAV)

    遍历训练集，按真实标签收集每个类的 z_raw 激活向量，计算类均值。
    同时收集 coarse_logits 用于后续阈值计算。

    Args:
        model: 已加载权重的 MultiTaskOSRNet 模型
        loader: 训练集 DataLoader (仅已知类)
        device: torch device
        num_classes: 已知类别总数

    Returns:
        mavs: dict {class_id: np.array [D]}  每类的均值激活向量
        class_activations: dict {class_id: np.array [N_i, D]}  每类的原始 z_raw
        all_logits: np.array [N_total, num_classes]  所有样本的 coarse_logits
        all_c_true: np.array [N_total]  所有样本的真实标签
    """
    model.eval()
    class_z_raw = {i: [] for i in range(num_classes)}
    class_logits = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for x, y_c, _, _ in tqdm(loader, desc="Computing MAVs"):
            x_dev = x.to(device)
            input_data = x_dev.unsqueeze(1) if x_dev.dim() == 2 else x_dev
            output = model(input_data)
            z_raw = output['z_raw'].cpu().numpy()              # [B, D]
            c_logits = output['coarse_logits'].cpu().numpy()   # [B, num_classes]
            c_true = y_c.numpy()                                # [B]

            for cls_id in range(num_classes):
                mask = c_true == cls_id
                if mask.sum() > 0:
                    class_z_raw[cls_id].append(z_raw[mask])
                    class_logits[cls_id].append(c_logits[mask])

    mavs = {}
    class_activations = {}
    all_logits_list = []
    all_c_true_list = []
    for cls_id in range(num_classes):
        if class_z_raw[cls_id]:
            acts = np.concatenate(class_z_raw[cls_id], axis=0)       # [N_i, D]
            lgs = np.concatenate(class_logits[cls_id], axis=0)       # [N_i, num_classes]
            class_activations[cls_id] = acts
            mavs[cls_id] = np.mean(acts, axis=0)                     # [D]
            all_logits_list.append(lgs)
            all_c_true_list.append(np.full(len(acts), cls_id))
            print(f"  类别 {cls_id}: {len(acts)} 个样本, MAV dim={mavs[cls_id].shape}")
        else:
            class_activations[cls_id] = np.array([])
            mavs[cls_id] = None
            print(f"  类别 {cls_id}: 无样本!")

    all_logits = np.concatenate(all_logits_list, axis=0) if all_logits_list else np.array([])
    all_c_true = np.concatenate(all_c_true_list, axis=0) if all_c_true_list else np.array([])
    return mavs, class_activations, all_logits, all_c_true
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

    special_targets = {3: 0.81, }

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
    print("🔍 OSR 深度诊断报告 (Known vs Unknown) - OpenMax")
    print("="*50)

    # 建立已知类和未知类的掩码 (假设标签 -1 为未知类)
    known_mask = (res['c_true'] != -1)
    unknown_mask = ~known_mask

    metrics = {
        'OpenMax Unknown Prob': res.get('final_score'),
        'MSP (Max Softmax)': res.get('msp'),
        'Normalized Entropy': res.get('entropy'),
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

def plot_results(res, cfg, threshold, coarse_map, fine_map, closed_oa, open_oa):
    """
    集成了雷达图、分组细标签曲线、OSR分布及T-SNE的综合绘图函数
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
    EXTENDED_NAMES = COARSE_LABEL_NAMES + ["Background"]
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
    # 2. Fine-level 分组准确率曲线 (使用 FINE_LABEL_NAMES)
    # =========================================================================
    for c_idx in range(num_coarse):
        c_name = COARSE_LABEL_NAMES[c_idx]
        relevant_fine_indices = np.unique(res['f_true'][(res['c_true'] == c_idx) & known_mask])
        
        if len(relevant_fine_indices) == 0: continue

        fine_acc_list = []
        fine_row_names = []

        plt.figure(figsize=(10, 6))
        for f_idx in relevant_fine_indices:
            # 直接使用预定义的细分类名称
            f_name = FINE_LABEL_NAMES[f_idx] if f_idx < len(FINE_LABEL_NAMES) else f"F-{f_idx}"
            f_accs = []
            for s in snr_vals:
                # 1. 找出在该 SNR 下，真实标签确实是 f_idx 的掩码
                mask = (res['snr'] == s) & (res['f_true'] == f_idx)
                
                if mask.sum() > 0:
                    # 2. 计算准确率： (预测值 == 真实值).sum() / 总数
                    # 注意：这里必须是判断相等，而不是直接对 pred 取平均
                    acc = np.mean(res['f_pred'][mask] == f_idx)
                else:
                    acc = np.nan # 如果没样本，设为 NaN
                f_accs.append(acc)
            
            fine_acc_list.append(f_accs)
            fine_row_names.append(f_name)
            
            avg_f_acc = np.nanmean(f_accs) * 100
            plt.plot(snr_vals, f_accs, marker='s', linestyle='--', label=f"{f_name} ({avg_f_acc:.1f}%)")
        # 保存该粗类下的 Fine CSV
        df_fine = pd.DataFrame(fine_acc_list, index=fine_row_names, columns=[f"{s}dB" for s in snr_vals])
        df_fine['AvgAcc'] = df_fine.mean(axis=1)
        df_fine.to_csv(os.path.join(cfg.acc_dir, f"fine_accuracy_{c_name}.csv"), float_format="%.4f")

        plt.title(f"Fine-level Accuracy: {c_name} Components", fontsize=14, fontweight='bold')
        plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.ylim(0, 1.05); 
        plt.grid(True, linestyle='--', alpha=0.6); plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.acc_dir, f"fine_acc_{c_name}.png"), dpi=300)
        plt.close(); gc.collect()

    # =========================================================================
    # 3. OSR 分数分布图 (OpenMax Unknown Probability)
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
    plt.title("OSR OpenMax Decision Score Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("OpenMax Unknown Probability (0=Known, 1=Unknown)"); plt.ylabel("Density"); plt.legend()
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

def plot_all_confusion_matrices(res, cfg, threshold, coarse_names, fine_names):
    """
    生成粗标签、细标签和开集识别的混淆矩阵 (图片+CSV)
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
    # -------------------------------------------------------------------------
    # 3. 细标签混淆矩阵 (针对每个粗标签绘制 3x3)
    # -------------------------------------------------------------------------
    for c_idx, c_name in enumerate(coarse_names):
        # 筛选属于该粗类的已知样本
        mask = (res['c_true'] == c_idx)
        if mask.sum() == 0: continue
        
        # 获取该类下存在的细标签索引
        relevant_f_indices = sorted(np.unique(res['f_true'][mask]))
        f_display_names = [fine_names[i] for i in relevant_f_indices]
        
        cm_fine = confusion_matrix(res['f_true'][mask], res['f_pred'][mask], labels=relevant_f_indices)
        cm_fine_perc = cm_fine.astype('float') / cm_fine.sum(axis=1)[:, np.newaxis]
        cm_fine_perc = np.nan_to_num(cm_fine_perc)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_fine_perc, annot=True, fmt='.2f', cmap='Oranges',
                    xticklabels=f_display_names, yticklabels=f_display_names)
        plt.title(f"{c_name}")
        plt.savefig(os.path.join(cm_dir, f"fine_cm_{c_name}.png"), dpi=300)
        plt.close()
        
        # 保存细标签 CSV
        pd.DataFrame(cm_fine, index=f_display_names, columns=f_display_names).to_csv(
            os.path.join(cm_dir, f"fine_cm_{c_name}.csv"))

    print(f"All confusion matrices (images & CSVs) saved to: {cm_dir}")
# =========================================================================
# 1. 绘图增强：专门的 EVT 概率直方图
# =========================================================================
def plot_evt_distribution(test_res, threshold, cfg):
    """绘制 OpenMax Unknown Probability 分布直方图"""
    plt.figure(figsize=(10, 6))
    known_mask = test_res['c_true'] != -1

    # 画出已知类和未知类的概率分布
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
    plt.title("OpenMax Unknown Probability Distribution\n(0=Likely Known, 1=Likely Unknown)", fontsize=13, fontweight='bold')
    plt.xlabel("OpenMax Unknown Probability"); plt.ylabel("Density")
    plt.grid(alpha=0.3)

    save_path = os.path.join(cfg.results_dir, "openmax_probability_distribution.png")
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"📊 OpenMax 概率分布图已保存至: {save_path}")

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
    import collections
    cfg = Config()

    # --- 0. 环境与模型初始化 ---
    model = MultiTaskOSRNet(cfg).to(cfg.device)
    if os.path.exists(cfg.model_save_path):
        checkpoint = torch.load(cfg.model_save_path, map_location=cfg.device)
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
    # PHASE 1: 使用训练集计算 MAV + 验证集搜索阈值
    # =========================================================================
    num_coarse = len(COARSE_LABEL_NAMES)

    # --- 1. 加载训练集 (仅已知类) 用于计算 MAV ---
    train_pkl_path = cfg.eval_data_path.replace('.pkl', '_train.pkl')
    print(f"\n🚀 [PHASE 1a] 使用训练集计算 OpenMax MAV...")
    if not os.path.exists(train_pkl_path):
        print(f"❌ 错误: 未找到训练集文件 {train_pkl_path}")
        return
    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)
    print(f"DEBUG: 训练集标签分布: {collections.Counter(train_data['coarse_labels'])}")

    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_data['features']).float(),
        torch.tensor(train_data['coarse_labels']).long(),
        torch.tensor(train_data['fine_labels']).long(),
        torch.tensor(train_data['snrs']).float()
    ), batch_size=cfg.batch_size, shuffle=False)

    # 在训练集上推理，计算每类 MAV 并收集原始激活向量和 logits
    mavs, class_activations, train_logits, train_c_true = get_AV(model, train_loader, cfg.device, num_coarse)

    # 初始化并拟合 OpenMax 校准器 (内部用 class_activations 计算 MAV + 拟合 Weibull)
    calibrator = OpenMaxCalibrator(tail_size=20)
    calibrator.fit(class_activations, num_coarse)

    # 将 class_activations 与 train_logits 展平为对齐数组
    train_z_raw_flat = []
    train_logits_flat = []
    train_labels_for_thr = []
    for cls_id in range(num_coarse):
        acts = class_activations.get(cls_id, np.array([]))
        if len(acts) == 0:
            continue
        start = sum(len(class_activations[c]) for c in range(cls_id))
        for j in range(len(acts)):
            train_z_raw_flat.append(acts[j])
            train_logits_flat.append(train_logits[start + j])
            train_labels_for_thr.append(cls_id)
    train_z_raw_flat = np.array(train_z_raw_flat)
    train_logits_flat = np.array(train_logits_flat)
    train_labels_for_thr = np.array(train_labels_for_thr)

    # 在训练集上计算 OpenMax unknown 概率 (用于参考阈值，仅可视化)
    train_openmax_unknown = []
    for i in range(len(train_z_raw_flat)):
        probs, _, _ = calibrator.predict_openmax(train_z_raw_flat[i], train_logits_flat[i])
        train_openmax_unknown.append(probs[-1])
    train_openmax_unknown = np.array(train_openmax_unknown)

    # 从训练集计算类特异性参考阈值 (95th 百分位，仅用于绘图参考线)
    class_thresholds = {}
    for cls_id in range(num_coarse):
        mask = train_labels_for_thr == cls_id
        if mask.sum() > 0:
            class_thresholds[cls_id] = np.percentile(train_openmax_unknown[mask], 95)
        else:
            class_thresholds[cls_id] = np.percentile(train_openmax_unknown, 95)
    avg_threshold = np.mean(list(class_thresholds.values()))
    print(f"🎯 训练集参考阈值 (95th percentile, 仅可视化): {class_thresholds}")
    print(f"🛡️ 平均参考阈值: {avg_threshold:.4f}")

    # =========================================================================
    # PHASE 2: 在测试集上使用 OpenMax 进行最终评估
    # =========================================================================
    test_pkl_path = cfg.eval_data_path.replace('.pkl', '_test.pkl')
    print(f"\n🚀 [PHASE 2] 正在测试集上执行 OpenMax 评估 ...")

    if not os.path.exists(test_pkl_path):
        print(f"❌ 错误: 未找到测试集文件 {test_pkl_path}")
        return

    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(test_data['features']).float(),
        torch.tensor(test_data['coarse_labels']).long(),
        torch.tensor(test_data['fine_labels']).long(),
        torch.tensor(test_data['snrs']).float()
    ), batch_size=cfg.batch_size, shuffle=False)

    # --- 测试集推理 (内联，收集所有评估所需指标) ---
    model.eval()
    test_res = {
        'c_true': [], 'f_true': [], 'snr': [],
        'c_pred': [], 'f_pred': [], 'embs': [],
        'msp': [], 'entropy': [],
    }
    openmax_unknown_probs_test = []
    openmax_preds_test = []

    with torch.no_grad():
        for x, y_c, y_f, snr in tqdm(test_loader, desc="Test Inference"):
            x_dev = x.to(cfg.device)
            input_data = x_dev.unsqueeze(1) if x_dev.dim() == 2 else x_dev
            output = model(input_data)

            c_logits = output['coarse_logits']          # [B, num_coarse]
            f_logits = output['fine_logits']            # [B, num_fine]
            z_raw = output['z_raw'].cpu().numpy()       # [B, D]
            c_logits_np = c_logits.cpu().numpy()        # [B, num_coarse]

            test_res['c_true'].append(y_c.numpy())
            test_res['f_true'].append(y_f.numpy())
            test_res['snr'].append(snr.numpy())
            test_res['c_pred'].append(c_logits.argmax(1).cpu().numpy())
            test_res['f_pred'].append(f_logits.argmax(1).cpu().numpy())
            test_res['embs'].append(output['embedding'].cpu().numpy())

            # MSP & Entropy
            probs = torch.softmax(c_logits, dim=1)
            msp, _ = torch.max(probs, dim=1)
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            test_res['msp'].append(msp.cpu().numpy())
            test_res['entropy'].append((ent / np.log(max(probs.shape[1], 2))).cpu().numpy())

            # OpenMax 推理 (逐样本)
            for i in range(len(y_c)):
                om_probs, _, _ = calibrator.predict_openmax(z_raw[i], c_logits_np[i])
                openmax_unknown_probs_test.append(om_probs[-1])
                openmax_preds_test.append(np.argmax(om_probs))

    # 合并
    for k in test_res:
        test_res[k] = np.concatenate(test_res[k])
    test_res['final_score'] = np.array(openmax_unknown_probs_test)
    openmax_preds_test = np.array(openmax_preds_test)

    known_mask_test = (test_res['c_true'] != -1)
    unknown_mask_test = ~known_mask_test

    known_scores = test_res['final_score'][known_mask_test]
    unknown_scores = test_res['final_score'][unknown_mask_test]
    print(f"DEBUG: OpenMax Unknown Prob - 已知类分位数 [50, 90, 95, 99]: {np.percentile(known_scores, [50, 90, 95, 99])}")
    print(f"DEBUG: OpenMax Unknown Prob - 未知类分位数 [50, 90, 95, 99]: {np.percentile(unknown_scores, [50, 90, 95, 99])}")

    # --- OpenMax argmax 判定：unknown 类 (索引 num_coarse) 获胜 → 判为 -1 ---
    open_pred = np.where(openmax_preds_test == num_coarse, -1, openmax_preds_test)

    # --- 计算指标 ---
    known_mask = test_res['c_true'] != -1
    closed_oa = accuracy_score(test_res['c_true'][known_mask], test_res['c_pred'][known_mask])
    open_oa = accuracy_score(test_res['c_true'], open_pred)

    # --- 计算召回率 ---
    true_labels = test_res['c_true'][known_mask]
    pred_labels = open_pred[known_mask]
    class_indices = list(range(num_coarse))
    per_class_recall = recall_score(true_labels, pred_labels, labels=class_indices, average=None)
    macro_recall = np.mean(per_class_recall)
    print(f"\n📈 [召回率报告 - OpenMax Argmax]")
    print(f"   - 总体平均召回率 (Known Recall): {macro_recall*100:.2f}%")
    for i in range(len(per_class_recall)):
        label_name = COARSE_LABEL_NAMES[i]
        recall_val = per_class_recall[i]
        print(f"   - 类别 {i} ({label_name}) 召回率: {recall_val*100:.2f}%")

    print(f"\n📊 [OpenMax 评测结果]")
    print(f"   - 闭集分类准确率 (Known): {closed_oa*100:.2f}%")
    print(f"   - 开集识别准确率 (Total): {open_oa*100:.2f}%")

    # --- Polar 类深度诊断 ---
    polar_idx = 3
    polar_fp_mask = (test_res['c_true'] == -1) & (open_pred == polar_idx)
    known_polar_mask = (test_res['c_true'] == polar_idx)

    if polar_fp_mask.any():
        print(f"\n🔍 [深度诊断: Polar 误判分析 - OpenMax]")
        print(f" - [误判为 Polar 的未知类数量]: {polar_fp_mask.sum()}")
        print(f" - [误判样本] OpenMax Unknown Prob 均值: {np.mean(test_res['final_score'][polar_fp_mask]):.4f}")
        print(f" - [真正 Polar] OpenMax Unknown Prob 均值: {np.mean(test_res['final_score'][known_polar_mask]):.4f}")
        print(f" - [真正 Polar] OpenMax Unknown Prob 95分位数: {np.percentile(test_res['final_score'][known_polar_mask], 95):.4f}")

    # --- 绘制 OpenMax Unknown Prob 分布 ---
    plt.figure(figsize=(8, 5))
    plt.hist(test_res['final_score'][known_polar_mask], bins=100, alpha=0.5,
             label='True Polar (OpenMax Unknown Prob)', color='blue', density=True)
    if polar_fp_mask.any():
        plt.hist(test_res['final_score'][polar_fp_mask], bins=100, alpha=0.5,
                 label='Misclassified Unknown (OpenMax Unknown Prob)', color='red', density=True)
    plt.axvline(class_thresholds.get(polar_idx, avg_threshold), color='k', linestyle='--',
                label=f'Polar Ref Thr (95%): {class_thresholds.get(polar_idx, avg_threshold):.4f}')
    plt.yscale('log')
    plt.title("OpenMax Diagnosis: Unknown Probability Distribution (Log Scale)")
    plt.xlabel("OpenMax Unknown Probability")
    plt.ylabel("Density (Log)")
    plt.legend()
    plt.savefig(os.path.join(cfg.results_dir, 'polar_openmax_diagnosis.png'), dpi=300)
    plt.close()

    test_res['c_pred_original'] = test_res['c_pred'].copy()
    test_res['c_pred'] = open_pred
    diagnose_osr_scores(test_res, cfg)
    # =========================================================================
    # 5. 绘图与结果持久化 (使用测试集结果)
    # =========================================================================
    # 这里我们把测试集推理出的 test_res 传给绘图函数
    
    plot_all_confusion_matrices(
        test_res, cfg, class_thresholds, COARSE_LABEL_NAMES, FINE_LABEL_NAMES
    )
    
    plot_evt_distribution(test_res, class_thresholds, cfg)

    plot_results(test_res, cfg, class_thresholds, test_data['coarse_map'], test_data['fine_map'], closed_oa, open_oa)

    print(f"\n✨ OpenMax 评估任务全部完成！所有图表和数据已保存至: {cfg.results_dir}")

if __name__ == "__main__":
    main()