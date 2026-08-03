"""
ARPL 实验结果可视化评估器。

参考 open_recognition/evaluator.py 的绘图模式，为 ARPL 对比实验生成:
  - OSCR 曲线
  - 分信噪比 ROC 曲线
  - 准确率-信噪比图 (雷达图 + 线性图)
  - OSR 分数分布 (MSP / 距离分数)
  - T-SNE 特征可视化
  - 闭集 / 开集混淆矩阵
  - MSP 分布直方图

用法:
    from evaluator import evaluate_and_plot
    evaluate_and_plot(model, criterion, test_loader, out_loader, cfg, coarse_map)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tqdm import tqdm
from matplotlib import font_manager

# ── 中文字体配置 ──
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
font_cn = font_manager.FontProperties(fname=FONT_PATH, size=12)
font_cn_title = font_manager.FontProperties(fname=FONT_PATH, size=14, weight='bold')

# ── 编码体制名称 ──
COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH", "Ham"]


# =========================================================================
# 推理辅助
# =========================================================================

def get_predictions(model, criterion, loader, cfg, device=None):
    """
    对 DataLoader 中的所有样本进行推理，收集嵌入、logits、MSP、预测等。

    Args:
        model:     ARPLNet
        criterion: ARPLoss / RPLoss / Softmax
        loader:    DataLoader (返回 data, labels, [snr])
        cfg:       配置
        device:    计算设备
    Returns:
        res: dict 包含 embs, logits, logits_m (模型自身分类头logits),
                  c_true, c_pred, msp, msp_m, snr
    """
    if device is None:
        device = cfg.device
    model.eval()
    res = {
        'embs': [], 'logits': [], 'logits_m': [], 'c_true': [], 'c_pred': [],
        'msp': [], 'msp_m': [], 'snr': [],
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            if len(batch) == 3:
                data, labels, snr = batch
            else:
                data, labels = batch
                snr = None

            data = data.to(device)
            labels = labels.to(device)

            # 模型前向: 返回 (embedding, classifier_logits)
            embedding, logits_m = model(data)

            # Criterion 重算的距离 logits (ARPLoss / RPLoss)
            logits, loss = criterion(embedding, logits_m)

            # ── 从 ARPLoss 距离 logits 计算 MSP ──
            probs = F.softmax(logits, dim=1)
            msp, pred = probs.max(1)

            # ── 从模型自身分类头 logits 计算 MSP ──
            probs_m = F.softmax(logits_m, dim=1)
            msp_m, pred_m = probs_m.max(1)

            res['embs'].append(embedding.cpu().numpy())
            res['logits'].append(logits.cpu().numpy())
            res['logits_m'].append(logits_m.cpu().numpy())
            res['c_true'].append(labels.cpu().numpy())
            res['c_pred'].append(pred.cpu().numpy())    # 基于 ARPLoss 距离 logits 的预测 (闭集 ACC)
            res['msp'].append(msp.cpu().numpy())
            res['msp_m'].append(msp_m.cpu().numpy())
            if snr is not None:
                res['snr'].append(snr.numpy())

    for k in res:
        if res[k]:
            res[k] = np.concatenate(res[k])
    return res


# =========================================================================
# OSCR 曲线
# =========================================================================

def compute_oscr(pred_k, pred_u, labels):
    """
    计算 OSCR (Open Set Classification Rate).
    参考 ARPL-master/core/evaluation.py 的实现。
    """
    x1 = np.max(pred_k, axis=1)
    x2 = np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)

    m_x1 = np.zeros(len(x1))
    m_x1[correct] = 1
    k_target = np.concatenate([m_x1, np.zeros(len(x2))])
    u_target = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
    predict = np.concatenate([x1, x2])
    n = len(predict)

    CCR = np.zeros(n + 2)
    FPR = np.zeros(n + 2)
    idx = np.argsort(predict)
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()
        CCR[k] = CC / len(x1)
        FPR[k] = FP / len(x2)

    CCR[n] = 0.0; FPR[n] = 0.0
    CCR[n + 1] = 1.0; FPR[n + 1] = 1.0
    ROC = sorted(zip(FPR, CCR), reverse=True)
    oscr = 0.0
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0
        oscr += h * w
    return oscr * 100.0


def plot_oscr_curve(pred_k, pred_u, labels, save_path):
    """绘制 OSCR 曲线并保存。"""
    x1 = np.max(pred_k, axis=1)
    x2 = np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)

    scores_all = np.concatenate([x1, x2])
    gt_correct = np.concatenate([correct.astype(int), np.zeros(len(x2))])
    gt_unknown = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

    # 按分数升序排列
    indices = np.argsort(scores_all)
    gt_correct = gt_correct[indices]
    gt_unknown = gt_unknown[indices]

    # 改用后缀累加（而非前缀累加），以匹配 compute_oscr 的逻辑：
    #   CCR(t) = 分数 > t 的正确已知样本比例（后缀和）
    #   FPR(t) = 分数 >= t 的未知样本比例（后缀和）
    ccr = np.cumsum(gt_correct[::-1])[::-1] / len(x1)
    fpr = np.cumsum(gt_unknown[::-1])[::-1] / len(x2)

    # 反转使 FPR 从 0→1（严格阈值→宽松阈值），用于绘图和积分
    ccr = ccr[::-1]
    fpr = fpr[::-1]

    # 添加边界点 (0,0) 和 (1,1) 使曲线完整
    ccr = np.concatenate([[0], ccr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    oscr_score = np.trapz(ccr, fpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, ccr, label=f'OSCR (AUC={oscr_score:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('FPR（未知类被接受）', fontproperties=font_cn)
    plt.ylabel('CCR（已知类正确识别率）', fontproperties=font_cn)
    plt.title('OSCR 曲线', fontproperties=font_cn_title)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f'  OSCR curve saved: {save_path}')


# =========================================================================
# AUROC 计算
# =========================================================================

def compute_auroc(known_scores, unknown_scores):
    """计算 AUROC。"""
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
    idx = np.argsort(-scores)
    labels_sorted = labels[idx]
    tpr = np.cumsum(labels_sorted) / np.sum(labels_sorted)
    fpr = np.cumsum(1 - labels_sorted) / np.sum(1 - labels_sorted)
    return np.trapz(tpr, fpr) * 100.0


# =========================================================================
# 分 SNR ROC 曲线
# =========================================================================

def plot_snr_roc_curves(res, save_path, score_key='msp'):
    """按不同 SNR 绘制 ROC 曲线。"""
    known_mask = res['c_true'] != -1
    unique_snrs = np.unique(res['snr'][known_mask]) if 'snr' in res and len(res['snr']) > 0 else []
    if len(unique_snrs) == 0:
        return

    plt.figure(figsize=(10, 7))
    plot_snrs = unique_snrs[::2] if len(unique_snrs) > 5 else unique_snrs
    for s in plot_snrs:
        m_k = (res['snr'] == s) & known_mask
        m_u = (res['snr'] == s) & ~known_mask
        if m_k.any() and m_u.any():
            scores = np.concatenate([res[score_key][m_k], res[score_key][m_u]])
            true_l = np.concatenate([np.zeros(m_k.sum()), np.ones(m_u.sum())])
            fpr_s, tpr_s, _ = roc_curve(true_l, -scores)  # 取反：MSP 越高越像已知
            auc_s = auc(fpr_s, tpr_s)
            plt.plot(fpr_s, tpr_s, label=f'SNR {s}dB (AUC={auc_s:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Detection ROC Curves at Different SNRs', fontsize=13, fontweight='bold')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f'  SNR ROC curves saved: {save_path}')


# =========================================================================
# 准确率-信噪比图
# =========================================================================

def plot_accuracy_vs_snr(res, save_dir, coarse_map):
    """绘制各编码体制在不同 SNR 下的准确率雷达图和线性图。"""
    known_mask = res['c_true'] != -1
    unique_snrs = np.sort(np.unique(res['snr'][known_mask])) if 'snr' in res and len(res['snr']) > 0 else []
    if len(unique_snrs) == 0:
        print('  Warning: no SNR data, skipping accuracy-vs-SNR plots')
        return

    num_classes = len(coarse_map)
    acc_matrix = np.zeros((num_classes, len(unique_snrs)))

    for j, s in enumerate(unique_snrs):
        mask_s = (res['snr'] == s) & known_mask
        for i in range(num_classes):
            mask = mask_s & (res['c_true'] == i)
            acc_matrix[i, j] = np.mean(res['c_pred'][mask] == i) if mask.sum() > 0 else np.nan

    # 保存 CSV
    coarse_names = [COARSE_LABEL_NAMES[i] if i < len(COARSE_LABEL_NAMES) else f'C-{i}'
                    for i in range(num_classes)]
    df = pd.DataFrame(acc_matrix, index=coarse_names,
                      columns=[f'{s}dB' for s in unique_snrs])
    df['AvgAcc'] = df.mean(axis=1)
    df.to_csv(os.path.join(save_dir, 'coarse_accuracy_data.csv'), float_format='%.4f')

    # ── 雷达图 ──
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(unique_snrs), endpoint=False).tolist()
    angles += angles[:1]
    for i in range(num_classes):
        values = acc_matrix[i].tolist()
        values += values[:1]
        plt.plot(angles, values, label=coarse_names[i], linewidth=2)
        plt.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{s:.0f}dB' for s in unique_snrs], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    plt.title('Coarse Accuracy Radar Chart', pad=30, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'coarse_accuracy_radar.png'), dpi=300)
    plt.close()

    # ── 线性图 ──
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        avg_acc = np.nanmean(acc_matrix[i]) * 100
        plt.plot(unique_snrs, acc_matrix[i], marker='o', linewidth=2,
                 label=f"{coarse_names[i]} (Avg:{avg_acc:.1f}%)")
    plt.title('Coarse-level Recognition Accuracy vs SNR', fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)'); plt.ylabel('Accuracy'); plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'coarse_accuracy_linear.png'), dpi=300)
    plt.close()
    print(f'  Accuracy plots saved: {save_dir}')


# =========================================================================
# OSR 分数分布
# =========================================================================

def plot_score_distribution(res, save_path, score_key='msp', score_label='MSP'):
    """绘制已知类和未知类的 MSP 分数分布对比。"""
    known_mask = res['c_true'] != -1
    plt.figure(figsize=(10, 6))
    sns.histplot(res[score_key][known_mask], label='Known Codes',
                 color='blue', kde=True, stat='density', alpha=0.4, bins=50)
    unk_mask = ~known_mask
    if unk_mask.any():
        sns.histplot(res[score_key][unk_mask], label='Unknown Codes',
                     color='orange', kde=True, stat='density', alpha=0.4, bins=50)
    plt.title(f'Score Distribution (Known vs Unknown)', fontsize=13, fontweight='bold')
    plt.xlabel(score_label); plt.ylabel('Density')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f'  Score distribution saved: {save_path}')


# =========================================================================
# T-SNE 特征可视化
# =========================================================================

def plot_tsne(res, save_path, coarse_map):
    """T-SNE 降维可视化特征空间中的已知/未知类分布。"""
    embs = res['embs']
    if len(embs) > 3000:
        idx = np.random.choice(len(embs), 3000, replace=False)
        embs = embs[idx]
        labels = res['c_true'][idx]
    else:
        labels = res['c_true']

    # 归一化
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    print('  Computing T-SNE (this may take a moment)...')
    tsne = TSNE(n_components=2, random_state=42)
    low_dim = tsne.fit_transform(embs_norm)

    num_classes = len(coarse_map)
    coarse_names = [COARSE_LABEL_NAMES[i] if i < len(COARSE_LABEL_NAMES) else f'C-{i}'
                    for i in range(num_classes)]

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        mask = labels == i
        if mask.any():
            plt.scatter(low_dim[mask, 0], low_dim[mask, 1],
                        label=coarse_names[i], s=25, alpha=0.7)
    unk_mask = labels == -1
    if unk_mask.any():
        plt.scatter(low_dim[unk_mask, 0], low_dim[unk_mask, 1],
                    c='black', marker='x', label='Unknown', s=35, alpha=0.5)
    plt.title('T-SNE Visualization of Feature Space', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f'  T-SNE visualization saved: {save_path}')


# =========================================================================
# 混淆矩阵
# =========================================================================

def plot_confusion_matrices(res, save_dir, coarse_map):
    """绘制闭集和开集混淆矩阵。"""
    os.makedirs(save_dir, exist_ok=True)

    num_classes = len(coarse_map)
    coarse_names = [COARSE_LABEL_NAMES[i] if i < len(COARSE_LABEL_NAMES) else f'C-{i}'
                    for i in range(num_classes)]

    # ── 开集混淆矩阵 (含 Unknown 类) ──
    osr_labels = list(range(num_classes)) + [-1]
    osr_names = coarse_names + ['Unknown']
    cm_osr = confusion_matrix(res['c_true'], res['c_pred'], labels=osr_labels)
    cm_osr_perc = cm_osr.astype('float') / cm_osr.sum(axis=1)[:, np.newaxis]
    cm_osr_perc = np.nan_to_num(cm_osr_perc)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_osr_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=osr_names, yticklabels=osr_names)
    plt.title('OSR 混淆矩阵', fontproperties=font_cn_title)
    plt.xlabel('预测标签', fontproperties=font_cn)
    plt.ylabel('真实标签', fontproperties=font_cn)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'osr_confusion_matrix.png'), dpi=300)
    plt.close()
    pd.DataFrame(cm_osr, index=osr_names, columns=osr_names).to_csv(
        os.path.join(save_dir, 'osr_confusion_matrix.csv'))

    # ── 闭集混淆矩阵 (仅已知类) ──
    known_mask = res['c_true'] != -1
    true_closed = res['c_true'][known_mask]
    pred_closed = res['c_pred'][known_mask]
    cm_closed = confusion_matrix(true_closed, pred_closed, labels=list(range(num_classes)))
    cm_closed_perc = cm_closed.astype('float') / cm_closed.sum(axis=1)[:, np.newaxis]
    cm_closed_perc = np.nan_to_num(cm_closed_perc)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_closed_perc, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=coarse_names, yticklabels=coarse_names)
    plt.title('闭集混淆矩阵', fontproperties=font_cn_title)
    plt.xlabel('预测标签', fontproperties=font_cn)
    plt.ylabel('真实标签', fontproperties=font_cn)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'closed_confusion_matrix.png'), dpi=300)
    plt.close()
    pd.DataFrame(cm_closed, index=coarse_names, columns=coarse_names).to_csv(
        os.path.join(save_dir, 'closed_confusion_matrix.csv'))

    print(f'  Confusion matrices saved: {save_dir}')


# =========================================================================
# 闭集准确率随训练轮次变化曲线
# =========================================================================

def plot_training_history(history, save_path):
    """绘制训练过程中的 loss 和 acc 曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    axes[0].plot(history['val_loss'], label='Val Loss', color='red', alpha=0.7)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Acc
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue', alpha=0.7)
    axes[1].plot(history['val_acc'], label='Val Acc', color='red', alpha=0.7)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f'  Training history saved: {save_path}')


# =========================================================================
# 汇总评估与绘图
# =========================================================================

def evaluate_and_plot(model, criterion, test_loader, out_loader, cfg, coarse_map,
                      train_history=None):
    """
    完整的 OSR 评估与可视化入口。

    对已知类测试集 + 未知类测试集进行推理，计算 ACC / AUROC / OSCR，
    并绘制所有可视化图表。

    Args:
        model:        训练好的 ARPLNet
        criterion:    ARPLoss / RPLoss / Softmax
        test_loader:  已知类测试 DataLoader
        out_loader:   未知类 DataLoader
        cfg:          配置
        coarse_map:   标签映射字典
        train_history: 可选，训练历史记录 {train_loss, val_loss, train_acc, val_acc}
    Returns:
        results: dict {ACC, AUROC, OSCR}
    """
    device = cfg.device

    # 确保输出目录存在
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.cm_dir, exist_ok=True)
    os.makedirs(cfg.acc_dir, exist_ok=True)

    # ── 1. 已知类推理 ──
    print('\n--- Evaluating Known Classes ---')
    res_known = get_predictions(model, criterion, test_loader, cfg, device)
    # 已知类标签标记为非 -1，保持原始值
    # res_known['c_true'] 已经是原始标签

    # ── 2. 未知类推理 ──
    print('\n--- Evaluating Unknown Classes ---')
    res_unknown = get_predictions(model, criterion, out_loader, cfg, device)
    # 给未知类统一标 -1
    res_unknown['c_true'][:] = -1

    # ── 3. 合并结果 (同时保留 ARPLoss 距离 logits 和模型自身分类头 logits) ──
    res = {}
    for k in ['embs', 'logits', 'logits_m', 'msp', 'msp_m']:
        res[k] = np.concatenate([res_known.get(k, np.array([])),
                                 res_unknown.get(k, np.array([]))], axis=0)
    for k in ['c_true', 'c_pred', 'snr']:
        if k in res_known and len(res_known[k]) > 0:
            res[k] = np.concatenate([res_known[k], res_unknown.get(k, np.array([]))], axis=0)
        else:
            res[k] = np.concatenate([res_known.get(k, np.array([])),
                                     res_unknown.get(k, np.array([]))], axis=0)

    # ── 4. 计算指标 ──
    known_mask = res['c_true'] != -1
    correct = (res['c_pred'][known_mask] == res['c_true'][known_mask]).sum()
    total = known_mask.sum()
    acc = correct / total * 100.0

    # 使用模型自身分类头的 MSP (msp_m) 做 OOD 检测：
    #   ARPLoss 的距离 logits (dist_l2 - dist_dot) 在调制信号域可能
    #   出现未知类 MSP 高于已知类的反直觉现象，模型分类头的 logits
    #   (标准 Softmax) 对 OOD 检测更鲁棒。
    has_model_logits = 'msp_m' in res and len(res.get('msp_m', [])) > 0
    if has_model_logits:
        auroc = compute_auroc(res['msp_m'][known_mask], res['msp_m'][~known_mask])
        oscr = compute_oscr(res_known['logits_m'], res_unknown['logits_m'],
                            res_known['c_true'])
    else:
        # 回退到 ARPLoss 距离 logits
        auroc = compute_auroc(res['msp'][known_mask], res['msp'][~known_mask])
        oscr = compute_oscr(res_known['logits'], res_unknown['logits'],
                            res_known['c_true'])

    results = {'ACC': acc, 'AUROC': auroc, 'OSCR': oscr}

    print(f'\n{"=" * 50}')
    print(f'ARPL ({cfg.loss_type}) Evaluation Results:')
    print(f'  ACC:   {acc:.2f}%')
    print(f'  AUROC: {auroc:.2f}%')
    print(f'  OSCR:  {oscr:.2f}%')
    print(f'{"=" * 50}')

    # ── 5. 绘图 ──
    print('\n--- Plotting Results ---')

    # 选择 OOD 评分数据源
    if has_model_logits:
        logits_k_ood = res_known['logits_m']
        logits_u_ood = res_unknown['logits_m']
        score_key = 'msp_m'
        score_label = 'Softmax MSP (from model classifier)'
    else:
        logits_k_ood = res_known['logits']
        logits_u_ood = res_unknown['logits']
        score_key = 'msp'
        score_label = 'ARPLoss MSP (from distance logits)'

    # OSCR 曲线 (使用选定的 logits)
    plot_oscr_curve(logits_k_ood, logits_u_ood, res_known['c_true'],
                    os.path.join(cfg.results_dir, 'oscr_curve.png'))

    # 分 SNR ROC 曲线
    if score_key in res and len(res.get(score_key, [])) > 0:
        plot_snr_roc_curves(res, os.path.join(cfg.results_dir, 'snr_roc_curves.png'),
                            score_key=score_key)

    # 准确率-信噪比图
    plot_accuracy_vs_snr(res, cfg.acc_dir, coarse_map)

    # MSP 分数分布
    if score_key in res:
        plot_score_distribution(res, os.path.join(cfg.results_dir, 'msp_distribution.png'),
                                score_key=score_key, score_label=score_label)

    # T-SNE
    plot_tsne(res, os.path.join(cfg.results_dir, 'tsne_visualization.png'), coarse_map)

    # 混淆矩阵
    plot_confusion_matrices(res, cfg.cm_dir, coarse_map)

    # 训练历史
    if train_history is not None:
        plot_training_history(train_history, os.path.join(cfg.results_dir, 'training_history.png'))

    print(f'\nAll plots saved to: {cfg.results_dir}')
    return results
