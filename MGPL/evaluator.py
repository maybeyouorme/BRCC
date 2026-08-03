"""
MGPL Open-Set Recognition Evaluator.

Uses minimum distance to the nearest prototype as the OSR score:
  - known classes → small distance (close to some prototype)
  - unknown classes → large distance (far from all prototypes)

Evaluation metrics:
  - Closed-set accuracy (nearest-prototype classification)
  - AUROC for open-set detection
  - OSCR curve
  - Per-SNR ROC curves
  - t-SNE feature visualization
  - Confusion matrices
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, recall_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from thop import profile, clever_format
import gc

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from MGPL.config import Config
from MGPL.models import MGPLNet

from matplotlib import font_manager
font_cn = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    size=12
)

COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH","Ham"]


# =========================================================================
# MGPL Inference
# =========================================================================

def get_predictions(model, loader, device):
    """
    MGPL inference: extract features, distances, and predictions.

    Returns dict with:
      - embs:       latent features [N, latent_dim]
      - c_true:     true coarse labels [N]
      - snr:        SNR values [N]
      - c_pred:     predicted labels (nearest prototype) [N]
      - dist_min:   minimum distance to prototypes [N]  (OSR score)
      - dists:      full distance matrix [N, n_prototypes]
      - kl_min:     minimum KL divergence to prototypes [N]
    """
    model.eval()
    res = {
        'embs': [], 'c_true': [], 'snr': [],
        'c_pred': [], 'dist_min': [], 'dists': [],
        'kl_min': [],
    }

    with torch.no_grad():
        for x, y_c, snr in tqdm(loader, desc='MGPL Inference'):
            x_dev = x.to(device)
            if x_dev.dim() == 2:
                x_dev = x_dev.unsqueeze(1)

            latent_z, dist, kl_div = model(x_dev)

            # Nearest-prototype prediction
            n_classes = model.n_classes
            n_sub = model.n_sub_prototypes
            dist_reshaped = dist.cpu().reshape(len(x), n_classes, n_sub)
            dist_class_min, _ = dist_reshaped.min(dim=2)   # [B, n_classes]
            dist_min, _ = dist_class_min.min(dim=1)        # [B] — OSR score
            preds = dist_class_min.argmin(dim=1)            # [B]

            # KL min (alternative OSR score)
            kld_reshaped = kl_div.cpu().reshape(len(x), n_classes, n_sub)
            kld_class_min, _ = kld_reshaped.min(dim=2)
            kld_min, _ = kld_class_min.min(dim=1)

            res['embs'].append(latent_z.cpu().numpy())
            res['c_true'].append(y_c.numpy())
            res['snr'].append(snr.numpy())
            res['c_pred'].append(preds.numpy())
            res['dist_min'].append(dist_min.numpy())
            res['dists'].append(dist.cpu().numpy())
            res['kl_min'].append(kld_min.numpy())

    for k in res:
        res[k] = np.concatenate(res[k])

    print(f'Unknown samples in results: {(res["c_true"] == -1).sum()}')
    return res


# =========================================================================
# Per-Class Threshold Search
# =========================================================================

def find_class_specific_thresholds(scores, c_true, c_pred, target_recall=0.95):
    """
    For each known class, find a distance threshold such that
    `target_recall` of correctly-classified samples are accepted.

    Args:
        scores:    outlier scores (larger = more unknown) [N]
        c_true:    true labels [N]  (-1 for unknown)
        c_pred:    predicted labels [N]
        target_recall: desired recall (e.g., 0.95 = accept 95% of knowns)
    Returns:
        dict: {class_id: threshold}
    """
    thresholds = {}
    known_mask = c_true != -1
    unique_classes = np.unique(c_true[known_mask])

    for cls_id in unique_classes:
        # Only correctly classified samples to model "normal" behavior
        mask = (c_true == cls_id) & (c_pred == cls_id)
        if mask.sum() < 10:
            mask = c_true == cls_id  # fallback

        cls_scores = np.sort(scores[mask])
        idx = int(len(cls_scores) * target_recall)
        thresholds[cls_id] = cls_scores[min(idx, len(cls_scores) - 1)]

    return thresholds


# =========================================================================
# Plotting Functions
# =========================================================================

def plot_osr_results(res, results_dir, acc_dir, threshold, coarse_map, closed_oa, open_oa):
    """Plot OSCR, SNR-ROC, radar chart, t-SNE, and score distribution."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)

    known_mask = res['c_true'] != -1
    snr_vals = sorted(np.unique(res['snr'][known_mask]))
    num_snr = len(snr_vals)
    num_coarse = len(coarse_map)
    coarse_names = [COARSE_LABEL_NAMES[i] for i in range(num_coarse)]

    x1 = res['dist_min'][known_mask]       # known scores
    x2 = res['dist_min'][~known_mask]      # unknown scores
    pred = res.get('c_pred_original', res['c_pred'])[known_mask]
    labels_known = res['c_true'][known_mask]
    correct = (pred == labels_known)

    # ========== OSCR Curve ==========
    scores_all = np.concatenate([x1, x2])
    gt_correct = np.concatenate([correct.astype(int), np.zeros(len(x2))])
    gt_unknown = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

    indices = np.argsort(scores_all)
    gt_correct = gt_correct[indices]
    gt_unknown = gt_unknown[indices]

    ccr = np.concatenate([[0], np.cumsum(gt_correct) / len(x1)])
    fpr_os = np.concatenate([[0], np.cumsum(gt_unknown) / len(x2)])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_os, ccr, label=f'OSCR (AUC={np.trapz(ccr, fpr_os):.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (Unknown accepted)', fontproperties=font_cn)
    plt.ylabel('CCR (Known correctly classified)', fontproperties=font_cn)
    plt.title('OSCR Curve (MGPL)', fontproperties=font_cn)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'oscr_curve.png'), dpi=300)
    plt.close()
    gc.collect()

    # ========== Overall ROC Curve (AUROC) ==========
    y_true_roc = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
    y_score_roc = np.concatenate([x1, x2])
    fpr_roc, tpr_roc, _ = roc_curve(y_true_roc, y_score_roc)
    auroc_val = auc(fpr_roc, tpr_roc)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_roc, tpr_roc, label=f'ROC Curve (AUROC={auroc_val:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random (AUROC=0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (1 - Specificity)', fontproperties=font_cn)
    plt.ylabel('TPR (Recall / Sensitivity)', fontproperties=font_cn)
    plt.title('Overall ROC Curve (MGPL)', fontproperties=font_cn)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    gc.collect()
    print(f'   AUROC curve saved to: {os.path.join(results_dir, "roc_curve.png")}')

    # ========== Per-SNR ROC ==========
    plt.figure(figsize=(10, 7))
    plot_snrs = snr_vals[::2] if len(snr_vals) > 5 else snr_vals
    for s in plot_snrs:
        m_k = (res['snr'][known_mask] == s)
        m_u = (res['snr'][~known_mask] == s)
        if m_k.any() and m_u.any():
            s_score = np.concatenate([x1[m_k], x2[m_u]])
            s_true = np.concatenate([np.zeros(m_k.sum()), np.ones(m_u.sum())])
            fpr_s, tpr_s, _ = roc_curve(s_true, s_score)
            auc_s = auc(fpr_s, tpr_s)
            plt.plot(fpr_s, tpr_s, label=f'SNR {s}dB (AUC = {auc_s:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('MGPL Detection ROC at Different SNRs')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'snr_roc_curves.png'), dpi=300)
    plt.close()
    gc.collect()

    # ========== Coarse Accuracy Matrix ==========
    coarse_pred = res.get('c_pred_original', res['c_pred'])
    coarse_acc_matrix = np.zeros((num_coarse, num_snr))
    for j, s in enumerate(snr_vals):
        mask_s = (res['snr'] == s) & known_mask
        for i in range(num_coarse):
            mask = mask_s & (res['c_true'] == i)
            coarse_acc_matrix[i, j] = np.mean(coarse_pred[mask] == i) if mask.sum() > 0 else np.nan

    df_coarse = pd.DataFrame(
        coarse_acc_matrix, index=coarse_names,
        columns=[f'{s}dB' for s in snr_vals]
    )
    df_coarse['AvgAcc'] = df_coarse.mean(axis=1)
    df_coarse.to_csv(os.path.join(acc_dir, 'coarse_accuracy_data.csv'), float_format='%.4f')

    # --- Radar Chart ---
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, num_snr, endpoint=False).tolist()
    angles += angles[:1]
    for i in range(num_coarse):
        values = coarse_acc_matrix[i].tolist() + [coarse_acc_matrix[i][0]]
        ax.plot(angles, values, label=coarse_names[i], linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{s:.0f}dB' for s in snr_vals])
    ax.set_ylim(0, 1.05)
    ax.set_title(f'MGPL Coarse Accuracy Radar (Closed OA: {closed_oa * 100:.2f}%)',
                 pad=30, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(acc_dir, 'coarse_accuracy_radar.png'), dpi=300)
    plt.close()
    gc.collect()

    # --- Linear Accuracy vs SNR ---
    plt.figure(figsize=(10, 6))
    for i in range(num_coarse):
        avg = np.nanmean(coarse_acc_matrix[i]) * 100
        plt.plot(snr_vals, coarse_acc_matrix[i], marker='o', linewidth=2,
                 label=f'{coarse_names[i]} (Avg:{avg:.1f}%)')
    plt.title(f'MGPL Coarse Accuracy vs SNR (Open OA: {open_oa * 100:.2f}%)',
              fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(acc_dir, 'coarse_accuracy_linear.png'), dpi=300)
    plt.close()
    gc.collect()

    # ========== OSR Score Distribution ==========
    plt.figure(figsize=(10, 6))
    sns.histplot(res['dist_min'][known_mask], label='Known Codes',
                 color='blue', kde=True, stat='density', alpha=0.4)
    sns.histplot(res['dist_min'][~known_mask], label='Unknown Codes',
                 color='orange', kde=True, stat='density', alpha=0.4)
    avg_thr = np.mean(list(threshold.values()))
    min_thr = min(threshold.values())
    max_thr = max(threshold.values())
    plt.axvline(avg_thr, color='red', linestyle='--', linewidth=2,
                label=f'Avg Threshold: {avg_thr:.4f}')
    plt.axvspan(min_thr, max_thr, color='red', alpha=0.1, label='Threshold Range')
    plt.title('MGPL Distance Score Distribution (larger = more unknown)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Min Distance to Prototypes')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'osr_score_distribution.png'), dpi=300)
    plt.close()
    gc.collect()

    # ========== T-SNE ==========
    print('Computing T-SNE...')
    tsne = TSNE(n_components=2, random_state=42)
    max_pts = 3000
    idx = np.random.choice(len(res['embs']), min(max_pts, len(res['embs'])), replace=False)
    embs_norm = res['embs'][idx] / (np.linalg.norm(res['embs'][idx], axis=1, keepdims=True) + 1e-9)
    low_dim = tsne.fit_transform(embs_norm)
    subset_labels = res['c_true'][idx]

    plt.figure(figsize=(10, 8))
    for i in range(num_coarse):
        m = subset_labels == i
        plt.scatter(low_dim[m, 0], low_dim[m, 1], label=coarse_names[i], s=25, alpha=0.7)
    m_un = subset_labels == -1
    if m_un.any():
        plt.scatter(low_dim[m_un, 0], low_dim[m_un, 1], c='black', marker='x',
                    label='Unknown', s=35, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()
    gc.collect()
    print('T-SNE done.')


def plot_confusion_matrices(res, results_dir, threshold, coarse_names):
    """Plot open-set and closed-set confusion matrices."""
    cm_dir = os.path.join(results_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    open_pred = res['c_pred'].copy()
    avg_thr = np.mean(list(threshold.values()))
    for i in range(len(open_pred)):
        cls = open_pred[i]
        thr = threshold.get(cls, avg_thr)
        if res['dist_min'][i] > thr:
            open_pred[i] = -1

    # --- Open-set CM ---
    osr_labels = list(range(len(coarse_names))) + [-1]
    osr_display = coarse_names + ['Unknown']
    cm_osr = confusion_matrix(res['c_true'], open_pred, labels=osr_labels)
    cm_osr_pct = cm_osr.astype(float) / cm_osr.sum(axis=1)[:, np.newaxis]
    cm_osr_pct = np.nan_to_num(cm_osr_pct)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_osr_pct, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=osr_display, yticklabels=osr_display)
    plt.title('MGPL OSR Confusion Matrix', fontproperties=font_cn)
    plt.xlabel('Predicted', fontproperties=font_cn)
    plt.ylabel('True', fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, 'osr_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    pd.DataFrame(cm_osr, index=osr_display, columns=osr_display).to_csv(
        os.path.join(cm_dir, 'osr_confusion_matrix.csv'))

    # --- Closed-set CM ---
    known_mask = res['c_true'] != -1
    true_closed = res['c_true'][known_mask]
    pred_closed = res.get('c_pred_original', res['c_pred'])[known_mask]
    cm_closed = confusion_matrix(true_closed, pred_closed, labels=list(range(len(coarse_names))))
    cm_closed_pct = cm_closed.astype(float) / cm_closed.sum(axis=1)[:, np.newaxis]
    cm_closed_pct = np.nan_to_num(cm_closed_pct)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_closed_pct, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=coarse_names, yticklabels=coarse_names)
    plt.title('MGPL Closed-Set Confusion Matrix', fontproperties=font_cn)
    plt.xlabel('Predicted', fontproperties=font_cn)
    plt.ylabel('True', fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, 'closed_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrices saved to: {cm_dir}')


# =========================================================================
# Main Evaluation Entry Point
# =========================================================================

def main():
    cfg = Config()
    device = torch.device(cfg.device)

    EVAL_DATA_PATH = cfg.eval_data_path
    val_data_path = EVAL_DATA_PATH.replace('.pkl', '_val.pkl')
    test_data_path = EVAL_DATA_PATH.replace('.pkl', '_test.pkl')

    # --- 0. Load model ---
    model = MGPLNet(cfg).to(device)
    if not os.path.exists(cfg.model_save_path):
        print(f'❌ Model not found: {cfg.model_save_path}')
        return

    checkpoint = torch.load(cfg.model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    print(f'✅ Loaded model: {cfg.model_save_path}')

    # Model stats
    dummy = torch.randn(1, 1, cfg.seq_len).to(device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    print(f'📊 Model: params={params}, flops={flops}')

    # =========================================================================
    # PHASE 1: Find per-class thresholds on validation set
    # =========================================================================
    print(f'\n🚀 [PHASE 1] Finding MGPL distance thresholds on validation set...')

    if not os.path.exists(val_data_path):
        print(f'❌ Validation data not found: {val_data_path}')
        return

    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_data['features']).float(),
            torch.tensor(val_data['coarse_labels']).long(),
            torch.tensor(val_data['snrs']).float()
        ),
        batch_size=cfg.batch_size, shuffle=False
    )

    val_res = get_predictions(model, val_loader, device)
    val_res['final_score'] = val_res['dist_min']  # MGPL OSR score

    # Find per-class thresholds
    class_thresholds = find_class_specific_thresholds(
        val_res['dist_min'], val_res['c_true'], val_res['c_pred'],
        target_recall=cfg.osr_confidence_level
    )
    avg_threshold = np.mean(list(class_thresholds.values()))

    print(f'🎯 Per-class distance thresholds:')
    for cls in sorted(class_thresholds.keys()):
        name = COARSE_LABEL_NAMES[cls] if cls < len(COARSE_LABEL_NAMES) else f'Class-{cls}'
        cls_scores = val_res['dist_min'][val_res['c_true'] == cls]
        print(f'   {name}: thr={class_thresholds[cls]:.4f} '
              f'(mean={cls_scores.mean():.4f}, p95={np.percentile(cls_scores, 95):.4f})')
    print(f'   Global avg threshold: {avg_threshold:.4f}')

    # =========================================================================
    # PHASE 2: Test set evaluation
    # =========================================================================
    print(f'\n🚀 [PHASE 2] Testing on evaluation set...')

    if not os.path.exists(test_data_path):
        print(f'❌ Test data not found: {test_data_path}')
        return

    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data['features']).float(),
            torch.tensor(test_data['coarse_labels']).long(),
            torch.tensor(test_data['snrs']).float()
        ),
        batch_size=cfg.batch_size, shuffle=False
    )

    test_res = get_predictions(model, test_loader, device)
    test_res['final_score'] = test_res['dist_min']

    known_mask = test_res['c_true'] != -1
    unknown_mask = ~known_mask

    # Stats
    print(f'\n📊 Distance distribution:')
    print(f'   Known:  mean={test_res["dist_min"][known_mask].mean():.4f}, '
          f'median={np.median(test_res["dist_min"][known_mask]):.4f}')
    if unknown_mask.any():
        print(f'   Unknown: mean={test_res["dist_min"][unknown_mask].mean():.4f}, '
              f'median={np.median(test_res["dist_min"][unknown_mask]):.4f}')

    # --- AUROC ---
    known_scores = test_res['dist_min'][known_mask]
    unknown_scores = test_res['dist_min'][unknown_mask]
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    auroc = roc_auc_score(y_true, y_score)
    # Youden's J threshold
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    youden_idx = (tpr - fpr).argmax()
    opt_threshold = thr_roc[youden_idx]

    print(f'\n📈 [MGPL AUROC]')
    print(f'   Known avg dist: {known_scores.mean():.4f}')
    print(f'   Unknown avg dist: {unknown_scores.mean():.4f}')
    print(f'   AUROC: {auroc:.4f}')
    print(f'   Optimal threshold (Youden): {opt_threshold:.4f}')

    # --- Closed-set accuracy ---
    closed_oa = accuracy_score(test_res['c_true'][known_mask],
                                test_res['c_pred'][known_mask])
    print(f'\n📊 [MGPL Classification]')
    print(f'   Closed-set accuracy: {closed_oa * 100:.2f}%')

    # --- Open-set accuracy with per-class thresholds ---
    open_pred = test_res['c_pred'].copy()
    for i in range(len(open_pred)):
        cls = open_pred[i]
        thr = class_thresholds.get(cls, avg_threshold)
        if test_res['dist_min'][i] > thr:
            open_pred[i] = -1

    open_oa = accuracy_score(test_res['c_true'], open_pred)

    # Per-class recall
    true_labels = test_res['c_true'][known_mask]
    pred_labels = open_pred[known_mask]
    class_indices = list(range(len(COARSE_LABEL_NAMES)))
    per_class_recall = recall_score(true_labels, pred_labels,
                                     labels=class_indices, average=None)
    macro_recall = np.mean(per_class_recall)

    print(f'\n📊 [MGPL Open-Set Evaluation]')
    print(f'   Open-set accuracy: {open_oa * 100:.2f}%')
    print(f'   Macro recall: {macro_recall * 100:.2f}%')
    for i, r in enumerate(per_class_recall):
        print(f'   {COARSE_LABEL_NAMES[i]} recall: {r * 100:.2f}%')

    # Save results
    test_res['c_pred_original'] = test_res['c_pred'].copy()
    test_res['c_pred'] = open_pred

    # --- Plotting ---
    print(f'\n🎨 Generating plots...')
    plot_osr_results(test_res, cfg.results_dir, cfg.acc_dir,
                     class_thresholds, test_data['coarse_map'],
                     closed_oa, open_oa)
    plot_confusion_matrices(test_res, cfg.results_dir,
                            class_thresholds, COARSE_LABEL_NAMES)

    print(f'\n✅ MGPL Evaluation Complete! All results saved to: {cfg.results_dir}')


if __name__ == '__main__':
    main()
'''
# 仅训练
python main.py --mode train

# 仅评估 (需预训练模型)
python main.py --mode eval

# 训练 + 评估
python main.py --mode both
'''