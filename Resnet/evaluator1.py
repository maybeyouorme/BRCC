import matplotlib
matplotlib.use('Agg')
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
from models import ResNetOSRNet
from config import Config
import gc
from matplotlib import font_manager

font_cn = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    size=12,
)

COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH","Ham"]


# =========================================================================
# 核心推理函数
# =========================================================================
def get_AV(model, loader, device):
    """提取模型推理结果，包括 OpenMax 所需的 z_raw 激活向量"""
    model.eval()
    res = {
        'embs': [], 'c_true': [], 'snr': [],
        'c_pred': [],
        'z_raw_list': [],
        'logits': [],
    }
    raw_metrics = {
        'msp': [],
        'entropy': [],
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            if len(batch) == 4:
                x, y_c, y_f, snr = batch
            elif len(batch) == 3:
                x, y_c, snr = batch
            else:
                x, y_c = batch
                snr = None

            x_dev = x.to(device)
            input_data = x_dev.unsqueeze(1) if x_dev.dim() == 2 else x_dev
            output = model(input_data)

            logits = output['logits']

            res['c_true'].append(y_c.numpy())
            if snr is not None:
                res['snr'].append(snr.numpy())
            res['c_pred'].append(logits.argmax(1).cpu().numpy())
            emb = output['embedding'].cpu().numpy()
            res['embs'].append(emb)
            res['z_raw_list'].append(emb)   # OpenMax: z_raw, [B, 128]
            res['logits'].append(logits.cpu().numpy())

            # MSP
            probs = torch.softmax(logits, dim=1)
            msp, _ = torch.max(probs, dim=1)
            raw_metrics['msp'].append(msp.cpu().numpy())

            # 归一化熵
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            num_coarse_cls = probs.shape[1]
            raw_metrics['entropy'].append((ent / np.log(max(num_coarse_cls, 2))).cpu().numpy())

    for k in res:
        res[k] = np.concatenate(res[k])
    for k in raw_metrics:
        raw_metrics[k] = np.concatenate(raw_metrics[k])

    print(f"Total Background samples in results: {np.sum(res['c_true'] == -1)}")
    res.update(raw_metrics)
    return res


# =========================================================================
# OpenMax 校准器
# =========================================================================
class OpenMaxCalibrator:
    """OpenMax 开集识别校准器 (基于 Bendale & Boult, CVPR 2016)"""
    def __init__(self, tail_size=20):
        self.tail_size = tail_size
        self.mavs = {}
        self.weibull_params = {}
        self.num_classes = 0
        self.activation_dim = None

    def fit(self, class_activations, num_classes):
        self.num_classes = num_classes
        print("\n⚙️  OpenMax Calibration (MAV + Weibull)...")

        for cls_id in range(num_classes):
            acts = class_activations.get(cls_id, None)
            if acts is None or len(acts) == 0:
                print(f"  ⚠️  Class {cls_id}: no activations, using zero MAV")
                if self.activation_dim is not None:
                    self.mavs[cls_id] = np.zeros(self.activation_dim)
                continue

            if self.activation_dim is None:
                self.activation_dim = acts.shape[1]

            self.mavs[cls_id] = np.mean(acts, axis=0)
            dists = np.linalg.norm(acts - self.mavs[cls_id], axis=1)

            if len(dists) >= self.tail_size:
                tail = np.sort(dists)[-self.tail_size:]
                if np.std(tail) < 1e-9:
                    tail = tail + np.random.normal(0, 1e-7, len(tail))
                    print(f"  ⚠️  Class {cls_id}: low variance, added noise")
                self.weibull_params[cls_id] = weibull_max.fit(tail + 1e-6)
                c, loc, scale = self.weibull_params[cls_id]
                print(f"  ✅  Class {cls_id}: samples={len(acts)}, "
                      f"Weibull(c={c:.4f}, loc={loc:.4f}, scale={scale:.4f})")
            else:
                print(f"  ⚠️  Class {cls_id}: insufficient samples ({len(dists)} < {self.tail_size})")

    def predict_openmax(self, z_raw, logits):
        num_classes = self.num_classes

        distances = np.zeros(num_classes)
        for cls_id in range(num_classes):
            if cls_id in self.mavs:
                distances[cls_id] = np.linalg.norm(z_raw - self.mavs[cls_id])
            else:
                distances[cls_id] = np.inf

        stable_logits = logits - np.max(logits)
        exp_logits = np.exp(stable_logits)
        scores = exp_logits / exp_logits.sum()

        w = np.zeros(num_classes)
        for cls_id in range(num_classes):
            if cls_id in self.weibull_params:
                params = self.weibull_params[cls_id]
                w[cls_id] = np.clip(weibull_max.cdf(distances[cls_id], *params), 0, 1)
            else:
                w[cls_id] = 0.5

        s_hat = np.zeros(num_classes + 1)
        for i in range(num_classes):
            s_hat[i] = scores[i] * (1.0 - w[i])
        s_hat[num_classes] = np.sum(scores * w)

        # s_hat = s_hat - np.max(s_hat)
        # s_hat = np.exp(s_hat) / np.exp(s_hat).sum()

        return s_hat, distances, w


# =========================================================================
# 辅助函数
# =========================================================================
def find_class_specific_thresholds(fusion_scores, c_true, c_pred, target_recall=0.95):
    class_thresholds = {}
    unique_classes = np.unique(c_true[c_true != -1])
    special_targets = {3: 0.81}

    for cls_id in unique_classes:
        if cls_id == -1: continue
        current_recall = special_targets.get(cls_id, target_recall)
        mask = (c_true == cls_id) & (c_pred == cls_id)
        if mask.sum() > 10:
            scores = np.sort(fusion_scores[mask])
            idx = int(len(scores) * current_recall)
            class_thresholds[cls_id] = scores[min(idx, len(scores) - 1)]
        else:
            class_thresholds[cls_id] = np.percentile(fusion_scores[c_true != -1], target_recall * 100)
    return class_thresholds


def diagnose_osr_scores(res, cfg):
    print("\n" + "=" * 50)
    print("OSR Diagnosis (Known vs Unknown)")
    print("=" * 50)

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
            k_val = np.mean(data[known_mask])
            u_val = np.mean(data[unknown_mask])
            diff = ((u_val - k_val) / (k_val + 1e-9)) * 100
            print(f"{name:<25} | {k_val:<12.6f} | {u_val:<12.6f} | {diff:>+8.2f}%")
        else:
            print(f"{name:<25} | N/A")

    print("=" * 50 + "\n")


# =========================================================================
# 绘图
# =========================================================================
def plot_results(res, cfg, threshold, coarse_map, closed_oa, open_oa):
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.acc_dir, exist_ok=True)

    known_mask = res['c_true'] != -1
    snr_vals = sorted(np.unique(res['snr'][known_mask]))
    num_snr = len(snr_vals)
    num_coarse = len(coarse_map)
    coarse_names = [COARSE_LABEL_NAMES[i] for i in range(num_coarse)]

    # OSCR 曲线
    x1 = res['final_score'][known_mask]
    x2 = res['final_score'][~known_mask]
    pred = res.get('c_pred_original', res['c_pred'])[known_mask]
    labels = res['c_true'][known_mask]
    correct = (pred == labels)

    scores_all = np.concatenate([x1, x2])
    gt_correct = np.concatenate([correct.astype(int), np.zeros(len(x2))])
    gt_unknown = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

    indices = np.argsort(scores_all)
    gt_correct = gt_correct[indices]
    gt_unknown = gt_unknown[indices]

    ccr = np.cumsum(gt_correct) / len(x1)
    fpr = np.cumsum(gt_unknown) / len(x2)
    ccr = np.concatenate([[0], ccr])
    fpr = np.concatenate([[0], fpr])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, ccr, label=f'OSCR (AUC={np.trapezoid(ccr, fpr):.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("FPR（未知类被接受）", fontproperties=font_cn)
    plt.ylabel("CCR (已知类正确识别率）", fontproperties=font_cn)
    plt.title('OSCR 曲线', fontproperties=font_cn)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.results_dir, "oscr_curve.png"), dpi=300); plt.close(); gc.collect()

    # SNR ROC
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
    plt.title('Detection ROC Curves at Different SNRs'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.results_dir, "snr_roc_curves.png"), dpi=300); plt.close(); gc.collect()

    # Coarse 准确率矩阵 (使用 OSR 前的原始预测)
    coarse_pred = res.get('c_pred_original', res['c_pred'])
    coarse_acc_matrix = np.zeros((num_coarse, num_snr))
    for j, s in enumerate(snr_vals):
        mask_s = (res['snr'] == s) & known_mask
        for i in range(num_coarse):
            mask = mask_s & (res['c_true'] == i)
            coarse_acc_matrix[i, j] = np.mean(coarse_pred[mask] == i) if mask.sum() > 0 else np.nan

    df_coarse = pd.DataFrame(coarse_acc_matrix, index=coarse_names, columns=[f"{s}dB" for s in snr_vals])
    df_coarse['AvgAcc'] = df_coarse.mean(axis=1)
    df_coarse.to_csv(os.path.join(cfg.acc_dir, "coarse_accuracy_data.csv"), float_format="%.4f")

    # 雷达图
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, num_snr, endpoint=False).tolist()
    angles += angles[:1]
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
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.acc_dir, "coarse_accuracy_radar.png"), dpi=300)
    plt.close(); gc.collect()

    # 线性图
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

    # OSR 分布
    plt.figure(figsize=(10, 6))
    sns.histplot(res['final_score'][known_mask], label="Known Codes", color="blue", kde=False, stat="density", alpha=0.4)
    sns.histplot(res['final_score'][~known_mask], label="Unknown Codes", color="orange", kde=False, stat="density", alpha=0.4)
    avg_thr = np.mean(list(threshold.values()))
    plt.axvline(avg_thr, color='red', linestyle='--', linewidth=2, label=f'Avg Threshold: {avg_thr:.6f}')
    plt.title("OSR OpenMax Decision Score Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("OpenMax Unknown Probability (0=Known, 1=Unknown)"); plt.ylabel("Density"); plt.legend()
    plt.savefig(os.path.join(cfg.results_dir, "osr_score_distribution.png"), dpi=300)
    plt.close(); gc.collect()

    # # T-SNE (注释掉，t-SNE 在高维数据上非常慢)
    # print("Computing T-SNE for visualization...")
    # tsne = TSNE(n_components=2, random_state=42)
    # max_pts = 3000
    # indices = np.random.choice(len(res['embs']), min(max_pts, len(res['embs'])), replace=False)
    # embs_norm = res['embs'][indices] / (np.linalg.norm(res['embs'][indices], axis=1, keepdims=True) + 1e-9)
    # low_dim = tsne.fit_transform(embs_norm)
    # subset_labels = res['c_true'][indices]
    #
    # plt.figure(figsize=(10, 8))
    # for i in range(num_coarse):
    #     m = subset_labels == i
    #     name = COARSE_LABEL_NAMES[i]
    #     plt.scatter(low_dim[m, 0], low_dim[m, 1], label=name, s=25, alpha=0.7)
    #
    # m_un = subset_labels == -1
    # if m_un.any():
    #     plt.scatter(low_dim[m_un, 0], low_dim[m_un, 1], c='black', marker='x', label='Unknown', s=35, alpha=0.5)
    #
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    # plt.savefig(os.path.join(cfg.results_dir, "tsne_visualization.png"), dpi=300)
    # plt.close(); gc.collect()


def plot_all_confusion_matrices(res, cfg, threshold, coarse_names):
    cm_dir = os.path.join(cfg.results_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # 使用传入的 OSR 预测结果（已在 main() 中判定），不再额外做阈值二次过滤
    open_pred = res['c_pred']

    # OSR 混淆矩阵
    osr_labels = list(range(len(coarse_names))) + [-1]
    osr_display_names = coarse_names + ["Unknown"]
    cm_osr = confusion_matrix(res['c_true'], open_pred, labels=osr_labels)
    cm_osr_perc = cm_osr.astype('float') / cm_osr.sum(axis=1)[:, np.newaxis]
    cm_osr_perc = np.nan_to_num(cm_osr_perc)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_osr_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=osr_display_names, yticklabels=osr_display_names)
    plt.title("OSR 混淆矩阵", fontproperties=font_cn)
    plt.xlabel("预测标签", fontproperties=font_cn); plt.ylabel("真实标签", fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, "osr_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    pd.DataFrame(cm_osr, index=osr_display_names, columns=osr_display_names).to_csv(
        os.path.join(cm_dir, "osr_confusion_matrix.csv"))

    # 闭集混淆矩阵 (使用 OSR 前的原始预测)
    known_mask = res['c_true'] != -1
    closed_pred = res.get('c_pred_original', res['c_pred'])
    cm_closed = confusion_matrix(res['c_true'][known_mask], closed_pred[known_mask],
                                 labels=list(range(len(coarse_names))))
    cm_closed_perc = cm_closed.astype('float') / cm_closed.sum(axis=1)[:, np.newaxis]
    cm_closed_perc = np.nan_to_num(cm_closed_perc)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_closed_perc, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=coarse_names, yticklabels=coarse_names)
    plt.title("闭集混淆矩阵", fontproperties=font_cn)
    plt.xlabel("预测标签", fontproperties=font_cn); plt.ylabel("真实标签", fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, "closed_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All confusion matrices saved to: {cm_dir}")


def plot_evt_distribution(test_res, threshold, cfg):
    plt.figure(figsize=(10, 6))
    known_mask = test_res['c_true'] != -1
    sns.histplot(test_res['final_score'][known_mask], label="Known Classes",
                 color="royalblue", kde=False, stat="density", alpha=0.4, bins=50)
    if (~known_mask).any():
        sns.histplot(test_res['final_score'][~known_mask], label="Unknown Classes",
                     color="crimson", kde=False, stat="density", alpha=0.4, bins=50)

    colors = ['r', 'g', 'b', 'c', 'm']
    for idx, (cls_idx, thr) in enumerate(threshold.items()):
        plt.axvline(thr, color=colors[idx % len(colors)], linestyle=':', alpha=0.7,
                    label=f'Class {cls_idx} Thr: {thr:.3f}')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title("OpenMax Unknown Probability Distribution\n(0=Likely Known, 1=Likely Unknown)", fontsize=13, fontweight='bold')
    plt.xlabel("OpenMax Unknown Probability"); plt.ylabel("Density"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.results_dir, "openmax_probability_distribution.png"), dpi=300)
    plt.close()


# =========================================================================
# 主程序
# =========================================================================
def main():
    import collections
    cfg = Config()

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.acc_dir, exist_ok=True)

    # --- 0. 模型初始化 ---
    model = ResNetOSRNet(cfg).to(cfg.device)
    if os.path.exists(cfg.model_save_path):
        checkpoint = torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model: {cfg.model_save_path}")
    else:
        print(f"Error: Model not found at {cfg.model_save_path}")
        return

    dummy_in = torch.randn(1, 1, cfg.seq_len).to(cfg.device)
    flops, params = profile(model, inputs=(dummy_in,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model: Params={params}, FLOPs={flops}")

    num_coarse = len(COARSE_LABEL_NAMES)

    # =========================================================================
    # PHASE 1: 训练集 - 计算 MAV + Weibull 拟合
    # =========================================================================
    train_pkl_path = cfg.eval_data_path.replace('.pkl', '_train.pkl')
    print(f"\n[PHASE 1a] Computing OpenMax MAV on training set...")
    if not os.path.exists(train_pkl_path):
        print(f"Error: Training file not found at {train_pkl_path}")
        return

    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)
    print(f"Train label distribution: {collections.Counter(train_data['coarse_labels'])}")

    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_data['features']).float(),
        torch.tensor(train_data['coarse_labels']).long(),
        torch.tensor(train_data['snrs']).float(),
    ), batch_size=cfg.batch_size, shuffle=False)

    train_res = get_AV(model, train_loader, cfg.device)

    class_activations = {}
    for i in range(num_coarse):
        mask = train_res['c_true'] == i
        class_activations[i] = train_res['z_raw_list'][mask]
        print(f"  Class {i} ({COARSE_LABEL_NAMES[i]}): {mask.sum()} samples for MAV")

    calibrator = OpenMaxCalibrator(tail_size=20)
    calibrator.fit(class_activations, num_coarse)

    # 训练集 OpenMax 概率
    train_openmax_unknown = []
    for i in range(len(train_res['c_true'])):
        z_raw_i = train_res['z_raw_list'][i]
        logits_i = train_res['logits'][i]
        probs, _, _ = calibrator.predict_openmax(z_raw_i, logits_i)
        train_openmax_unknown.append(probs[-1])
    train_openmax_unknown = np.array(train_openmax_unknown)

    class_thresholds = {}
    for cls_id in range(num_coarse):
        mask = train_res['c_true'] == cls_id
        if mask.sum() > 0:
            class_thresholds[cls_id] = np.percentile(train_openmax_unknown[mask], 95)
        else:
            class_thresholds[cls_id] = np.percentile(train_openmax_unknown, 95)
    avg_threshold = np.mean(list(class_thresholds.values()))
    print(f"Train reference thresholds: {class_thresholds}")
    print(f"Avg threshold: {avg_threshold:.4f}")

    # =========================================================================
    # PHASE 2: 测试集 OpenMax 评估
    # =========================================================================
    test_pkl_path = cfg.eval_data_path.replace('.pkl', '_test.pkl')
    print(f"\n[PHASE 2] Running OpenMax evaluation on test set...")

    if not os.path.exists(test_pkl_path):
        print(f"Error: Test file not found at {test_pkl_path}")
        return

    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(test_data['features']).float(),
        torch.tensor(test_data['coarse_labels']).long(),
        torch.tensor(test_data['snrs']).float(),
    ), batch_size=cfg.batch_size, shuffle=False)

    test_res = get_AV(model, test_loader, cfg.device)

    known_mask_test = (test_res['c_true'] != -1)
    unknown_mask_test = (test_res['c_true'] == -1)

    # 应用 OpenMax
    openmax_unknown_probs_test = []
    openmax_preds_test = []

    for i in range(len(test_res['c_true'])):
        z_raw_i = test_res['z_raw_list'][i]
        logits_i = test_res['logits'][i]
        probs, _, _ = calibrator.predict_openmax(z_raw_i, logits_i)
        openmax_unknown_probs_test.append(probs[-1])
        openmax_preds_test.append(np.argmax(probs))

    test_res['final_score'] = np.array(openmax_unknown_probs_test)
    openmax_preds_test = np.array(openmax_preds_test)

    known_scores = test_res['final_score'][known_mask_test]
    unknown_scores = test_res['final_score'][unknown_mask_test]
    print(f"OpenMax Unknown Prob - Known percentiles [50,90,95,99]: {np.percentile(known_scores, [50, 90, 95, 99])}")
    print(f"OpenMax Unknown Prob - Unknown percentiles [50,90,95,99]: {np.percentile(unknown_scores, [50, 90, 95, 99])}")

    open_pred = np.where(openmax_preds_test == num_coarse, -1, openmax_preds_test)

    known_mask = test_res['c_true'] != -1
    closed_oa = accuracy_score(test_res['c_true'][known_mask], test_res['c_pred'][known_mask])
    open_oa = accuracy_score(test_res['c_true'], open_pred)

    true_labels = test_res['c_true'][known_mask]
    pred_labels = open_pred[known_mask]
    per_class_recall = recall_score(true_labels, pred_labels, labels=list(range(num_coarse)), average=None)
    macro_recall = np.mean(per_class_recall)

    print(f"\n[Recall Report]")
    print(f"   - Macro Recall: {macro_recall*100:.2f}%")
    for i in range(len(per_class_recall)):
        print(f"   - Class {i} ({COARSE_LABEL_NAMES[i]}): {per_class_recall[i]*100:.2f}%")

    print(f"\n[OpenMax Results]")
    print(f"   - Closed-set Accuracy: {closed_oa*100:.2f}%")
    print(f"   - Open-set Accuracy: {open_oa*100:.2f}%")

    # Polar 诊断
    polar_idx = 3
    polar_fp_mask = (test_res['c_true'] == -1) & (open_pred == polar_idx)
    known_polar_mask = (test_res['c_true'] == polar_idx)

    if polar_fp_mask.any():
        print(f"\n[Polar Diagnosis]")
        print(f"   Unknown misclassified as Polar: {polar_fp_mask.sum()}")
        print(f"   Unknown -> Polar OpenMax prob mean: {np.mean(test_res['final_score'][polar_fp_mask]):.4f}")
        print(f"   True Polar OpenMax prob mean: {np.mean(test_res['final_score'][known_polar_mask]):.4f}")
        print(f"   True Polar OpenMax prob 95th: {np.percentile(test_res['final_score'][known_polar_mask], 95):.4f}")

    # Polar 分布图
    plt.figure(figsize=(8, 5))
    plt.hist(test_res['final_score'][known_polar_mask], bins=100, alpha=0.5,
             label='True Polar', color='blue', density=True)
    if polar_fp_mask.any():
        plt.hist(test_res['final_score'][polar_fp_mask], bins=100, alpha=0.5,
                 label='Misclassified Unknown', color='red', density=True)
    plt.axvline(class_thresholds.get(polar_idx, avg_threshold), color='k', linestyle='--',
                label=f'Polar Ref Thr: {class_thresholds.get(polar_idx, avg_threshold):.4f}')
    plt.yscale('log')
    plt.title("Polar OpenMax Diagnosis: Unknown Probability (Log Scale)")
    plt.xlabel("OpenMax Unknown Probability"); plt.ylabel("Density (Log)"); plt.legend()
    plt.savefig(os.path.join(cfg.results_dir, 'polar_openmax_diagnosis.png'), dpi=300)
    plt.close()

    test_res['c_pred_original'] = test_res['c_pred'].copy()
    test_res['c_pred'] = open_pred
    diagnose_osr_scores(test_res, cfg)

    # --- 绘图 ---
    plot_all_confusion_matrices(test_res, cfg, class_thresholds, COARSE_LABEL_NAMES)
    plot_evt_distribution(test_res, class_thresholds, cfg)
    plot_results(test_res, cfg, class_thresholds, test_data['coarse_map'], closed_oa, open_oa)

    print(f"\nOpenMax evaluation complete! Results saved to: {cfg.results_dir}")


if __name__ == "__main__":
    main()
