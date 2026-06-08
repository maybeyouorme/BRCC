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
# 导入模型
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from open_recognition.models import MultiTaskOSRNet
from open_recognition.config import Config
import gc
from matplotlib import font_manager
font_cn = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    size=12
)

COARSE_LABEL_NAMES = ["Conv", "LDPC", "Turbo", "Polar", "BCH"]


def get_predictions(model, loader, device):
    model.eval()
    res = {
        'embs': [], 'c_true': [], 'snr': [],
        'c_pred': [], 'max_sim': []
    }
    raw_metrics = {
        'dist': [],
        'recon': [],
        'msp': [],
        'entropy': []
    }

    with torch.no_grad():
        for x, y_c, snr in tqdm(loader, desc="Inference"):
            x_dev = x.to(device)
            input_data = x_dev.unsqueeze(1) if x_dev.dim() == 2 else x_dev  # [B, 1, SeqLen]
            output = model(input_data)

            c_logits = output['coarse_logits']

            res['c_true'].append(y_c.numpy())
            res['snr'].append(snr.numpy())
            res['c_pred'].append(c_logits.argmax(1).cpu().numpy())
            res['embs'].append(output['embedding'].cpu().numpy())

            # MSP (Maximum Softmax Probability)
            probs = torch.softmax(c_logits / 32.0, dim=1)
            msp, _ = torch.max(probs, dim=1)
            raw_metrics['msp'].append(msp.cpu().numpy())

            # 归一化熵
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            raw_metrics['entropy'].append((ent / np.log(5)).cpu().numpy())

            # 重构误差
            recon = output['reconstruction']
            target = output['target_signal']
            if target.dim() == 3: target = target.squeeze(1)
            if recon.dim() == 3: recon = recon.squeeze(1)

            cos_sim = F.cosine_similarity(recon, target, dim=1)
            r_err = torch.abs(torch.mean(torch.abs(recon - target), dim=1) / (cos_sim + 1e-8))
            raw_metrics['recon'].append(r_err.cpu().numpy())

            # 余弦相似度
            z = F.normalize(output['embedding'], p=2, dim=1)
            prototypes = F.normalize(model.coarse_head.weight, p=2, dim=1)
            sim_matrix = torch.matmul(z, prototypes.t())
            max_sim, _ = torch.max(sim_matrix, dim=1)
            res['max_sim'].append(max_sim.cpu().numpy())
            raw_metrics['dist'].append((1.0 - max_sim).cpu().numpy())

    for k in res:
        res[k] = np.concatenate(res[k])
    for k in raw_metrics:
        raw_metrics[k] = np.concatenate(raw_metrics[k])

    print(f"Total Background samples in results: {np.sum(res['c_true'] == -1)}")
    res.update(raw_metrics)
    res['re_score_raw'] = res['recon']
    return res


# =========================================================================
# 绘图函数
# =========================================================================

def plot_results(res, results_dir, acc_dir, threshold, coarse_map, closed_oa, open_oa):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)

    known_mask = res['c_true'] != -1
    snr_vals = sorted(np.unique(res['snr'][known_mask]))
    num_snr = len(snr_vals)
    num_coarse = len(coarse_map)
    coarse_names = [COARSE_LABEL_NAMES[i] for i in range(num_coarse)]

    # ========== OSCR 曲线 ==========
    x1 = res['final_score'][known_mask]
    x2 = res['final_score'][~known_mask]
    pred = res['c_pred'][known_mask]
    labels = res['c_true'][known_mask]
    correct = (pred == labels)

    scores_all = np.concatenate([x1, x2])
    gt_correct = np.concatenate([correct.astype(int), np.zeros(len(x2))])
    gt_unknown = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

    indices = np.argsort(scores_all)
    gt_correct = gt_correct[indices]
    gt_unknown = gt_unknown[indices]

    num_k = len(x1)
    num_u = len(x2)

    ccr = np.cumsum(gt_correct) / num_k
    fpr_os = np.cumsum(gt_unknown) / num_u

    ccr = np.concatenate([[0], ccr])
    fpr_os = np.concatenate([[0], fpr_os])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_os, ccr, label=f'OSCR (AUC={np.trapz(ccr, fpr_os):.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("FPR（未知类被接受）", fontproperties=font_cn)
    plt.ylabel("CCR (已知类正确识别率）", fontproperties=font_cn)
    plt.title('OSCR 曲线', fontproperties=font_cn)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "oscr_curve.png"), dpi=300); plt.close(); gc.collect()

    # ========== 分 SNR 的 ROC 曲线 ==========
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
    plt.savefig(os.path.join(results_dir, "snr_roc_curves.png"), dpi=300); plt.close(); gc.collect()

    # ========== Coarse 级准确率矩阵 ==========
    coarse_acc_matrix = np.zeros((num_coarse, num_snr))
    for j, s in enumerate(snr_vals):
        mask_s = (res['snr'] == s) & known_mask
        for i in range(num_coarse):
            mask = mask_s & (res['c_true'] == i)
            coarse_acc_matrix[i, j] = np.mean(res['c_pred'][mask] == i) if mask.sum() > 0 else np.nan

    coarse_names = [COARSE_LABEL_NAMES[i] for i in range(num_coarse)]
    df_coarse = pd.DataFrame(coarse_acc_matrix, index=coarse_names, columns=[f"{s}dB" for s in snr_vals])
    df_coarse['AvgAcc'] = df_coarse.mean(axis=1)
    coarse_csv_path = os.path.join(acc_dir, "coarse_accuracy_data.csv")
    df_coarse.to_csv(coarse_csv_path, float_format="%.4f")
    print(f"Coarse accuracy CSV saved to: {coarse_csv_path}")

    # --- 雷达图 ---
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2*np.pi, num_snr, endpoint=False).tolist()
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
    ax.legend(loc='upper right',bbox_to_anchor=(1.3, 1.1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(acc_dir, "coarse_accuracy_radar.png"), dpi=300)
    plt.close(); gc.collect()

    # --- 线性图 ---
    plt.figure(figsize=(10, 6))
    for i in range(num_coarse):
        avg_acc = np.nanmean(coarse_acc_matrix[i]) * 100
        name = COARSE_LABEL_NAMES[i]
        plt.plot(snr_vals, coarse_acc_matrix[i], marker='o', linewidth=2, label=f"{name} (Avg:{avg_acc:.1f}%)")
    plt.title(f"Coarse-level Recognition Accuracy vs SNR\n(Open OA: {open_oa*100:.2f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(os.path.join(acc_dir, "coarse_accuracy_linear.png"), dpi=300)
    plt.close(); gc.collect()

    # ========== OSR 分数分布图 ==========
    plt.figure(figsize=(10, 6))
    sns.histplot(res['final_score'][known_mask], label="Known Codes", color="blue", kde=True, stat="density", alpha=0.4)
    sns.histplot(res['final_score'][~known_mask], label="Unknown Codes", color="orange", kde=True, stat="density", alpha=0.4)
    avg_thr = np.mean(list(threshold.values()))
    min_thr = min(threshold.values())
    max_thr = max(threshold.values())
    plt.axvline(avg_thr, color='red', linestyle='--', linewidth=2, label=f'Threshold: {avg_thr:.4f}')
    plt.axvspan(min_thr, max_thr, color='red', alpha=0.1, label='Threshold Range')
    plt.title("OSR Decision Score Distribution (final_score = 1 - MSP)", fontsize=13, fontweight='bold')
    plt.xlabel("Score (0=Known, 1=Unknown)"); plt.ylabel("Density"); plt.legend()
    plt.savefig(os.path.join(results_dir, "osr_score_distribution.png"), dpi=300)
    plt.close(); gc.collect()

    # ========== T-SNE ==========
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

    m_un = subset_labels == -1
    if m_un.any():
        plt.scatter(low_dim[m_un, 0], low_dim[m_un, 1], c='black', marker='x', label='Unknown', s=35, alpha=0.5)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tsne_visualization.png"), dpi=300)
    plt.close(); gc.collect()


def plot_all_confusion_matrices(res, results_dir, threshold, coarse_names):
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    open_pred = res['c_pred'].copy()
    for i in range(len(open_pred)):
        assigned_cls = open_pred[i]
        thr = threshold.get(assigned_cls, np.mean(list(threshold.values())))
        if res['final_score'][i] > thr:
            open_pred[i] = -1

    # --- 开集混淆矩阵 (6x6) ---
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

    # --- 闭集混淆矩阵 (5x5) ---
    known_mask = res['c_true'] != -1
    true_labels_closed = res['c_true'][known_mask]
    pred_labels_closed = res['c_pred'][known_mask]
    closed_labels = list(range(len(coarse_names)))
    closed_display_names = coarse_names
    cm_closed = confusion_matrix(true_labels_closed, pred_labels_closed, labels=closed_labels)
    cm_closed_perc = cm_closed.astype('float') / cm_closed.sum(axis=1)[:, np.newaxis]
    cm_closed_perc = np.nan_to_num(cm_closed_perc)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_closed_perc, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=closed_display_names, yticklabels=closed_display_names)
    plt.title("闭集混淆矩阵", fontproperties=font_cn)
    plt.xlabel("预测标签", fontproperties=font_cn); plt.ylabel("真实标签", fontproperties=font_cn)
    plt.savefig(os.path.join(cm_dir, "closed_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All confusion matrices saved to: {cm_dir}")


# =========================================================================
# 主程序
# =========================================================================

def main(model_path, val_data_path, test_data_path, save_dir, device='cuda', batch_size=64, msp_threshold=0.8,cfg=None):
    """
    基于 MSP (Maximum Softmax Probability) 的开集识别评估

    参数:
        model_path:     模型权重文件路径 (.pth)
        val_data_path:  验证集 pickle 文件路径 (_val.pkl)
        test_data_path: 测试集 pickle 文件路径 (_test.pkl)
        save_dir:       结果保存目录
        device:         计算设备 ('cuda' 或 'cpu')
        batch_size:     推理批次大小
        msp_threshold:  MSP 阈值，最大 softmax 概率低于此值时判为未知类
    """
    results_dir = os.path.join(save_dir, "results")
    acc_dir = os.path.join(save_dir, "accuracy_tables")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)

    # --- 0. 加载模型 ---
    device = torch.device(device)
    model = MultiTaskOSRNet(cfg)  # 默认构造，权重通过 load_state_dict 加载
    model = model.to(device)

    if not os.path.exists(model_path):
        print(f"❌ 错误: 未找到模型文件 {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    # 尝试加载 centers（如果存在）
    centers = None
    if 'center_loss' in checkpoint:
        centers = checkpoint['center_loss']['centers'].to(device)
        print("✅ 成功加载中心点")
    elif hasattr(model, 'center_loss_fn'):
        centers = model.center_loss_fn.centers.detach()
        print("✅ 成功从模型中提取 Center Loss 聚类中心")
    print(f"✅ 成功加载模型权重: {model_path}")

    # 打印模型复杂度
    dummy_in = torch.randn(1, 1, 8192).to(device)
    flops, params = profile(model, inputs=(dummy_in,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"📊 模型统计: 参数量={params}, 计算量={flops}")

    # =========================================================================
    # PHASE 1: 验证集 - 确认 MSP 阈值
    # =========================================================================
    print(f"\n🚀 [PHASE 1] 验证集 MSP 阈值检查...")
    if not os.path.exists(val_data_path):
        print(f"❌ 错误: 未找到验证集文件 {val_data_path}")
    else:
        with open(val_data_path, 'rb') as f:
            val_data = pickle.load(f)
        import collections
        print(f"DEBUG: 标签分布: {collections.Counter(val_data['coarse_labels'])}")
        val_loader = DataLoader(TensorDataset(
            torch.tensor(val_data['features']).float(),
            torch.tensor(val_data['coarse_labels']).long(),
            torch.tensor(val_data['snrs']).float()
        ), batch_size=batch_size, shuffle=True)

        val_res = get_predictions(model, val_loader, device)
        set1 = set(val_res['msp'])
        #print(f"所有概率取值：{set1}")
        known_mask_v = val_res['c_true'] != -1
        if known_mask_v.any():
            print(f"🔍 [验证集] 已知类 MSP 均值: {val_res['msp'][known_mask_v].mean():.4f}")

    # 设定阈值
    class_thresholds = {i: 1.0 - msp_threshold for i in range(len(COARSE_LABEL_NAMES))}
    avg_threshold = 1.0 - msp_threshold
    print(f"🎯 MSP 阈值已锁定: p_max < {msp_threshold} 判为未知类")
    print(f"   (等价 final_score = 1 - MSP > {avg_threshold:.4f})")

    # =========================================================================
    # PHASE 2: 测试集评估
    # =========================================================================
    print(f"\n🚀 [PHASE 2] 正在测试集上执行 MSP 开集识别评估 ...")

    if not os.path.exists(test_data_path):
        print(f"❌ 错误: 未找到测试集文件 {test_data_path}")
        return

    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(test_data['features']).float(),
        torch.tensor(test_data['coarse_labels']).long(),
        torch.tensor(test_data['snrs']).float()
    ), batch_size=batch_size, shuffle=False)

    test_res = get_predictions(model, test_loader, device)

    known_mask_test = (test_res['c_true'] != -1)
    unknown_mask_test = (test_res['c_true'] == -1)

    # 使用 MSP 计算 final_score：1 - MSP，越大越不确定
    test_res['final_score'] = 1.0 - test_res['msp']

    # --- 打印 MSP 详细分布 ---
    print("\n📊 [MSP 分布诊断]")
    for name, mask in [("已知类", known_mask_test), ("未知类", unknown_mask_test)]:
        if mask.any():
            msp_vals = test_res['msp'][mask]
            print(f"   {name}: mean={msp_vals.mean():.4f}, "
                  f"min={msp_vals.min():.4f}, "
                  f"med={np.median(msp_vals):.4f}, "
                  f"p5={np.percentile(msp_vals, 5):.4f}")
            for thr in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.32]:
                pct = (msp_vals < thr).mean() * 100
                print(f"       MSP < {thr}: {pct:.1f}%")
    print(f"\n📊 [final_score = 1-MSP 分布]")
    for name, mask in [("已知类", known_mask_test), ("未知类", unknown_mask_test)]:
        if mask.any():
            fs = test_res['final_score'][mask]
            print(f"   {name}: mean={fs.mean():.4f}, min={fs.min():.4f}, max={fs.max():.4f}, "
                  f"p95={np.percentile(fs, 95):.4f}")
    print(f"   [阈值对比] avg_threshold(current)={avg_threshold:.4f} 等价于 MSP < {msp_threshold}")
    print(f"   [提示] 要使当前阈值生效，需要未知类的 MSP 低于 {msp_threshold}")

    # 应用判定：MSP < msp_threshold 判为未知类
    open_pred = test_res['c_pred'].copy()
    for i in range(len(open_pred)):
        if test_res['final_score'][i] > avg_threshold:
            open_pred[i] = -1

    # --- 计算指标 ---
    known_mask = test_res['c_true'] != -1
    closed_oa = accuracy_score(test_res['c_true'][known_mask], test_res['c_pred'][known_mask])
    open_oa = accuracy_score(test_res['c_true'], open_pred)

    # 召回率
    true_labels = test_res['c_true'][known_mask]
    pred_labels = open_pred[known_mask]
    class_indices = list(range(len(COARSE_LABEL_NAMES)))
    per_class_recall = recall_score(true_labels, pred_labels, labels=class_indices, average=None)
    macro_recall = np.mean(per_class_recall)
    print(f"\n📈 [真实召回率报告]")
    print(f"   - 总体平均召回率: {macro_recall*100:.2f}%")
    for i in range(len(per_class_recall)):
        print(f"   - 类别 {i} ({COARSE_LABEL_NAMES[i]}) 召回率: {per_class_recall[i]*100:.2f}%")

    print(f"\n📊 [MSP 开集识别评测结果报告]")
    print(f"   - MSP 判定阈值: p_max < {msp_threshold}")
    print(f"   - 闭集分类准确率 (Known): {closed_oa*100:.2f}%")
    print(f"   - 开集识别准确率 (Total): {open_oa*100:.2f}%")

    # --- 保存预测结果 ---
    test_res['c_pred_original'] = test_res['c_pred'].copy()
    test_res['c_pred'] = open_pred

    # --- 绘图 ---
    plot_all_confusion_matrices(test_res, results_dir, class_thresholds, COARSE_LABEL_NAMES)
    plot_results(test_res, results_dir, acc_dir, class_thresholds, test_data['coarse_map'], closed_oa, open_oa)

    print(f"\n✨ 评估完成！结果已保存至: {save_dir}")


if __name__ == "__main__":
    # ====== 用户在此配置路径 ======
    cfg = Config()
    main(
        model_path="../open_recognition/checkpoits1/best_model.pth",
        val_data_path="/data/project_lyb/Open data/val_data_for_eval_val.pkl",
        test_data_path="/data/project_lyb/Open data/val_data_for_eval_test.pkl",
        save_dir="./output",
        device="cuda",
        batch_size=64,
        msp_threshold=0.42,
        cfg=cfg
    )
