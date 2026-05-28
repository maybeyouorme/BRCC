import torch
import time
import numpy as np

from models import MultiTaskOSRNet
from config import Config


def load_model(cfg, model_path):
    """
    加载训练好的模型
    """
    model = MultiTaskOSRNet(cfg).to(cfg.device)

    checkpoint = torch.load(model_path, map_location=cfg.device)

    # 读取模型参数
    model.load_state_dict(checkpoint['model'])

    model.eval()

    return model


def generate_dummy_signal(seq_len):
    """
    生成模拟信号 (用于测试推理时间)
    """
    signal = np.random.randn(seq_len)

    signal = torch.tensor(signal, dtype=torch.float32)

    signal = signal.unsqueeze(0)  # batch = 1

    return signal


def measure_inference_time(model, input_tensor, warmup=20, runs=100):
    """
    测试推理时间
    """

    input_tensor = input_tensor.to(next(model.parameters()).device)

    # GPU同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------
    # warmup (GPU预热)
    # -------------------------
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------
    # 正式测试
    # -------------------------
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()

    total_time = end - start

    avg_latency = total_time / runs

    fps = 1 / avg_latency

    return avg_latency, fps


def main():

    cfg = Config()

    print("Device:", cfg.device)

    model_path = cfg.model_save_path

    print("Loading model from:", model_path)

    model = load_model(cfg, model_path)

    # 构造测试输入
    input_signal = generate_dummy_signal(cfg.seq_len)

    # 测试推理速度
    latency, fps = measure_inference_time(model, input_signal)

    print("===================================")
    print("Average Inference Latency: %.4f ms" % (latency * 1000))
    print("FPS: %.2f" % fps)
    print("===================================")

if __name__ == "__main__":
    main()