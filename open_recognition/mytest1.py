# import h5py
# import numpy as np

# # 修改此处文件名即可读取不同文件
# filename = '../new_train/test_bch_15_11.mat'

# with h5py.File(filename, 'r') as f:
#     # 获取变量名
#     var_names = [k for k in f.keys()]
#     print(f'变量: {var_names}')
#     var = var_names[0]

#     # 形状 (8195, 13000), 行=特征, 列=样本
#     data = f[var][:]
#     print(f'形状: {data.shape}, dtype: {data.dtype}')
#     print()

#     # 转置为 (样本, 特征) 后打印前3行
#     data_t = data.T  # (13000, 8195)
#     print('前3行 (样本0~2), 只显示LLR特征列1~6 及 后3列标签:')
#     print('-' * 80)

#     # 列名
#     header = [f'LLR{i}' for i in range(1, 7)] + ['...', 'SNR', '粗标签', '细标签']
#     print(f'{"样本":>4s}' + ''.join(f'{h:>8s}' for h in header))

#     for i in range(3):
#         llr_preview = data_t[i, :6]       # 前6个LLR值
#         labels = data_t[i, -3:]           # SNR, 粗标签, 细标签
#         row = list(llr_preview) + [0] + list(labels)
#         print(f'{i:>4d}' + ''.join(f'{v:>8d}' for v in row))
import torch
import torch.nn.functional as F
a = torch.tensor([[3.0, 4.0]])
a_norm = F.normalize(a, p=2, dim=1)
print(a_norm)