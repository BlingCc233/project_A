import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# 读取数据
k_values = []
phi_values = []

for i in range(5, 6):
    df_k = pd.read_csv(f'./raw_data/kz_{i}.csv')
    df_phi = pd.read_csv(f'./raw_data/Theta_{i}.csv')
    k_values.append(df_k.values.flatten())
    phi_values.append(df_phi.values.flatten())

k_values = np.array(k_values)
phi_values = np.array(phi_values)


# 定义最小二乘法函数来估计高程
def estimate_height_lstsq(k_values, phi_values):
    num_measurements, num_pixels = phi_values.shape
    heights = np.zeros(num_pixels)

    for i in range(num_pixels):
        A = k_values[:, i].reshape(-1, 1)  # 变成列向量
        B = phi_values[:, i]

        # 使用最小二乘法进行求解
        h, residuals, rank, s = lstsq(A, B)
        heights[i] = h[0]

    return heights


# 估计高程
heights = estimate_height_lstsq(k_values, phi_values)

# 重新形状为原始的图像大小
heights = heights.reshape(804, 2001)

# 绘制色度图
plt.imshow(heights, cmap='viridis')
plt.colorbar(label='Height (m)')
plt.title('Estimated Heights')
plt.show()

# 输出四个顶点的高程值
print("Top-left:", heights[0, 0])
print("Top-right:", heights[0, -1])
print("Bottom-left:", heights[-1, 0])
print("Bottom-right:", heights[-1, -1])
