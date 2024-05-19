---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.6
  nbformat: 4
  nbformat_minor: 5
---

::: {#3953b155b1da4613 .cell .markdown}
raw文件读入
:::

::: {#588ee8be30b8ce04 .cell .code execution_count="1" ExecuteTime="{\"end_time\":\"2024-05-19T08:39:21.830553Z\",\"start_time\":\"2024-05-19T08:39:16.879530Z\"}"}
``` python
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.ndimage import convolve


    
df_k = []
df_theta = []

for i in range(1,6):
    df_k.append(pd.read_csv(f'./raw_data/kz_{i}.csv', header=None))
    df_theta.append(pd.read_csv(f'./raw_data/Theta_{i}.csv', header=None ))
    


import re


for i in range(5):
    print(f'kz_{i+1}.csv shape: {df_k[i].shape}')
    print(f'Theta_{i+1}.csv shape: {df_theta[i].shape}')
    
    for j in range(df_k[i].shape[1]):
        k_j = df_k[i].iloc[0 ,j]
        k_j_str = str(k_j)
        pattern = r"-0\.\d+"
        matches = re.findall(pattern,k_j_str)
        df_k[i].loc[0, j] = float(matches[0])
        
    df_k[i] = pd.DataFrame(np.array(df_k[i]).astype(float))
      
        
num_rows = 805
num_cols = 2001  
        
    
# 未解缠的 n_i
n_i = 1

```

::: {.output .stream .stderr}
    /var/folders/5n/wnngr1j51d91yrx_z36vh6jr0000gn/T/ipykernel_66588/2037963179.py:16: DtypeWarning: Columns (257,317) have mixed types. Specify dtype option on import or set low_memory=False.
      df_k.append(pd.read_csv(f'./raw_data/kz_{i}.csv', header=None))
    /var/folders/5n/wnngr1j51d91yrx_z36vh6jr0000gn/T/ipykernel_66588/2037963179.py:16: DtypeWarning: Columns (668,890,1035,1050,1089,1152,1162,1170,1172,1318,1421,1859) have mixed types. Specify dtype option on import or set low_memory=False.
      df_k.append(pd.read_csv(f'./raw_data/kz_{i}.csv', header=None))
    /var/folders/5n/wnngr1j51d91yrx_z36vh6jr0000gn/T/ipykernel_66588/2037963179.py:16: DtypeWarning: Columns (406,599,828,1205,1287,1288,1640) have mixed types. Specify dtype option on import or set low_memory=False.
      df_k.append(pd.read_csv(f'./raw_data/kz_{i}.csv', header=None))
    /var/folders/5n/wnngr1j51d91yrx_z36vh6jr0000gn/T/ipykernel_66588/2037963179.py:16: DtypeWarning: Columns (517,781,1004,1024,1096,1130,1132,1156,1166,1179,1185,1201,1238,1247,1261,1284,1288,1297,1302,1341,1520,1522,1559) have mixed types. Specify dtype option on import or set low_memory=False.
      df_k.append(pd.read_csv(f'./raw_data/kz_{i}.csv', header=None))
    /var/folders/5n/wnngr1j51d91yrx_z36vh6jr0000gn/T/ipykernel_66588/2037963179.py:16: DtypeWarning: Columns (149,153,156,819) have mixed types. Specify dtype option on import or set low_memory=False.
      df_k.append(pd.read_csv(f'./raw_data/kz_{i}.csv', header=None))
:::

::: {.output .stream .stdout}
    kz_1.csv shape: (805, 2001)
    Theta_1.csv shape: (805, 2001)
    kz_2.csv shape: (805, 2001)
    Theta_2.csv shape: (805, 2001)
    kz_3.csv shape: (805, 2001)
    Theta_3.csv shape: (805, 2001)
    kz_4.csv shape: (805, 2001)
    Theta_4.csv shape: (805, 2001)
    kz_5.csv shape: (805, 2001)
    Theta_5.csv shape: (805, 2001)
:::
:::

::: {#3b2556e28dfb02cf .cell .markdown}
重写用于打表的plot函数
:::

::: {#909fb73e2be9df46 .cell .code execution_count="70" ExecuteTime="{\"end_time\":\"2024-05-19T09:27:23.367654Z\",\"start_time\":\"2024-05-19T09:27:23.332514Z\"}"}
``` python
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw3D(H):
    N, M = H.shape
    x = np.linspace(1, M, M)
    y = np.linspace(N, 1, N)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, H, cmap='viridis')

    ax.set_title('Estimated Height')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height(m)')

    # Rotate view by 180 degrees
    ax.view_init(elev=20, azim=230)
    # Set aspect ratio and scale
    ax.set_aspect('equal', 'box')
    ax.set_box_aspect([2001, 805, 500]) 

    plt.show()
    

def plot(*argv, titles=None):
  """
  plots a given number of phase maps
  """
  if len(argv) == 1:
    f, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
    if titles is not None:
      ax.set_title(titles)
    a = ax.imshow(argv[0].squeeze(), cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(a, cax=cax)
    plt.show()
  else:
    f, axes = plt.subplots(1, len(argv), sharey=True, figsize=(10, 10))
    for i in range(len(argv)):
        if titles is not None:
          axes[i].set_title(titles[i])
        a = axes[i].imshow(argv[i].squeeze(), cmap='jet')
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(a, cax=cax)
    plt.show()
    f.colorbar(a, cax=cax)

def plot_hist(*argv, titles):
  """
  plots the historgram of the input phase maps
  """
  for i in range(len(argv)):
      hist = np.histogram(argv[i].ravel(), bins=100)
      plt.plot(hist[1][1:], hist[0])
  plt.xlabel("phase values")
  plt.ylabel("frequency")
  plt.title("Histogram Analysis")
  plt.grid()
  if titles is not None:
    plt.legend(titles)
  plt.show()  
```
:::

::: {#7bc5037c1aed2333 .cell .code execution_count="3" ExecuteTime="{\"end_time\":\"2024-05-19T08:39:21.855037Z\",\"start_time\":\"2024-05-19T08:39:21.852385Z\"}"}
``` python
def raw_pic(index):
    
    # 遍历每个数据点，计算结果
    results = -1 * ( np.array(df_theta[index]) + 2 * np.pi * n_i ) / np.array(df_k[index])
            
    return results
```
:::

::: {#8b237124403d3137 .cell .markdown}
# 原始数据打表

> 先跑一遍原始数据，将n置为0，在不还原高程的情况下看一下相位纠缠导致的结果
:::

::: {#d7162a3b0b495119 .cell .markdown}
另外打一个关于 theta 和 k 的表，查看相位分布情况 与 K 和 raw图的关系
:::

::: {#466400000bc9f5e6 .cell .code execution_count="4" ExecuteTime="{\"end_time\":\"2024-05-19T08:39:24.757726Z\",\"start_time\":\"2024-05-19T08:39:21.857440Z\"}"}
``` python
for i in range(5):
    plot(raw_pic(i), df_theta[i], df_k[i], titles=["Height", "$\phi$", "$K_z$"])
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/c8753e9fed53973aa1521716c790979bc44988e1.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/2bc48d8045dd72903ad87a2c36aa2171e32303b6.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/9db074f36c009196989f097eea6b716214869ae3.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/4dbb6791b04b347ada40e852a1a8abcb07cd770c.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/52462eaf322ef7ab90b388b4ab24f8a4c3f3fff4.png)
:::
:::

::: {#2fbd04a96520da7c .cell .markdown}
发现相位图与raw图在形式上是那啥的关系
:::

::: {#c556d844e28b8fb1 .cell .markdown}
基于连续性假设（相位展开）的解缠算法：
:::

::: {#d53b00c889e40649 .cell .code execution_count="5" ExecuteTime="{\"end_time\":\"2024-05-19T08:39:24.770564Z\",\"start_time\":\"2024-05-19T08:39:24.763163Z\"}"}
``` python
def unwrap_phase(theta):
    # 初始化 phi 矩阵
    phi = np.zeros_like(theta)
    phi[0, 0] = theta[0, 0]
    
    # 展开行
    for i in range(1, theta.shape[0]):
        delta = theta[i, 0] - theta[i-1, 0]
        delta_wrapped = np.mod(delta + np.pi, 2 * np.pi) - np.pi
        phi[i, 0] = phi[i-1, 0] + delta_wrapped
    
    # 展开列
    for j in range(1, theta.shape[1]):
        delta = theta[0, j] - theta[0, j-1]
        delta_wrapped = np.mod(delta + np.pi, 2 * np.pi) - np.pi
        phi[0, j] = phi[0, j-1] + delta_wrapped
    
    # 展开其余矩阵
    for i in range(1, theta.shape[0]):
        for j in range(1, theta.shape[1]):
            delta_row = theta[i, j] - theta[i-1, j]
            delta_col = theta[i, j] - theta[i, j-1]
            delta_row_wrapped = np.mod(delta_row + np.pi, 2 * np.pi) - np.pi
            delta_col_wrapped = np.mod(delta_col + np.pi, 2 * np.pi) - np.pi
            phi[i, j] = phi[i-1, j] + delta_row_wrapped
            phi[i, j] = phi[i, j-1] + delta_col_wrapped
            
    return phi
```
:::

::: {#3442dc325e4c1d37 .cell .markdown}
对第二组数据（考虑到raw图纠缠最简洁）输出3D高程图和2D高程图
:::

::: {#b865a422d2838c2e .cell .code execution_count="63" ExecuteTime="{\"end_time\":\"2024-05-19T09:22:44.023841Z\",\"start_time\":\"2024-05-19T09:22:38.475469Z\"}"}
``` python
phi = unwrap_phase(np.array(df_theta[1]))

# while np.min(phi) < 0:
#     phi += 2 * np.pi
    
results = np.zeros((num_rows, num_cols))

results = -1 * phi / np.array(df_k[1])

plot(results, titles=["Height (m)"])
draw3D(results)
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/9cb8358c2e65d48469fb3fd2cd3f6cdfcae66eb5.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/2d0ec8001e2ecab617775f215d4d4dc56603cd31.png)
:::
:::

::: {#692d8a93dded792e .cell .markdown}
发现解缠效果优秀，但是最后高程范围来到了负数，对此整体迭代添加周期，直至下限为150
:::

::: {#e361f9c88ecaf9b .cell .code execution_count="67" ExecuteTime="{\"end_time\":\"2024-05-19T09:24:53.064199Z\",\"start_time\":\"2024-05-19T09:24:51.309098Z\"}"}
``` python
while np.min(results) < 150:
    results -= (2 * np.pi / df_k[1])
    
plot(results, raw_pic(1))

# draw3D(results)
H = results
N, M = H.shape
x = np.linspace(1, M, M)
y = np.linspace(N, 1, N)
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize=(12, 6))  # Set the figure size to be wide enough for two subplots
ax1 = fig.add_subplot(121, projection='3d')  # Change this to 121 for a 1x2 grid (first subplot)
ax1.plot_surface(x, y, H, cmap='viridis')
ax1.set_title('Estimated Height')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Height(m)')
# Rotate view by 180 degrees
ax1.view_init(elev=20, azim=230)
# Set aspect ratio and scale
ax1.set_aspect('equal', 'box')
ax1.set_box_aspect([2001, 805, 500])  

# Second plot
H = -1 * (np.array(df_theta[1])) / np.array(df_k[1])
N, M = H.shape
x = np.linspace(1, M, M)
y = np.linspace(1, N, N)
x, y = np.meshgrid(x, y)
ax2 = fig.add_subplot(122, projection='3d')  # Change this to 122 for a 1x2 grid (second subplot)
ax2.plot_surface(x, y, H, cmap='viridis')
ax2.set_title('Estimated Height')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Height(m)')
ax2.view_init(elev=20, azim=150)
# Set aspect ratio and scale
ax2.set_aspect('equal', 'box')
ax2.set_box_aspect([2001, 805, 500]) 

plt.show()
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/1877af36d0099e431d699d9802bed0fc53cd130f.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/c86fe7fbd3c55e19af044df143d177ffa0099f45.png)
:::
:::

::: {#aaa1f3449dd475b9 .cell .markdown}
发现结果正好在题目给定范围内，对此做验证（2～5)组：
:::

::: {#b2ca7b515c9adfd .cell .code execution_count="8" ExecuteTime="{\"end_time\":\"2024-05-19T08:39:46.149462Z\",\"start_time\":\"2024-05-19T08:39:31.127114Z\"}"}
``` python
results = np.zeros((4, num_rows, num_cols))
for o in range(1, 5):
    phi = unwrap_phase(np.array(df_theta[o]))
                
    results[o-1] = -1 * phi / np.array(df_k[o])
  
plot(raw_pic(1), raw_pic(2), raw_pic(3), raw_pic(4), titles=[f"raw pic{i+1}" for i in range(1, 5)])      
plot(results[0], results[1], results[2], results[3], titles=[f"pic{i+1}" for i in range(1, 5)])
plot(df_theta[1], df_theta[2], df_theta[3], df_theta[4], titles=[f"theta{i+1}" for i in range(1, 5)])
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/9909052aef13e132e288e174265ac64200640559.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/ff256757cc8d20b4e3fbdf6be956a6f1eb052587.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/99ed7263d9f56e19ae1685036ee02d5e23c1207f.png)
:::
:::

::: {#81df051cd55e56fd .cell .markdown}
这里使用numpy自带的解缠方法对两个轴向上进行解缠分别查看效果（以第五组为例）
:::

::: {#e4bcd7059cae777d .cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2024-05-19T08:39:50.270410Z\",\"start_time\":\"2024-05-19T08:39:46.150299Z\"}"}
``` python
theta = np.array(df_theta[1]) 
k4 = np.array(df_k[1])

# 使用numpy的unwrap函数对theta进行处理
# 默认情况下，unwrap会沿着最后一个轴处理，对于二维矩阵来说，就是沿着每一行处理
# 如果你想沿着其他轴处理，可以使用axis参数
phix = np.unwrap(theta, axis=1)
phiy = np.unwrap(theta, axis=0)
phi = (np.unwrap(phix, axis=0))

results = np.zeros((3 ,num_rows, num_cols))


for i in range(3):
    if not i:
        results[i] = -1 * phix / k4
        continue
    elif i % 2:
        results[i] = -1 * phiy / k4
    elif not i % 2:
        results[i] = -1 * phi / k4

plot(raw_pic(1), results[0], results[1], results[2], titles=["raw" , "unwrapped x" ,"unwrap y", "both dir"])

        
results[0] = -1 / k4 * unwrap_phase(theta)


plot(results[2], results[0]  , titles=[ "fixed both dir", "old unwrap"])
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/68ab5b7a32d262eae8aae9c8b9e6b21a9d429aca.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/ccc3fa539c7325e5b1bbed2567f7ab9d861f09ee.png)
:::
:::

::: {#5de4c5230c103fff .cell .markdown}
可见两种方法得出的效果非常相近
:::

::: {#c071bc533938acbe .cell .markdown}
遍历所有数据，查看每组数据高程图还原的步长和结果,还原是对相位调整$2 n \pi$
:::

::: {#dc288fd3ea02ae2c .cell .code execution_count="71" ExecuteTime="{\"end_time\":\"2024-05-19T09:27:57.550823Z\",\"start_time\":\"2024-05-19T09:27:39.345703Z\"}"}
``` python
h = np.zeros((num_rows, num_cols))

for i in range(1,5):
    step = (i+1) * 100
    phi = unwrap_phase(np.array(df_theta[i]))
    h = -1 * phi / df_k[i]
    while np.min(h) < 150 or np.max(h) > 300:
        step += 1
        phi += 2 * np.pi
        h = -1 * phi / df_k[i]
        if np.min(h) >= 150 and np.max(h) <= 300:
            plot(df_theta[i], phi, h, titles=["raw $\Theta$", "new $\phi$", f"{step}"])
            draw3D(h)
            

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/579466e6fce81103c380b1aa19242b36f38ed814.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/a69516d16585c76199ba1de4ec46171fc3e36bf8.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/906967a69d029cb7b69c233eb4a6148e37b12816.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/5c8bcbb95d0752a1391d349366ec06b296e1a1e0.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/5d872f1e4d8d0e68488fdee4bfd2534a81a930c8.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/199f26a5ed247ac384c4f4c69c3a522384dc1f65.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/4f3193754fe1a1819610ecd9cbacfa73d23e70fc.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/e5e2d339a933f935530b439b0ea0546de4a99476.png)
:::
:::

::: {#9196471090b578ca .cell .markdown}
发现随着原始相位图纠缠条纹变密集，还原高程图的步骤逐渐变长，而最后还原出的色度图相差无几，验证了题目测量的是同一地区的题干
:::

::: {#65b40d7b5d299f84 .cell .markdown}
自主模拟生产相位图 与 对应的纠缠图
:::

::: {#5c291cde842781c0 .cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2024-05-19T08:40:05.922616Z\",\"start_time\":\"2024-05-19T08:40:05.916074Z\"}"}
``` python
from mpl_toolkits.axes_grid1 import make_axes_locatable


def simulate(size, m_1, m_2, C, A, mu_x, mu_y, sigma_x, sigma_y):
  """
  creates an arbitrary phase map by mixing gaussian blobs and adding ramps
  """
  x = np.arange(0, size[0], 1)
  y = np.arange(0, size[0], 1)
  xx, yy = np.meshgrid(x, y, sparse=True)
  I = np.zeros(size)
  ## mix randomly shaped and placed gaussian blobs
  
  for i in range(len(sigma_x)):
      a = (xx-mu_x[i])**2/(2*sigma_x[i]**2) + (yy-mu_y[i])**2/(2*sigma_y[i]**2)
      I += A[i]*np.exp(-a)
  ## add ramp phase with random gradients and shifts
  I = m_1*xx + m_2*yy + C + 0.1*I
  return I

def wrap(phi):
  """
  wraps the true phase signal within [-pi, pi]
  """
  return np.angle(np.exp(1j*phi))

def rescale(im, range):
  """
  mini-max rescales the input image
  """
  im_std = (im - im.min()) / (im.max() - im.min())
  im_scaled = im_std * (range[1] - range[0]) + range[0]
  return im_scaled

def create_random_image(size):
  """
  creates an randomly simulated true phase map
  """ 
  array_len = np.random.randint(2, 5)
  m = np.random.uniform(0, 0.5, [2])
  C = np.random.randint(1, 10)
  A = np.random.randint(50, 1000, array_len)
  mu_x = np.random.randint(round(size[0] / 4), round(size[1] * 3 / 4), array_len)
  mu_y = np.random.randint(round(size[0] / 4), round(size[1] * 3 / 4), array_len)
  sigma_x = np.random.randint(10, 45, array_len)
  sigma_y = np.random.randint(10, 45, array_len)
  I = simulate(size, m[0], m[1], C, A, mu_x, mu_y, sigma_x, sigma_y)
  return I

```
:::

::: {#44fa68d282746cf6 .cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2024-05-19T08:40:06.146208Z\",\"start_time\":\"2024-05-19T08:40:05.923705Z\"}"}
``` python
## example
size = (256, 256)

I = create_random_image(size)
I_wrap = wrap(I)
plot(I, I_wrap, titles=["$\phi$", "$\psi$"])
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/3b2b5a5065a61c1f0204c5add9684df9e195c9bf.png)
:::
:::

::: {#8e35448110950e78 .cell .markdown}
用相位展开方法解缠
:::

::: {#afef0bdef079802b .cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2024-05-19T08:40:06.566070Z\",\"start_time\":\"2024-05-19T08:40:06.149616Z\"}"}
``` python
phi = unwrap_phase(I_wrap)

plot(phi)
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/edccad97f1b9a1600e1bcb0afb3773c8c5e86677.png)
:::
:::

::: {#3ad2dedb48dc50cf .cell .markdown}
以上结果发现对随机生成的相位图拟合效果不错能大体上解缠，但是依旧有一些区域效果不太好，故打算引入神经网络
:::

::: {#605c0482662f03bc .cell .markdown}
以下是神经网络方法，训练过程参考tutorial

用自主模拟的相位图来训练一个模型，此处参考[一种联合性的机器学习相位解缠方法](https://ieeexplore.ieee.org/document/9414748)，使用它们的预训练模型，微调后接入我们的模型
:::

::: {#c19bfa257db7af50 .cell .code execution_count="14" ExecuteTime="{\"end_time\":\"2024-05-19T08:40:09.852264Z\",\"start_time\":\"2024-05-19T08:40:06.567818Z\"}"}
``` python
import numpy as np
import h5py
import os
import matplotlib as mpl
mpl.style.use('default')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from keras.layers import Activation, BatchNormalization, Conv2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.layers import MaxPooling2D, Dropout, Input, AveragePooling2D, Reshape, Permute, UpSampling2D
from keras.layers import SimpleRNN, Bidirectional, LSTM
from keras.layers import Lambda
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
from keras.optimizers import *
import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
K.set_image_data_format('channels_last')
```
:::

::: {#600255019bee8120 .cell .markdown}
引用论文模型原型用于导入模型（修改了尺寸）
:::

::: {#5701873013970697 .cell .code execution_count="15" ExecuteTime="{\"end_time\":\"2024-05-19T08:40:09.861495Z\",\"start_time\":\"2024-05-19T08:40:09.853183Z\"}"}
``` python
def get_joint_conv_sqd_lstm_net():
    """
    Defines the joint convoltional and spatial quad-directional LSTM network
    """
    ## input to the network
    input = Input((256, 256, 1))

    ## encoder network
    c1 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(input)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = AveragePooling2D()(c1)

    c2 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = AveragePooling2D()(c2)

    c3 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = AveragePooling2D()(c3)

    c4 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = AveragePooling2D()(c4)

    # SQD-LSTM Block
    x_hor_1 = Reshape((16 * 16, 128))(p4)
    x_ver_1 = Reshape((16 * 16, 128))(Permute((2, 1, 3))(p4))

    h_hor_1 = Bidirectional(LSTM(units=32, activation='tanh', return_sequences=True, go_backwards=False))(x_hor_1)
    h_ver_1 = Bidirectional(LSTM(units=32, activation='tanh', return_sequences=True, go_backwards=False))(x_ver_1)

    H_hor_1 = Reshape((16, 16, 64))(h_hor_1)
    H_ver_1 = Permute((2, 1, 3))(Reshape((16, 16, 64))(h_ver_1))

    c_hor_1 = Conv2D(filters=64, kernel_size=(3, 3),
                     kernel_initializer='he_normal', padding='same')(H_hor_1)
    c_ver_1 = Conv2D(filters=64, kernel_size=(3, 3),
                     kernel_initializer='he_normal', padding='same')(H_ver_1)

    H = concatenate([c_hor_1, c_ver_1])

    # decoder Network
    u5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(H)
    u5 = concatenate([u5, c4])
    c5 = Conv2D(filters=128, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c3])
    c6 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    u7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c2])
    c7 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    u8 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c1])
    c8 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    ## output layer
    output = Conv2D(filters=1, kernel_size=(1, 1), padding='same', name='out1')(c8)
    output = Activation('linear')(output)

    model = Model(inputs=[input], outputs=[output])
    return model
```
:::

::: {#98433d87e14f52ca .cell .markdown}
加载模型
:::

::: {#28103fdf62ce2879 .cell .code execution_count="17" ExecuteTime="{\"end_time\":\"2024-05-19T08:40:47.794157Z\",\"start_time\":\"2024-05-19T08:40:47.632441Z\"}"}
``` python
Model = lambda df_theta: [unwrap_phase(np.array(df_theta[i])) for i in range(5)]
Model = get_joint_conv_sqd_lstm_net()
model_path = './model/LSTM_model.h5'
Model.load_weights(model_path)
```

::: {.output .error ename="TypeError" evalue="__init__() missing 1 required positional argument: 'pool_size'"}
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    Cell In[17], line 2
          1 Model = lambda df_theta: [unwrap_phase(np.array(df_theta[i])) for i in range(5)]
    ----> 2 Model = get_joint_conv_sqd_lstm_net()
          3 model_path = './model/LSTM_model.h5'
          4 Model.load_weights(model_path)

    Cell In[15], line 12, in get_joint_conv_sqd_lstm_net()
         10 c1 = BatchNormalization()(c1)
         11 c1 = Activation('relu')(c1)
    ---> 12 p1 = AveragePooling2D()(c1)
         14 c2 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p1)
         15 c2 = BatchNormalization()(c2)

    TypeError: __init__() missing 1 required positional argument: 'pool_size'
:::
:::

::: {#e92a9fe034401975 .cell .markdown}
由于一开始加载了df_theta直接使用，接下来使用模型输出解缠值
:::

::: {#5eb1159ca395fa93 .cell .code execution_count="18" ExecuteTime="{\"end_time\":\"2024-05-19T08:41:09.078308Z\",\"start_time\":\"2024-05-19T08:40:50.370347Z\"}"}
``` python
X_test = np.array(df_theta)
Y_pred = np.zeros_like(X_test)

Y_pred = Model(X_test)
```
:::

::: {#e7b383f9542e3992 .cell .code execution_count="19" ExecuteTime="{\"end_time\":\"2024-05-19T08:41:27.933179Z\",\"start_time\":\"2024-05-19T08:41:11.580445Z\"}"}
``` python
# 输出图像
for i in range(1,5):
    plot(Y_pred[i], unwrap_phase(np.array(df_theta[i])), titles=[f"$\phi${i+1} by LSTM", f"$\phi${i+1} by Math Unwrap"])
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/26182adddac2373884f777ce96a2b6c2ceb48d25.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/34eea694cddabaeb069e17a5e81bc14840f7a0e6.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/91f2b730d7d2f16da21d5825bf1b38a744395552.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/b3023eba6b095711041d1cafc24a2318b9fdd398.png)
:::
:::

::: {#9076b7ccc4f04762 .cell .markdown}
肉眼并不能看出这两种方法明显的区别，但是考虑到该机器学习论文训练集足够大，且在其测试集上表现非常优良，而数学方法在前面自主生成相位纠缠的图像上表现欠佳（有图像证据和论文证据），故后续全部使用LSTM方法计算$\phi$
:::

::: {#994b035bc3404718 .cell .markdown}
对于第一题
需要引入一种评估方案评估$\phi$，考虑到干涉是物理光学性质，由此可以知道，当干涉条纹间隔越短，采样率更高，但随之带来的误差也会提高（假设）。
故数据5采样率最高，数据1连续性最好。我们基于先前的unwrap
函数，定义一个三维方向上的unwrap函数（在二维方向上依旧使用LSTM），假设将数据中的某n组包装到一起，就会得到一个n
x 805 x 2001的矩阵，而且对每层n是极大消除误差的。
:::

::: {#898150006c3de333 .cell .markdown}
:::

::: {#4040787c3f1e9314 .cell .code execution_count="20" ExecuteTime="{\"end_time\":\"2024-05-19T08:41:32.604879Z\",\"start_time\":\"2024-05-19T08:41:32.595625Z\"}"}
``` python
def unwrap_3D_phase(theta):
    # 初始化 phi 矩阵
    phi = np.zeros_like(theta)
    phi[0, 0, 0] = theta[0, 0, 0]
    
    
    # 展开每一个平面的其余矩阵
    
    for k in range(0, theta.shape[2]):
        # 展开第一个平面的行
        for i in range(1, theta.shape[0]):
            delta = theta[i, 0, k] - theta[i-1, 0, k]
            delta_wrapped = np.mod(delta + np.pi, 2 * np.pi) - np.pi
            phi[i, 0, k] = phi[i-1, 0, k] + delta_wrapped
        
        # 展开第一个平面的列
        for j in range(1, theta.shape[1]):
            delta = theta[0, j, k] - theta[0, j-1, k]
            delta_wrapped = np.mod(delta + np.pi, 2 * np.pi) - np.pi
            phi[0, j, k] = phi[0, j-1, k] + delta_wrapped
            
        for i in range(1, theta.shape[0]):
            for j in range(1, theta.shape[1]):
                delta_row = theta[i, j, k] - theta[i-1, j, k]
                delta_col = theta[i, j, k] - theta[i, j-1, k]
                delta_row_wrapped = np.mod(delta_row + np.pi, 2 * np.pi) - np.pi
                delta_col_wrapped = np.mod(delta_col + np.pi, 2 * np.pi) - np.pi
                phi[i, j, k] = phi[i-1, j, k] + delta_row_wrapped
                phi[i, j, k] = phi[i, j-1, k] + delta_col_wrapped
               
    #展开深度方向
    depth = lambda theta, phi: np.array([[[phi[i-1, j, k] + (np.mod(theta[i, j, k] - theta[i-1, j, k] + np.pi, 2 * np.pi) - np.pi) if i > 0 else phi[i, j, k],
                                         phi[i, j-1, k] + (np.mod(theta[i, j, k] - theta[i, j-1, k] + np.pi, 2 * np.pi) - np.pi) if j > 0 else phi[i, j, k],
                                         phi[i, j, k-1] + (np.mod(theta[i, j, k] - theta[i, j, k-1] + np.pi, 2 * np.pi) - np.pi) if k > 0 else phi[i, j, k]]
                                        for j in range(theta.shape[1])]])
 
    
    return phi
```
:::

::: {#a95e403558afa14b .cell .markdown}
分别用数学方法和神经网络方法计算每组数据的高程值
数学方法在前面已经做过，以下是神经网络方法求出的高程值，我们主要考虑迭代步数s
:::

::: {#1c1ae651bc8fbf39 .cell .code execution_count="21" ExecuteTime="{\"end_time\":\"2024-05-19T08:41:52.175972Z\",\"start_time\":\"2024-05-19T08:41:35.354605Z\"}"}
``` python
h = np.zeros((num_rows, num_cols))
s = []
s.append(0)
for i in range(1,5):
    step = (i+1) * 100
    phi = unwrap_phase(Y_pred[i])
    h = -1 * phi / df_k[i]
    while np.min(h) < 150 or np.max(h) > 300:
        step += 1
        phi += 2 * np.pi
        h = -1 * phi / df_k[i]
        if np.min(h) >= 150 and np.max(h) <= 300:
            plot(df_theta[i], phi, h, titles=["raw $\Theta$", "LSTM $\phi$", f"{step}"])
            s.append(step%100)
            
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/ddac2f7eab0179d5211e99eafbd794f5a8ebab53.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/1e90f6444d62b0722ae025e5456d0640cc4deb27.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/b0fd91a4a75e8bd70ad0cb15218471601b7fe20c.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/de36dc3bd272c47ee3024607f43ad4ee54264e65.png)
:::
:::

::: {#6f850a77a5dd524a .cell .markdown}
上图也可以看出随着原始样本复杂性变高得到的phi的极差也变大了，说明在同等大小信息容器下，第五组数据相较于第一组，信息密度更高，即采样率越大验证了我们评估方案合理性。
:::

::: {#7fa7b949f768aa29 .cell .markdown}
我们得到的s是迭代次数，反映了采样率，这将作为我们混合量化的权重考虑参数，次数越多，权重越大.
对给定极大消除误差过的生成的n x 805 x
2001的矩阵，我们要将其最后分权转为805 x
2001的矩阵，就需要乘上一个权重矩阵。
:::

::: {#377d030696bb7bc5 .cell .markdown}
我们先选取第一组和第五组尝试计算
:::

::: {#d0a0fb42b4b44e17 .cell .markdown}
2 5
:::

::: {#e05fdcc17c9914c5 .cell .code execution_count="72" ExecuteTime="{\"end_time\":\"2024-05-19T09:29:01.831143Z\",\"start_time\":\"2024-05-19T09:28:50.590869Z\"}"}
``` python
combine_2_5 = np.zeros((num_rows, num_cols, 2))

for i in range(num_rows):
    for j in range(num_cols):
        combine_2_5[i][j][0] = Y_pred[1][i][j]
        combine_2_5[i][j][1] = Y_pred[4][i][j]

weight = np.zeros((2,1))
weight1 = s[1] / (s[1] + s[4])
weight2 = s[4] / (s[1] + s[4])
weight[0][0] = weight1
weight[1][0] = weight2

combine_2_5 = unwrap_3D_phase(combine_2_5)
combine = np.zeros((num_rows, num_cols))

tmp = np.zeros((num_rows, num_cols))

combine = combine_2_5 @ weight

# for i in range(num_rows):
#     for j in range(num_cols):
#         tmp[i][j] = combine_1_5[i][j][0] * weight1 + combine_1_5[i][j][1] * weight2
        
combine = combine.ravel().reshape((num_rows,num_cols))
# combine = tmp
# print(combine.shape)
# for i in range(num_rows):
#     for j in range(num_cols):
#         combine[i][j] = combine_1_5[0][i][j] * s[0] / (s[0] + s[4]) + combine_1_5[1][i][j] * s[4] / (s[0] + s[4])
        
plot(combine, titles="combined $\phi$")

plot((-1 * combine / df_k[1]), (-1 * combine / df_k[4]), titles=["Height 0", "Height 1"])

H = -1 * combine / df_k[1]

N, M = H.shape
x = np.linspace(1, M, M)
y = np.linspace(1, N, N)
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize=(12, 6))  # Set the figure size to be wide enough for two subplots
ax1 = fig.add_subplot(121, projection='3d')  # Change this to 121 for a 1x2 grid (first subplot)
ax1.plot_surface(x, y, H, cmap='viridis')
ax1.set_title('Estimated Height')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Height(m)')
ax1.view_init(elev=10, azim=135)

# Second plot
H = -1 * combine / df_k[4]

N, M = H.shape
x = np.linspace(1, M, M)
y = np.linspace(1, N, N)
x, y = np.meshgrid(x, y)
ax2 = fig.add_subplot(122, projection='3d')  # Change this to 122 for a 1x2 grid (second subplot)
ax2.plot_surface(x, y, H, cmap='viridis')
ax2.set_title('Estimated Height')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Height(m)')
ax2.view_init(elev=20, azim=180)
# Set aspect ratio and scale
ax2.set_aspect('equal', 'box')
ax2.set_box_aspect([2001, 805, 500]) 

plt.show()
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/cd3b98338e1736067b86dd6b6120ba342c809eea.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/cf7e33f989a281b43850e82130f169b97a49486e.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/b1df25b7eac6eac0b40591715912cca35f5cf12f.png)
:::
:::

::: {#e3aa7d616b8fa0b5 .cell .markdown}
用先前的方法将高程移至区间内
:::

::: {#570d73bfece17daa .cell .code execution_count="73" ExecuteTime="{\"end_time\":\"2024-05-19T09:29:57.450592Z\",\"start_time\":\"2024-05-19T09:29:55.832970Z\"}"}
``` python
phi = combine

H = -1 * phi / df_k[1]
while np.min(H) < 150 :
    phi += 2 * np.pi
    H = -1 * phi / df_k[1]


N, M = H.shape
x = np.linspace(1, M, M)
y = np.linspace(N, 1, N)
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize=(12, 6))  # Set the figure size to be wide enough for two subplots
ax1 = fig.add_subplot(121, projection='3d')  # Change this to 121 for a 1x2 grid (first subplot)
ax1.plot_surface(x, y, H, cmap='viridis')
ax1.set_title('Estimated Height')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Height(m)')
ax1.view_init(elev=20, azim=230)
# Set aspect ratio and scale
ax1.set_aspect('equal', 'box')
ax1.set_box_aspect([2001, 805, 500]) 


# plt.show()
# Second plot
phi = combine
H = -1 * phi / df_k[4]
while np.min(H) < 150 :
    phi += 2 * np.pi
    H = -1 * phi / df_k[4]
N, M = H.shape
x = np.linspace(1, M, M)
y = np.linspace(N, 1, N)
x, y = np.meshgrid(x, y)
ax2 = fig.add_subplot(122, projection='3d')  # Change this to 122 for a 1x2 grid (second subplot)
ax2.plot_surface(x, y, H, cmap='viridis')
ax2.set_title('Estimated Height')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Height(m)')
ax2.view_init(elev=20, azim=230)
# Set aspect ratio and scale
ax2.set_aspect('equal', 'box')
ax2.set_box_aspect([2001, 805, 500])  # Adjust the third value to control the height scaling

plt.show()

plot(H,titles="Height with combined $\phi$")
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/faf16bb4bd28fcc417ce6b5cc588475372d362ba.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/b0334b582f82aac8bf2fefbf99b9fd70e781d0a6.png)
:::
:::

::: {#c08e03bd1481b031 .cell .markdown}
验证第五组数据上界
:::

::: {#92f801e84c07fd79 .cell .code execution_count="24" ExecuteTime="{\"end_time\":\"2024-05-19T08:42:38.368486Z\",\"start_time\":\"2024-05-19T08:42:38.358551Z\"}"}
``` python
print(np.max(H))
```

::: {.output .stream .stdout}
    248.30459484076502
:::
:::

::: {#583a6e62a021044d .cell .markdown}
综上，由于第五组数据采样率最高，故选择第五组数据的k来高程恢复，由此得出四个顶点的高程值
:::

::: {#5f9cb24d594cbc2e .cell .code execution_count="25" ExecuteTime="{\"end_time\":\"2024-05-19T08:42:41.860072Z\",\"start_time\":\"2024-05-19T08:42:41.732906Z\"}"}
``` python
print(H.shape)

final = np.array(H)
# 获取四个顶点的值
top_left = round(final[0, 0])
top_right = round(final[0, num_cols-1])
bottom_left = round(final[num_rows-1, 0])
bottom_right = round(final[num_rows-1, num_cols-1])

# 创建一个2x2的图
fig, axs = plt.subplots(2, 2)

# 设置四个顶点的值
axs[0, 0].text(0.5, 0.5, str(top_left), ha='center', va='center', size=20)
axs[0, 0].set_title('Top Left')
axs[0, 1].text(0.5, 0.5, str(top_right), ha='center', va='center', size=20)
axs[0, 1].set_title('Top Right')
axs[1, 0].text(0.5, 0.5, str(bottom_left), ha='center', va='center', size=20)
axs[1, 0].set_title('Bottom Left')
axs[1, 1].text(0.5, 0.5, str(bottom_right), ha='center', va='center', size=20)
axs[1, 1].set_title('Bottom Right')

# 隐藏坐标轴
for ax in axs.flat:
    ax.set(xticks=[], yticks=[])

# 调整子图间距
plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()
```

::: {.output .stream .stdout}
    (805, 2001)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/bd0147c00ceb36718ad7409668374fdd399cad40.png)
:::
:::

::: {#42c005ab4ed23dfc .cell .markdown}
由于后面需要用到直方图，故写一个函数
:::

::: {#39386cbb3035d327 .cell .code execution_count="26" ExecuteTime="{\"end_time\":\"2024-05-19T08:42:45.574005Z\",\"start_time\":\"2024-05-19T08:42:45.563357Z\"}"}
``` python
def histogram(data_array, y_label, title):
    # 创建直方图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(data_array)), data_array, color='b')
    bars[-1].set_color('r')
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)
    
    plt.xlabel('Data Set Index')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(range(len(data_array)), [f'{i+1}' for i in range(len(data_array))])
    
    # 显示图表
    plt.show()
```
:::

::: {#2dc607bf5692e240 .cell .markdown}
n越大采样率越高，对于第一题，我们既要保证采样率，又要保证误差最小，以下代码，反映五组数据的方差
:::

::: {#7c641a115c7be832 .cell .code execution_count="27" ExecuteTime="{\"end_time\":\"2024-05-19T08:43:03.759812Z\",\"start_time\":\"2024-05-19T08:42:46.960255Z\"}"}
``` python
vars = []
vars_for_dis = []
hs = []

h = np.zeros((num_rows, num_cols))
for i in range(1,5):
    phi = unwrap_phase(Y_pred[i])
    h = -1 * phi / df_k[i]
    while np.min(h) < 150 or np.max(h) > 300:
        phi += 2 * np.pi
        h = -1 * phi / df_k[i]
        if np.min(h) >= 150 and np.max(h) <= 300:
            hs.append(h)
            vars.append(np.array(h).flatten())
    
            
for i in range(1,5):
    vars_for_dis.append(np.var(vars[i-1], ddof=1))

vars_for_dis.append(np.var(final.flatten(), ddof=1))
hs.append(final)

# print(vars_for_dis[i] for i in range(6))

histogram(vars_for_dis, "Variance", "Variance of Each Data Set")

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/1d0a9fdbd5743090bfea744c436068bde6aabcdf.png)
:::
:::

::: {#9d1c1ec1af8c7790 .cell .markdown}
由以上直方图可以看出我们的模型相较于原始模型有非常明显的方差降低，代表误差与精度的提升
:::

::: {#f2cfda6b8245246a .cell .markdown}
接下来使用梯度平均值查看我们模型的精度
:::

::: {#d5203ef4a95802d3 .cell .code execution_count="28" ExecuteTime="{\"end_time\":\"2024-05-19T08:44:09.115326Z\",\"start_time\":\"2024-05-19T08:44:08.725008Z\"}"}
``` python
hs_for_dis = []

for i in range(1, 6):   
    gradient_x, gradient_y = np.gradient(np.array(hs[i-1]))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # 计算梯度的平均值作为光滑程度的指标
    smoothness_gradient = np.mean(gradient_magnitude)
    hs_for_dis.append(smoothness_gradient)
   
histogram(hs_for_dis, "Gradient", "Gradient of Each Data Set")

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/04e3f1bf158c1016fa1d9ac13888f95121e40048.png)
:::
:::

::: {#5ac778ede5766ed4 .cell .markdown}
由以上直方图可以看出我们的模型相较于原始模型有一定程度的梯度降低，图像光滑程度也有所提高
:::

::: {#c963ecc193212cd4 .cell .markdown}
接下来遍历$\underset{i=2}{\overset{4}{C^i_4}}$
:::

::: {#5fbb920610e748f6 .cell .code execution_count="29" ExecuteTime="{\"end_time\":\"2024-05-19T08:45:03.410731Z\",\"start_time\":\"2024-05-19T08:44:12.369779Z\"}"}
``` python
binom = []
hs = []


for i in range(1, 5):
    for j in range(i+1, 5):  
        binom_2_4 = np.zeros((num_rows, num_cols, 2))
        for x in range(num_rows):
            for y in range(num_cols):
                binom_2_4[x][y][0] = Y_pred[i][x][y]
                binom_2_4[x][y][1] = Y_pred[j][x][y]
        weight = np.zeros((2,1))
        weight1 = s[i] / (s[i] + s[j])
        weight2 = s[j] / (s[i] + s[j])
        weight[0][0] = weight1
        weight[1][0] = weight2
        binom_2_4 = unwrap_3D_phase(binom_2_4)
        binom24 = np.zeros((num_rows, num_cols))
        binom24 = binom_2_4 @ weight
        binom24 = binom24.ravel().reshape((num_rows, num_cols))
        while np.min(-1 * binom24 / df_k[j]) < 150:
            binom24 += 2 * np.pi
        binom.append(-1 * binom24 / df_k[j])
 

binom.append(final)

            
```
:::

::: {#27b528e5900ca56a .cell .code execution_count="30" ExecuteTime="{\"end_time\":\"2024-05-19T08:45:05.995755Z\",\"start_time\":\"2024-05-19T08:45:05.786187Z\"}"}
``` python

vars = []
vars_for_dis = []

for i in range(len(binom)):
    vars.append(np.array(binom[i]).flatten() )
 
for i in range(len(vars)):
   vars_for_dis.append(np.var(vars[i], ddof=1))

histogram(vars_for_dis, "Variance", "Variance of Each Data Set")

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/87e6b1b9bae5fd94ad4b76f2c4d6475e19f6a9a5.png)
:::
:::

::: {#198085d2fc67de36 .cell .code execution_count="31" ExecuteTime="{\"end_time\":\"2024-05-19T08:45:09.478990Z\",\"start_time\":\"2024-05-19T08:45:09.173090Z\"}"}
``` python
hs_for_dis = []

for i in range(len(binom)):   
    gradient_x, gradient_y = np.gradient(np.array(binom[i]))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # 计算梯度的平均值作为光滑程度的指标
    smoothness_gradient = np.mean(gradient_magnitude)
    hs_for_dis.append(smoothness_gradient)
  
histogram(hs_for_dis, "Gradient", "Gradient of Each Data Set") 
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/5505eb2d51025b2ef81cf1509706c4720078b7fe.png)
:::
:::

::: {#d3bcdba7f334510 .cell .code execution_count="32" ExecuteTime="{\"end_time\":\"2024-05-19T08:46:08.034154Z\",\"start_time\":\"2024-05-19T08:45:16.675989Z\"}"}
``` python
binom = []


for i in range(1, 5):
    for j in range(i+1, 5):  
        for k in range(j+1,5):
            binom_3_4 = np.zeros((num_rows, num_cols, 3))
            for x in range(num_rows):
                for y in range(num_cols):
                    binom_3_4[x][y][0] = Y_pred[i][x][y]
                    binom_3_4[x][y][1] = Y_pred[j][x][y]
                    binom_3_4[x][y][2] = Y_pred[k][x][y]
                    
            weight = np.zeros((3,1))
            weight1 = s[i] / (s[i] + s[j] + s[k])
            weight2 = s[j] / (s[i] + s[j] + s[k])
            weight3 = s[k] / (s[i] + s[j] + s[k])
            weight[0][0] = weight1
            weight[1][0] = weight2
            weight[2][0] = weight3
            binom_3_4 = unwrap_3D_phase(binom_3_4)
            binom34 = np.zeros((num_rows, num_cols))
            binom34 = binom_3_4 @ weight
            binom34 = binom34.ravel().reshape((num_rows, num_cols))
            while np.min(-1 * binom34 / df_k[k]) < 150:
                binom34 += 2 * np.pi
            binom.append(-1 * binom34 / df_k[k])
 

binom.append(final)

            
```
:::

::: {#460baba248649162 .cell .code execution_count="33" ExecuteTime="{\"end_time\":\"2024-05-19T08:46:10.781651Z\",\"start_time\":\"2024-05-19T08:46:10.447376Z\"}"}
``` python

vars = []
vars_for_dis = []

for i in range(len(binom)):
    vars.append(np.array(binom[i]).flatten() )
 
for i in range(len(vars)):
   vars_for_dis.append(np.var(vars[i], ddof=1))

histogram(vars_for_dis, "Variance", "Variance of Each Data Set")

# 创建直方图
```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/5b6ef939f921b2a9d89fe010a86c8a3db66ec810.png)
:::
:::

::: {#7d0750e62b58bb6f .cell .code execution_count="34" ExecuteTime="{\"end_time\":\"2024-05-19T08:46:13.565056Z\",\"start_time\":\"2024-05-19T08:46:13.205720Z\"}"}
``` python
hs_for_dis = []

for i in range(len(binom)):   
    gradient_x, gradient_y = np.gradient(np.array(binom[i]))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # 计算梯度的平均值作为光滑程度的指标
    smoothness_gradient = np.mean(gradient_magnitude)
    hs_for_dis.append(smoothness_gradient)
   
# 创建直方图
histogram(hs_for_dis, "Gradient", "Gradient of Each Data Set") 

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/93b56ee1724e494e89f97e552990fb9f11675a13.png)
:::
:::

::: {#13e1c55c1676b590 .cell .code execution_count="35" ExecuteTime="{\"end_time\":\"2024-05-19T08:46:37.534726Z\",\"start_time\":\"2024-05-19T08:46:20.112333Z\"}"}
``` python
binom = []

binom_4_4 = np.zeros((num_rows, num_cols, 4))
for x in range(num_rows):
    for y in range(num_cols):
        binom_3_4[x][y][0] = Y_pred[1][x][y]
        binom_3_4[x][y][1] = Y_pred[2][x][y]
        binom_3_4[x][y][2] = Y_pred[3][x][y]
        binom_3_4[x][y][2] = Y_pred[4][x][y]
        
weight = np.zeros((4,1))
weight1 = s[0] / (s[1] + s[2] + s[3] + s[4])
weight2 = s[1] / (s[1] + s[2] + s[3] + s[4])
weight3 = s[2] / (s[1] + s[2] + s[3] + s[4])
weight4 = s[3] / (s[1] + s[2] + s[3] + s[4])
weight[0][0] = weight1
weight[1][0] = weight2
weight[2][0] = weight3
weight[3][0] = weight4
binom_4_4 = unwrap_3D_phase(binom_4_4)
binom44 = np.zeros((num_rows, num_cols))
binom44 = binom_4_4 @ weight
binom44 = binom34.ravel().reshape((num_rows, num_cols))
while np.min(-1 * binom44 / df_k[4]) < 150:
    binom44 += 2 * np.pi
binom.append(-1 * binom44 / df_k[4])

binom.append(final)
```
:::

::: {#2304177e6ce1767f .cell .code execution_count="36" ExecuteTime="{\"end_time\":\"2024-05-19T08:46:45.011544Z\",\"start_time\":\"2024-05-19T08:46:44.888659Z\"}"}
``` python

vars = []
vars_for_dis = []

for i in range(len(binom)):
    vars.append(np.array(binom[i]).flatten() )
 
for i in range(len(vars)):
   vars_for_dis.append(np.var(vars[i], ddof=1))


# 创建直方图
histogram(vars_for_dis, "Variance", "Variance of Each Data Set")

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/03dfeb72e618f33a605da7368320e831ceb6a9ae.png)
:::
:::

::: {#889229304bb3b9da .cell .code execution_count="37" ExecuteTime="{\"end_time\":\"2024-05-19T08:46:46.270906Z\",\"start_time\":\"2024-05-19T08:46:46.073088Z\"}"}
``` python
hs_for_dis = []

for i in range(len(binom)):   
    gradient_x, gradient_y = np.gradient(np.array(binom[i]))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # 计算梯度的平均值作为光滑程度的指标
    smoothness_gradient = np.mean(gradient_magnitude)
    hs_for_dis.append(smoothness_gradient)
   
# 创建直方图
histogram(hs_for_dis, "Gradient", "Gradient of Each Data Set") 

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/706cbb0f719f858530825f2b57cf2bfa5d24bd1e.png)
:::
:::

::: {#3d4deb870237bc0f .cell .markdown}
综合所有数据的高程情况
:::

::: {#daa2470c05a78329 .cell .code execution_count="38" ExecuteTime="{\"end_time\":\"2024-05-19T08:49:04.893262Z\",\"start_time\":\"2024-05-19T08:46:48.134235Z\"}"}
``` python
vars = []
vars_for_dis = []
binom = []
hs_for_dis = []


h = np.zeros((num_rows, num_cols))
for i in range(1,5):
    phi = unwrap_phase(Y_pred[i])
    h = -1 * phi / df_k[i]
    while np.min(-1 * phi / df_k[i]) < 150:
        phi += 2 * np.pi
    binom.append(-1 * phi / df_k[i])
    
            
for i in range(1, 5):
    for j in range(i+1, 5):  
        binom_2_4 = np.zeros((num_rows, num_cols, 2))
        for x in range(num_rows):
            for y in range(num_cols):
                binom_2_4[x][y][0] = Y_pred[i][x][y]
                binom_2_4[x][y][1] = Y_pred[j][x][y]
        weight = np.zeros((2,1))
        weight1 = s[i] / (s[i] + s[j])
        weight2 = s[j] / (s[i] + s[j])
        weight[0][0] = weight1
        weight[1][0] = weight2
        binom_2_4 = unwrap_3D_phase(binom_2_4)
        binom24 = np.zeros((num_rows, num_cols))
        binom24 = binom_2_4 @ weight
        binom24 = binom24.ravel().reshape((num_rows, num_cols))
        while np.min(-1 * binom24 / df_k[j]) < 150:
            binom24 += 2 * np.pi
        binom.append(-1 * binom24 / df_k[j])

for i in range(1, 5):
    for j in range(i+1, 5):  
        for k in range(j+1,5):
            binom_3_4 = np.zeros((num_rows, num_cols, 3))
            for x in range(num_rows):
                for y in range(num_cols):
                    binom_3_4[x][y][0] = Y_pred[i][x][y]
                    binom_3_4[x][y][1] = Y_pred[j][x][y]
                    binom_3_4[x][y][2] = Y_pred[k][x][y]
                    
            weight = np.zeros((3,1))
            weight1 = s[i] / (s[i] + s[j] + s[k])
            weight2 = s[j] / (s[i] + s[j] + s[k])
            weight3 = s[k] / (s[i] + s[j] + s[k])
            weight[0][0] = weight1
            weight[1][0] = weight2
            weight[2][0] = weight3
            binom_3_4 = unwrap_3D_phase(binom_3_4)
            binom34 = np.zeros((num_rows, num_cols))
            binom34 = binom_3_4 @ weight
            binom34 = binom34.ravel().reshape((num_rows, num_cols))
            while np.min(-1 * binom34 / df_k[k]) < 150:
                binom34 += 2 * np.pi
            binom.append(-1 * binom34 / df_k[k])
            
binom_4_4 = np.zeros((num_rows, num_cols, 4))
for x in range(num_rows):
    for y in range(num_cols):
        binom_3_4[x][y][0] = Y_pred[1][x][y]
        binom_3_4[x][y][1] = Y_pred[2][x][y]
        binom_3_4[x][y][2] = Y_pred[3][x][y]
        binom_3_4[x][y][2] = Y_pred[4][x][y]
        
weight = np.zeros((4,1))
weight1 = s[0] / (s[1] + s[2] + s[3] + s[4])
weight2 = s[1] / (s[1] + s[2] + s[3] + s[4])
weight3 = s[2] / (s[1] + s[2] + s[3] + s[4])
weight4 = s[3] / (s[1] + s[2] + s[3] + s[4])
weight[0][0] = weight1
weight[1][0] = weight2
weight[2][0] = weight3
weight[3][0] = weight4
binom_4_4 = unwrap_3D_phase(binom_4_4)
binom44 = np.zeros((num_rows, num_cols))
binom44 = binom_4_4 @ weight
binom44 = binom34.ravel().reshape((num_rows, num_cols))
while np.min(-1 * binom44 / df_k[4]) < 150:
    binom44 += 2 * np.pi
binom.append(-1 * binom44 / df_k[4])


binom.append(final)
print(len(binom))
```

::: {.output .stream .stdout}
    16
:::
:::

::: {#9abdb532906dd10 .cell .code execution_count="39" ExecuteTime="{\"end_time\":\"2024-05-19T08:49:10.702936Z\",\"start_time\":\"2024-05-19T08:49:09.700624Z\"}"}
``` python
for i in range(len(binom)):
    vars.append(np.array(binom[i]).flatten() )
 
for i in range(len(vars)):
   vars_for_dis.append(np.var(vars[i], ddof=1))


# 创建直方图
histogram(vars_for_dis, "Variance", "Variance of Each Data Set")


hs_for_dis = []

for i in range(len(binom)):   
    gradient_x, gradient_y = np.gradient(np.array(binom[i]))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # 计算梯度的平均值作为光滑程度的指标
    smoothness_gradient = np.mean(gradient_magnitude)
    hs_for_dis.append(smoothness_gradient)
   
# 创建直方图
histogram(hs_for_dis, "Gradient", "Gradient of Each Data Set") 

```

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/006c41d88f1e3f0dfd9db5e4c6d9f52776cf0170.png)
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/51a66f19d9893423f7cca210bde95fcea62a84a8.png)
:::
:::

::: {#87f5aa0de24ce49 .cell .markdown}
-   精细度我们从三个方向给分：梯度越低越高，方差越小越好，分辨率越高越好。
-   复杂度从两个方向惩罚：时间复杂度和空间复杂度
:::

::: {#ba4b591f64951aad .cell .markdown}
### 精细度
:::

::: {#b62e0d2005094068 .cell .code execution_count="40" ExecuteTime="{\"end_time\":\"2024-05-19T08:49:14.155296Z\",\"start_time\":\"2024-05-19T08:49:13.988806Z\"}"}
``` python
score = np.zeros((3, 16))
# 方差
score[0] = vars_for_dis
# 梯度
score[1] = hs_for_dis

# 分辨率即s即迭代次数
s_for_dis = []
for i in range(1,5):
    s_for_dis.append(s[i])
    
for i in range(1,5):
    for j in range(i+1, 5 ):
        s_for_dis.append(s[i] + s[j])
        
for i in range(1,5):
    for j in range(i+1, 5):
        for k in range(j+1, 5):
            s_for_dis.append(s[i] + s[j] + s[k])
            
s_for_dis.append(s[1] + s[2] + s[3] + s[4])

s_for_dis.append(s[1] + s[4])

# 迭代次数
score[2] = s_for_dis

final_score = []

for i in range(3):
    max = np.max(score[i])
    # 比例
    k = 1 / max
    score[i] =  score[i] * k
    if i == 0 or i == 1:
        score[i] = 1 - score[i]
    # print(score[i])

for i in range(16):
    final_score.append(score[0][i] + score[1][i] + score[2][i])
    
print(final_score)
histogram(final_score, "Score", "Positive Score of Each Data Set")
```

::: {.output .stream .stdout}
    [0.39229337381897345, 0.6287653254553656, 0.6584713392989737, 0.7398634869100815, 1.2892248523376673, 1.51258241714962, 1.6004415926362292, 1.4946467575150804, 1.5258408919321442, 1.258251415404719, 1.7841603171195408, 2.0400434356103596, 1.9041249047215882, 1.8719704425607588, 2.0247482203385365, 1.6004415926362292]
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/202a046f3027d783129941fec3bd9c3550a02850.png)
:::
:::

::: {#87cf0a5c626dd654 .cell .markdown}
### 复杂度
:::

::: {#de8fa96fc447ae55 .cell .markdown}
采用指数惩罚方案 惩罚因子为复杂度O(n)
由于题目给定的数据量超过160w，故采用对数来平滑化复杂度
:::

::: {#3978b758a8a0d2d8 .cell .code execution_count="41" ExecuteTime="{\"end_time\":\"2024-05-19T08:49:16.682391Z\",\"start_time\":\"2024-05-19T08:49:16.483626Z\"}"}
``` python
punish = []

for i in range(1,5):
    punish.append(1)
    
for i in range(1,5):
    for j in range(i+1, 5):
        punish.append(2)
        
for i in range(1,5):
    for j in range(i+1,5):
        for k in range(j+1,5):
            punish.append(3)
            
punish.append(4)

punish.append(2)

magnitude = num_cols * num_rows

for i in range(len(punish)):
    punish[i] = math.log(magnitude ** punish[i])  + math.log(magnitude ** punish[i])
 
print(punish)   
# 创建直方图
histogram(punish, "Punish", "Negtive Score of Each Data Set")
```

::: {.output .stream .stdout}
    [28.584489224004592, 28.584489224004592, 28.584489224004592, 28.584489224004592, 57.168978448009184, 57.168978448009184, 57.168978448009184, 57.168978448009184, 57.168978448009184, 57.168978448009184, 85.75346767201378, 85.75346767201378, 85.75346767201378, 85.75346767201378, 114.33795689601837, 57.168978448009184]
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/cf73e4b00eda9a85d8cf0941c1c14dd66d1a4dcc.png)
:::
:::

::: {#de3d758a2a562b25 .cell .markdown}
综合分数 = 正向分数 / 负向分数
:::

::: {#ff1ee2dc9e0b9ebe .cell .code execution_count="42" ExecuteTime="{\"end_time\":\"2024-05-19T08:49:18.338441Z\",\"start_time\":\"2024-05-19T08:49:18.167544Z\"}"}
``` python
f_score = np.zeros(16)
f_score = np.array(final_score) / np.array(punish)

print(f_score)
histogram(f_score, "Final Score"," Final Score of Each Data Set")
```

::: {.output .stream .stdout}
    [0.01372399 0.02199673 0.02303597 0.02588339 0.02255113 0.0264581
     0.02799493 0.02614437 0.02669001 0.02200934 0.02080569 0.02378963
     0.02220464 0.02182968 0.01770845 0.02799493]
:::

::: {.output .display_data}
![](vertopal_507370557942491881ca46ff406db7bf/bd93b49679fb14504409e139eb36f66439621a44.png)
:::
:::

::: {#8198f8903166f55c .cell .code}
``` python
```
:::
