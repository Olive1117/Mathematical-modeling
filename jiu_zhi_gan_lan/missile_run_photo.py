import  numpy as np
import matplotlib.pyplot as plt
radius: float = 7.0
height: float = 10.0
# 1. 离散圆柱
N = 16
ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
dR = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
bottom = np.array([0.0, 200.0, 0.0]) + dR * radius
top = bottom + np.array([0, 0, height])

# 2. 生成所有边（底面、顶面、母线）
edges = []
for i in range(N):
    j = (i + 1) % N
    edges.append((bottom[i], bottom[j]))  # 底边
    edges.append((top[i], top[j]))  # 顶边
    edges.append((bottom[i], top[i]))  # 竖边

# 3. 对每条边采样
n_sample = 8  # 每条边采样点数
ts = np.linspace(0, 1, n_sample)

for A, B in edges:
    for t in ts:
        p = (1 - t) * A + t * B

# 4. 收集所有采样点
points = []
for A, B in edges:
    for t in ts:
        p = (1 - t) * A + t * B
        points.append(p)
points = np.array(points)          # shape: (M, 3)

# 5. 绘制 3D 散点图
from mpl_toolkits.mplot3d import Axes3D   # 必须导入，哪怕看似没用到
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           s=15, c='tab:blue', marker='o')

# 6. 美化
ax.set_box_aspect([1, 1, 1.2])          # 保持比例
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Discrete Cylinder Point Cloud')
ax.view_init(elev=20, azim=-45)
ax.tick_params(axis='x', labelsize=8)
ax.xaxis.pane.fill = False
plt.tight_layout()
plt.show()

