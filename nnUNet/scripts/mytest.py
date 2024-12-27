import numpy as np
import matplotlib.pyplot as plt

# 定义 x 和 y
x = np.linspace(0, 200, 1000)  # 生成从0到200的1000个点
y = np.where(x <= 150, 1 - x / 150, 0)  # 当 x <= 150 时，y = 1 - x / 150；否则 y = 0

# 绘制图形
plt.plot(x, y, label="y = 1 - x / 150 for x <= 150, y = 0 for x > 150")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the function')
plt.axvline(150, color='r', linestyle='--', label='x = 150')  # 在x = 150处添加垂直线
plt.legend()
plt.grid(True)
plt.show()
