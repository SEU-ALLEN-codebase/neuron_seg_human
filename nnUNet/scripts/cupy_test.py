import cupy as cp

# 确保 GPU 设备可用


x = cp.array([1, 2, 3])
y = cp.array([4, 5, 6])
z = x + y
print("Test array addition on GPU:", z)
