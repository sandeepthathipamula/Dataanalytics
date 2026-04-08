import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(10)
y = np.random.randint(1, 100, size=10)

# Bar plot
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.bar(x, y)
plt.title('Bar Plot')

# Column plot
plt.subplot(2, 3, 2)
plt.plot(x, y, 'r')
plt.title('Column Plot')

# Line plot
plt.subplot(2, 3, 3)
plt.plot(x, y, 'g')
plt.title('Line Plot')

# Scatter plot
plt.subplot(2, 3, 4)
plt.scatter(x, y)
plt.title('Scatter Plot')

# 3D Cubes
ax = plt.subplot(2, 3, 5, projection='3d')
x = np.random.normal(size=500)
y = np.random.normal(size=500)
z = np.random.normal(size=500)
ax.scatter(x, y, z)
plt.title('3D Cubes')

plt.tight_layout()
plt.show()
