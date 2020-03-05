import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n = 1000
t = np.linspace(-2, 2., num=n)
x = np.outer(np.linspace(-2, 2., num=n), np.ones(n))
y = np.outer(np.ones(n), np.linspace(-2, 2., num=n))

f = x * y - 0.5 * x**2 - y

plt.figure(1)
# plt.contour(x, y, f, levels=np.linspace(-1, 1, 10))
cs = plt.contour(x, y, f, levels=np.linspace(-4.5, 4.5, 10))
# cs = plt.contour(x, y, f)
plt.clabel(cs, inline=True, fontsize=10)
plt.plot(t, t**2, color='black')
plt.plot(t, t * 0, color='orange')
plt.plot(t * 0, t, color='orange')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()