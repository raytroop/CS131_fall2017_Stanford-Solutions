from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

sigma1 = 1
sigma2 = 2

x = np.linspace(-3*sigma2, 3*sigma2, 600)

xx, yy = np.meshgrid(x[np.newaxis, :], x[:, np.newaxis])

def gaussian2d(sigma, x, y):
    return np.exp(-(x**2 + y**2)/2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

zz1 = gaussian2d(sigma1, xx, yy)
zz2 = gaussian2d(sigma2, xx, yy)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(xx, yy, zz2-zz1, color='b')

plt.show()