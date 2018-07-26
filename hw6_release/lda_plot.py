import numpy as np
import matplotlib.pyplot as plt


mean1 = [1, 1]
cov1 = [[0.3, 0.1], [0.1, 0.3]]
X1 = np.random.multivariate_normal(mean1, cov1, 500)

mean2 = [-1, -1]
cov2 = [[0.3, 0.1], [0.1, 0.3]]
X2 = np.random.multivariate_normal(mean2, cov2, 500)

plt.scatter(X1[:, 0], X1[:, 1], marker='s', s=40, label='X1')
plt.scatter(X2[:, 0], X2[:, 1], marker='^', s=40, label='X2')


mu1 = np.mean(X1, axis=0, keepdims=True)
mu2 = np.mean(X2, axis=0, keepdims=True)

SIGMA1 = np.cov(X1.T)
SIGMA2 = np.cov(X2.T)

Sb = np.dot((mu1 - mu2).T, (mu1 - mu2))
Sw = SIGMA1 + SIGMA2

W_opt = np.linalg.inv(Sw).dot((mu1 - mu2).T)
print(W_opt)
x_opt = W_opt[0, 0]
y_opt = W_opt[1, 0]

x_end = 4
y_end = y_opt * x_end / x_opt
plt.plot([-x_end, x_end], [-y_end, y_end], label='lda', c='r', linewidth=2)
plt.legend(loc='best', fontsize=20)
# plt.axis('equal')
plt.show()