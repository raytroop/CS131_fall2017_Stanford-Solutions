import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 1000)
u, s, vt = np.linalg.svd(X)

plt.scatter(X[:, 0], X[:, 1])
Xb = X - np.mean(X, axis=0, keepdims=True)
cov_mx = np.dot(Xb.T, Xb) / (X.shape[0] - 1)
eigenValues, eigenVectors = np.linalg.eig(cov_mx)   # outs are not necessarily ordered. 
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print('eigenvalue:', eigenValues)
print('eigenvector:', eigenVectors)

plt.scatter(X[:, 0], X[:, 1])
x_p = 2.5
y_p = eigenVectors[1, 0] * x_p / eigenVectors[0, 0]
plt.plot([-x_p, x_p], [-y_p, y_p], 
        'r', label='1st pca', linewidth=3.0)

x_p = 2
y_p = eigenVectors[1, 1] * x_p / eigenVectors[0, 1]
plt.plot([-x_p, x_p], [-y_p, y_p], 
        '--g', label='2nd pca', linewidth=3.0)
plt.legend(loc = 'best', fontsize= 20)
plt.axis('equal')
plt.tight_layout()
plt.show()
