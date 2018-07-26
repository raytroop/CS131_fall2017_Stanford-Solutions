from skimage import io
import numpy as np
import matplotlib.pyplot as plt

img = io.imread('pitbull.jpg', as_gray=True)
u, s, vt = np.linalg.svd(img)
ss = s**2
sscum = np.cumsum(ss)
sscum_normalize = sscum / np.sum(ss)
fig1 = plt.figure(1)
plt.plot(np.arange(200) + 1, sscum_normalize[:200], '-*')
plt.show()

fig2 = plt.figure()
def ext(n):
    return u[:, :n].dot(np.diag(s[:n])).dot(vt[:n, :])

sig = [1, 5, 10, 15, 20, 30]
for i, n in enumerate(sig):
    plt.subplot(2, 3, i+1)
    img_re = ext(n)
    plt.imshow(img_re, cmap='gray')
    plt.title('svd:{}'.format(n))
    plt.xticks([])
    plt.yticks([])
plt.show()