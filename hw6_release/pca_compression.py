# http://www.dsc.ufcg.edu.br/~hmg/disciplinas/posgraduacao/rn-copin-2014.3/material/SignalProcPCA.pdf
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import io


img = io.imread('pitbull.jpg', as_gray=True)
img =  img_as_float(img)
mu = np.mean(img, axis=0, keepdims=True)
img_b = img - mu
U, S, VT = np.linalg.svd(img_b)
variances = S**2
W = VT.T
n_pc = 10
W_project = W[:, :n_pc]
Y_project = img_b.dot(W_project)
img_recovery = Y_project.dot(W_project.T) + mu
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(img_recovery, cmap='gray')
plt.show()
