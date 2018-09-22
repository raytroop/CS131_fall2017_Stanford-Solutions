import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.draw import ellipse
from skimage.transform import rotate

image = np.zeros((600, 600))
rr, cc = ellipse(300, 350, 100, 220)
image[rr, cc] = 1
image = rotate(image, angle=15, order=0)

label_img = measure.label(input=image)
print(label_img.shape, label_img.max(), label_img.min(), np.unique(label_img))
print(np.sum(image - label_img))    #
regions = measure.regionprops(label_image=label_img)
print(regions)
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)  # pylint: disable=E1101
