import numpy as np


def mavg_filter(img, pad=1, kind='max'):
    """
    For now, only support gray img

    Args:
        img: numpy array (height, width)
    Returns:
        img_filtered: numpy array (height, width)
    """
    kernel_supported = {'max': np.amax, 'mean': np.mean, 'min': np.amin, 'median': np.median}
    kernel=  kernel_supported[kind]
    h, w = img.shape
    # pad pixel around img
    img_padded = np.zeros((h+2*pad, w+2*pad), dtype=np.float64)
    img_padded[pad:-pad, pad:-pad] = img
    # plt.imshow(img_padded, cmap='gray')
    img_out = np.empty_like(img, dtype=np.float64)
    for i in range(pad, h+pad):
        for j in range(pad, w+pad):
            img_out[i-pad, j-pad] = kernel(img_padded[i-pad:i+pad, j-pad:j+pad])
    # print(img_out.dtype)
    return img_out

if __name__ == '__main__':
    from skimage import io
    from skimage import color
    import matplotlib.pyplot as plt
    # from skimage import filters
    from skimage.util import random_noise
    img = io.imread('p2179382362.png')  # output uint8 [0, 255]
    img_gray = color.rgb2grey(img)      # output float64 [0, 1]
    # img_blur = filters.gaussian(img_gray, sigma=5) 
    img_noise = random_noise(img_gray, mode='pepper')
    img_filtered = mavg_filter(img_noise, pad=2, kind='mean')

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    for ax_i, img_i in zip(ax.flatten(), [img_noise, img_filtered]):
        ax_i.imshow(img_i, cmap="gray")

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    