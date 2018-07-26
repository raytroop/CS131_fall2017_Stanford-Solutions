import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    kernel = kernel[::-1, ::-1]
    padh = Hk // 2
    padw = Wk // 2
    ### YOUR CODE HERE
    img_pad = np.zeros((Hi + 2*padh, Wi + 2*padw))
    img_pad[padh : Hi + padh, padw : Wi + padw] = image
    for i in range(Hi):
        for j in range(Wi):
            val = 0.0
            for Ki in range(i, i+Hk):
                for Kj in range(j, j+Wk):
                    val += img_pad[Ki, Kj] * kernel[Ki-i, Kj-j]
            out[i, j] = val
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height : H + pad_height, pad_width : W+pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = kernel[::-1, ::-1]
    padh = Hk // 2
    padw = Wk // 2
    img_pad = zero_pad(image, padh, padw)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = (img_pad[i:i+Hk, j:j+Wk] * kernel).sum()
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padh = Hk // 2
    padw = Wk // 2
    img_pad = zero_pad(f, padh, padw)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = (img_pad[i:i+Hk, j:j+Wk] * g).sum()
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g -= np.mean(g)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padh = Hk // 2
    padw = Wk // 2
    img_pad = zero_pad(f, padh, padw)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = (img_pad[i:i+Hk, j:j+Wk] * g).sum()
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = (g - np.mean(g))/np.std(g)
    f = (f - np.mean(f))/np.std(f)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padh = Hk // 2
    padw = Wk // 2
    img_pad = zero_pad(f, padh, padw)
    for i in range(Hi):
        for j in range(Wi):
            img_patch = img_pad[i:i+Hk, j:j+Wk]
            img_normalize = (img_patch - np.mean(img_patch)) / np.std(img_patch)
            out[i, j] = (img_normalize * g).sum()
    ### END YOUR CODE

    return out
