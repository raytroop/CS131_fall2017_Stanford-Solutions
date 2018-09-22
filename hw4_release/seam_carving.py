import numpy as np
from skimage import color


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: you can use np.gradient here
    The gradient is computed using second order accurate central differences in the interior points
    and either first or second order accurate one-sides (forward or backwards) differences
    at the boundaries. The returned gradient hence has the same shape as the input array.

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))

    ### YOUR CODE HERE
    img_gray = color.rgb2gray(image)
    dy, dx = np.gradient(img_gray)
    out = np.abs(dy) + np.abs(dx)
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    for i in range(1, H):
        left = np.concatenate(([np.Inf], cost[i-1, :W-1]))
        upper = cost[i-1, :]
        right = np.concatenate((cost[i-1, 1:], [np.Inf]))
        cost_i = np.c_[left, upper, right]
        idx_sort = np.argsort(cost_i, axis=1)
        paths[i] = idx_sort[:, 0] - 1
        cost[i] = energy[i] + np.min(cost_i, axis=1)
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H-1, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    seam = np.zeros(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    ### YOUR CODE HERE
    for i in range(H-1, 0, -1):
        seam[i-1] = seam[i] + paths[i, seam[i]]
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    idx = np.c_[np.arange(H), seam]
    idx_flatten = (idx[:, 0] * W + idx[:, 1]) * C
    idx_collect = []
    for i in range(C):
        idx_collect.append(idx_flatten + i)
    idx_collect = np.stack(idx_collect).T.flatten()
    out = np.delete(image, idx_collect)
    out = np.reshape(out, (H, W-1, C))
    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    energy = efunc(out)
    vcost, vpaths = cfunc(out, energy)  # Here `out` is dummy arg
    for _ in range(W - size):
        seam = backtrack_seam(vpaths, np.argmin(vcost[-1]))
        out = remove_seam(out, seam)
        energy = efunc(out)
        vcost, vpaths = compute_cost(out, energy)   # workaround hardcoding
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i, j in enumerate(seam):
        out[i, :j, :] = image[i, :j, :]
        out[i, j+1:, :] = image[i, j:, :]
    out[range(H), seam, :] = image[range(H), seam, :]
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    energy = efunc(out)
    vcost, vpaths = cfunc(out, energy)  # Here `out` is dummy arg
    for _ in range(size - W):
        seam = backtrack_seam(vpaths, np.argmin(vcost[-1]))
        out = duplicate_seam(out, seam)
        energy = efunc(out)
        vcost, vpaths = compute_cost(out, energy)   # workaround hardcoding
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)
    ### YOUR CODE HERE
    seams = find_seams(out, size - W)

    seam = np.full(shape=H, fill_value=np.inf)
    for i in range(1, size - W + 1):
        _, seam_i = np.nonzero(seams == i)
        seam_new = np.copy(seam_i)
        seam_i[seam_i > seam] = seam_i[seam_i > seam] + i - 1
        out = duplicate_seam(out, seam_i)
        seam = seam_new
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)
    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    # _, gradient_h = np.gradient(energy)
    # gradient_h[:, 1:-1] = 2 * gradient_h[:, 1:-1]
    # gradient_h[:, [0, -1]]= energy[:, [0, -1]]
    # gradient_h = np.abs(gradient_h)
    # for i in range(1, H):
    #     cl_i = np.concatenate((np.array([np.inf]), np.abs(energy[i, :-1] - energy[i-1, 1:])))
    #     cv_i = 0.0
    #     cr_i = np.concatenate((np.abs(energy[i, 1:] - energy[i-1, :-1]), np.array([np.inf])))

    #     cl_i = cl_i + gradient_h[i] + np.concatenate((np.array([np.inf]), cost[i-1, :-1]))
    #     cv_i = cv_i + gradient_h[i] + cost[i-1, :]
    #     cr_i = cr_i + gradient_h[i] + np.concatenate((cost[i-1, 1:], np.array([np.inf])))
    #     cost_i = np.c_[cl_i, cv_i, cr_i]
    #     idx_sort = np.argsort(cost_i, axis=1)
    #     paths[i] = idx_sort[:, 0] - 1
    #     cost[i] = np.min(cost_i, axis=1)

    # https://github.com/mikucy/CS131/blob/master/hw4_release/seam_carving.py
    for i in range(1, H):
        m1 = np.insert(image[i, 0:W-1], 0, 0, axis=0)
        m2 = np.insert(image[i, 1:W], W-1, 0, axis=0)
        m3 = image[i-1]
        c_v = np.abs(m1 - m2)
        c_v[[0, -1]] = 0
        c_l = c_v + np.abs(m3 - m1)
        c_r = c_v + np.abs(m3 - m2)
        c_l[0] = 0
        c_r[-1] = 0
        i1 = np.insert(cost[i-1, 0:W-1], 0, 1e10, axis=0)
        i2 = cost[i-1]
        i3 = np.insert(cost[i-1, 1:W], W-1, 1e10, axis=0)
        C = np.r_[i1 + c_l, i2 + c_v, i3 + c_r].reshape(3, -1)  # pylint: disable=E1121, E1111
        cost[i] = energy[i] + np.min(C, axis=0)
        paths[i] = np.argmin(C, axis=0) - 1
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    out = np.copy(image)

    ### YOUR CODE HERE
    # Refer to @lgqfhwy
    # https://github.com/mikucy/CS131/issues/3#issue-328719868
    from skimage import measure
    label_image = measure.label(mask)
    regions = measure.regionprops(label_image)
    region = regions[0]
    if len(regions) != 1:
        print("Maybe two objects to remove?")
        # Find the biggest area of region
        for i in regions:
            if i.area > region.area:
                region = i
    transposeImage = False
    # Bounding box (min_row, min_col, max_row, max_col)
    if region.bbox[2] - region.bbox[0] < region.bbox[3] - region.bbox[1]:
        out = np.transpose(out, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))
        transposeImage = True
    count = 0   # count time for all iteration
    # def rowcol(mat):
    #     cols = np.sum(np.sum(mat, axis=0) > 0)
    #     rows = np.sum(np.sum(mat, axis=1) > 0)
    #     return rows, cols
    # print('row {} cols {}'.format(*(rowcol(mask))))
    while not np.all(mask == 0):
        # print(count, out.shape)
        energy_image = energy_function(out)
        energy_image = energy_image + energy_image * mask * (-100)
        vcost, vpaths = compute_forward_cost(out, energy_image)
        end = np.argmin(vcost[-1])
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(out, seam)
        # print(count, out.shape)
        mask = remove_seam(mask, seam)
        # print('row {} cols {}'.format(*(rowcol(mask))))
        # print(out.shape, mask.shape)
        count += 1
    #print("count = ", count)
    out = enlarge(out, out.shape[1] + count)
    if transposeImage:
        out = np.transpose(out, (1, 0, 2))
    ### END YOUR CODE
    return out
