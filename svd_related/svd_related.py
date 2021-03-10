#############################################################
# Author: Chng Soon Siang                                   #
# Date: 03 OCT 2021                                         #
#                                                           #
# Compilation of Utility Functions relating to image        #
# reconstruction using truncated SVD                        #
#                                                           #
# !Codes which are not written by me are attributed to      #
# others appropriately!                                     #
#############################################################

import cv2
import numpy as np
import os
import matplotlib.image
import matplotlib.pyplot as plt
from optht import optht

# Define randomized SVD function
def rSVD(X, r=None, q=1, p=5):
    """
    >>> From www.databookuw.com/ <<<
    Given a matrix X, returns its SVD using randomized SVD. 

    Inputs
    q: Power Iterations, default 1
    p: Oversampling Parameter, default 5
    r: Target Rank, default min(X.shape)

    Returns
    U, S, VT: with obvious meanings from the literature
    """
    if r is None:
        r = min(X.shape)

    # Step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny, r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY

    return U, S, VT


def rSVD_3Darray(X, *args):
    """
    Given a matrix 3D-array, X, of shape (m, n, c), returns its SVD using randomized SVD. 

    Inputs
    q: Power Iterations, default 1
    p: Oversampling Parameter, default 5
    r: Target Rank, default min(X.shape)

    Returns
    U, S, VT: with obvious meanings from the literature
    """
    m, n, c = X.shape
    s_size = min(m, n)

    A, B, C = rSVD(X[..., 0], *args)

    for i in range(1, X.shape[-1]):
        U, S, VT = rSVD(X[..., i], *args)
        A = np.dstack([A, U])
        B = np.column_stack([B, S])
        C = np.dstack([C, VT])

    return A, B, C


def reconstruct_from_truncated_rSVD_2Darray(*SVDorX,  r=None, **kwargs):
    """
    Given input 2D-array, X, reconstructs the 2D-array using the truncated SVD of X.

    Inputs
    -> *SVDorX, must either of length 1 or 4. i.e. either X or (X, S, U, VT) where S, U, VT is the SVD of X.
    X <2d-array>: Input 2D-array    
    -> r <int>: The rank of the truncated 2D-array. Default is None and its value is then determined using optht().
    -> **kwargs: Other keywords parameters for rSVD() and optht()

    Output
    -> Reconstruction the 2D-array using the truncated SVD of X, it has the same shape as X.    
    """
    # Parameters manipulation. Checks if r is int.
    if r is not None:
        if not isinstance(r, int):
            raise TypeError("r should be int.")

    # Parameters manipulation. Checks either X or (X, U, S, VT) is given as input(s).
    if len(SVDorX) == 4:
        X, U, S, VT = SVDorX
    elif len(SVDorX) == 1:
        X = SVDorX[0]
        # If only X is given, calculates its SVD.
        U, S, VT = rSVD(X, **kwargs)
    else:
        return

    # Computes the hard threshold and sets the optimal rank, r, to be that.
    if r is None:
        r = optht(X, sv=S, trace=None, **kwargs)

    # Returns the reconstructed matrix using the truncated SVD at the optimal rank, r.
    return U[:, :(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1), :]


def reconstruct_from_truncated_rSVD_3Darray(X,  r=None, **kwargs):
    """
    Given input a 3D-array, X, reconstructs the 3D-array using the truncated SVD of X.

    Inputs
    -> *SVDorX, must either of length 1 or 4: i.e. either X or (X, S, U, VT) where S, U, VT is the SVD of X.
    X <3d-array>: Input 3D-array of shape either (channels, height, width) or (height, width, channels). Number
    of channels must be 3 (representing the RGB channels).
    -> r <int>: The rank of the truncated 3D-array. Default is None and its value is then determined using optht().
    -> **kwargs: Other keywords parameters for rSVD() and optht()

    Output
    -> X_result <3d-array>: Reconstruction of the 3D-array using the truncated SVD of X, it has the same shape as X.    
    """
    # Strategy
    # Determine the position of the channel axis.
    # Using np.moveaxis, move the channel axis to the first position if it is not in the first position.
    # Using map, apply reconstruct_from_truncated_rSVD_2Darray() to the image layer of each channel.
    # Move back the channel axis if it is not in the first position.

    if r is not None:
        if not isinstance(r, int):
            raise TypeError("r should be int.")

    if len(X.shape) == 2:
        X_result = reconstruct_from_truncated_rSVD_2Darray(X,  r=r, **kwargs)

    elif X.shape[-1] == 3:

        X_swapped = np.moveaxis(X, -1, 0)

        def f(x): return reconstruct_from_truncated_rSVD_2Darray(
            x,  r=r, **kwargs)
        X_result = np.moveaxis(np.array([*map(f, X_swapped)]), 0, -1)

    elif X.shape[0] == 3:

        def f(x): return reconstruct_from_truncated_rSVD_2Darray(
            x,  r=r, **kwargs)
        X_result = np.array([*map(f, X)])

    return X_result


def show_Orig_Reconstructed_Imgs(*params, **kwargs):
    """
    Accepts an arbitrary number of iterables of images or an arbitrary number of images or
    a combination of both as positional arguments, returns a subplots of 2 columns, 
    such that, for each row, the first column shows the original image 
    and the second column shows the reconstructed image using truncated SVD.

    Inputs
    *params <iterables>: Iterables of images.
    **kwargs: Keyword arguments for plt.subplots, plt.text, 
    plt.reconstruct_from_truncated_rSVD2, scale_between_0_255 and plt.imshow

    Output
    A subplots of 2 columns, such that, for each row, the first column shows the original image 
    and the second column shows the reconstructed image using truncated SVD.
    """
    images = []
    ### Make all iterables into a single list of np.ndarray ###
    for i in params:
        if not isinstance(i, np.ndarray):
            for j in i:
                images.append(j)
        else:
            images.append(i)
    ### Set up the subplots ###
    n_rows = len(images)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows), **kwargs)
    ax = axes.flat

    j = 0
    i = 0
    for image in images:
        # For each image in images, 1. show the original image, 2. calculate the reconstructed image,
        # 3. show the reconstructed image.
        ax[j].imshow(image, **kwargs)  # 1
        ax[j].text(
            0.02, 1.04, f'Image {i}: Original Image', transform=ax[j].transAxes, **kwargs)
        reconstructed_image = reconstruct_from_truncated_rSVD_3Darray(
            image, **kwargs)  # 2
        reconstructed_image = scale_between_0_255(
            reconstructed_image, **kwargs)  # 3
        ax[j+1].imshow(reconstructed_image, **kwargs)
        ax[j+1].text(0.02, 1.04, f'Image {i}: Reconstructed from SVD-Image',
                     transform=ax[j+1].transAxes, **kwargs)
        j += 2
        i += 1


def scale_between_0_255(image):
    """
    Utility function to scale all pixel values of an integer to be between 0 and 255.
    Returns the scaled image.
    """
    image = 255*(image-image.min())/(image.max()-image.min())
    image = image.astype('uint8')
    return image