import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordinates of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    pixel = np.column_stack((pixels, np.ones((pixels.shape[0], 1), dtype=pixels.dtype))) 
    pixel_calibrated = pixel @ np.linalg.inv(K).T

    lamb = -t_wc[2] / (R_wc[2] @ pixel_calibrated.T)  

    Pw = (lamb[:, np.newaxis] * (R_wc @ pixel_calibrated.T).T) + t_wc

    return Pw
