import numpy as np

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    Find the Lucas-Kanade optical flow on a single square patch.
    The patch is centered at (y, x), therefore it generally extends
    from x-size//2 to x+size//2 (inclusive), same for y, EXCEPT when
    exceeding image boundaries!
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
        - x: SECOND coordinate of patch center - integer in range [0, W-1]
        - y: FIRST coordinate of patch center - integer in range [0, H-1]
        - size: optional parameter to change the side length of the patch in pixels
    
    Outputs:
        - flow: flow estimate for this patch - shape: (2,)
        - conf: confidence of the flow estimates - scalar
    """

    ### STUDENT CODE START ###

    H, W = Ix.shape
    d = size // 2
    y_start = max(y - d, 0)
    y_end   = min(y + d + 1, H)
    x_start = max(x - d, 0)
    x_end   = min(x + d + 1, W)
    
    patch_Ix = Ix[y_start : y_end, x_start : x_end].reshape(-1)
    patch_Iy = Iy[y_start : y_end, x_start : x_end].reshape(-1)
    patch_It = It[y_start : y_end, x_start : x_end].reshape(-1)

    A = np.stack([patch_Ix, patch_Iy], axis=1)  
    b = -patch_It
    flow, _, _, s = np.linalg.lstsq(A, b, rcond=-1)
    conf = s[-1] if len(s) > 1 else 0.0

    ### STUDENT CODE END ###
    return flow.ravel(), conf


def flow_lk(Ix, Iy, It, size=5):
    """
    Compute the Lucas-Kanade flow for all patches of an image.
    To do this, iteratively call flow_lk_patch for all possible patches.
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative
    Outputs:
        - image_flow: flow estimate for each patch - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
    """

    ### STUDENT CODE START ###
    
    H, W = Ix.shape
    image_flow = np.zeros((H, W, 2))
    confidence = np.zeros((H, W))
    
    for y in range(H):
        for x in range(W):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x] = flow      
            confidence[y, x] = conf     

    ### STUDENT CODE END ###
    return image_flow, confidence

    

