import numpy as np

def est_homography(X, Y):
    """ 
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out 
    what X and Y should be. 
    Input:
        X: 4x2 matrix of (x,y) coordinates 
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 transformation matrix s.t. Y ~ H*X
        
    """

    A = []
    for (x, y), (x_, y_) in zip(X, Y):
        A.append([x, y, 1, 0, 0, 0, -x*x_, -y*x_, -x_])
        A.append([0, 0, 0, x, y, 1, -x*y_, -y*y_, -y_])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)

    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    
    return H
