import numpy as np

def pose_candidates_from_E(E):
    transform_candidates = []
    ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
    """ 

    Computes four possible camera poses (rotation and translation) from the essential matrix.

    Inputs:
    - E: 3x3 Matrix, Essential Matrix

    Outputs:
    - transform_candidates: List of 4 dictionaries, each containing:
        - "T": 3x1 Vector, Translation vector
        - "R": 3x3 Matrix, Rotation matrix
    """

    ##### STUDENT CODE START #####
    U, _, Vt = np.linalg.svd(E)
    
    Rz_plus = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Rz_minus = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    T = U[:, -1]
    
    R1 = U @ Rz_plus.T @ Vt
    R2 = U @ Rz_minus.T @ Vt

    transform_candidates.append({"T": T, "R": R1})
    transform_candidates.append({"T": T, "R": R2})
    transform_candidates.append({"T": -T, "R": R1})
    transform_candidates.append({"T": -T, "R": R2})
    ##### STUDENT CODE END #####

    return transform_candidates