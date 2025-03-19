from est_homography import est_homography
import numpy as np


def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
    """

    ##### STUDENT CODE START #####
    H = est_homography(Pw[:, :2], Pc)  
    H_normalized = np.linalg.inv(K) @ H  

    r1 = H_normalized[:, 0]
    r2 = H_normalized[:, 1]
    t = H_normalized[:, 2]

    R_approx = np.column_stack((r1, r2, np.cross(r1, r2)))  
    U, _, Vt = np.linalg.svd(R_approx) 
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt  
    ##### STUDENT CODE END #####

    return R.T, -R.T @ (t / np.linalg.norm(r1))

if __name__ == "__main__":
    K = np.array([[823.8, 0.0, 304.8],
                [0.0, 822.8, 236.3],
                [0.0, 0.0, 1.0]])