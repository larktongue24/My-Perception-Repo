import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)
        Pw = R@Pc + t
    """

    Pc = np.column_stack((Pc, np.ones((Pc.shape[0], 1), dtype=Pc.dtype))) 
    K_inv = np.linalg.inv(K)
    Pc_calibrated = (K_inv @ Pc.T).T  

    a = np.linalg.norm(Pw[1] - Pw[2])
    b = np.linalg.norm(Pw[0] - Pw[2])
    c = np.linalg.norm(Pw[0] - Pw[1])

    j1, j2, j3 = P3P_Unit_Vectors(Pc_calibrated[0], Pc_calibrated[1], Pc_calibrated[2])

    cos_alpha = np.dot(j2, j3)
    cos_beta = np.dot(j1, j3)
    cos_gamma = np.dot(j1, j2)

    coefficient = P3P_coefficients(a, b, c, cos_alpha, cos_beta, cos_gamma)
    root = np.roots(coefficient)
    v = root[np.isclose(root.imag, 0)].real
    u = ((-1 + ((a**2 - c**2) / b**2)) * v**2 - 2 * ((a**2 - c**2) / b**2) * cos_beta * v + 1 
            + ((a**2 - c**2) / b**2)) / (2 * (cos_gamma - v * cos_alpha))
    
    d1 = np.sqrt(a**2 / (u**2 + v**2 - 2 * u * v * cos_alpha))
    d2, d3 = u * d1, v * d1

    Pc_cal_norm = Pc_calibrated[:3, :] / np.linalg.norm(Pc_calibrated[:3, :], axis=1, keepdims=True)
    Pc_set = np.array([
        Pc_cal_norm[:3, :] * np.array([d1_val, d2_val, d3_val])[:, np.newaxis]
        for d1_val, d2_val, d3_val in zip(d1, d2, d3)
    ])

    min_error = float('inf')
    best_R, best_t = None, None
    
    for candidate in Pc_set:
        R, t = Procrustes(candidate, Pw[:3, :]) 
        p = K @ R.T @ (Pw[3] - t)
        projected = p / p[-1]

        error = np.linalg.norm(projected[:2] - Pc[3, :2])

        if error < min_error:
            min_error = error
            best_R = R
            best_t = t

    return best_R, best_t


def P3P_Unit_Vectors(q1, q2, q3):
    """
    Determine the unit vectors given q1, q2, q3

    Input:
        q1, q2, q3: camera coordinates
    Returns:
        j1, j2, j3: unit vectors for camera coordinates
        """

    q_vectors = np.array([q1, q2, q3])
    scale_factor = np.linalg.norm(q_vectors, axis=1, keepdims=True)
    unit_vectors = q_vectors / scale_factor

    return unit_vectors[0], unit_vectors[1], unit_vectors[2]


def P3P_coefficients(a, b, c, cos_alpha, cos_beta, cos_gamma):
    """
    Solve Perspective-3-Point coefficients, given side lengths and angles

    Input:
        a: side length between p2 and p3
        b: side length between p1 and p3
        c: side length between p1 and p2
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)
        Pw = R@Pc + t
    """

    a4 = ((a**2 - c**2) / b**2 - 1)**2 - (4 * c**2 / b**2) * cos_alpha**2
    a3 = 4 * (((a**2 - c**2) / b**2) * (1 - ((a**2 - c**2) / b**2)) * cos_beta
            - (1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma + 2 * (c**2 / b**2) * cos_alpha**2 * cos_beta)
    a2 = 2 * (((a**2 - c**2) / b**2)**2 - 1 + 2 * ((a**2 - c**2) / b**2)**2 * cos_beta**2
            + 2 * ((b**2 - c**2) / b**2) * cos_alpha**2 - 4 * ((a**2 + c**2) / b**2) * cos_alpha * cos_beta * cos_gamma
            + 2 * ((b**2 - a**2) / b**2) * cos_gamma**2)
    a1 = 4 * (-((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * cos_beta
            + 2 * (a**2 / b**2) * cos_gamma**2 * cos_beta - (1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma)
    a0 = (1 + ((a**2 - c**2) / b**2))**2 - 4 * (a**2 / b**2) * cos_gamma**2

    return np.array([a4, a3, a2, a1, a0], dtype=float)


def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    H = Y_centered.T @ X_centered

    U, _, Vt = np.linalg.svd(H)

    R = U @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ Vt

    t = Y_mean - R @ X_mean

    return R, t