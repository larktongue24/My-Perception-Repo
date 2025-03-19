from lse import least_squares_estimation
import numpy as np
# import tqdm
# import scipy.spatial as ss

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        
        """
        E: ndarray of shape (3,3)
            Essential matrix.
        inliers: ndarray of shape (n,)
            Indices of inlier matches.
        """

        E = least_squares_estimation(X1[sample_indices, :], X2[sample_indices, :])
        inliers = list(sample_indices) 
        e3_cross = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        # Compare distances to the epipolars for each pairs in the test group
        for row in test_indices:
            d_x2x1 = (X2[row] @ E @ X1[row].T) ** 2 / np.linalg.norm(e3_cross @ E @ X1[row]) ** 2
            d_x1x2 = (X1[row] @ E.T @ X2[row].T) ** 2 / np.linalg.norm(e3_cross @ E.T @ X2[row]) ** 2
            if d_x2x1 + d_x1x2 < eps:
                inliers.append(row)
        inliers = np.array(inliers)
        
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers

# if __name__ == "__main__":
#     T = np.array([0.9, 2, 1.5])
#     Tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
#     Rot = ss.transform.Rotation.from_euler('xyz', [0.5, 0.3, 0.2]).as_matrix()
#     E = Tx.dot(Rot)
#     q = np.random.rand(9, 3)
#     q[:, 0] /= q[:, 2]
#     q[:, 1] /= q[:, 2]
#     q[:, 2] /= q[:, 2]
#     L = q.dot(E)
#     p = np.random.rand(9, 3)
#     p[:, 0] /= p[:, 2]
#     p[:, 1] /= p[:, 2]
#     p[:, 2] /= p[:, 2]
#     p[:, 0] = (-L[:, 2]-L[:, 1]*p[:, 1])/L[:, 0]
#     ransac_estimator(p, q)
