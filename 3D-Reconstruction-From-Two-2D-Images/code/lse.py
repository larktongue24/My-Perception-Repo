import numpy as np

def least_squares_estimation(X1, X2):
	""" 
	inputs:
	X1: Nx3 Matrix Calibrated Point
	X2: Nx3 Matrix Calibrated Point
	X1 ~ X2

	rets:
	E: Essential Matrix using 8 point algorithm
	"""
	n = X1.shape[0]
	A = np.zeros((n, 9))

	# Write A (n*9)
	for i in range(n):
		x1, y1, z1 = X1[i]
		x2, y2, z2 = X2[i]
		A[i] = [x1 * x2, x1 * y2, x1 * z2,
				y1 * x2, y1 * y2, y1 * z2,
				z1 * x2, z1 * y2, z1 * z2]
		
	# Solving for E'
	U, S, Vt = np.linalg.svd(A)
	E = Vt[-1].reshape(3, 3).T

	# Redecomposing E'
	U, S, Vt = np.linalg.svd(E)
	E = U @ np.diag([1, 1, 0]) @ Vt

	return E