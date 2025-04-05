import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread("data/insight23.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("data/insight25.png", cv2.IMREAD_GRAYSCALE)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

flow = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 5, 3, 5, 1.2, 0)

mask = np.full_like(img0, False, dtype=bool)
for i in range(0, img0.shape[0], 20):
    for j in range(0, img0.shape[1], 20):
        mask[i, j] = True

X, Y = np.meshgrid(np.arange(0, img0.shape[1]), np.arange(0, img0.shape[0]))
plt.imshow(img0, cmap='gray')
plt.quiver(X[mask], Y[mask], flow[mask,0], flow[mask,1], color='red', width=0.01)
plt.show()