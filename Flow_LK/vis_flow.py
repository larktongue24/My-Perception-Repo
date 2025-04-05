import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    ### PLEASE DO NOT MODIFY ###
    
    H, W = image.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    # Filter flow vectors based on confidence threshold
    valid_indices = np.where(confidence > threshmin)
    u = flow_image[:,:,0][valid_indices]
    v = flow_image[:,:,1][valid_indices]
    X = X[valid_indices]
    Y = Y[valid_indices]
    
    # Plot the flow field
    plt.imshow(image, cmap='gray')
    plt.quiver(X, Y, u, -v, color='red')
    
    return



    

