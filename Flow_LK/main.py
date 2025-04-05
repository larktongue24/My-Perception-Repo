import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from compute_grad import compute_Ix, compute_Iy, compute_It
from compute_flow import flow_lk
from vis_flow import plot_flow
# from depth import depth
from epipole import epipole


# DO NOT CHANGE THIS DEFINITION
K = np.array([
    [1118, 0, 357],
    [0, 1121, 268],
    [0, 0, 1]
])

if __name__ == "__main__":

    # TODO: Edit this parameter to see different results.
    threshmin = 30

    # load images
    data_folder = "data"
    images = [
        mpimg.imread(os.path.join(data_folder, "insight{}.png".format(i))) * 255
        for i in range(20, 27)
    ]
    images = np.stack(images, axis=-1)
    print(f'Loaded images with shape: {images.shape}')

    # vid = cv2.VideoWriter('flow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2, (images.shape[1], images.shape[0]), True)
    # for i in range(images.shape[-1]):
    #     vid.write(images[..., i].astype(np.uint8)[:, :, None].repeat(3, axis=-1))
    # vid.release()

    # find gradients
    Ix = compute_Ix(images)
    Iy = compute_Iy(images)
    It = compute_It(images)
    
    print(f'Computed gradients with shape: {Ix.shape}, {Iy.shape}, {It.shape}')

    # only take the image in the middle for flow computations
    valid_idx = 3
    flow, confidence = flow_lk(Ix[..., valid_idx], Iy[..., valid_idx], It[..., valid_idx])
    
    print(f'Computed LK flow with shape: {flow.shape}, {confidence.shape}')

    # visualize flow
    plt.figure()
    plot_flow(images[..., valid_idx], flow, confidence, threshmin=threshmin)
    plt.title(f'LK optical flow for threshold: {threshmin}')
    plt.savefig(f"flow_{threshmin}.png")
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()

    # compute and visualize epipole
    print('Computing epipole - this may take a while...')
    
    
    ep = epipole(flow[..., 0], flow[..., 1], confidence, threshmin, num_iterations=1000)
    ep = ep / ep[2]
    print("EPIPOLE: ", ep)

    # visualize flow and epipole

    plot_flow(images[..., valid_idx], flow, confidence, threshmin=threshmin)
    plt.figure()
    plt.scatter(ep[0] + flow.shape[0]//2, ep[1] + flow.shape[1]//2, c='g', s=20, marker='*')
    plt.title(f'Epipole for threshold: {threshmin}')
    plt.savefig(f"epipole_{threshmin}.png")
    plt.show()








