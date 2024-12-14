""" Visualisation tools to save gif files of model outputs. 
    Expected inputs should be torch.Tensor.
    Author: Keren Collins
    Date: 4/12/2024

    Modification record:
    N/A
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

# Alternate implementation.
# def animateRGB(imgTensor, filenameBase="test", anim=False, dpi = 100):
#     """
#     Visualise a designated snapshot of the grid specified by idx
#     Input in form (frames, channels, height, width)
#     """ 
#     if len(imgTensor.shape) < 4:
#         imgTensor.unsqueeze(0)

    
#     def update(imgIdx):
#         # We're only interested in the RGBalpha channels, and need numpy representation for plt
#         img = imgTensor[imgIdx].squeeze().permute(1, 2, 0)

#         plt.suptitle("Update " + str(imgIdx))

#         # Plot RGB channels
#         plt.subplot(1, 2, 1)
#         plt.imshow(img[:, :, 0:3].clip(0, 1).detach().numpy())
#         plt.title("RGB")

#         # Plot Alpha channel
#         plt.subplot(1, 2, 2)
#         plt.imshow(img[:, :, 3].clip(0, 1).detach().numpy())
#         plt.title("Alpha (alive/dead)")

#         print(f"Rendered frame {imgIdx}")

#     frames = []

#     for imgIdx in range(imgTensor.shape[0]):
#         fig = plt.figure()
#         plt.xticks([])
#         plt.yticks([])
#         update(imgIdx=imgIdx)
#         frames.append(figToRGBArray(fig, dpi))
#         plt.close(fig)
    
#     imgs = [Image.fromarray(img) for img in frames]
#     # duration is the number of milliseconds between frames; this is 40 frames per second
#     imgs[0].save(f"{filenameBase}.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)


# def figToRGBArray(fig, dpi):

#     io_buf = io.BytesIO()
#     fig.savefig(io_buf, format='raw', dpi=dpi)
#     io_buf.seek(0)
#     img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
#                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
#     io_buf.close()
#     return img_arr



def plotRGB(imgTensor, filenameBase="test", save = True):
    imgTensor = imgTensor.detach().cpu()
    fig = plt.figure()

    # Get canvases for Images
    print("plotRGB", imgTensor.shape)
    canvasRGB = plt.imshow((imgTensor[0].squeeze().permute(1, 2, 0))[:, :, 0:3].clip(0, 1).detach().numpy())
    plt.title("RGB")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filenameBase + ".png")

    plt.close(fig)


def animateRGB(imgTensor, filenameBase="test", alpha = True, save=True):
    """
    Visualise a designated snapshot of the grid specified by idx
    Input in form (channels, height, width)
    """
    imgTensor = imgTensor.detach().cpu()

    if len(imgTensor.shape) < 4:
        imgTensor.unsqueeze(0)

    fig = plt.figure()
    title = plt.suptitle("Update 0")

    if alpha:
        # Get canvases for Images
        plt.subplot(1, 2, 1)
        canvasRGB = plt.imshow((imgTensor[0].squeeze().permute(1, 2, 0))[:, :, 0:3].clip(0, 1).detach().numpy())
        plt.title("RGB")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        canvasAlpha = plt.imshow((imgTensor[0].squeeze().permute(1, 2, 0))[:, :, 3].clip(0, 1).detach().numpy())
        plt.title("Alpha (alive/dead)")
        plt.xticks([])
        plt.yticks([])
    else: 
        canvasRGB = plt.imshow((imgTensor[0].squeeze().permute(1, 2, 0))[:, :, 0:3].clip(0, 1).detach().numpy())
        plt.title("RGB")
        plt.xticks([])
        plt.yticks([])

    
    def update(imgIdx):

        # We're only interested in the RGBalpha channels, and need numpy representation for plt
        img = imgTensor[imgIdx].squeeze().permute(1, 2, 0)

        title.set_text("Update " + str(imgIdx))

        # Plot RGB channels
        canvasRGB.set_data(img[:, :, 0:3].clip(0, 1).detach().numpy())

        if alpha:
            # Plot Alpha channel
            canvasAlpha.set_data(img[:, :, 3].clip(0, 1).detach().numpy())

    ani = animation.FuncAnimation(fig, update, frames=len(imgTensor), repeat=False)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=15, metadata=dict(artist="Me"), bitrate=-1
    )
    ani.save(filenameBase + ".gif", writer=writer)

    plt.close(fig)

    return fig, ani


def visualiseHidden(imgTensor, channels_idxs = [], filenameBase="test", columns=4, sigmoid = False):
    """
    Visualise a designated snapshot of the grid specified by idx
    imgTensor should be in the form (batch, channels, height, width) OR (channels, height, width)
    Outputs a saved GIF format file saved to <filenameBase>.gif
    """    
    imgTensor = imgTensor.detach().cpu()

    if len(imgTensor.shape) < 4:
        imgTensor.unsqueeze(0)

    if (len(channels_idxs) == 0): # Pixels
        channels_idxs = [i for i in range(imgTensor.shape[1])] # This should be the number of channels

    if (sigmoid):
        imgTensor = torch.sigmoid(imgTensor)
        imgTensor[:, :, 0:3, 0] = torch.Tensor([0, 0.5, 1])
    else:
        ch_mag = torch.max(torch.abs(imgTensor)).item()

        print(f"Max value in the tensor is: {torch.max(imgTensor).item()}")
        print(f"Min value in the tensor is: {torch.min(imgTensor).item()}")

        imgTensor[:, :, 0:3, 0] = torch.Tensor([-ch_mag, 0.5, ch_mag])

    fig = plt.figure()
    title = plt.suptitle("Update 0")
    canvases = []

    # Get canvases for Images
    for i in range(len(channels_idxs)):
        plt.subplot(len(channels_idxs)//columns + 1, columns , i+1)
        canvases.append(plt.imshow((imgTensor[0].squeeze().permute(1, 2, 0))[:, :, channels_idxs[i]].detach().numpy(), cmap="bwr"))
        plt.title(f"Channel {channels_idxs[i]}", fontdict={'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 12 }, x = 0.5, y = 0.95)
        plt.xticks([])
        plt.yticks([])

    
    def update(imgIdx):

        title.set_text("Update " + str(imgIdx))

        img = imgTensor[imgIdx].squeeze().permute(1, 2, 0)
        
        # Add reference scale (also combats plt autoscale)

        for i in range(len(channels_idxs)):
            canvases[i].set_data(img[:, :, channels_idxs[i]].detach().numpy())

    
    ani = animation.FuncAnimation(fig, update, frames=len(imgTensor), repeat=False)
    # Display image
    
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=15, metadata=dict(artist="Me"), bitrate=-1
    )
    ani.save(filenameBase + ".gif", writer=writer)

    plt.close(fig)

    return fig, ani
