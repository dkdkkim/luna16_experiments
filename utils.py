# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import cupy
import cupyx.scipy.ndimage

def fig_3views(img, clim=(0., 1.0)):
    
    fig, ax = plt.subplots(4, 6, figsize=(12, 8))
    for pidx in range(8):
        ax[pidx % 4, int(pidx / 4)].imshow(img[int(img.shape[0] / 2 - 4 + pidx),:,:],
                                      cmap='gray', clim=clim)
        ax[pidx % 4, int(pidx / 4)].axis('off')

        ax[pidx % 4, int(2 + pidx / 4)].imshow(img[:, int(img.shape[1] / 2 - 4 + pidx), :],
                                          cmap='gray', clim=clim)
        ax[pidx % 4, int(2 + pidx / 4)].axis('off')

        ax[pidx % 4, int(4 + pidx / 4)].imshow(img[:, :, int(img.shape[2] / 2 - 4 + pidx)],
                                          cmap='gray', clim=clim)
        ax[pidx % 4, int(4 + pidx / 4)].axis('off')

    ax[0, 0].text(1, 4, '<transverse>', color='white')
    ax[0, 2].text(1, 4, '<coronal>', color='white')
    ax[0, 4].text(1, 4, '<sagittal>', color='white')
    return fig, ax

def interpolation(img, spacing, new_spacing):
    spacing = np.array(spacing)
    new_spacing = np.array(new_spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / img.shape
    new_spacing = spacing / real_resize_factor
    return scipy.ndimage.zoom(img, real_resize_factor)

def interpolation_cupy(image, spacing, new_spacing, gpu=0):
    spacing = np.array(spacing)
    new_spacing = np.array(new_spacing)
    with cupy.cuda.Device(gpu):
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        x = cupy.array(image)
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        us = cupyx.scipy.ndimage.zoom(x, real_resize_factor)
        us_cpu = us.get()
        del x
        del us
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    return us_cpu