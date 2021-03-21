import datetime
import matplotlib.pyplot as plt
import numpy as np
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline

class Cloudless_tools:
	#def find_cloud(user_analysis):
        
        def overlay_cloud_mask(image, mask=None, factor=1./255, figsize=(15, 15), fig=None):
            """
            Utility function for plotting RGB images with binary mask overlayed.
            """
            if fig == None:
                plt.figure(figsize=figsize)
            rgb = np.array(image)
            plt.imshow(rgb * factor)
            if mask is not None:
                cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                cloud_image[mask == 255] = np.asarray([255, 255, 0, 100], dtype=np.uint8)
                plt.imshow(cloud_image)
                
        def plot_probability_map(rgb_image, prob_map, factor=1./255, figsize=(15, 30)):
            """
            Utility function for plotting a RGB image and its cloud probability map next to each other. 
            """
            plt.figure(figsize=figsize)
            plot = plt.subplot(1, 2, 1)
            plt.imshow(rgb_image * factor)
            plot = plt.subplot(1, 2, 2)
            plot.imshow(prob_map, cmap=plt.cm.inferno)
            
        def plot_cloud_mask(mask, figsize=(15, 15), fig=None):
            """
            Utility function for plotting a binary cloud mask.
            """
            if fig == None:
                plt.figure(figsize=figsize)
            plt.imshow(mask, cmap=plt.cm.gray)
            
        def plot_previews(data, dates, cols=4, figsize=(15, 15)):
            """
            Utility to plot small "true color" previews.
            """
            width = data[-1].shape[1]
            height = data[-1].shape[0]
            
            rows = data.shape[0] // cols + (1 if data.shape[0] % cols else 0)
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
            for index, ax in enumerate(axs.flatten()):
                if index < data.shape[0]:
                    caption = '{}: {}'.format(index, dates[index].strftime('%Y-%m-%d'))
                    ax.set_axis_off()
                    ax.imshow(data[index] / 255., vmin=0.0, vmax=1.0)
                    ax.text(0, -2, caption, fontsize=12, color='g')
                else:
                    ax.set_axis_off()
        
