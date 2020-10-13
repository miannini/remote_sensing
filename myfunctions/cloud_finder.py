import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
import datetime
from sentinelhub import WmsRequest, BBox, CRS, MimeType, CustomUrlParam, get_area_dates
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
from myfunctions.tools import Cloudless_tools
#cloud detection
class Cloud_finder:
    def cloud_process(bounding_box, Date_Ini, Date_Fin, x_width, y_height,analysis_area,clouds_folder):
        INSTANCE_ID = 'e8757ad9-2099-4dd9-89c9-6d20fcaa96ab' #From Sentinel HUB Python Instance ID /change to dynamic user input
        LAYER_NAME = 'TRUE-COLOR-S2-L1C' # e.g. TRUE-COLOR-S2-L1C
        #Obtener imagenes por fecha (dentro de rango) dentro de box de interés
        wms_true_color_request = WmsRequest(layer=LAYER_NAME,
                                            bbox=bounding_box,
                                            time=(Date_Ini, Date_Fin), #cambiar a fechas de interés
                                            width=x_width, height=y_height,
                                            image_format=MimeType.PNG,
                                            time_difference=datetime.timedelta(hours=2),
                                            instance_id=INSTANCE_ID)
        wms_true_color_imgs = wms_true_color_request.get_data()
        #Cloudless_tools.plot_previews(np.asarray(wms_true_color_imgs), wms_true_color_request.get_dates(), cols=4, figsize=(15, 10))
        
        #count of 0's to know how empty is the image
        count_of_zeros = []
        for n in range(0,len(wms_true_color_imgs)):
            # zeros / 4 channels * width * height (pixels)
            count_of_zeros.append((np.count_nonzero(wms_true_color_imgs[n]==0))/(4*wms_true_color_imgs[n][:,:,0].shape[0]*wms_true_color_imgs[n][:,:,0].shape[1]))
        
        #Calculo de probabilidades y obtención de mascaras de nubes
        bands_script = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'
        wms_bands_request = WmsRequest(layer=LAYER_NAME,
                                       custom_url_params={
                                           CustomUrlParam.EVALSCRIPT: bands_script,
                                           CustomUrlParam.ATMFILTER: 'NONE'
                                       },
                                       bbox=bounding_box, 
                                       time=(Date_Ini, Date_Fin),
                                       width=x_width, height=y_height,
                                       image_format=MimeType.TIFF_d32f,
                                       time_difference=datetime.timedelta(hours=2),
                                       instance_id=INSTANCE_ID)
        wms_bands = wms_bands_request.get_data()
        wms_bands_request.get_filename_list()
        wms_bands_request.get_url_list()
        
        cloud_detector = S2PixelCloudDetector(threshold=0.35, average_over=8, dilation_size=3) #change threshold to test
        #cloud_probs = cloud_detector.get_cloud_probability_maps(np.array(wms_bands))
        cloud_masks = cloud_detector.get_cloud_masks(np.array(wms_bands))
        all_cloud_masks = CloudMaskRequest(ogc_request=wms_bands_request, threshold=0.35)
        
        
        #Mostrar las probabilidades de nubes para cada imagen por fecha en el rango de analisis
        n_cols = 4
        n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))
        fig = plt.figure(figsize=(n_cols*4,n_rows*3)) #, constrained_layout=False
        for idx, [prob, mask, data] in enumerate(all_cloud_masks):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            image = wms_true_color_imgs[idx]
            Cloudless_tools.overlay_cloud_mask(image, mask, factor=1, fig=fig)
        plt.tight_layout()
        plt.savefig(clouds_folder+analysis_area+'/real_and_cloud.png')
        #Mostrar las mascaras de nubes para cada imagen por fecha en el rango de analisis
        #fig = plt.figure(figsize=(15, 10))
        n_cols = 4
        n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))
        fig = plt.figure(figsize=(n_cols*4,n_rows*3))
        each_cld_mask = all_cloud_masks.get_cloud_masks(threshold=0.35)
        for idx, cloud_mask in enumerate(all_cloud_masks.get_cloud_masks(threshold=0.35)):
        #for idx, cloud_mask in enumerate(each_cld_mask): #se repite con linea 101
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            Cloudless_tools.plot_cloud_mask(cloud_mask, fig=fig)  
        plt.tight_layout()
        plt.savefig(clouds_folder+analysis_area+'/cloud_masks.png')
        #Calculo y extracción de imagenes con cobertura de nubes menor a x%
        cld_per_idx = []
        #each_cld_mask = all_cloud_masks.get_cloud_masks(threshold=0.35)                 #se repite con linea 94
        for a in range(0,len(each_cld_mask)):
            n_cloud_mask = np.shape(np.concatenate(each_cld_mask[a]))
            cloud_perc = sum(np.concatenate(each_cld_mask[a])== 1)/n_cloud_mask
            cld_per_idx.append(cloud_perc.astype(float))
        x = pd.DataFrame(cld_per_idx)<0.6 #Menor a 60% de cobertura de nubes
        all_dates = pd.DataFrame(all_cloud_masks.get_dates())
        #valid_dates = pd.DataFrame(all_cloud_masks.get_dates())[x[0]]
        valid_dates = all_dates[x[0]]
        #all_dates = pd.DataFrame(all_cloud_masks.get_dates())
        all_dates['cld_percent'] = cld_per_idx
        all_dates.to_csv (clouds_folder+analysis_area+'/fechas_porcentaje_nubes.csv', index = False, header=True)
        #print("[INFO] valid dates ... {:f})".format(valid_dates))
        
        #filter clouds dataframe with only valid dates
        clouds_data = cloud_masks[x[0]]
        minIndex = cld_per_idx.index(min(cld_per_idx))
        best_date = valid_dates[valid_dates.index==minIndex]
        best_date = best_date.iloc[0,0]
        
        #Mostrar las mascaras de nubes para cada imagen por fecha valida
        #fig = plt.figure(figsize=(15, 10))
        n_cols = 4
        n_rows = int(np.ceil(len(clouds_data) / n_cols))
        fig = plt.figure(figsize=(n_cols*4,n_rows*3))
        for idx, cloud_mask in enumerate(clouds_data):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            Cloudless_tools.plot_cloud_mask(cloud_mask, fig=fig)
        plt.tight_layout()
        plt.savefig(clouds_folder+analysis_area+'/cloud_masks_valid.png')
        
        clear_pct = len(valid_dates)/len(cld_per_idx)
        return best_date, valid_dates, clouds_data, clear_pct