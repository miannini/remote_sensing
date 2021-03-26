import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#import shapefile
import datetime
from sentinelhub import WmsRequest, BBox, CRS, MimeType, CustomUrlParam, get_area_dates, SentinelHubRequest, SentinelHubDownloadClient, bbox_to_dimensions, DownloadRequest, DataCollection, SHConfig
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
from myfunctions.tools import Cloudless_tools

#cloud detection

##### for new eval script 3, saving CONFIG details in other place
#from sentinelhub import SHConfig
# In case you put the credentials into the configuration file you can leave this unchanged
CLIENT_ID = 'a956fb12-aa17-44f6-b790-0b80304fdd82'
CLIENT_SECRET = 'K|rZ6kOW<1szMf{Zw]P(c.&[*L%k9R>vYQq;6^|O'
config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")
######

class Cloud_finder:
    def cloud_process(bounding_box, Date_Ini, Date_Fin, x_width, y_height,analysis_area,clouds_folder, lote_aoi,municipio, departamento):
        INSTANCE_ID = '48c7fc4a-3862-40a8-be4a-b18db33c2d60' #From Sentinel HUB Python Instance ID /change to dynamic user input
        LAYER_NAME = 'TRUE-COLOR-S2-L1C' # e.g. TRUE-COLOR-S2-L1C
        #Obtener imagenes por fecha (dentro de rango) dentro de box de interés
        
        #batch de fechas
        start = datetime.datetime.strptime(Date_Ini,'%Y-%m-%d')#(2019,1,1)
        end = datetime.datetime.strptime(Date_Fin,'%Y-%m-%d')#(2019,12,31)
        n_chunks = math.ceil((end-start).days/5) #13
        tdelta = (end - start) / n_chunks
        edges = [(start + i*tdelta).date().isoformat() for i in range(n_chunks+1)]
        slots = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
        
        print('Total time windows:\n')
        for slot in slots:
            print(slot)
        
        #evalscript V3 //nuevo
        evalscript_true_color = """
            //VERSION=3
        
            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }
        
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """
        #cloud layer
        evalscript_clm = """
        //VERSION=3
        function setup() {
          return {
            input: ["B02", "B03", "B04", "CLM"],
            output: { bands: 4 }
          }
        }
        
        function evaluatePixel(sample,scene) {
          return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02, sample.CLM, scene.date];
          
        }
        
        """
        #if (sample.CLM == 1) {
        #    return [0.75 + sample.B04, sample.B03, sample.B02]
        #  }
        #second output in case metadata is available to download
        #function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
        #    outputMetadata.userData = { "metadata":  JSON.stringify(scenes) }
        #}
        
        #multi download
        def get_true_color_request(time_interval):
            return SentinelHubRequest(
                evalscript=evalscript_clm,
                #evalscript=evalscript_true_color,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L1C,
                        time_interval=time_interval,
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.PNG)#,
                    #{
                    #    "identifier": "userdata",
                    #    "format": {
                    #        "type": "application/json"}
                    #}
                ],
                bbox=bounding_box,
                size=(x_width,y_height),
                config=config
            )
        # create a list of requests
        list_of_requests = [get_true_color_request(slot) for slot in slots]
        list_of_requests = [request.download_list[0] for request in list_of_requests]
        
        # download data with multiple threads
        wms_true_color_imgs = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=None)#5)
        #wms_true_color_imgs = data
        '''
        #single download
        #betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
        resolution = 30
        #betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
        box_size = bbox_to_dimensions(bounding_box, resolution=resolution)
        request_true_color = SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=(Date_Ini, Date_Fin),
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG)
            ],
            bbox=bounding_box, #betsiboka_bbox,
            size=(x_width,y_height),
            #size=box_size,
            config=config
        )
        #wms_true_color_request = WmsRequest(layer=LAYER_NAME,
        #                                    bbox=bounding_box,
        #                                    time=(Date_Ini, Date_Fin), #cambiar a fechas de interés
        #                                    width=x_width, height=y_height,
        #                                    image_format=MimeType.PNG,
        #                                    time_difference=datetime.timedelta(hours=2),
        #                                    instance_id=INSTANCE_ID)
        wms_true_color_imgs = request_true_color.get_data()
        print(f'Returned data is of type = {type(wms_true_color_imgs)} and length {len(wms_true_color_imgs)}.')
        print(f'Single element in the list is of type {type(wms_true_color_imgs[-1])} and has shape {wms_true_color_imgs[-1].shape}')
        
        image = wms_true_color_imgs[0]
        print(f'Image type: {image.dtype}')
        
        # plot function
        # factor 1/255 to scale between 0-1
        # factor 3.5 to increase brightness
        #plot_image(image, factor=3.5/255, clip_range=(0,1))
        '''
        #plot_image(data_with_cloud_mask[0], factor=1/255)
        plt.imshow(wms_true_color_imgs[0][:,:,0:3])
        plt.savefig(clouds_folder+analysis_area+'/cloud_test1.png')
        
        #count of 0's to know how empty is the image
        count_of_zeros = []
        for n in range(0,len(wms_true_color_imgs)):
            # zeros / 3 channels * width * height (pixels)
            count_of_zeros.append((np.count_nonzero(wms_true_color_imgs[n][:,:,0:3]==0))/(3*wms_true_color_imgs[n][:,:,0].shape[0]*wms_true_color_imgs[n][:,:,0].shape[1]))
        
        '''
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
        #wms_bands_request.get_filename_list()
        #wms_bands_request.get_url_list()
        #wms_bands_request.get_dates()
        cloud_detector = S2PixelCloudDetector(threshold=0.35, average_over=8, dilation_size=3) #change threshold to test
        #cloud_probs = cloud_detector.get_cloud_probability_maps(np.array(wms_bands))
        cloud_masks = cloud_detector.get_cloud_masks(np.array(wms_bands))
        all_cloud_masks = CloudMaskRequest(ogc_request=wms_bands_request, threshold=0.35)
        #cloud_masks = all_cloud_masks.get_cloud_masks()
        '''
        #Mostrar las probabilidades de nubes para cada imagen por fecha en el rango de analisis
        n_cols = 4
        n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))
        fig = plt.figure(figsize=(n_cols*4,n_rows*3)) #, constrained_layout=False
        for idx,data in enumerate(wms_true_color_imgs): #[prob, mask, data]
            #print (idx)
            #print (data)
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            image = wms_true_color_imgs[idx][:,:,0:3]
            mask = wms_true_color_imgs[idx][:,:,3]
            Cloudless_tools.overlay_cloud_mask(image, mask, factor=1, fig=fig)
        plt.tight_layout()
        plt.savefig(clouds_folder+analysis_area+'/real_and_cloud.png')
        
        #Mostrar las mascaras de nubes para cada imagen por fecha en el rango de analisis
        n_cols = 4
        n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))
        fig = plt.figure(figsize=(n_cols*4,n_rows*3))
        #each_cld_mask = all_cloud_masks.get_cloud_masks(threshold=0.35)
        cld_per_idx = []
        for idx, data in enumerate(wms_true_color_imgs):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            #correct mask, when no data is in the image, to mask non values
            mask = data[:,:,3] #wms_true_color_imgs[idx]
            mask[data[:,:,0]==0] = 2 #to check this
            cloud_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            cloud_mask[mask >0] = 1 #np.asarray([255, 255, 0, 100], dtype=np.uint8)
            
            Cloudless_tools.plot_cloud_mask(cloud_mask, fig=fig)
            n_cloud_mask = np.shape(np.concatenate(cloud_mask))
            cloud_perc = sum(np.concatenate(cloud_mask)== 1)/n_cloud_mask
            cld_per_idx.append(cloud_perc.astype(float))
        plt.tight_layout()
        plt.savefig(clouds_folder+analysis_area+'/cloud_masks.png')
        
        #Calculo y extracción de imagenes con cobertura de nubes menor a x%
        x = pd.DataFrame(cld_per_idx)<0.6 #Menor a 60% de cobertura de nubes // or lote is visible TO ADD
        #all_dates = pd.DataFrame(all_cloud_masks.get_dates())
        #valid_dates = slots[x[0]]
        valid_dates = [slots[i] for i in range(len(slots)) if x[0][i]]  # or also [i for (i, v) in zip(slots, x[0]) if v]
        all_dates = pd.DataFrame(slots)
        all_dates['year-month'] =  all_dates[0].str[:7] #.dt.to_period('M')
        all_dates['cld_percent'] = cld_per_idx
        all_dates['empty_percent'] = count_of_zeros
        all_dates = all_dates.rename(columns={0:'date_start', 1:'date_end'})
        
        all_dates['date_start'] = pd.to_datetime(all_dates['date_start'], format='%Y-%m-%d') 
        all_dates['date_end'] = pd.to_datetime(all_dates['date_end'], format='%Y-%m-%d') 
        all_dates['year-month'] = pd.to_datetime(all_dates['year-month'], format='%Y-%m').dt.to_period('M') 
        #summary
        '''
        summary_clds = all_dates[['year-month','cld_percent','dates']].groupby('year-month').agg({'dates':lambda x: x.diff().mean(), 'cld_percent': ['count', lambda x: (x<0.6).sum(), lambda x: x.mean(), 'min']}) \
            .reset_index()
        '''
        def f_mi(x):
            d = []
            d.append(x['date_start'].diff().mean())
            d.append(x['cld_percent'].count())
            d.append((x['cld_percent'] <0.6).sum())
            d.append(x['cld_percent'].mean())
            d.append(x['cld_percent'].min())
            d.append(x[x['cld_percent']<0.6]['date_start'].max())
            d.append(x[x['cld_percent']<0.6]['date_start'].min())
            d.append(x['empty_percent'].max())
            d.append(x['empty_percent'].min())
            return pd.Series(d, index=['time_between_pass','count_pass','clear_images','mean_cloud_cover','min_cloud_cover','last_good_date','first_good_date','max_empty_space','min_empty_space']) #
        
        summary_clds = all_dates.groupby('year-month').apply(f_mi)
        summary_clds['centroid_x'], summary_clds['centroid_y'], summary_clds['terrain_name'], summary_clds['terrain_code'], summary_clds['municipio'], summary_clds['departamento']= lote_aoi['x'][0], lote_aoi['y'][0],lote_aoi['name'][0],analysis_area,municipio, departamento 
        #export data
        summary_clds.to_csv (clouds_folder+analysis_area+'/Analisis_nubes.csv', index = True, header=True)
        
        #filter clouds dataframe with only valid dates
        #cloud_masks = np.empty(shape=(0,mask.shape[0], mask.shape[1]),dtype='object') 
        #cloud_masks = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        #cloud_data = np.empty(shape=(mask.shape[0], mask.shape[1]))
        for n in range(len(wms_true_color_imgs)):
            if n == 0:
                cloud_masks = np.array(wms_true_color_imgs[n][:,:,3])
            else:
                temp = np.array(wms_true_color_imgs[n][:,:,3])  
                cloud_masks = np.dstack((cloud_masks,temp))        
        
        clouds_data = cloud_masks[:,:,x[0]]
        minIndex = cld_per_idx.index(min(cld_per_idx))
        best_date = slots[minIndex]
        best_date = best_date[0]
        
        #Mostrar las mascaras de nubes para cada imagen por fecha valida
        n_cols = 4
        n_rows = int(np.ceil(clouds_data.shape[2] / n_cols))
        fig = plt.figure(figsize=(n_cols*4,n_rows*3))
        for n in range(clouds_data.shape[2]):
            ax = fig.add_subplot(n_rows, n_cols, n + 1)
            Cloudless_tools.plot_cloud_mask(clouds_data[:,:,n], fig=fig)
        plt.tight_layout()
        plt.savefig(clouds_folder+analysis_area+'/cloud_masks_valid.png')
        
        clear_pct = len(valid_dates)/len(cld_per_idx)
        number_cld_analysis = len(cld_per_idx)
        return best_date, valid_dates, clouds_data, clear_pct, number_cld_analysis