# -*- coding: utf-8 -*-
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from osgeo import gdal_array
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
#import math
#from rasterio.mask import mask
#Define custom fucntions to compare bands
#more complex than rest

class Satellite_tools:
    def crop_sat(folder, name, aoi, analysis_area,output_folder):
        with rio.open(folder+name) as src:
            out_image, out_transform = rio.mask.mask(src, aoi.geometry,crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
            src.close()
        date,band = name.split("_")[1],name.split("_")[2] #agregué tercer argumento
        newname = output_folder+analysis_area+'/'+date[:8]+""+band[:3]+".tif"
        with rio.open(newname, "w", **out_meta) as dest:
            dest.write(out_image)
            dest.close()
    
    def area_crop(date,aoi2,analysis_area,source, destination, output_folder): #"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" 
        with rio.open(output_folder+analysis_area+'/'+date[:8]+source) as src:
            out_image, out_transform = rio.mask.mask(src, aoi2.geometry,crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})    
        with rio.open(output_folder+analysis_area+'/'+date[:8]+destination, "w", **out_meta) as dest:
            dest.write(out_image)
            
    def cld_msk(date, clouds_data, ind_mask, analysis_area,output_folder):
        b4a = rio.open(output_folder+analysis_area+'/'+date[:8]+"B04.tif") #Automatizar nombre a 4 elemento del directorio
        x_width = b4a.width
        y_height = b4a.height
        b4a.close()
        #x_width,y_height
        #cloud mask generate file
        src = output_folder+analysis_area+'/'+date[:8]+"B04.tif"
        arr = gdal_array.LoadFile(src) 
        output = gdal_array.SaveArray(clouds_data[ind_mask], output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        #cloud mask generate file 2nd pass
        src = output_folder+analysis_area+'/'+date[:8]+"B04.tif"
        arr = gdal_array.LoadFile(src) 
        output = gdal_array.SaveArray(clouds_data[ind_mask], output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        return x_width, y_height
    
    def ndvi_calc(date, analysis_area,crop,output_folder):
        # Open b4(red),b8
        msk_cloud = rio.open(output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif")       
        b4a = rio.open(output_folder+analysis_area+'/'+date[:8]+"B04.tif")
        b8a = rio.open(output_folder+analysis_area+'/'+date[:8]+"B08.tif")
        # read Red(b4) and NIR(b8) as arrays
        red = b4a.read()
        nir = b8a.read()
        cld = msk_cloud.read()
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (nir.astype(float)-red.astype(float))/(nir+red)
        ndvi[cld==1] = None
        if crop=='grass':
            lai=0.001*np.exp(8.7343*ndvi.astype(rio.float32))
            bm = 451.99*lai.astype(rio.float32) + 1870.7
        else:
            lai=None
            bm=None
        msk_cloud.close()
        b4a.close()
        b8a.close()
        # Write the NDVI image
        meta = b4a.meta
        meta.update(driver='GTiff')
        meta.update(dtype=rio.float32)
        with rio.open(output_folder+analysis_area+'/'+date[:8]+"_NDVI.tif", 'w', **meta) as dst:
            dst.write(ndvi.astype(rio.float32))
            dst.close()
            b4a.close()
            b8a.close()
        with rio.open(output_folder+analysis_area+'/'+date[:8]+"_LAI.tif", 'w', **meta) as dst:
            dst.write(lai.astype(rio.float32))
            dst.close()
            b4a.close()
            b8a.close()
        with rio.open(output_folder+analysis_area+'/'+date[:8]+"_BM.tif", 'w', **meta) as dst:
            dst.write(bm.astype(rio.float32))
            dst.close()
            b4a.close()
            b8a.close()
        return meta
    
    def band_calc(date, analysis_area, band1, band2, name,x_width,output_folder): #'B04', 'B08', 'NDWI'
        msk_cloud = rio.open(output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif")       
        b1 = rio.open(output_folder+analysis_area+'/'+date[:8]+band1+".tif")
        b2 = rio.open(output_folder+analysis_area+'/'+date[:8]+band2+".tif")
        # read Red(b4) and NIR(b8) as arrays
        b1r = b1.read()
        b2r = b2.read()
        cld = msk_cloud.read()
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate (could be conditional for rest, sum or other complex)
        calculated = (b1r.astype(float)-b2r.astype(float))/(b1r+b2r)
        #for bands with 20m resolution, reshape cloud mask to scale size       
        if x_width > b1.width:
            img = cv2.imread(output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif",-1)
            scale_percent = b1.width/x_width # percent of original size
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            resized = np.expand_dims(resized, axis=0)
            calculated[resized==1] = None
        else:
            calculated[cld==1] = None
        #calculated[cld==1] = None
        msk_cloud.close()
        b1.close()
        b2.close()
        # Write the calculated image
        meta = b1.meta
        meta.update(driver='GTiff')
        meta.update(dtype=rio.float32)
        with rio.open(output_folder+analysis_area+'/'+date[:8]+name+".tif", 'w', **meta) as dst:
            dst.write(calculated.astype(rio.float32))
            dst.close()
            b1.close()
            b2.close()
    
    def plot_ndvi(date, analysis_area, source, destination,output_folder,cmap,min_loc,max_loc): #source="_NDVI.tif", destination="_NDVI_export.png", dest_folder="Output_Images/", cmap='RdYlGn'
        ndvi_plt = rio.open(output_folder+analysis_area+'/'+date[:8]+source) #"_NDVI.tif"
        ndvi = ndvi_plt.read(1)
        ndvi[ndvi==0] = None
        fig = plt.figure(figsize=(16,16))
        #area chart
        ax = plt.gca()
        im = ax.imshow(ndvi, vmin=min_loc, vmax=max_loc, cmap=cmap) #'RdYlGn'
        #plot
        plt.title('NDVI in Analysis area')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        #to locate colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(output_folder+analysis_area+'/'+date[:8]+destination,bbox_inches='tight',dpi=500)
        plt.clf()
        
    def trns_coor(area,meta):
        x, y = meta['transform'][2]+area[0]*10, meta['transform'][5]+area[1]*-10
        return x, y