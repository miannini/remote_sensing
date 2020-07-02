# -*- coding: utf-8 -*-
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from osgeo import gdal_array
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from rasterio.mask import mask

class Satellite_tools:
    def crop_sat(folder, name, aoi, analysis_area):
        with rio.open(folder+name) as src:
            out_image, out_transform = rio.mask.mask(src, aoi.geometry,crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
            src.close()
        date,band = name.split("_")[1],name.split("_")[2] #agregué tercer argumento
        newname = "Output_Images/"+analysis_area+'/'+date[:8]+""+band[:3]+".tif"
        with rio.open(newname, "w", **out_meta) as dest:
            dest.write(out_image)
            dest.close()
    
    def area_crop(date,aoi2,analysis_area,source, destination, dest_folder): #"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" 
        with rio.open("Output_Images/"+analysis_area+'/'+date[:8]+source) as src:
            out_image, out_transform = rio.mask.mask(src, aoi2.geometry,crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})    
        with rio.open(dest_folder+analysis_area+'/'+date[:8]+destination, "w", **out_meta) as dest:
            dest.write(out_image)
            
    def cld_msk(date, clouds_data, ind_mask, analysis_area):
        b4a = rio.open("Output_Images/"+analysis_area+'/'+date[:8]+"B04.tif") #Automatizar nombre a 4 elemento del directorio
        x_width = b4a.width
        y_height = b4a.height
        b4a.close()
        #x_width,y_height
        #cloud mask generate file
        src = "Output_Images/"+analysis_area+'/'+date[:8]+"B04.tif"
        arr = gdal_array.LoadFile(src) 
        output = gdal_array.SaveArray(clouds_data[ind_mask], "Output_Images/"+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        #cloud mask generate file 2nd pass
        src = "Output_Images/"+analysis_area+'/'+date[:8]+"B04.tif"
        arr = gdal_array.LoadFile(src) 
        output = gdal_array.SaveArray(clouds_data[ind_mask], "Output_Images/"+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        return x_width, y_height
    
    def ndvi_calc(date, analysis_area):
        # Open b4(red),b8
        msk_cloud = rio.open("Output_Images/"+analysis_area+'/'+date[:8]+"_cldmsk.tif")       
        b4a = rio.open("Output_Images/"+analysis_area+'/'+date[:8]+"B04.tif")
        b8a = rio.open("Output_Images/"+analysis_area+'/'+date[:8]+"B08.tif")
        # read Red(b4) and NIR(b8) as arrays
        red = b4a.read()
        nir = b8a.read()
        cld = msk_cloud.read()
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (nir.astype(float)-red.astype(float))/(nir+red)
        ndvi[cld==1] = None
        msk_cloud.close()
        b4a.close()
        b8a.close()
        # Write the NDVI image
        meta = b4a.meta
        meta.update(driver='GTiff')
        meta.update(dtype=rio.float32)
        with rio.open("Output_Images/"+analysis_area+'/'+date[:8]+"_NDVI.tif", 'w', **meta) as dst:
            dst.write(ndvi.astype(rio.float32))
            dst.close()
            b4a.close()
            b8a.close()
        return meta
    
    def plot_ndvi(date, analysis_area, source, destination,dest_folder): #source="_NDVI.tif", destination="_NDVI_export.png", dest_folder="Output_Images/"
        ndvi_plt = rio.open("Output_Images/"+analysis_area+'/'+date[:8]+source) #"_NDVI.tif"
        ndvi = ndvi_plt.read(1)
        ndvi[ndvi==0] = None
        fig = plt.figure(figsize=(16,16))
        #area chart
        ax = plt.gca()
        im = ax.imshow(ndvi, vmin=-1, vmax=1, cmap='RdYlGn')
        #plot
        plt.title('NDVI in Analysis area for date ' + date)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        #to locate colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(dest_folder+analysis_area+'/'+destination,bbox_inches='tight',dpi=500)
        plt.clf()
        
    def trns_coor(area,meta):
        x, y = meta['transform'][2]+area[0]*10, meta['transform'][5]+area[1]*-10
        return x, y