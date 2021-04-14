# -*- coding: utf-8 -*-
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from osgeo import gdal_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import interpolation
#from myfunctions.tools import MidpointNormalize
import cv2
#import math
#from rasterio.mask import mask
#Define custom fucntions to compare bands
#more complex than rest

import matplotlib.colors as colors



class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class Satellite_tools:
    '''def crop_sat(folder, name, aoi, analysis_area,output_folder, x_width):
        #R10,ba,aoi,analysis_area,output_folder
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
        #in case of asymmetryc shapes  
        dif_dims = out_image.shape[1] - out_image.shape[2]
        max_dim = max(out_image.shape[1],out_image.shape[2])
        dif_perc = dif_dims/max_dim
        if dif_perc > 0.25:
            skip = True
        else:
            skip = False
        if skip == False:
            if dif_dims < 0 :
                out_image = out_image[:,:,abs(dif_dims):]
            elif dif_dims > 0 :
                out_image = out_image[:,abs(dif_dims):,:]
            #out_image =  out_image_c      
            
            scale = x_width/len(out_image[0])
            if scale > 1.1:
                data = out_image[0]
                data_interpolated = interpolation.zoom(data,scale)
                data_interpolated = np.expand_dims(data_interpolated, axis=0)
                out_image = data_interpolated
            with rio.open(newname, "w", **out_meta) as dest:
                dest.write(out_image)
                dest.close()
        return skip
    
    def area_crop(date,aoi2,analysis_area,source, destination, output_folder): #"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" 
        with rio.open(output_folder+analysis_area+'/'+date[:8]+source) as src:
            out_image, out_transform = rio.mask.mask(src, aoi2.geometry,crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})  
            src.close()
        with rio.open(output_folder+analysis_area+'/'+date[:8]+destination, "w", **out_meta) as dest:
            dest.write(out_image)
            dest.close()
            
    def cld_msk(date, clouds_data, ind_mask, analysis_area,output_folder):
        #define construction fixed band at 512 pxl and use always same
        #base = rio.open(output_folder+analysis_area+'/'+"CLOUD_BASE.tif")
        base = rio.open(output_folder+analysis_area+'/'+date[:8]+"B04.tif")
        x_width = base.width
        y_height = base.height
        base.close()
        #cloud mask generate file
        #src = output_folder+analysis_area+'/'+"CLOUD_BASE.tif"
        src = output_folder+analysis_area+'/'+date[:8]+"B04.tif"
        data = clouds_data[ind_mask]
        scale = x_width/len(data[0])
        data = data[0]
        data_interpolated = interpolation.zoom(data,scale)
        gdal_array.LoadFile(src) 
        gdal_array.SaveArray(data_interpolated, output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        #gdal_array.SaveArray(clouds_data[ind_mask], output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        #cloud mask generate file 2nd pass
        #src = output_folder+analysis_area+'/'+date[:8]+"B04.tif"
        #arr = gdal_array.LoadFile(src) 
        #output = gdal_array.SaveArray(clouds_data[ind_mask], output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
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
        b1r = b1.read()
        b2r = b2.read()
        cld = msk_cloud.read()
            
        #scale here also
        img = cv2.imread(output_folder+analysis_area+'/'+date[:8]+band1+".tif",-1)
        dif_dims = img.shape[0] - img.shape[1]
        if dif_dims < 0 :
            img = img[:,abs(dif_dims):]
        elif dif_dims > 0 :
            img = img[abs(dif_dims):,:]
        #for bands with resolution less than 10m, reshape band to scale size       
        if x_width > b1.width:
            #commented
            #img = cv2.imread(output_folder+analysis_area+'/'+date[:8]+band1+".tif",-1)
            #dif_dims = img.shape[0] - img.shape[1]
            #if dif_dims < 0 :
            #    img = img[:,abs(dif_dims):]
            #elif dif_dims > 0 :
            #    img = img[abs(dif_dims):,:]
            #scale_percent = x_width/b1.width # percent of original size
            
            scale_percent = x_width/img.shape[1]
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            resized = np.expand_dims(resized, axis=0)
            b1r = resized
        
        img2 = cv2.imread(output_folder+analysis_area+'/'+date[:8]+band2+".tif",-1)
        dif_dims2 = img2.shape[0] - img2.shape[1]
        if dif_dims2 < 0 :
            img2 = img2[:,abs(dif_dims2):]
        elif dif_dims2 > 0 :
            img2 = img2[abs(dif_dims2):,:]
        if x_width > b2.width:
            #commented
            #img = cv2.imread(output_folder+analysis_area+'/'+date[:8]+band2+".tif",-1)
            #dif_dims = img.shape[0] - img.shape[1]
            #if dif_dims < 0 :
            #    img = img[:,abs(dif_dims):]
            #elif dif_dims > 0 :
            #    img = img[abs(dif_dims):,:]
            #scale_percent = x_width/b2.width # percent of original size
                        scale_percent2 = x_width/img2.shape[1]
            width2 = int(img2.shape[1] * scale_percent2)
            height2 = int(img2.shape[0] * scale_percent2)
            dim2 = (width2, height2)
            resized2 = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)
            resized2 = np.expand_dims(resized2, axis=0)
            b2r = resized2
            
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate (could be conditional for rest, sum or other complex)
        calculated = (b1r.astype(float)-b2r.astype(float))/(b1r+b2r)
        #correct by cloud mask
        calculated[cld==1] = None
        msk_cloud.close()
        b1.close()
        b2.close()
        # Write the calculated image
        meta = msk_cloud.meta
        meta.update(driver='GTiff')
        meta.update(dtype=rio.float32)
        with rio.open(output_folder+analysis_area+'/'+date[:8]+name+".tif", 'w', **meta) as dst:
            dst.write(calculated.astype(rio.float32))
            dst.close()
            b1.close()
            b2.close()
            msk_cloud.close()
    
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
        ndvi_plt.close()
    '''    
    def trns_coor(area,meta):
        try:
            x, y = meta['transform'][2]+area[0]*10, meta['transform'][5]+area[1]*-10
        except:
            x, y = meta[0]['transform'][2]+area[0]*10, meta[0]['transform'][5]+area[1]*-10    
        return x, y
    
    def plot_figura2(image, analysis_area, date, output_folder, png_folder, lote_name,index_str, cmap, **kwargs): #, vmin=-1, vmax=1
            ext=False
            cen = False
            for key,value in kwargs.items():
                if key == 'vmin':
                    vmin1=value
                    ext=True
                elif key == 'vmax':
                    vmax1=value
                    ext=True
                elif key == 'vcen':
                    vcen1=value
                    ext=True
                    cen = True
                        
            fig = plt.figure(figsize=(3,3))
            #area chart
            ax = plt.gca()
            #remove 0 to None
            image[image==0] = None
            if cen == True:
                im = ax.imshow(image[0], cmap=cmap, clim=(vmin1, vmax1), norm=MidpointNormalize(midpoint=vcen1,vmin=vmin1, vmax=vmax1))
            elif ext==True:
                im = ax.imshow(image[0], cmap=cmap, vmin=vmin1, vmax=vmax1)
            else:
                im = ax.imshow(image[0], cmap=cmap, vmin=np.nanpercentile(image,1), vmax=np.nanpercentile(image,99)) #'RdYlGn'
            #plot
            plt.title(index_str+' in ' + str(lote_name))
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            #to locate colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig(png_folder+analysis_area+'/'+date[:8]+"_"+str(lote_name)+index_str+".png",bbox_inches='tight',dpi=100)
            plt.clf()