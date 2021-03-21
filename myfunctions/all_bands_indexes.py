# -*- coding: utf-8 -*-
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show
from osgeo import gdal_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import interpolation
import cv2
from varname import nameof, Wrapper, varname
import inspect
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from myfunctions.tools import MidpointNormalize
from myfunctions.tools import Satellite_tools
#Define custom fucntions to compare bands
#more complex than rest


    
class Satellite_proc:
    def crop_sat(folder, name, aoi, analysis_area,output_folder, x_width):
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
        dif_perc = abs(dif_dims)/max_dim
        if dif_perc > 0.05:
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
            try:
                out_image, out_transform = rio.mask.mask(src, aoi2.geometry,crop=True)
            except:
                out_image, out_transform = rio.mask.mask(src, aoi2,crop=True)  
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})  
            src.close()
        with rio.open(output_folder+analysis_area+'/'+date[:8]+destination, "w", **out_meta) as dest:
            dest.write(out_image)
            dest.close()
    '''        
    def cld_msk(date, clouds_data, ind_mask, analysis_area,output_folder):
        #define construction fixed band at 512 pxl and use always same
        base = rio.open(output_folder+analysis_area+'/'+date[:8]+"B04.tif")
        x_width = base.width
        y_height = base.height
        base.close()
        #cloud mask generate file
        #src = output_folder+analysis_area+'/'+"CLOUD_BASE.tif"
        src = output_folder+analysis_area+'/'+date[:8]+"B04.tif"
        data = clouds_data[:,:,ind_mask]
        data = data[:,:,0]
        scale = x_width/len(data[0])
        #data = data[0]
        data_interpolated = interpolation.zoom(data,scale)
        data_interpolated = np.expand_dims(data_interpolated, axis=0)
        save = gdal_array.LoadFile(src) 
        save = gdal_array.SaveArray(data_interpolated, output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        save = gdal_array.SaveArray(data_interpolated, output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif", format="GTiff", prototype=src)
        save = None
        return x_width, y_height
    '''
    
    def band_calc(date, analysis_area,x_width,output_folder): 
        #msk_cloud = rio.open(output_folder+analysis_area+'/'+date[:8]+"_cldmsk.tif")       
        b1 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B01.tif")
        b2 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B02.tif")
        b3 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B03.tif")
        b4 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B04.tif")
        b5 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B05.tif")
        b6 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B06.tif")
        b7 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B07.tif")
        b8 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B08.tif")
        b8a = rio.open(output_folder+analysis_area+'/'+date[:8]+"B8A.tif")
        b9 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B09.tif")
        b10 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B10.tif")
        b11 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B11.tif")
        b12 = rio.open(output_folder+analysis_area+'/'+date[:8]+"B12.tif")
        
        
        b1r = b1.read()
        blue = b2.read()
        green =b3.read()
        red = b4.read()
        b5r = b5.read()
        b6r = b6.read()
        b7r = b7.read()
        nir = b8.read()
        b8ar = b8a.read()
        b9r = b9.read()
        b10r = b10.read()
        b11r = b11.read()
        b12r = b12.read()
        #cld = msk_cloud.read()
        
        
        
        #20m convert
        def scale_20(band):
            scale = x_width/len(band[0])
            data = band[0]
            data_interpolated = interpolation.zoom(data,scale)
            data_interpolated = np.expand_dims(data_interpolated, axis=0)
            return data_interpolated
        
        def mirror_dims(band):
            dif_dims = band.shape[1] - band.shape[2]
            max_dim = max(band.shape[1],band.shape[2])
            dif_perc = abs(dif_dims)/max_dim
            if dif_perc > 0.05:
                skip = True
            else:
                skip = False
            if skip == False:
                if dif_dims < 0 :
                    band = band[:,:,abs(dif_dims):]
                elif dif_dims > 0 :
                    band = band[:,abs(dif_dims):,:]
                return band     
                
                
        
        b1r_c = scale_20(mirror_dims(b1r))
        b5r_c = scale_20(mirror_dims(b5r))
        b6r_c = scale_20(mirror_dims(b6r))
        b7r_c = scale_20(mirror_dims(b7r))
        b8ar_c = scale_20(mirror_dims(b8ar))
        b9r_c = scale_20(mirror_dims(b9r))
        b10r_c = scale_20(mirror_dims(b10r))
        b11r_c = scale_20(mirror_dims(b11r))
        b12r_c = scale_20(mirror_dims(b12r))
        
        '''
        blue=492 +/- 66, green=560 +/- 36, red=664 +/- 31, nir=833 +/- 106  [10 meter] RGB and NIR
        b5=704.1 +/- 15, b6=740.5 +/- 15, b7=782.8 +/- 20, b8a=864.7 +/- 21 [20 meter] Veg rededge / narrow NIR
        b11=1613 +/- 91, b12=2202 +/- 175                                   [20 meter] SWIR
        b1=442.7 +/- 21, b9=945 +/- 20, b10=1373 +/- 31                     [60 meter] Cloud & Atm
        b1=coastal aerosol, b5-7=vegetation red edge, b8a=narrow NIR, b9=water vapour, b10=SWIR-cirrus, b11-12=SWIR 
        '''
        
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (nir.astype(float)-red.astype(float))/(nir+red) #normalized difference vegetation index
        ndwi = (green.astype(float)-nir.astype(float))/(nir+green) #normalized difference water index
        rededge = b5r_c.astype(float)/red
        rededge2 = (b5r_c.astype(float) - red.astype(float))/(b5r_c.astype(float)+red)
        ccci = ((rededge.astype(float)-nir.astype(float))/(nir.astype(float)+rededge.astype(float)))/ndvi.astype(float) #canopy chlorophyll content index
        ci_g = (nir.astype(float)/green) -1 #chlorophyll  index green
        X=0.08; a=1.22; b=0.03; L=0.5
        atsavi = a*((nir-a*red-b)/(a*nir+red-a*b+X*(1+a*a))) #adjusted transformed soil-adjusted Vegetation index
        savi = (1+L)*(nir.astype(float)-red.astype(float))/(nir+red+L) #soil adjusted vegetation index
        ndmi = (nir.astype(float)-b11r_c.astype(float))/(nir+b11r_c) #normalized diferenve moisture index
        gvmi = ((nir+0.1)-(b11r_c+0.02))/((nir+0.1)+(b11r_c+0.02)) #Global vegetation moisture index
        cvi = nir.astype(float) * (red.astype(float)/(green.astype(float)*green.astype(float))) # chlorophyll vegetation index
        dswi = (nir.astype(float)-green.astype(float)) / (b11r_c+red) # 	Disease water stress index
        lai=0.001*np.exp(8.7343*ndvi.astype(float)) #leaf area index
        bm = 451.99*lai.astype(float) + 1870.7 #biomass
        bwdrvi = (0.1*nir.astype(float)-blue.astype(float))/(0.1*nir+blue) #blue wide dynamic range vegetation index
        bri = ((1/green.astype(float))-(1/b5r_c.astype(float)))/nir.astype(float)  #brwoning reflectance index
        #s2 CP and NDF from paper, b1=560, b2=665, b3=865, b4=2202
        cp = 7.63-54.39*green.astype(float)/10000 + 31.37*red.astype(float)/10000 +21.23*b8ar_c.astype(float)/10000 -25.33*b12r_c.astype(float)/10000 #Crude Protein
        ndf = 54.49+207.53*green.astype(float)/10000 -193.99*red.astype(float)/10000 -54.19*b8ar_c.astype(float)/10000 +111.15*b12r_c.astype(float)/10000 #Neutral Detergent Fiber
        
        #cld_pxl_count = (np.count_nonzero(cld==0))/(cld[0].shape[0]*cld[0].shape[1])
        
        #custom own cloud detection algorithm
        ndgr = (green.astype(float)-red.astype(float))/(green+red) #normalized difference green/red
        tao = 0.2
        cld_custom = np.where( ((b11r_c/10000>tao) & ( ((green/10000>0.175) & (ndgr>0)) | (green/10000>0.39))) , 1, 0)
        cld_shadow = np.where( (green/10000<0.319) & (b8ar_c/10000<0.166) & (((green-b7r_c/10000<0.027) & (b9r_c-b11r_c/10000>-0.097)) | ((green-b7r_c/10000>0.027) & (b9r_c-b11r_c/10000>0.021))), 1, 0)
        cld_shadow2 = np.where( (green/10000<0.319) & (b5r_c/b11r_c>4.33) & (green/10000<0.525) & (b1r_c/b5r_c>1.184), 1, 0)                      
        cld_shd_tot = np.where( (cld_shadow2==1) |(cld_shadow==1) , 1, 0)     
           
        #CV para quitar ruido (pequenos puntos) y ampliar bordes para quitar sombras y zona de duda
        cld2 = cld_custom[0,:,:]
        cld2 = cld2.astype('uint8')
        kernel = np.ones((3,3),np.uint8)
        kernel2 = np.ones((5,5),np.uint8)
        erosion = cv2.erode(cld2,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 3)
        cld_custom = np.expand_dims(dilation, axis=0)
        
        cld_all = np.where(cld_shd_tot==1, 2, np.where(cld_custom==1, 1, 0)) #1=cloud, 2=shadow  
        cld_pxl_count = (np.count_nonzero(cld_custom==0))/(cld_custom[0].shape[0]*cld_custom[0].shape[1])
        
        indexes = [ndvi,ndwi,ccci,ci_g,atsavi,savi,ndmi,gvmi,cvi,dswi,lai,bm,bwdrvi,bri,cp,ndf]
        
        #corrections based on clouds
        def remove_clouds(index):
            index[cld_custom>=1] = None #>1
        for ind in indexes:
            remove_clouds(ind)
        
        indexes = [ndvi,ndwi,ccci,ci_g,atsavi,savi,ndmi,gvmi,cvi,dswi,lai,bm,bwdrvi,bri,cp,ndf,cld_custom]
        #indexes.append(cld_custom) #//no trae el nombre correctamente para TIF

        #cp[cp<5] = None
        #ndf[ndf>60] = None
        cp[ndvi<0.35] = None
        ndf[ndvi<0.35] = None
        
        
        def retrieve_name(var):
            callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var]
        
        #np.nanpercentile(cvi,90)

        def plot_figura(index, cmap, **kwargs): #, vmin=-1, vmax=1
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
                        
            fig = plt.figure(figsize=(10,10))
            #area chart
            ax = plt.gca()
            if cen == True:
                im = ax.imshow(index[0], cmap=cmap, clim=(vmin1, vmax1), norm=MidpointNormalize(midpoint=vcen1,vmin=vmin1, vmax=vmax1))
            elif ext==True:
                im = ax.imshow(index[0], cmap=cmap, vmin=vmin1, vmax=vmax1)
            else:
                im = ax.imshow(index[0], cmap=cmap, vmin=np.nanpercentile(index,1), vmax=np.nanpercentile(index,99)) #'RdYlGn'
            #plot
            plt.title(str(retrieve_name(index)[0])+' in Analysis area')
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            #to locate colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig(output_folder+analysis_area+'/'+date[:8]+"_"+str(retrieve_name(index)[0])+".png",bbox_inches='tight',dpi=300)
            plt.clf()
        
        #cmaps = 'RdYlGn', 'RdYlBu','nipy_spectral_r'
        plot_figura(ndvi,'RdYlGn',vmin=-1,vmax=1)
        plot_figura(ndwi,'RdYlBu', vmin=-1, vmax=0.4,vcen=0)
        plot_figura(ccci,'RdYlGn')
        plot_figura(ci_g,'RdYlGn')
        plot_figura(atsavi,'RdYlGn', vmin=0, vmax=1)
        plot_figura(savi,'RdYlGn')
        plot_figura(dswi,'RdYlGn')
        plot_figura(lai,'nipy_spectral_r', vmin=0, vmax=3)
        plot_figura(bm,'nipy_spectral_r', vmin=2000, vmax=3500)
        plot_figura(bwdrvi,'jet')
        plot_figura(bri,'terrain')
        plot_figura(cp,'RdYlGn', vmin=5, vmax=18)
        plot_figura(ndf,'RdYlGn', vmin=30, vmax=60)
        plot_figura(cld_custom,'gray',vmin=0,vmax=1)
        plot_figura(cld_all,'copper',vmin=0,vmax=2)
        
        #RGB     
        rgb=np.zeros((green.shape[1],green.shape[2],3))
        bands = [red, green, blue]
        for i,b in enumerate(bands):
            #copy of original RGB bands
            transformed_band = b.astype(float).copy()
            transformed_band[cld_custom>=1] = None
            #stats of bands color without clouds to normalize colors
            tr_avg = np.nanmean(transformed_band)
            tr_std= np.nanstd(transformed_band)
            #scale
            scaler = MinMaxScaler()
            x_sample = [tr_avg-2*tr_std, tr_avg+2*tr_std] #rango de AVG +/- STD DEV
            scaler.fit(np.array(x_sample)[:, np.newaxis]) #ajustado al rango de cada banda
            #reshape to apply same transformation to all rows and cols of image
            ascolumns = transformed_band[0,:,:].reshape(-1,1)
            t = scaler.transform(ascolumns) #fit_
            transformed = t.reshape(transformed_band.shape)
            #include again the clouds, but with a mask yellow over it
            #transformed[cld_custom>=1] = b 
            #include in big RGB file
            rgb[...,i] = transformed#_band
            
        rgb2 = (rgb.copy()*255).astype('uint8')
        #Reverse Red-Blue Channels as open cv will reverse again upon writing image on disk
        cv2.imwrite(output_folder+analysis_area+'/'+date[:8]+"_True_Color.jpg",rgb2[...,::-1])
        
        #export indexes as tif format
        meta = b4.meta
        meta.update(driver='GTiff')
        meta.update(dtype=rio.float32)
        def export_tif(index):           
            with rio.open(output_folder+analysis_area+'/'+date[:8]+"_"+str(retrieve_name(index)[0])+".tif", 'w', **meta) as dst:
                dst.write(index.astype(rio.float32))
                dst.close()
                b4.close()
     
        for ind in indexes:
            export_tif(ind)
        
        #Close opened bands
        bands = [b1,b2,b3,b4,b5,b6,b7,b8,b8a,b9,b10,b11,b12]#msk_cloud
        for ban in bands:
            ban.close()
        
        #store expanded bands // but can be more space required
        '''
        #should create a dictionary of bands vs real name band to save with same names
        expanded_bands = [b1r_c, b5r_c, b6r_c, b7r_c, b8ar_c, b9r_c, b10r_c, b11r_c, b12r_c]
        for exb in expanded_bands:
            export_tif(exb)
        '''
        return meta, cld_pxl_count  
        
    def trns_coor(area,meta):
        try:
            x, y = meta['transform'][2]+area[0]*10, meta['transform'][5]+area[1]*-10
        except:
            x, y = meta[0]['transform'][2]+area[0]*10, meta[0]['transform'][5]+area[1]*-10    
        return x, y
    
    def mosaic_files(unzipped_folder,analysis_area,qs):
        src_files_to_mosaic = []
        for fp in qs:
            src = rio.open(fp)
            src_files_to_mosaic.append(src)
        mosaic, out_trans = merge(src_files_to_mosaic)
        #show(mosaic, cmap='terrain')
        try:
            route = qs[0].split("/")[0]
            name = qs[0].split("/")[-1].split(".")[0]  
            folder_safe = qs[0].split("/")[-5] 
        except:
            route = qs[0].split("\\")[0]
            name = qs[0].split("\\")[-1].split(".")[0]
            folder_safe = qs[0].split("\\")[-5]
        name1 = name.split("_")[0][:3]+"_"+name.split("_")[1]+"_"+name.split("_")[2]
        # Copy the metadata
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update({"driver": "GTiff", #"JP2ECW", 
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                         }
                        ) 
        # Write the mosaic raster to disk
        with rio.open(unzipped_folder+analysis_area+"/mosaic/"+name1+".tif", "w", **out_meta) as dest: #".jp2" ".tif"
            dest.write(mosaic)
            dest.close()
        src.close()
        return route, name1, folder_safe
    
    #function to plot indexes by lote
    def small_area_crop_plot(date,aoig_near,analysis_area, source, output_folder): #aoi2,"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" ,lote_name
        with rio.open(output_folder+analysis_area+'/'+date[:8]+source+".tif") as src:
            indexes = ["_ndvi","_atsavi","_lai","_bm","_cp","_ndf"]
            #incluir aqui el loop para recortar los lotes, evitando I/O por cada lote
            for n, geo in enumerate (aoig_near.geometry):
                out_image, out_transform = rio.mask.mask(src, [geo],crop=True)  
                lote_name = aoig_near['name'][n] 
                #out_image, out_transform = rio.mask.mask(src, aoi2,crop=True) 
                if source==indexes[0]: #ndvi
                    Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder,lote_name,source,'RdYlGn',vmin=-1,vmax=1)
                elif source==indexes[1]: #atsavi
                    Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder,lote_name,source,'RdYlGn',vmin=0,vmax=1)
                elif source==indexes[2]: #lai
                    Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder,lote_name,source,'nipy_spectral_r',vmin=0,vmax=3)
                elif source==indexes[3]: #bm
                    Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder,lote_name,source,'nipy_spectral_r',vmin=2000,vmax=3500)
                elif source==indexes[4]: #cp
                    Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder,lote_name,source,'RdYlGn',vmin=5,vmax=18)
                elif source==indexes[5]: #ndf
                    Satellite_tools.plot_figura2(out_image, analysis_area, date, output_folder,lote_name,source,'RdYlGn',vmin=30,vmax=60)
                
            src.close()
            
            #output_folder+analysis_area+'/'+date[:8]+destination
