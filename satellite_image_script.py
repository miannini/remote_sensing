# USAGE
# python satellite_image_script_v3.py --user XnpUeu6f9oaV7YMXG2NKTSa75EE3/-MAYVhF63-q6yNaIvqhs --download yes --date_ini 2020-01-01 --date_fin 2020-01-31

## leer librerias
import numpy as np
import pandas as pd
import os
import geopandas as gpd
from pathlib import Path
import datetime
from os import listdir
from os.path import isfile, join
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from myfunctions import Fire_down
from myfunctions import Cloud_finder
from myfunctions import Sentinel_downloader
from myfunctions.tools import Satellite_tools
from myfunctions import Contour_detect
from myfunctions.temp_stats import Stats_charts
from myfunctions import Upload_fire
import argparse

## variables dinamicas para correr en terminal unicamente
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", required=True,
	help="path of working user/terrain")
ap.add_argument("-d", "--download", type=str, default='no',
	help="define if download is required yes/no")
ap.add_argument("-i", "--date_ini", type=str, default='2020-01-01',
	help="define initial date yyyy-mm-dd")
ap.add_argument("-f", "--date_fin", type=str, default='2020-02-15',
	help="define final date yyyy-mm-dd")
args = vars(ap.parse_args())

#inicializacion de variables - fechas
Date_Ini = (args["date_ini"]) #'2020-01-01' #replace for dynamic dates
Date_Fin = (args["date_fin"]) #'2020-01-31' #replace for dynamic dates

#inicializacion de variables - user / area
user_analysis = (args["user"]) #'7x27nHWFRKZhXePiHbVfkHBx9MC3/-M9nlimuyRUhXIlsAicA' # 
#user_analysis = 'XnpUeu6f9oaV7YMXG2NKTSa75EE3/-MAYVhF63-q6yNaIvqhs'
analysis_area = user_analysis.split("/")[1]
x_width = 768*2    #16km width
y_height = 768*2   #16km height

#funcion para leer firebase
lote_aoi,lote_aoi_loc,minx,maxx,miny,maxy,bounding_box = Fire_down.find_poly(user_analysis)
print("[INFO] box coordinates (min_x, max_x = {:.2f}, {:.2f})".format(minx,maxx))
print("[INFO] box coordinates (min_y, max_y = {:.2f}, {:.2f})".format(miny,maxy))

#leer shapefile
aoi = gpd.read_file('shapefiles/'+analysis_area+'/big_box.shp') #para imagen satelital
aoi.crs = {'init':'epsg:32618', 'no_defs': True}
aoi_universal= aoi.to_crs(4326)                                 #para API sentinel
footprint = None
for i in aoi_universal['geometry']:                             #area
    footprint = i

#folder de imagenes nubes
Path('output_clouds/'+analysis_area).mkdir(parents=True, exist_ok=True)   
#cloud detection            
best_date, valid_dates, clouds_data = Cloud_finder.cloud_process(bounding_box, Date_Ini, Date_Fin, x_width, y_height,analysis_area)
print(best_date)

#download images
Path('Zipped_Images/').mkdir(parents=True, exist_ok=True)
Path('Unzipped_Images/'+analysis_area).mkdir(parents=True, exist_ok=True)

down_yes = (args["download"])
if down_yes == 'yes':
    Sentinel_downloader.image_down(footprint, Date_Ini, Date_Fin, valid_dates, analysis_area)
direcciones = Sentinel_downloader.get_routes(analysis_area)
print(direcciones)

#crop satellite images
Path('Output_Images/'+analysis_area).mkdir(parents=True, exist_ok=True)
R10=''
date=''
for dire in direcciones:
    R10=dire+'\\'
    date= R10.split("_")[-2][:8]
    #find cloud mask date file    
    ind_mask = []
    date_obj = datetime.datetime.strptime(date, '%Y%m%d')
    for i in range(0,len(valid_dates)):
        date_msk = valid_dates.iloc[i,0].date()    
        if date_obj.date() == date_msk:
            ind_mask.append(i)
    #list bands
    onlyfiles = [f for f in listdir(R10) if isfile(join(R10, f))]
    #crop bands
    for ba in onlyfiles:
        if 'TCI' not in ba:          
            Satellite_tools.crop_sat(R10,ba,aoi,analysis_area)
    
    #cloud mask
    x_width_band, y_height_band = Satellite_tools.cld_msk(date, clouds_data, ind_mask, analysis_area)
    
    #calculate NDVI
    meta = Satellite_tools.ndvi_calc(date, analysis_area) #calculate NDVI
    Satellite_tools.plot_ndvi(date, analysis_area, "_NDVI.tif", "_NDVI_export.png","Output_Images/") #export png file
    Satellite_tools.area_crop(date,lote_aoi_loc,analysis_area,"_NDVI.tif", "_NDVI_lote.tif","Output_Images/" ) #crop tif to small area analysis
    Satellite_tools.plot_ndvi(date, analysis_area, "_NDVI_lote.tif", "_NDVI_analysis_lotes.png","Output_Images/" ) #export png of small area analysis
    
#contornos
edged2, canvas = Contour_detect.read_image_tif(best_date,analysis_area) #leer imagen y generar fondo negro
lotesa_res = Contour_detect.identif_forms(edged2,50,10,200,1) #deteccion de contornos basado en NDVI 50/255 separador
lotesb_res = Contour_detect.identif_forms(edged2,205,10,200,3) #deteccion de contornos basado en NDVI 205/255 separador
#transformar coordenadas
lotes_a = Contour_detect.trns_lst(lotesa_res,meta) #transformacion de coordenadas x,y pixel a EPSG32618
lotes_b = Contour_detect.trns_lst(lotesb_res,meta) #transformacion de coordenadas x,y pixel a EPSG32618
#graficar contornos
fig = plt.figure(figsize=(10,10))
plt.imshow(cv2.drawContours(canvas, lotesa_res, -1, (255, 255, 255), 1))
plt.imshow(cv2.drawContours(canvas, lotesb_res, -1, (255, 255, 255), 1))
plt.savefig("shapefiles/"+analysis_area+'/detected_contours.png',bbox_inches='tight',dpi=200)

#crear geodataframe
aoig_a = Contour_detect.create_geodata(lotes_a)
aoig_b = Contour_detect.create_geodata(lotes_b)
crs = {'init': 'epsg:32618'}
aoig_c =gpd.GeoDataFrame( pd.concat( [aoig_a,aoig_b], ignore_index=False) , crs=crs)
aoig_c_plot = aoig_c.to_crs("epsg:4326")
#export shapefile
aoig_c.to_file("shapefiles/"+analysis_area+'/lotes_auto_.shp')

#lotes cercanos
aoig = gpd.GeoDataFrame(pd.concat([lote_aoi_loc,aoig_c], ignore_index=True), crs=lote_aoi_loc.crs)
aoig["x"] = aoig.centroid.map(lambda p: p.x) 
aoig["y"] = aoig.centroid.map(lambda p: p.y)
aoig['distance']=99999
#calculate distance between centroids
for n in range(1,len(aoig)):
    aoig.iloc[n,3] = np.sqrt((aoig.iloc[n,1]-aoig.iloc[0,1])**2+(aoig.iloc[n,2]-aoig.iloc[0,2])**2)
#get 10 nearest polygons, with minimum distance of 500m
aoig = aoig[aoig['distance']>=500]
aoig_near = aoig.nsmallest(10, ['distance']) 
aoig_near = gpd.GeoDataFrame( pd.concat( [lote_aoi_loc,aoig_near], ignore_index=True) , crs=lote_aoi_loc.crs)

#estadisticas
Path('Images_to_firebase/'+analysis_area).mkdir(parents=True, exist_ok=True)
arr = os.listdir('Output_Images/'+analysis_area)
out = list( [x[0:8] for x in arr])
out = set(out)
out = sorted(out)
list_dates = [out[0],out[round(len(out)/2)],out[-1]]
big_proto = []
for data_i in list_dates : #cambiar por list_dates
    analysis_date='Output_Images/'+analysis_area+'/'+ data_i
    folder_out = 'Images_to_firebase/'+analysis_area+'/'+ data_i
    Satellite_tools.area_crop(data_i,aoig_near,analysis_area,"_NDVI.tif", "_NDVI_lotes.tif",'Output_Images/') #agregar folder origen y destion, lo mismo para la funcion
    Satellite_tools.plot_ndvi(data_i, analysis_area, "_NDVI_lotes.tif", "_NDVI_benchmarking_lotes.png", "Images_to_firebase/" ) #export png of small area analysis
    size_flag, datag = Stats_charts.data_g(data_i,analysis_date, aoig_near)
    if size_flag:
        print(data_i)
    else:
        pd.DataFrame(big_proto.append(datag ))


big_proto_F = pd.concat(big_proto, axis = 0)
big_proto_F = big_proto_F.sort_values(by=['date' , 'poly'])
plgn_1 = big_proto_F[big_proto_F['poly'].isin([1])]
seasonplot = sns.boxplot(x="date", y="data_pixel",data=plgn_1, palette="Set3")
plt.title('polygon:'+'1')
plt.setp(seasonplot.get_xticklabels(), rotation=80)
plt.savefig(folder_out+'_NDVI_lote1.png',bbox_inches='tight',dpi=200) #Esto va al firebase
plt.clf()

#otras graficas
for ld in list_dates:
    ssn_1 = big_proto_F[big_proto_F['date'].isin([ld])]
    # plgnplot = None
    plgnplot = sns.boxplot(x="poly", y="data_pixel",
                           data=ssn_1, palette="Set3")
    plt.title(ld)
    plt.savefig('Images_to_firebase/'+analysis_area+'/'+ld+'_NDVI_lotes_oneDate.png',bbox_inches='tight',dpi=200) #Esto va al firebase
    plt.clf()

    all_plot = sns.lineplot(x="date", y="data_pixel",hue = 'poly' ,
                            err_style="bars" , data=big_proto_F, palette="Set3")
    plt.setp(all_plot.get_xticklabels(), rotation=80)
    plt.clf()
    
    median_array = big_proto_F.groupby(['date', 'poly'])[['data_pixel']].median()
    median_array.reset_index(inplace=True)
    median_array  = median_array[median_array['poly'].isin([1,2,3,4,5,6,7,8,9,10,11])]

    #median_array  = median_array[median_array['poly'].isin([0])]
    median_plot = sns.lineplot(x="date", y="data_pixel", hue = 'poly',
                               data=median_array, palette="Set3")
    plt.setp(median_plot.get_xticklabels(), rotation=80)
    plt.savefig(folder_out+'_NDVI_lotes_median.png',bbox_inches='tight',dpi=200) # esto al firebas
    plt.clf()
    
    #mover imagenes NDVI lote0 de las fechas a carpeta firebase
#crear de una vez base de datos de pixeles


#upload a firebase
Upload_fire.upload_image('Images_to_firebase/',analysis_area,user_analysis)
print("[INFO] images uploaded to Firebase")
