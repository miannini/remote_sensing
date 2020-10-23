# USAGE
#todo declarado
# python satellite_image_script_v3.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MAa0O5PMyE81I_AFC6E --download yes --date_ini 2020-01-01 --date_fin 2020-01-31 --shape external_shape --own no --erase yes
#todo declarado - sin external file
# python satellite_image_script_v3.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIyk5QHhyIGAnlsH3RJ --download yes --date_ini 2019-10-05 --date_fin 2020-09-30
#
#sin declarar nada
# python satellite_image_script_v3.py
# declarando solo fecha
# python satellite_image_script_v3.py --download no --date_ini 2020-01-01 --date_fin 2020-01-31
# declarando solo user
# python satellite_image_script_v3.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MAa0O5PMyE81I_AFC6E --download no
#analisis de imagenes locales
# python satellite_image_script_v5_locals.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIAbLizOODQRp_OCDFX --date_ini 2020-01-05 --date_fin 2020-01-30 --local local

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import shutil
import time
from myfunctions import Fire_down
from myfunctions import Cloud_finder
from myfunctions import Sentinel_downloader
from myfunctions import Satellite_proc
from myfunctions import Contour_detect
from myfunctions.temp_stats import Stats_charts
from myfunctions import Upload_fire
from myfunctions import Ext_shape
import argparse

## variables dinamicas para correr en terminal unicamente
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", default='no',
	help="path of working user/terrain")
ap.add_argument("-d", "--download", type=str, default='no',
	help="define if download is required yes/no")
ap.add_argument("-i", "--date_ini", type=str, default='no',
	help="define initial date yyyy-mm-dd")
ap.add_argument("-f", "--date_fin", type=str, default='no',
	help="define final date yyyy-mm-dd")
ap.add_argument("-s", "--shape", type=str, default='no',
	help="optional external shapes folder")
ap.add_argument("-o", "--own", type=str, default='yes',
	help="keep own terrain (only in conjunction with external shapes)")
ap.add_argument("-e", "--erase", type=str, default='no',
	help="define if erase unzipped images yes/no")
ap.add_argument("-l", "--local", type=str, default='no',
	help="define path of local images to use user_local/cloud/no")
args = vars(ap.parse_args())

#inicializacion de variables - fechas
Date_Ini = (args["date_ini"]) 
Date_Fin = (args["date_fin"]) 
#Date_Ini='2019-10-01'
#Date_Fin='2020-10-20'

#inicializacion de variables - user / area
user_analysis = (args["user"])
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MAa0O5PMyE81I_AFC6E' # Simijaca
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIAbLizOODQRp_OCDFX' #Riopaila
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIAboyeRxGdI7C0oWCl' #Sogamoso
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIAdI8GCpbfGLPY3KyM' #Uberrimo
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-M9nlimuyRUhXIlsAicA' #Rivera_llanos
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIykncYYNNNfVETEh-v' #Bioenergy_puerto_lopez
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIyk5QHhyIGAnlsH3RJ' #hoyo rico
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIzHZ9Kr8cUvkQyBgY5' #Ibague_arroz
#user_analysis = '7x27nHWFRKZhXePiHbVfkHBx9MC3/-MKK7Su7NDUn_-iY1DdQ' #guasimo
#user_analysis = 'no' # auto

if user_analysis != 'no' : 
    analysis_area = user_analysis.split("/")[1]

x_width = 768*2    #16km width
y_height = 768*2   #16km height
x_width_cloud = 512    #faster
y_height_cloud = 512   #faster
shape_folder = '../Data/shapefiles/'
#funcion para leer firebase
lote_aoi,lote_aoi_loc,minx,maxx,miny,maxy,bounding_box, user_analysis, analysis_area, Date_Ini, Date_Fin = Fire_down.find_poly(user_analysis,Date_Ini, Date_Fin, shape_folder)       

print("[INFO] box coordinates (min_x, max_x = {:.2f}, {:.2f})".format(minx,maxx))
print("[INFO] box coordinates (min_y, max_y = {:.2f}, {:.2f})".format(miny,maxy))
print("[INFO] tiempo de analisis (fecha_inicial, fecha_final = {}, {})".format(Date_Ini, Date_Fin))
print("[INFO] usuario y terreno de analisis (usuario, terreno, finca= {} / {})".format(user_analysis,lote_aoi["name"]))

    
#leer shapefile
aoi = gpd.read_file(shape_folder+analysis_area+'/big_box.shp') #para imagen satelital
aoi.crs = {'init':'epsg:32618', 'no_defs': True}
aoi_universal= aoi.to_crs(4326)                                 #para API sentinel
footprint = None
for i in aoi_universal['geometry']:                             #area
    footprint = i

#leer departamentos y municipios
'''
deptos = gpd.read_file(shape_folder+'/Colombia/depto/depto.shp')
deptos.crs = {'init':'epsg:21897', 'no_defs': True} #crs 21897 determinado usando QGIS
deptos = deptos.to_crs(4326)
departamento = deptos[deptos['geometry'].contains(footprint)]
'''
mpos = gpd.read_file(shape_folder+'/Colombia/mpos/MGN_MPIO_POLITICO.shp')
mpos.crs = {'init':'epsg:4326', 'no_defs': True}
mpos = gpd.overlay(mpos, lote_aoi, how='intersection')
municipio, departamento = mpos.loc[0,'MPIO_CNMBR'], mpos.loc[0,'DPTO_CNMBR']


database_folder = '../Data/Database/'
Path(database_folder+analysis_area).mkdir(parents=True, exist_ok=True)
    
#leer shapefiles externos si es requerido
folder_name = (args["shape"])
#folder_name = 'no' 'external_shape' #'no'
keep_own = (args["own"]) 
#keep_own = 'yes' 'no' #'yes' 
if folder_name != "no":
    todos_lotes, todos_lotes_loc = Ext_shape.merge_shapes(shape_folder,lote_aoi,lote_aoi_loc,folder_name,analysis_area)
    #si se definio external shapefiles, aoig_near contendra la misma info
    if keep_own != 'yes':
        todos_lotes=todos_lotes.iloc[1:,:] #remove row 0 is firebase terrain
        todos_lotes_loc=todos_lotes_loc.iloc[1:,:] #remove row 0 is firebase terrain
        print("[info] lote de firebase, removido del analisis")
    aoig_near = todos_lotes_loc
    lote_aoi_loc = todos_lotes_loc
    print("[info] lotes totales incluyendo de archivo externo = {}".format(len(todos_lotes)))
    #lote_aoi_loc.to_csv (r'export_lotes.csv', index = False, header=True)
    #todos_lotes.to_csv (r'lotes_universal.csv', index = False, header=True)
    #restart indexes
    aoig_near.reset_index(drop=True, inplace=True)
    lote_aoi_loc.reset_index(drop=True, inplace=True)   
    todos_lotes.reset_index(drop=True, inplace=True)
    #export to geojson
    Path(shape_folder+analysis_area+'/multiples_json/').mkdir(parents=True, exist_ok=True)
    for i in range(0,len(todos_lotes)):
        temp = todos_lotes[todos_lotes.index == i] #filter geodataframe, keeping same format to export
        temp.to_file(shape_folder+analysis_area+'/multiples_json/'+todos_lotes.iloc[i,0]+'.geojson', driver ='GeoJSON')

#folder de imagenes nubes
clouds_folder = '../Data/output_clouds/'
Path(clouds_folder+analysis_area).mkdir(parents=True, exist_ok=True)   
#cloud detection
#check if local clouds will be used
local_files = (args["local"]) 
#local_files = 'local' 'no' #'cloud' 
if local_files =='no':  
    start = time.time()          
    best_date, valid_dates, clouds_data, clear_pct, number_cld_analysis = Cloud_finder.cloud_process(bounding_box, Date_Ini, Date_Fin, x_width_cloud, y_height_cloud,analysis_area,clouds_folder,lote_aoi,municipio, departamento)
    end = time.time()
    dif = end - start
    print("[INFO] best date without clouds={}, total time (secs)={:.2f}, clear pctg of all dates={}%".format(best_date,dif,clear_pct*100))


#download images
zipped_folder='../Data/Zipped_Images/'
unzipped_folder='../Data/Unzipped_Images/'
Path(zipped_folder).mkdir(parents=True, exist_ok=True)
Path(unzipped_folder+analysis_area).mkdir(parents=True, exist_ok=True)

down_yes = (args["download"])
#down_yes = 'no' 'yes' 'no' 'yes'
if down_yes == 'yes':
    #Join area of multiple AOI, to download when partial satellite image contains the small AOI
    lotes_uni = lote_aoi_loc.to_crs(4326) 
    lotes_uni = lotes_uni['geometry'].unary_union 
    Sentinel_downloader.image_down(footprint, Date_Ini, Date_Fin, valid_dates, analysis_area,zipped_folder,unzipped_folder,lotes_uni)
direcciones = Sentinel_downloader.get_routes(analysis_area,unzipped_folder)
print(direcciones)

#release some memory
del(lote_aoi,minx,maxx,miny,maxy,bounding_box,aoi_universal,footprint)

#crop satellite images
output_folder='../Data/Output_Images/'
Path(output_folder+analysis_area).mkdir(parents=True, exist_ok=True)
start = time.time() 
big_proto = []
resumen_bandas = pd.DataFrame()
table_bandas = pd.DataFrame()
if local_files =='no': 
    R10=''
    date=''
    for dire in direcciones:
        R10=dire+'/'
        date= R10.split("_")[-2][:8]
        zone= R10.split("_")[-4][:]
        #find cloud mask date file    
        ind_mask = []
        date_obj = datetime.datetime.strptime(date, '%Y%m%d')
        print("[INFO] Date to Analyze = {}".format(date))
        for i in range(0,len(valid_dates)):
            date_msk = valid_dates.iloc[i,0].date()    
            if date_obj.date() == date_msk:
                ind_mask.append(i)
        #list bands
        onlyfiles = [f for f in listdir(R10) if isfile(join(R10, f))]
        #crop bands
        for ba in onlyfiles:
            if 'TCI' not in ba:          
                skip = Satellite_proc.crop_sat(R10,ba,aoi,analysis_area,output_folder,x_width)
        if skip == True:
            print("[INFO] fecha {}, zona {} recortada ... skip".format(date,zone))
            continue
        #cloud mask
        x_width_band, y_height_band = Satellite_proc.cld_msk(date, clouds_data, ind_mask, analysis_area,output_folder)
        
        #calculate NDVI
        meta = Satellite_proc.band_calc(date, analysis_area,x_width,output_folder) #calculate NDVI, crop='grass'
        matplotlib.pyplot.close("all")
        #database
        if folder_name != "no":
            analysis_date=output_folder+analysis_area+'/'+ date
            size_flag, datag, short_ordenado, short_resume = Stats_charts.data_g(date,analysis_date, aoig_near, todos_lotes, output_folder, analysis_area) #//aoig_near
            if size_flag:
                print(date)
            else:
                pd.DataFrame(big_proto.append(datag )) 
                resumen_bandas = pd.concat([resumen_bandas,short_resume])
                table_bandas = pd.concat([table_bandas,short_ordenado])
            #clear memory
            #del(size_flag, datag, short_ordenado, short_resume)
        print("[INFO] Date, zone Analyzed = {} - {}".format(date,zone))
    #write log to DB at end of process
    #processed_data = pd.read_csv (r'../Data/Database/DB_datos_proecsados.csv', index_col=0)   
    processed_df = pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'municipio':[municipio] , 'departamento':[departamento] ,'initial_date' : [Date_Ini] ,'final_date' : [Date_Fin], 'last_valid_date' : [max(valid_dates[0])], 'number_valid_date' : [len(valid_dates[0])], 'number_analyzed_images': [number_cld_analysis] , 'processed_date' : [datetime.date.today()]})    
    processed_df.to_csv (r'../Data/Database/DB_datos_proecsados.csv', index = True, header=False, mode='a')
        
    #local files
elif local_files == 'local':
    bands = ["B01.tif","B02.tif","B03.tif","B04.tif","B05.tif","B06.tif","B07.tif","B08.tif","B09.tif","B10.tif","B11.tif","B12.tif","B8A.tif","_cldmsk.tif"]
    #list bands
    onlyfiles = [f for f in listdir(output_folder+analysis_area) if isfile(join(output_folder+analysis_area, f))]
    matching = [s for s in onlyfiles if any(s[-7:] ==  b for b in bands)]
    dates = list(set([(m[:8]) for m in matching ]))
    count_of_clouds = pd.DataFrame()       
    for date in dates:
        #calculate indexes
        print("[INFO] Date to Analyze = {}".format(date))
        meta, cld_pxl_count = Satellite_proc.band_calc(date, analysis_area,x_width,output_folder) #calculate NDVI, crop='grass'
        matplotlib.pyplot.close("all")
        #dataframe for best_date image
        if count_of_clouds.empty:
            count_of_clouds = pd.DataFrame(data={'date' : [date], 'clear_pxl_count':[cld_pxl_count]})
        else:
            count_of_clouds = count_of_clouds.append(pd.DataFrame(data={'date' : [date], 'clear_pxl_count':[cld_pxl_count]}))
        
        #database
        if folder_name != "no":
            analysis_date=date
            size_flag, datag, short_ordenado, short_resume = Stats_charts.data_g(date,analysis_date, aoig_near, todos_lotes, output_folder, analysis_area) #//aoig_near
            if size_flag:
                print(date)
            else:
                pd.DataFrame(big_proto.append(datag )) 
                resumen_bandas = pd.concat([resumen_bandas,short_resume])
                table_bandas = pd.concat([table_bandas,short_ordenado])
            #clear memory
            #del(size_flag, datag, short_ordenado, short_resume)
        print("[INFO] Date, user_analysis Analyzed = {} - {}".format(date,analysis_area))
    #write log to DB at end of process
    form_date = datetime.datetime.strptime(max(dates),'%Y%m%d').strftime('%d/%m/%Y %H:%M')
    processed_df = pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'municipio':[municipio] , 'departamento':[departamento] ,'initial_date' : [Date_Ini] ,'final_date' : [Date_Fin], 'last_valid_date' : [form_date], 'number_valid_date' : [len(dates)], 'number_analyzed_images': [len(dates)] , 'processed_date' : [datetime.date.today()]})    
    processed_df.to_csv (r'../Data/Database/DB_datos_proecsados.csv', index = True, header=False, mode='a')
        
#else: #'cloud'
end = time.time()
print(end - start)   

best_date = count_of_clouds.loc[count_of_clouds.groupby('date')['clear_pxl_count'].idxmax()].date[0]

#exportar datos CSV
if folder_name != "no":
    #convert from pivot table to dataframe
    flattened = pd.DataFrame(table_bandas.to_records())
    flattened.columns = [hdr.replace("('", "").replace("')", "").replace("', '", ".") for hdr in flattened.columns]
    #Biomass corrected value
    #mean_value._BM, sum_value._BM, area, count_pxl._cldmsk, sum_value._cldmsk
    flattened['cld_percentage']=flattened["sum_value._cldmsk"]/flattened["count_pxl._cldmsk"]
    flattened['area_factor']= (flattened["count_pxl._BM"]/flattened["count_pxl._cldmsk"])*((100*flattened["count_pxl._BM"])/flattened["area"]) #mayor a 1 se debe reducir, menor a 1 se debe sumar
    flattened['biomass_corrected'] = flattened["mean_value._BM"]*(flattened["area"]/(100*100))
    flattened.to_csv (r'../Data/Database/'+analysis_area+'/resumen_lotes_medidas.csv', index = True, header=True)
    resumen_bandas.to_csv (r'../Data/Database/'+analysis_area+'/resumen_vertical_lotes_medidas.csv', index = True, header=True)
    print("[INFO] data table exported as CSV")

    
#pasar esto a una funcion.
#si se definir external shapes, no hacer esto. 
if folder_name == "no":
    #contornos
    if local_files =='no':
        best_date = str(best_date.date()).replace('-','')
    
    edged2, canvas = Contour_detect.read_image_tif(best_date,analysis_area,"_ndvi.tif",output_folder) #leer imagen y generar fondo negro
    lotesa_res = Contour_detect.identif_forms(edged2,50,10,200,1) #50 deteccion de contornos basado en NDVI 50/255 separador
    lotesb_res = Contour_detect.identif_forms(edged2,205,10,200,3) #205 deteccion de contornos basado en NDVI 205/255 separador
    #transformar coordenadas
    lotes_a = Contour_detect.trns_lst(lotesa_res,meta) #transformacion de coordenadas x,y pixel a EPSG32618
    lotes_b = Contour_detect.trns_lst(lotesb_res,meta) #transformacion de coordenadas x,y pixel a EPSG32618
    #graficar contornos
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cv2.drawContours(canvas, lotesa_res, -1, (255, 255, 255), 1))
    plt.imshow(cv2.drawContours(canvas, lotesb_res, -1, (255, 255, 255), 1))
    plt.savefig(shape_folder+analysis_area+'/detected_contours.png',bbox_inches='tight',dpi=200)
    
    #crear geodataframe
    aoig_a = Contour_detect.create_geodata(lotes_a)
    aoig_b = Contour_detect.create_geodata(lotes_b)
    crs = {'init': 'epsg:32618'}
    aoig_c =gpd.GeoDataFrame( pd.concat( [aoig_a,aoig_b], ignore_index=False) , crs=crs)
    aoig_c_plot = aoig_c.to_crs("epsg:4326")
    #export shapefile
    aoig_c.to_file(shape_folder+analysis_area+'/lotes_auto_.shp')
    
    #lotes cercanos
    aoig = gpd.GeoDataFrame(pd.concat([lote_aoi_loc,aoig_c], ignore_index=True), crs=lote_aoi_loc.crs)
    aoig["x"] = aoig.centroid.map(lambda p: p.x) 
    aoig["y"] = aoig.centroid.map(lambda p: p.y)
    aoig["area"] = aoig.area
    aoig['distance']=99999
    #calculate distance between centroids
    for n in range(1,len(aoig)):
        #aoig.iloc[n,3] = np.sqrt((aoig.iloc[n,2]-aoig.iloc[0,2])**2+(aoig.iloc[n,3]-aoig.iloc[0,3])**2)
        aoig.loc[n,'distance'] = np.sqrt((aoig.loc[n,'x']-aoig.loc[0,'x'])**2+(aoig.loc[n,'y']-aoig.loc[0,'y'])**2)
    #get 10 nearest polygons, with minimum distance of 500m
    aoig = aoig[aoig['distance']>=500]
    aoig_near = aoig.nsmallest(10, ['distance']) 
    aoig_near = gpd.GeoDataFrame( pd.concat( [lote_aoi_loc,aoig_near], ignore_index=True) , crs=lote_aoi_loc.crs)


#estadisticas
firebase_folder = '../Data/Images_to_firebase/'
Path(firebase_folder+analysis_area).mkdir(parents=True, exist_ok=True)
arr = os.listdir(output_folder+analysis_area)
out = list( [x[0:8] for x in arr])
out = set(out)
out = sorted(out)
list_dates = [out[0],out[round(len(out)/2)],out[-1]]
cnt = 0 #counter for image names
for data_i in list_dates : #cambiar por list_dates
    cnt = cnt + 1
    analysis_date=output_folder+analysis_area+'/'+ data_i
    folder_out = firebase_folder+analysis_area+'/'+ data_i
    Satellite_proc.area_crop(data_i,aoig_near,analysis_area,"_ndvi.tif", "_NDVI_lotes.tif",output_folder)# //aoig_near
    #Satellite_proc.plot_ndvi(data_i, analysis_area, "_NDVI_lotes.tif", "NDVI_Lote_&_Neighbors"+str(cnt)+".png", output_folder,'RdYlGn', -1, 1 ) # added cmap and limits
    if folder_name == "no":
        todos_lotes = aoig_near
        size_flag, datag, short_ordenado, short_resume = Stats_charts.data_g(data_i,analysis_date, aoig_near, todos_lotes, output_folder,analysis_area) #//aoig_near
        if size_flag:
            print(data_i)
        else:
            pd.DataFrame(big_proto.append(datag )) 
            resumen_bandas = pd.concat([resumen_bandas,short_resume])
            table_bandas = pd.concat([table_bandas,short_ordenado])
    #move NDVI images from output to firebase folder
    #shutil.move(output_folder+analysis_area+'/'+ data_i+"NDVI_Lote_&_Neighbors"+str(cnt)+".png", firebase_folder+analysis_area+'/'+ data_i+"NDVI_Lote_&_Neighbors"+str(cnt)+".png")
    #shutil.move(output_folder+analysis_area+'/'+ data_i+"_NDVI_analysis_lotes.png", firebase_folder+analysis_area+'/'+ data_i+"_NDVI_analysis_lotes.png")

#exportar datos CSV
if folder_name == "no":
    table_bandas.to_csv (r'../Data/Database/'+analysis_area+'/resumen_lotes_medidas.csv', index = True, header=True)
    resumen_bandas.to_csv (r'../Data/Database/'+analysis_area+'/resumen_vertical_lotes_medidas.csv', index = True, header=True)
    print("[INFO] data table exported as CSV")


big_proto_F = pd.concat(big_proto, axis = 0)
big_proto_F = big_proto_F.sort_values(by=['date' , 'poly'])
plgn_1 = big_proto_F[big_proto_F['poly'].isin([1])]
seasonplot = sns.boxplot(x="date", y="data_pixel",data=plgn_1, palette="Set3")
plt.title('polygon:'+'1')
plt.setp(seasonplot.get_xticklabels(), rotation=80)
plt.savefig(folder_out[:-8]+'NDVI_Lote_Over_Time.png',bbox_inches='tight',dpi=200) #Esto va al firebase
plt.clf()

#otras graficas
cnt = 0
for ld in list_dates:
    #ld = '20200102'
    cnt = cnt + 1
    ssn_1 = big_proto_F[big_proto_F['date'].isin([ld])]
    # plgnplot = None
    plgnplot = sns.boxplot(x="poly", y="data_pixel",
                           data=ssn_1, palette="Set3")
    plt.title(ld)
    plt.savefig(firebase_folder+analysis_area+'/'+'Lotes_Boxplot'+str(cnt)+'.png',bbox_inches='tight',dpi=200) #Esto va al firebase
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
plt.savefig(folder_out[:-8]+'NDVI_Lotes_MedianOVT.png',bbox_inches='tight',dpi=200) # esto al firebas
plt.clf()
    

#upload a firebase
Upload_fire.upload_image(firebase_folder,analysis_area,user_analysis)
print("[INFO] images uploaded to Firebase")

#delete downloaded unzipped images
erase_yes = (args["erase"])
#erase_yes='no'
if erase_yes == 'yes':
    shutil.rmtree(unzipped_folder, ignore_errors=True)
    print("[INFO] unzipped images erased")