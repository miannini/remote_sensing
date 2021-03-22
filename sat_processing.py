# USAGE
#todo declarado
# python sat_processing.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MAa0O5PMyE81I_AFC6E --download yes --date_ini 2020-03-01 --date_fin 2020-03-20 --shape external_shape --own no --erase yes
#todo declarado - sin external file
# python sat_processing.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIyk5QHhyIGAnlsH3RJ --download yes --date_ini 2019-10-05 --date_fin 2020-09-30
# python sat_processing.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MIyk5QHhyIGAnlsH3RJ --shape ID_CLIENTE-1
#
#sin declarar nada
# python sat_processing.py
# declarando solo fecha
# python sat_processing.py --download no --date_ini 2020-01-01 --date_fin 2020-01-31
# declarando solo user
# python sat_processing.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MAa0O5PMyE81I_AFC6E --download no
#analisis de imagenes locales
# python sat_processing.py --date_ini 2019-10-01 --date_fin 2020-10-20 --download yes --erase yes

### leer librerias
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
import glob
from myfunctions import Fire_down
#from myfunctions import Cloud_finder
from myfunctions import Sentinel_downloader
from myfunctions import Satellite_proc
from myfunctions import Contour_detect
from myfunctions.temp_stats import Stats_charts
from myfunctions import Upload_fire
from myfunctions import Ext_shape
from myfunctions.tools import GCP_Functions
import argparse


### variables dinamicas para correr en terminal unicamente
# argparse permite ingresar variables en CMD para correr python desde hi ocn inputs
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", default='no',
	help="path of working user/terrain")
ap.add_argument("-d", "--download", type=str, default='yes',
	help="define if download is required yes/no")
ap.add_argument("-i", "--date_ini", type=str, default='no',
	help="define initial date yyyy-mm-dd")
ap.add_argument("-f", "--date_fin", type=str, default='no',
	help="define final date yyyy-mm-dd")
ap.add_argument("-s", "--shape", type=str, default='no',
	help="optional external shapes folder")
ap.add_argument("-o", "--own", type=str, default='no',
	help="keep own terrain (only in conjunction with external shapes)")
ap.add_argument("-e", "--erase", type=str, default='yes',
	help="define if erase unzipped images yes/no")
ap.add_argument("-l", "--local", type=str, default='no',
	help="define path of local images to use user_local/cloud/no")
args = vars(ap.parse_args())

#inicializacion de variables - fechas
Date_Ini = (args["date_ini"]) 
Date_Fin = (args["date_fin"]) 
#Date_Ini= '2021-03-10'#'2020-09-11'
#Date_Fin= '2021-03-20'#'2021-02-02'

if Date_Ini == 'no':
    Date_Ini = (datetime.date.today()-datetime.timedelta(days=5)).strftime("%Y-%m-%d")
if Date_Fin == 'no':
    Date_Fin = datetime.date.today().strftime("%Y-%m-%d")
    
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

### si el usuario se deifnio, tomarlo del string tomando la segunda parte despues de /
if user_analysis != 'no' : 
    analysis_area = user_analysis.split("/")[1]

### definir constantes basicas para medidas
# todo se hace con base 512, las nubes se descargan a 512pxl, mientras que las areas de descarga
# se hacen a nivel dde 1536pxl para hacer un analisis de area grande
# las nubes se interpolaran x 3 para ahorrar espacio y tiempo, haciendo compatible las imagenes
x_width = 768*2    #16km width
y_height = 768*2   #16km height
x_width_cloud = 512    #faster
y_height_cloud = 512   #faster

#shapes and municipios files
shape_folder = '../Data/shapefiles/'
folder = 'Colombia/mpos/'
objetos = list(GCP_Functions.list_all_blobs('shapefiles-storage',prefix=folder,delimiter='/')) #list fro cloud storage
destination = shape_folder + folder
Path(destination).mkdir(parents=True, exist_ok=True) #create folder
for n in objetos:
    GCP_Functions.download_blob('shapefiles-storage', n, destination + n.split('/')[-1]) #dowloand fro GCP storage
    

#unzipped_folder='../Data/Unzipped_Images/'
#Path(shape_folder).mkdir(parents=True, exist_ok=True)

###funcion para leer firebase
# de aqui se leen los lotes del json, coordenadas, caja de coordenadas GPS, entre otros
# si no se define usuario o fechas, estos se basaran en la informacion de Firebase 
'''
pasar esto para abajo. calclular caja, lote_aoi, centroide basado en union de lotes shapefiles
download files from cloud store "shapefiles-storage/Colombia/mpos" y "shapefiles-storage/ID_CLIENTE-1" and copy in .Shapes/ folder
read lotes, create box, generate AOI
avoid Firebase
'''
Path(shape_folder+analysis_area).mkdir(parents=True, exist_ok=True)
lote_aoi,lote_aoi_loc,minx,maxx,miny,maxy,bounding_box, user_analysis, analysis_area, Date_Ini, Date_Fin = Fire_down.find_poly(user_analysis,Date_Ini, Date_Fin, shape_folder)       


### print de lo que se encuentra en Firebase para mostrar que se analizara
print("[INFO] box coordinates (min_x, max_x = {:.2f}, {:.2f})".format(minx,maxx))
print("[INFO] box coordinates (min_y, max_y = {:.2f}, {:.2f})".format(miny,maxy))
print("[INFO] tiempo de analisis (fecha_inicial, fecha_final = {}, {})".format(Date_Ini, Date_Fin))
print("[INFO] usuario y terreno de analisis (usuario, terreno, finca= {} / {})".format(user_analysis,lote_aoi["name"]))

    
###leer shapefile
# el area big_box se basa en firebase, que previamente se guardo en el sistema
aoi = gpd.read_file(shape_folder+analysis_area+'/big_box.shp') #para imagen satelital

# el CRS y EPSG son terminos geo-espaciales, para definir referencia base de esfera a plano
aoi.crs = {'init':'epsg:32618', 'no_defs': True}
aoi_universal= aoi.to_crs(4326)                                 #para API sentinel

# el footprint es para hallar imagenes satelitales que contengan esta area
# y despues extraer la parte de la imagen que solo contiene esta area de interes
footprint = None
for i in aoi_universal['geometry']:                             #area
    footprint = i

### leer departamentos y municipios
#basado en archivo externo con shapes de los municipios y dptos de Colombia
mpos = gpd.read_file(shape_folder+'/Colombia/mpos/MGN_MPIO_POLITICO.shp')
#estandarizar coordenadas a un mismo sistema de referencia
mpos.crs = {'init':'epsg:4326', 'no_defs': True}
mpos = gpd.overlay(mpos, lote_aoi, how='intersection')
#traer solo el municipio y departamento mas cercano
municipio, departamento = mpos.loc[0,'MPIO_CNMBR'], mpos.loc[0,'DPTO_CNMBR']

# crear folder para guardar 'database'
# reemplazar por DB real de SQL o archivo de cloud storage
database_folder = '../Data/Database/'
Path(database_folder+analysis_area).mkdir(parents=True, exist_ok=True)
    
#leer shapefiles externos si es requerido
folder_name = (args["shape"])
#folder_name = 'ID_CLIENTE-1' 'external_shape' #'no'
keep_own = (args["own"]) 
#keep_own = 'no' 'yes' 'no' #'yes' 

#prefixes
fincas = GCP_Functions.list_gcs_directories('shapefiles-storage', folder_name+'/')
for f in fincas:
    destination = shape_folder + f
    Path(destination).mkdir(parents=True, exist_ok=True) #create folder
    #list and download files in corresponding folders
    objetos = list(GCP_Functions.list_all_blobs('shapefiles-storage',prefix=f,delimiter='/')) 
    for n in objetos:
        GCP_Functions.download_blob('shapefiles-storage', n, destination + n.split('/')[-1])
     

### si se pasa un archivo de shapefeiles, crear jsons de lotes individuales
#esto se puede cambiar a data en el storage de GCP, buscando si hay shapefile 
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
    #restart indexes
    aoig_near.reset_index(drop=True, inplace=True)
    lote_aoi_loc.reset_index(drop=True, inplace=True)   
    todos_lotes.reset_index(drop=True, inplace=True)
    #export to geojson
    Path(shape_folder+analysis_area+'/multiples_json/').mkdir(parents=True, exist_ok=True)
    for i in range(0,len(todos_lotes)):
        temp = todos_lotes[todos_lotes.index == i] #filter geodataframe, keeping same format to export
        temp.to_file(shape_folder+analysis_area+'/multiples_json/'+todos_lotes.iloc[i,0]+'.geojson', driver ='GeoJSON')


### folder de imagenes nubes
clouds_folder = '../Data/output_clouds/'
Path(clouds_folder+analysis_area).mkdir(parents=True, exist_ok=True)   

### cloud detection
#check si se usaran archivos locales de nubes .... es raro, solo en caso de fallas
local_files = (args["local"]) 
#local_files = 'no' 'local' 'no' #'cloud' 
'''
if local_files =='no':  
    start = time.time()   
    # fechas validas, mejores fechas, datos de nubes, porcentajes
    best_date, valid_dates, clouds_data, clear_pct, number_cld_analysis = Cloud_finder.cloud_process(bounding_box, Date_Ini, Date_Fin, x_width_cloud, y_height_cloud,analysis_area,clouds_folder,lote_aoi,municipio, departamento)
    end = time.time()
    dif = end - start
    print("[INFO] best date without clouds={}, total time (secs)={:.2f}, clear pctg of all dates={}%".format(best_date,dif,clear_pct*100))
'''

### download images
zipped_folder='../Data/Zipped_Images/'
unzipped_folder='../Data/Unzipped_Images/'
Path(zipped_folder).mkdir(parents=True, exist_ok=True)
Path(unzipped_folder+analysis_area).mkdir(parents=True, exist_ok=True)

down_yes = (args["download"])
#down_yes =  'yes' 'no' 'yes'

#check si se descargaran datos ... es raro que no, solo en caso de fallas o pruebas
if down_yes == 'yes':
    #Join area of multiple AOI, to download when partial satellite image contains the small AOI
    lotes_uni = lote_aoi_loc.to_crs(4326) 
    lotes_uni = lotes_uni['geometry'].unary_union 
    products_df = Sentinel_downloader.image_down(footprint, Date_Ini, Date_Fin, analysis_area,zipped_folder,unzipped_folder,lotes_uni,user_analysis,municipio, departamento)
    products_df['date'] = products_df['date'].astype(str)#.replace("-","")
    products_df['date'] = products_df['date'].str.split(' ').str[0].str.replace("-","")
    #products2 = products_df['date'].str.split(' ').str[0]
else:
    products_df = pd.read_csv('../Data/Database/DB_downloaded_files.csv')
    products_df = products_df[products_df['terrain']== analysis_area]
    products_df['date'] = products_df['date'].astype(str)
    products_df['date'] = products_df['date'].str.split(' ').str[0].str.replace("-","")
direcciones = Sentinel_downloader.get_routes(analysis_area,unzipped_folder)
print(direcciones)


### Recortar y procesar imagenes satelitales
output_folder='../Data/Output_Images/'
Path(output_folder+analysis_area).mkdir(parents=True, exist_ok=True)
#contabilizar tiempo
start = time.time() 
#inicializar listas y dataframes
big_proto = []
dates_ready = []
resumen_bandas = pd.DataFrame()
table_bandas = pd.DataFrame()
count_of_clouds = pd.DataFrame() 

#si no se tienen archivos locales //es lo normal
if local_files =='no': 
    #inicializar variables
    R10=''
    date=''
    hora=''
    date_max=date
    for dire in direcciones:
        R10=dire+'/'
        date= R10.split("_")[-2][:8]
        hora=R10.split("_")[-2][9:15]
        zone= R10.split("_")[-4][:]
        #find cloud mask date file    
        ind_mask = []
        datelong = date+" " +hora
        date_obj = datetime.datetime.strptime(datelong, '%Y%m%d %H%M%S')
        if date>date_max: date_max=date #para obtener fecha maxima analizada
        print("[INFO] Date to Analyze = {}".format(date))
        if date in dates_ready: 
            print("omit date due to mosaic")
            #delete safe folder
            try: #para windows
                file_name = dire.split("/")[-4]
            except: #para linux
                file_name = dire.split("\\")[-4]
            shutil.rmtree(unzipped_folder+analysis_area+"/"+file_name+"/", ignore_errors=True)
            continue

        '''
        for i in range(0,len(valid_dates)):
            #date_msk = valid_dates.iloc[i,0].date() 
            date_msk_1 = datetime.datetime.strptime(valid_dates[i][0], '%Y-%m-%d') #str(valid_dates.iloc[c,0]).split()[0]
            date_msk_2 = datetime.datetime.strptime(valid_dates[i][1], '%Y-%m-%d')
            if (date_obj >= date_msk_1 and date_obj <= date_msk_2):
                ind_mask.append(i)
        '''
        #list bands
        onlyfiles = [f for f in listdir(R10) if isfile(join(R10, f))]
        #review if date is in products_df and is contained
        
        tile_date = products_df[products_df['date'] == date]
        if (tile_date[tile_date["mode"] == 'contained_footprint'].size == 0): #si la geometria NO esta contenida en la imagen al 100%
            #activate alteranive tiles mosaic
            print("zona requiere 2 areas unidas para analisis")
            list_files_same_date=[]
            for dires2 in direcciones:
                try:
                    file_name = dires2.split("/")[-4].split(".")[0]
                except:
                    file_name = dires2.split("\\")[-4].split(".")[0]
                for tiles in tile_date.title:
                    if file_name == tiles:
                        list_files_same_date.append(dires2)
            #listado de bandas
            list_extensions = ['*B01.jp2', '*B02.jp2','*B03.jp2','*B04.jp2','*B05.jp2','*B06.jp2','*B07.jp2','*B08.jp2','*B8A.jp2','*B09.jp2','*B10.jp2','*B11.jp2','*B12.jp2']
            #get files and dirpath based on extensions
            for search_criteria in list_extensions:
                qs=[]
                for dirpath in list_files_same_date:
                    q = os.path.join(dirpath, search_criteria)
                    dem_fps = glob.glob(q, recursive=True)
                    qs = qs + dem_fps
                
                
                #mosaic files
                Path(unzipped_folder+analysis_area+"/mosaic/").mkdir(parents=True, exist_ok=True)
                route, name1, folder_safe = Satellite_proc.mosaic_files(unzipped_folder,analysis_area,qs)
            dates_ready.append(date)
            #delete original folder
            shutil.rmtree(unzipped_folder+analysis_area+"/"+folder_safe+"/", ignore_errors=True)
            print("[INFO] .SAFE folder unzipped erased")
            
            #list bands
            onlyfiles = [f for f in listdir(unzipped_folder+analysis_area+"/mosaic/") if isfile(join(unzipped_folder+analysis_area+"/mosaic/", f))]
            #crop bands
            for ba in onlyfiles:
                if 'TCI' not in ba:          
                    skip = Satellite_proc.crop_sat(unzipped_folder+analysis_area+"/mosaic/",ba,aoi,analysis_area,output_folder,x_width)
        
        else: #cuando la geometria si esta 100% en una sola imagen
            #crop bands, al tamano requerido 
            for ba in onlyfiles:
                if 'TCI' not in ba:          
                    skip = Satellite_proc.crop_sat(R10,ba,aoi,analysis_area,output_folder,x_width)
            if skip == True:
                print("[INFO] fecha {}, zona {} recortada ... skip".format(date,zone))
                continue
        #cloud mask // no requerido si las nubes se detectan con codigo propio
        #x_width_band, y_height_band = Satellite_proc.cld_msk(date, clouds_data, ind_mask, analysis_area,output_folder)
        
        #calculate indexes, clouds and all others
        meta, cld_pxl_count = Satellite_proc.band_calc(date, analysis_area,x_width,output_folder) #calculate NDVI, crop='grass'
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
        
        #clouds database
        if count_of_clouds.empty:
            count_of_clouds = pd.DataFrame(data={'date' : [date], 'clear_pxl_count':[cld_pxl_count]})
        else:
            count_of_clouds = count_of_clouds.append(pd.DataFrame(data={'date' : [date], 'clear_pxl_count':[cld_pxl_count]}))
        print("[INFO] Date, zone Analyzed = {} - {}".format(date,zone))
        
    #write log to DB at end of process
    #processed_data = pd.read_csv (r'../Data/Database/DB_datos_proecsados.csv', index_col=0)   
    processed_df = pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'municipio':[municipio] , 'departamento':[departamento] ,'initial_date' : [Date_Ini] ,'final_date' : [Date_Fin], 'last_valid_date' : [date_max] , 'number_valid_date' : [len(direcciones)], 'number_analyzed_images': [len(direcciones)]  , 'processed_date' : [datetime.date.today()]})  #[valid_dates[-1][0]], [number_cld_analysis]  
    processed_df.to_csv (r'../Data/Database/DB_datos_proecsados.csv', index = True, header=False, mode='a')
    shutil.rmtree(unzipped_folder+analysis_area+"/mosaic/", ignore_errors=True)    

#local files para nubes /// muy raro usarlo
'''
elif local_files == 'local':
    bands = ["B01.tif","B02.tif","B03.tif","B04.tif","B05.tif","B06.tif","B07.tif","B08.tif","B09.tif","B10.tif","B11.tif","B12.tif","B8A.tif","_cldmsk.tif"]
    #list bands
    onlyfiles = [f for f in listdir(output_folder+analysis_area) if isfile(join(output_folder+analysis_area, f))]
    matching = [s for s in onlyfiles if any(s[-7:] ==  b for b in bands)]
    dates = list(set([(m[:8]) for m in matching ]))      
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

        print("[INFO] Date, user_analysis Analyzed = {} - {}".format(date,analysis_area))
    #write log to DB at end of process
    form_date = datetime.datetime.strptime(max(dates),'%Y%m%d').strftime('%d/%m/%Y %H:%M')
    processed_df = pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'municipio':[municipio] , 'departamento':[departamento] ,'initial_date' : [Date_Ini] ,'final_date' : [Date_Fin], 'last_valid_date' : [form_date], 'number_valid_date' : [len(dates)], 'number_analyzed_images': [len(dates)] , 'processed_date' : [datetime.date.today()]})    
    processed_df.to_csv (r'../Data/Database/DB_datos_proecsados.csv', index = True, header=False, mode='a')
'''
        
#finalizar contadores y obtener tiempo total
end = time.time()
print(end - start)   
if 'best_date' not in locals():
    count_of_clouds.reset_index(drop=True, inplace=True)
    best_date = count_of_clouds.loc[count_of_clouds.groupby('date')['clear_pxl_count'].idxmax()].date[0]


#exportar datos CSV
if folder_name != "no": #shapefile fue provisto entonces
    #convert from pivot table to dataframe
    flattened = pd.DataFrame(table_bandas.to_records())
    #corregir titulos de las columnas, para qutar parentesis
    flattened.columns = [hdr.replace("('", "").replace("')", "").replace("', '", ".") for hdr in flattened.columns]    
    #._cldmsk cambiado a ._ind por nueva forma de detectar nubes
    flattened['cld_percentage']=flattened["sum_value._ind"]/flattened["count_pxl._ind"]
    flattened['area_factor']= (flattened["count_pxl._bm"]/flattened["count_pxl._ind"])*((100*flattened["count_pxl._bm"])/flattened["area"]) #mayor a 1 se debe reducir, menor a 1 se debe sumar
    #Biomass corrected value
    flattened['biomass_corrected'] = flattened["mean_value._bm"]*(flattened["area"]/(100*100))
    ''' aqui voy'''
    #esto enviarlo directo a storage
    #version resumida de 'resumen_lotes_medidas' enviar a database directamente [agregar finca, quitar percentiles, bandas y coordenadas]
    flattened.to_csv (r'../Data/Database/'+analysis_area+'/resumen_lotes_medidas'+Date_Fin+'.csv', index = True, header=True)
    resumen_bandas.to_csv (r'../Data/Database/'+analysis_area+'/resumen_vertical_lotes_medidas'+Date_Fin+'.csv', index = True, header=True)
    print("[INFO] data table exported as CSV")

    
#pasar esto a una funcion.
#si se definir external shapes, no hacer esto. 
if folder_name == "no":
    #contornos
    if local_files =='no':
        best_date = str(best_date.date()).replace('-','') #revisar esto ... puede fallar
    
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
#firebase_folder = '../Data/Images_to_firebase/'
#Path(firebase_folder+analysis_area).mkdir(parents=True, exist_ok=True)


#obtener fechas basado en nombres, organizar y extraer inicial, media y final
png_folder = '../Data/PNG_Images/'
Path(png_folder+analysis_area).mkdir(parents=True, exist_ok=True)

#listar los archivos en folder de Output_Images
arr = os.listdir(output_folder+analysis_area)

start = time.time() 
indexes = ["_ndvi","_atsavi","_lai","_bm","_cp","_ndf"]
dates_in = list( [x[0:8] for x in arr])
dates_in = sorted(set(dates_in))
for data_i in dates_in : #cambiar por list_dates
    for inde in indexes:
        analysis_date=output_folder+analysis_area+'/'+ data_i
        #folder_out = firebase_folder+analysis_area+'/'+ data_i
        #recortar para cada lote y fecha, tomando cada indice y guardando .png
        Satellite_proc.small_area_crop_plot(data_i,aoig_near,analysis_area, inde, output_folder, png_folder)
end = time.time()
print(end - start) 

'''
    if folder_name == "no": #no shapefile passed
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
'''



#exportar datos CSV
if folder_name == "no":
    table_bandas.to_csv (r'../Data/Database/'+analysis_area+'/resumen_lotes_medidas'+Date_Fin+'.csv', index = True, header=True)
    resumen_bandas.to_csv (r'../Data/Database/'+analysis_area+'/resumen_vertical_lotes_medidas'+Date_Fin+'.csv', index = True, header=True)
    print("[INFO] data table exported as CSV")


### graficas estadistocas /// creo que esto no se esta usano, y se podria hacer mejor en el tablero o app
'''
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
'''    

### upload a firebase
#esto se podria quitar, y dejar solo en google cloud store
#Upload_fire.upload_image(firebase_folder,analysis_area,user_analysis)
#print("[INFO] images uploaded to Firebase")

### delete downloaded unzipped images
#esto se podria quitar, y mas bien enviar a un coldstorage de las bandas cropped, no la imagen total
erase_yes = (args["erase"])
#erase_yes= 'yes'
if erase_yes == 'yes':
    shutil.rmtree(unzipped_folder, ignore_errors=True)
    print("[INFO] unzipped images erased")
    