import os
import pandas as pd
import zipfile
import datetime
from sentinelsat import SentinelAPI
from geopandas import GeoSeries
import geopandas as gpd
from shapely.geometry import Polygon
user = 'miannini'
password = 'An4lytics@89'
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

class Sentinel_downloader:
    def image_down(footprint, Date_Ini, Date_Fin, valid_dates, analysis_area,zipped_folder,unzipped_folder,lotes_uni,user_analysis,municipio, departamento):
        Date_Ini_c = Date_Ini.replace('-','')
        Date_Fin_c = Date_Fin.replace('-','')
        products = api.query(footprint,
                     date = (Date_Ini_c, Date_Fin_c),
                     #area_relation = 'Contains',
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-1C'#,
                     #cloudcoverpercentage = (0,30)
                    )
        #Listado de imagenes satelitales dentro de la API
        products_gdf = api.to_geodataframe(products)
        products_gdf_sorted = products_gdf.sort_values(['beginposition'], ascending=[True]) #products_gdf.sort_values(['cloudcoverpercentage']
        
        #Measure intersection areas, to omit images with partial data
        products_gdf_sorted['intersec'] = products_gdf_sorted['geometry'].intersection(footprint).area
        #intersec = products_gdf_sorted['geometry'].intersection(footprint)
        #intersec.area  #result = 0.019214
        products_gdf_sorted['contains'] = products_gdf_sorted['geometry'].contains(footprint)
        max_inter = max(products_gdf_sorted['intersec'])

        #order to omit duplicated dates, sorted by date and intersection area
        products_gdf_sorted = products_gdf_sorted.sort_values(['beginposition','contains','intersec'], ascending=[True,True,False])
        #Leer el listado de imagenes por fechas en las que el porcentaje de nubes calculado sea menor al valor indicado
        file_list = []
        intersec_dates = [] #store analyzed dates
        contained_dates = []
        products_df = pd.DataFrame()
        for b in range(0,len(products_gdf_sorted)):
            api_id = str(products_gdf_sorted.iloc[b,7]).split()[0]
            for c in range(0,len(valid_dates)):
                valid_dates_id = str(valid_dates.iloc[c,0]).split()[0]
                #date of sentinel file and date of clouds valid are equal / and / intersection area of tile is near maximum
                #if(api_id==valid_dates_id and products_gdf_sorted['geometry'][b].contains(footprint)):
                if(api_id==valid_dates_id and (valid_dates_id not in contained_dates)): 
                    if(products_gdf_sorted['contains'][b] == True):
                        file = products_gdf_sorted['uuid'][b]
                        title = products_gdf_sorted['title'][b]
                        file_list.append(file)
                        #ready_dates.append(valid_dates_id)
                        contained_dates.append(valid_dates_id)
                        print(file,title,'contained')
                        if products_df.empty:
                            products_df = pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'title':[title], 'file':[file], 'municipio':[municipio] , 'departamento':[departamento] ,'date' : [valid_dates_id.replace("-","")] ,'mode' : "contained_footprint", 'processed_date' : [datetime.date.today()]})    
                        else:
                            products_df = products_df.append(pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'title':[title], 'file':[file], 'municipio':[municipio] , 'departamento':[departamento] ,'date' : [valid_dates_id.replace("-","")] ,'mode' : "contained_footprint", 'processed_date' : [datetime.date.today()]}))
                    elif(products_gdf_sorted['geometry'][b].contains(lotes_uni) and products_gdf_sorted['intersec'][b] >= max_inter - max_inter*0.3):
                        file = products_gdf_sorted['uuid'][b]
                        title = products_gdf_sorted['title'][b]
                        file_list.append(file)
                        intersec_dates.append(valid_dates_id)
                        print(file,title,'intersected')
                        if products_df.empty:
                            products_df = pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'title':[title], 'file':[file], 'municipio':[municipio] , 'departamento':[departamento] ,'date' : [valid_dates_id.replace("-","")] ,'mode' : "intersected", 'processed_date' : [datetime.date.today()]})    
                        else:
                            products_df = products_df.append(pd.DataFrame(data={'user' : [user_analysis.split("/")[0]], 'terrain':[analysis_area], 'title':[title], 'file':[file], 'municipio':[municipio] , 'departamento':[departamento] ,'date' : [valid_dates_id.replace("-","")] ,'mode' : "intersected", 'processed_date' : [datetime.date.today()]}))
                        
        #unique files // remove because of duplicated dates in clouds analysis
        file_list = list(dict.fromkeys(file_list))
        products_df.to_csv (r'../Data/Database/DB_downloaded_files.csv', index = False, header=False, mode='a')
        #Descarga de Imagenes por bandas
        for n in file_list:
            api.download(n, zipped_folder)
        
        #Descomprimir Metadata y eliminación de archivos descargados
        for root, dirs, files in os.walk(zipped_folder):
            for file in files:
                if file.endswith(".zip"):
                        zipfile_path = os.path.join(root, file)
                        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                            zip_ref.extractall(unzipped_folder+analysis_area)
                        os.remove(zipfile_path)
        return products_df
    #obtener rutas de las imagenes .jp2                    
    def get_routes(analysis_area,unzipped_folder):
        direcciones=[]
        for root, dirs, files in os.walk(unzipped_folder+analysis_area):
            for file in files:
                if file.endswith(".jp2"):
                    x=os.path.join(root, file)
                    foldi = x.split(os.path.sep)[-2]
                    if foldi == 'IMG_DATA':
                        direcciones.append(os.path.join(root))
        direcciones=list(set(direcciones))
        return direcciones
        
