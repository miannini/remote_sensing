import os
import pandas as pd
import zipfile
from sentinelsat import SentinelAPI
user = 'miannini'
password = 'An4lytics@89'
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

class Sentinel_downloader:
    def image_down(footprint, Date_Ini, Date_Fin, valid_dates, analysis_area):
        Date_Ini_c = Date_Ini.replace('-','')
        Date_Fin_c = Date_Fin.replace('-','')
        products = api.query(footprint,
                     date = (Date_Ini_c, Date_Fin_c),
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-1C'#,
                     #cloudcoverpercentage = (0,30)
                    )
        #Listado de imagenes satelitales dentro de la API
        products_gdf = api.to_geodataframe(products)
        products_gdf_sorted = products_gdf.sort_values(['beginposition'], ascending=[True]) #products_gdf.sort_values(['cloudcoverpercentage']
        
        #Leer el listado de imagenes por fechas en las que el porcentaje de nubes calculado sea menor al valor indicado
        file_list = []
        for b in range(0,len(products_gdf_sorted)):
            api_id = str(products_gdf_sorted.iloc[b,7]).split()[0]
            for c in range(0,len(valid_dates)):
                valid_dates_id = str(valid_dates.iloc[c,0]).split()[0]
                if(api_id==valid_dates_id):
                    file = products_gdf_sorted['uuid'][b]
                    title = products_gdf_sorted['title'][b]
                    file_list.append(file)
                    print(file,title)
        #Descarga de Imagenes por bandas
        for n in file_list:
            api.download(n, "Zipped_Images/")
        
        #Descomprimir Metadata y eliminación de archivos descargados
        for root, dirs, files in os.walk("Zipped_Images"):
            for file in files:
                if file.endswith(".zip"):
                        zipfile_path = os.path.join(root, file)
                        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                            zip_ref.extractall("Unzipped_Images/"+analysis_area)
                        os.remove(zipfile_path)
    #obtener rutas de las imagenes .jp2                    
    def get_routes(analysis_area):
        direcciones=[]
        for root, dirs, files in os.walk("Unzipped_Images/"+analysis_area):
            for file in files:
                if file.endswith(".jp2"):
                    x=os.path.join(root, file)
                    foldi = x.split(os.path.sep)[-2]
                    if foldi == 'IMG_DATA':
                        direcciones.append(os.path.join(root))
        direcciones=list(set(direcciones))
        return direcciones
        
