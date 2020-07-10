# -*- coding: utf-8 -*-
from shapely.geometry import Polygon
import geopandas as gpd
import folium 
import pandas as pd
import shapefile
import os
from pyproj import Proj, transform

class Ext_shape:
    def merge_shapes(shape_folder,lote_aoi,lote_aoi_loc,folder_name):
        #loop to read shapefiles
        todos = gpd.GeoDataFrame() #columns=['geometry','x','y']
        todos_loc = gpd.GeoDataFrame()
        for root, dirs, files in os.walk(shape_folder+folder_name):
            for file in files:
                #print (root,dirs,files)
                crs = {'init': 'epsg:32618'}
                if file.endswith(".shp"):
                        shape_path = os.path.join(root, file)
                        temp_file = gpd.read_file(shape_path)
                        temp_file.columns = map(str.lower, temp_file.columns) #lower case column names
                        #maybe join name of area with name of file
                        file_name = file.split(".")[0].split("-")[0]
                        if 'name' not in temp_file:
                            temp_file['name'] = 'no_name' + "-" + temp_file.index.map(str)
                        if file_name != "POLYGON":
                            temp_file['name'] = file_name + "-" + temp_file['name'].map(str)
                        #local
                        temp_file_loc= temp_file.to_crs(32618)
                        temp_file_loc["x"] = temp_file_loc.centroid.map(lambda p: p.x)
                        temp_file_loc["y"] = temp_file_loc.centroid.map(lambda p: p.y)
                        temp_file_loc = temp_file_loc[['name','geometry','x','y']]
                        todos_loc = gpd.GeoDataFrame( pd.concat( [temp_file_loc,todos_loc], ignore_index=True) , crs=crs)
                        #universal
                        #temp_file= temp_file.to_crs(32618)
                        temp_file["x"] = temp_file.centroid.map(lambda p: p.x)
                        temp_file["y"] = temp_file.centroid.map(lambda p: p.y)
                        temp_file = temp_file[['name','geometry','x','y']]
                        todos = gpd.GeoDataFrame( pd.concat( [temp_file,todos], ignore_index=True) , crs=crs)
        todos_loc = gpd.GeoDataFrame( pd.concat( [lote_aoi_loc,todos_loc], ignore_index=True) , crs=crs)
        todos = gpd.GeoDataFrame( pd.concat( [lote_aoi,todos], ignore_index=True) , crs=crs)
        return todos, todos_loc
