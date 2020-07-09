# -*- coding: utf-8 -*-
from shapely.geometry import Polygon
import geopandas as gpd
import folium 
import pandas as pd
import shapefile
import os
from pyproj import Proj, transform

class Ext_shape:
    def merge_shapes(shape_folder,lote_aoi_loc,folder_name):
        #loop to read shapefiles
        todos = gpd.GeoDataFrame() #columns=['geometry','x','y']
        for root, dirs, files in os.walk(shape_folder+folder_name):
            for file in files:
                #print (root,dirs,files)
                crs = {'init': 'epsg:32618'}
                if file.endswith(".shp"):
                        shape_path = os.path.join(root, file)
                        temp_file = gpd.read_file(shape_path)
                        #print(temp_file)
                        temp_file= temp_file.to_crs(32618)
                        temp_file["x"] = temp_file.centroid.map(lambda p: p.x)
                        temp_file["y"] = temp_file.centroid.map(lambda p: p.y)
                        temp_file = temp_file[['geometry','x','y']]
                        todos = gpd.GeoDataFrame( pd.concat( [temp_file,todos], ignore_index=True) , crs=crs)
        todos = gpd.GeoDataFrame( pd.concat( [lote_aoi_loc,todos], ignore_index=True) , crs=crs)
        return todos
