# -*- coding: utf-8 -*-
from shapely.geometry import Polygon
from firebase import firebase
import geopandas as gpd
import folium 
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
from pyproj import Proj, transform
from sentinelhub import BBox, CRS
import time

firebase = firebase.FirebaseApplication('https://proyectopiloto-28591.firebaseio.com/', None)

class Fire_down:
	def find_poly(user_analysis,Date_Ini, Date_Fin, shape_folder):
         if user_analysis == 'no':
             alldb = firebase.get('coordinatesUser/', None)
             pending = []
             for item in alldb.items(): #itera por la bd
                 usuario = item[0]
                 for item_terreno in item[1].values():
                     try:
                         if item_terreno['status'] == "Pendiente":
                             pending.append(dict({"user" : usuario , "terrain": item_terreno['uid'], "timestamp" : item_terreno['timestamp']})) 
                     except:
                         pass
             pending = sorted(pending, key = lambda i: i['timestamp'],reverse=True)
             user_analysis = pending[0]['user']+"/"+pending[0]['terrain']
         if (Date_Ini == 'no'):
             request_date = firebase.get('coordinatesUser/'+user_analysis+'/timestamp', None) #Conseguir fecha
             time_window = firebase.get('coordinatesUser/'+user_analysis+'/years', None) #Conseguir ventana de tiempo
             Date_Ini = time.strftime('%Y-%m-%d', time.gmtime((int(request_date)/1000) - (int(time_window))*31536000))
             Date_Fin = time.strftime('%Y-%m-%d', time.gmtime(int(request_date)/1000))

         result = firebase.get('/coordinatesUser/'+user_analysis+'/Coordenadas', None)
         lote_aoi = Polygon(result)
         polygons = []
         polygons.append(Polygon(lote_aoi))
         lote_aoi = gpd.GeoDataFrame(gpd.GeoSeries(poly for poly in polygons), columns=['geometry']) 
         
         lote_aoi.crs = {'init':'epsg:4326', 'no_defs': True}  #epsg:4326 is standard world coordinates
         #Conversión de coordenadas especificas locales
         lote_aoi_loc= lote_aoi.to_crs(32618) 
         lote_aoi_loc["x"] = lote_aoi_loc.centroid.map(lambda p: p.x) 
         lote_aoi_loc["y"] = lote_aoi_loc.centroid.map(lambda p: p.y)
         lote_aoi["x"] = float(lote_aoi.centroid.map(lambda p: p.x))
         lote_aoi["y"] = float(lote_aoi.centroid.map(lambda p: p.y))
         minx = float(lote_aoi_loc["x"])- (767.5*10)
         maxx = float(lote_aoi_loc["x"])+ (767.5*10)
         miny = float(lote_aoi_loc["y"])- (767.5*10)
         maxy = float(lote_aoi_loc["y"])+ (767.5*10)
         
         #Creación del shape respecto a coordenas de vértices
         analysis_area = user_analysis.split("/")[1]
         w = shapefile.Writer(shape_folder+analysis_area+'/big_box')
         w.field('name', 'C')
         w.poly([
                     [[minx,miny], [minx,maxy], [maxx,maxy], [maxx,miny], [minx,miny]]
                     ])
         w.record('polygon')
         w.close()
         
         #creacion de bounding box respecot a vertices, para cloud_finder
         inProj = Proj(init='epsg:32618')
         outProj = Proj(init='epsg:4326')
         x1,y1 = transform(inProj,outProj,minx,miny)
         x2,y2 = transform(inProj,outProj,maxx,maxy)
         bbox_coords_wgs84 = [x1,y2,x2,y1]
         bounding_box = BBox(bbox_coords_wgs84, crs=CRS.WGS84)

         return lote_aoi, lote_aoi_loc, minx,maxx,miny,maxy, bounding_box, user_analysis,analysis_area, Date_Ini, Date_Fin
