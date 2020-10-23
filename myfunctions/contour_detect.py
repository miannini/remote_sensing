# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon
import geopandas as gpd
from myfunctions.tools import Satellite_tools

class Contour_detect:
    #function to identify forms in image based on contours
    def read_image_tif(best_date, analysis_area,im_name,output_folder): #im_name = "_NDVI.tif" or "_LAI.tif"
        #best_date = str(best_date.date()).replace('-','')
        imagen = cv2.imread(output_folder+analysis_area+'/'+best_date+im_name,-1)
        dimens = imagen.shape
        slice1Copy = (imagen*255).astype(np.uint8)
        canvas = np.zeros(slice1Copy.shape, np.uint8)
        kernel = np.ones((3,3),np.float32)/9
        edged2 = cv2.filter2D(slice1Copy,-1,kernel)
        return edged2, canvas
    
    #function to identify forms in image based on contours
    def identif_forms(imagen, threshold, area_min, max_forms,remove_initials):
        ret,thresh = cv2.threshold(imagen,threshold,255,cv2.THRESH_BINARY_INV)
        contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        forms=[]
        for cont in contours: #cnts
            if cv2.contourArea(cont) > area_min:
                perimeter = cv2.arcLength(cont,True)
                epsilon = 0.002*cv2.arcLength(cont,True)
                approx = cv2.approxPolyDP(cont,epsilon,True)
                forms.append(approx)
        forms = forms[remove_initials:max_forms]
        return forms
    
    #function to transform a list of points from x,y to epsg 32618
    def trns_lst(lista,meta):
        nombre_df=[]
        for h in range(0,len(lista)):
            temp_list=[]
            lote=lista[h]
            for i in range(0,len(lote)):
                point = lote[i][0][0],lote[i][0][1]
                #call another sub-function to transform each pixel to another coordinates
                result = Satellite_tools.trns_coor(point,meta)
                temp_list.append(tuple(result))
            nombre_df.append(temp_list)
        return nombre_df
    
    #function to transform list of coordinates to deodataframe polygon
    def create_geodata(lista):
        crs = {'init': 'epsg:32618'}
        gdf = gpd.GeoDataFrame()
        for i in range(0,len(lista)):
            r = LinearRing(lista[i])
            ring = Polygon(r)
            geo_poly = gpd.GeoDataFrame(index=[i], crs=crs, geometry=[ring]) 
            gdf = gpd.GeoDataFrame( pd.concat( [gdf,geo_poly], ignore_index=True) , crs=crs)
        return gdf
    
    
