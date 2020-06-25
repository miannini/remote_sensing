# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase import firebase
import cv2
import glob

class Upload_fire:
    def upload_image(folder,analysis_area,user_analysis):
        cred = credentials.Certificate('KeyTerrenos.json')
        #si no existe, inizializar, sino, saltar
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'proyectopiloto-28591.appspot.com'
        }) 
        firebase_ap = firebase.FirebaseApplication('https://proyectopiloto-28591.firebaseio.com/', None)        
        count1=0
        count2=0
        count3=0
        count4=0        
        for img in glob.glob(folder+analysis_area+'/'+"*.png"): #"Images_to_firebase/"
            name= str(img)
            #name = name[19:] #no dependiende de numero sino de particion
            name = name.split('\\')[-1]
            print (name)
            bucket = storage.bucket()
            blob = bucket.blob('Images/'+user_analysis+'/'+name)
                        
            image = blob.upload_from_filename('./'+img)
            blob.make_public()
            url = blob.public_url
            data =  { 'name' : img, 'url': url}
            
            if name.find("NDVI_lote1") >7:
                count1 = count1 +1
                result = firebase_ap.put('/Images/'+user_analysis, 'NDVI_Lote_'+ str(count1) ,data)
            
            elif name.find("NDVI_lotes_exp") >7:
                count2 = count2 +1
                result = firebase_ap.put('/Images/'+user_analysis, 'NDVI_Lote_&_Neighbors'+ str(count2) ,data)
            
            elif name.find("NDVI_lotes_oneDate") >7:
                count3 = count3 +1
                result = firebase_ap.put('/Images/'+user_analysis, 'Lotes_Boxplot'+ str(count3) ,data)
            
            elif name.find("NDVI_lote1") == 0:
                result = firebase_ap.put('/Images/'+user_analysis, 'NDVI_Lote_Over_Time' ,data)
            
            elif name.find("NDVI_lotes_median") ==0:
                result = firebase_ap.put('/Images/'+user_analysis, 'NDVI_Lotes_MedianOVT' ,data)
            
            elif name.find("NDVI_lotes_oneDate") == 0:
                result = firebase_ap.put('/Images/'+user_analysis, 'Lotes_Boxplot_OneDate' ,data) 
                
            else:
                count4 = count4 +1
                result = firebase_ap.put('/Images/'+user_analysis, 'Img'+ str(count4) ,data)