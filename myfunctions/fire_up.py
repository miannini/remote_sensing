# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase import firebase
#import cv2
import glob

class Upload_fire:
    def upload_image(folder,analysis_area,user_analysis):
        cred = credentials.Certificate('secrets/KeyTerrenos.json')
        #si no existe, inizializar, sino, saltar
        try:
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'proyectopiloto-28591.appspot.com'
            })
        except:
            pass
        firebase_ap = firebase.FirebaseApplication('https://proyectopiloto-28591.firebaseio.com/', None)        
                      
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
            
            result = firebase_ap.put('/Images/'+user_analysis, name[:-4] ,data)
        firebase_ap.put('/coordinatesUser/'+ user_analysis, "status" , "Finalizado")