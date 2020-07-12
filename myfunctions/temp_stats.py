# -*- coding: utf-8 -*-
#from __future__ import division
import numpy.ma as ma
import pandas as pd
import numpy as np
import os
import rasterstats as rs
import seaborn as sns



def reduct(x):return x

def data_bruta(data,x):
    VAR = False
    BB = data[x].get('reduct')
    data2 = ma.getdata(BB)[BB.mask == VAR]
    poly = np.repeat(x , len(data2))
    dataf = {'poly':poly, 'data':data2}
    dataf = pd.DataFrame(dataf) 
    return dataf
    #pd.DataFrame(data_bruta(0))
def full_append(data,x):
    FA = [data_bruta(data,ii) for ii in x]
    return pd.concat(FA)

class Stats_charts:
    def data_g(data_i,analysis_date, aoig_near,todos_lotes,output_folder):
        #bands_avail
        bands_avail = []
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if file.endswith(".tif") and "_lote" not in file:
                    bands_avail.append(file[8:])
        bands_avail = list(set(bands_avail))
        
        big = []
        for band in bands_avail:
            data = rs.zonal_stats(aoig_near, analysis_date+band,  #"_NDVI.tif"
                                  stats="count",
                                  add_stats={'reduct': reduct}, layer=1)
            porygons = full_append(data,range(0,len(aoig_near)))
            Ndate = np.shape(porygons)[0]
            dateAssign = np.repeat( data_i , Ndate)
            datag = { 'date':dateAssign,'poly':porygons['poly'],'data_pixel':porygons['data'],'band':band.split('.')[0]  }
            datag = pd.DataFrame(datag)
            big.append(datag)
        size_flag =  np.shape(datag)[0] == 0
        #df resumen
        short_resume = pd.DataFrame()
        for n in range(0,len(big)):
            temp_resume = big[n].groupby(['date','poly','band']).agg(mean_value=('data_pixel', 'mean'), sum_value=('data_pixel', 'sum'), max_value=('data_pixel', 'max'), 
                                                                     std_dev=('data_pixel', 'std'), count_pxl=('data_pixel', 'count'), perc_90 = ('data_pixel', lambda x: np.percentile(x,90)) )
            short_resume = pd.concat([short_resume,temp_resume])      
        #unir datos lotes
        short_resume = short_resume.merge(todos_lotes, left_on='poly', right_index=True, how="left")
        short_resume = short_resume.drop(['geometry'], axis=1)
        short_ord = pd.pivot_table(short_resume, index=['date',"poly","name","x","y"],columns="band", values=['mean_value','sum_value', 'max_value','std_dev','count_pxl','perc_90'])
        return size_flag, datag, short_ord, short_resume

        



