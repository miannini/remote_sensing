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
    def data_g(data_i,analysis_date, aoig_near):
        data = rs.zonal_stats(aoig_near, analysis_date+"_NDVI.tif", 
                              stats="count",
                              add_stats={'reduct': reduct}, layer=1)
        porygons = full_append(data,range(0,len(aoig_near)))
        Ndate = np.shape(porygons)[0]
        dateAssign = np.repeat( data_i , Ndate)
        datag = { 'date':dateAssign,'poly':porygons['poly'],'data_pixel':porygons['data']  }
        datag = pd.DataFrame(datag)
        size_flag =  np.shape(datag)[0] == 0
        return size_flag, datag

        



