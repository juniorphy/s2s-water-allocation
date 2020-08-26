# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#from f_pr_pet import main_pr, main_pet
#from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from glob import glob
import os
from gamma_correction import gamma_correction as gc

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="fcst or hind")
    parser.add_argument("--method", type=str, help="ensemble or member")
    return parser.parse_args()

            
def read_obs(date_fcst,):
   date_start_obs = datetime.strptime(date_fcst, '%Y%m%d') - relativedelta(months=48)  
   date_end_obs = datetime.strptime(date_fcst, '%Y%m%d') - relativedelta(days=1) 
   
   a, b,c , d =  np.loadtxt('data/obs/pr_daily_funceme_obs_19730101_20200731_thiessen_reservatorio-9-oros-CE.asc', unpack=True)
   pr = d
   a, b,c , d =  np.loadtxt('data/obs/pet_daily_inmet_obs_19730101_20200731_thiessen_reservatorio-9-oros-CE.asc', unpack=True)

   pet = d
   date1 = datetime(1973,1,1)
   date2 = datetime(2020,7,31)
   date = pd.date_range(date1.strftime('%Y%m%d'),date2.strftime('%Y%m%d'))
   date = date.to_pydatetime()
   dfpr=pd.DataFrame(data=pr, index=date)
   
   dfpet=pd.DataFrame(data=pet, index=date)
   
   return dfpr[date_start_obs:date_end_obs] 

def pet_clim_s2s(date_fcst):
   date_pet_1 = datetime.strptime(date_fcst, '%Y%m%d') #+ relativedelta(days=1)
   date_pet_2 = date_pet_1 + relativedelta(days=45)
   date_s2s = pd.date_range(date_pet_1,date_pet_2)
   
   dts2s = pd.DataFrame(index=date_s2s)

   a, b,c , d =  np.loadtxt('data/obs/pet_daily_inmet_obs_19730101_20200731_thiessen_reservatorio-9-oros-CE.asc', unpack=True)

   date1 = datetime(1973,1,1)
   date2 = datetime(2020,7,31)
   date = pd.date_range(date1.strftime('%Y%m%d'),date2.strftime('%Y%m%d'))
   date = date.to_pydatetime()

   dfpet=pd.DataFrame(data=d, index=date)
   petclim = dfpet['19910101':'20101231']
   petclim = petclim.groupby([petclim.index.month, petclim.index.day]).mean()
   petclim = petclim.loc[zip(dts2s.index.month, dts2s.index.day)]
   petclimperiod = pd.DataFrame(data=petclim.values, index=date_s2s)

   return petclimperiod


def pet_clim_back(date_fcst):
   date_pet_2 = datetime.strptime(date_fcst, '%Y%m%d') - relativedelta(days=1)
   date_pet_1 = datetime.strptime(date_fcst, '%Y%m%d') - relativedelta(months=48)
   date_s2s = pd.date_range(date_pet_1,date_pet_2)

   dt = pd.DataFrame(index=date_s2s)

   a, b,c , d =  np.loadtxt('data/obs/pet_daily_inmet_obs_19730101_20200731_thiessen_reservatorio-9-oros-CE.asc', unpack=True)

   date1 = datetime(1973,1,1)
   date2 = datetime(2020,7,31)
   date = pd.date_range(date1.strftime('%Y%m%d'),date2.strftime('%Y%m%d'))
   date = date.to_pydatetime()

   dfpet=pd.DataFrame(data=d, index=date)
   petclim = dfpet['19910101':'20101231']
   petclim = petclim.groupby([petclim.index.month, petclim.index.day]).mean()
   petclim = petclim.loc[zip(dt.index.month,dt.index.day)]
   petclimback = pd.DataFrame(index=date_s2s, data=petclim.values)
   
   return petclimback

def read_hind_and_obs(exp, date,date1,date2, year=None):
    mf=date.month
    df=date.day
    mh1=date1.month
    md1=date1.day
    mh2=date2.month
    md2=date2.day
    if exp == 'fcst':
        pr_h = []
        pr_h_obs = []
        pr_h_y = []     
        prh_dct = {}   
        for y in range(1998,2018):
            if y == year:
                continue
           
            datef = datetime(y,mf,df).strftime('%Y%m%d')
            dateh1 = datetime(y,mh1,md1).strftime('%Y-%m-%d')
            dateh2 = datetime(y,mh2,md2).strftime('%Y-%m-%d')
            fin = glob('data/{0}/qflow/{1}/*{2}*ensemble*'.format('hind',y,datef))
            pr_dummy = pd.read_csv(fin[0],header=None, sep=' ', index_col=0)
            pr_h.append(pr_dummy[dateh1:dateh2].mean().values[0])
        #prh_dct['{0:02d}{1:02d}'.format(mf,df)] = pr_h
        #print(prh_dct)
        #read obs
            obs = pd.read_csv('vazao_posto_iguatu_36160000.csv', header=None,index_col=0) 
            pr_h_obs.append(obs[dateh1:dateh2].mean().values[0])
            #print(pr_h_obs, dateh1, dateh2)
            #input()
        
        #reading hind year
        if year != 2018:
            datef = datetime(year,mf,df).strftime('%Y%m%d')
            dateh1 = datetime(year,mh1,md1).strftime('%Y-%m-%d')
            dateh2 = datetime(year,mh2,md2).strftime('%Y-%m-%d')
            fin = glob('data/{0}/qflow/{1}/*{2}*ensemble*'.format('hind',year,datef))
            #print('data/{0}/qflow/{1}/*{2}*ensemble*'.format('hind',year,datef))
            #print(fin)
            pr  = pd.read_csv(fin[0],header=None, sep=' ', index_col=0)
            pr_h_y = pr[dateh1:dateh2].mean().values[0]
    pr_h = np.array(pr_h)
    pr_h_obs = np.array(pr_h_obs)

    if year == 2018:
        return pr_h, pr_h_obs
    else:   
        return pr_h ,pr_h_obs, pr_h_y

def get_obs(date1, date2):
    mh1=date1.month
    md1=date1.day
    mh2=date2.month
    md2=date2.day
    dateh1 = date1.strftime('%Y-%m-%d')
    dateh2 = date2.strftime('%Y-%m-%d')
 
    obs = pd.read_csv('vazao_posto_iguatu_36160000.csv', header=None,index_col=0)
    pr_f_obs = (obs[dateh1:dateh2].mean().values[0])
    
    return pr_f_obs
   
def remove_bias_flow(dates, horizon):
    pr_f_cor_m = np.full((17,1), np.nan)
    pr_h_cor_m = np.full((17,20), np.nan)
    pr_h_obs_m = np.copy(pr_h_cor_m)
    pr_f_obs_m = np.copy(pr_f_cor_m)   
    for id, date in enumerate(dates):
        date1,date2 = get_bounds(date, horizon)
        pr_h_cor = []
        pr_f_cor = []
        date = datetime.strptime(date, '%Y%m%d')
        mf=date.month
        df=date.day
        mh1=date1.month
        md1=date1.day
        mh2=date2.month
        md2=date2.day  
        print(date.strftime('%m%d'), horizon)
        
        ## correction of hindcast 
        for iy, yy in enumerate(range(1998,2018)):
            hind, obs, hind_y = read_hind_and_obs('fcst',date, date1,date2,yy)
            
            cor = gc(obs,hind,hind_y)
            pr_h_cor.append(cor)
        #pr_h_cor = np.array(pr_h_cor)

        # correcting forecasst   
        # reading fcst data
        exp = 'fcst'
        fin = glob('data/{0}/qflow/{1}/*{2}*ensemble*'.format(exp,2018,'2018{0:02d}{1:02d}'.format(mf,df)))   
        pr = pd.read_csv(fin[0],header=None, sep=' ', index_col=0)
        pr_f = pr[date1.strftime('%Y-%m-%d'):date2.strftime('%Y-%m-%d')].mean().values[0]

        #reading obs fcst file
        pr_f_obs = get_obs(datetime(2018,mh1,md1),datetime(2018,mh2,md2))
        pr_f_obs_m[id, :] = pr_f_obs
        #reading hindcast and obs hind for correcting fcst
        pr_h, pr_h_obs = read_hind_and_obs('fcst',date, date1,date2,2018) 

        pr_f_cor.append(gc(pr_h_obs, pr_h, pr_f))

        pr_f_cor_m[id,:] = pr_f_cor[0]
        pr_h_cor_m[id,:] = np.array(pr_h_cor)
        pr_h_obs_m[id,:] = pr_h_obs

        
    return np.squeeze(pr_f_cor_m), np.squeeze(pr_f_obs_m), pr_h_cor_m, pr_h_obs_m
            
            

def get_dates(year):
    dates_fcst = []
    if year == 2018:
        exp = 'fcst'
    else:
        exp = 'hind'
    a = np.sort(glob('data/{1}/qflow/{0}/*ensemble*.txt'.format(year,exp)))
    for ia in a:
        date = ia.split('/')[-1].split('_')[6]
        dates_fcst.append(date)
    return dates_fcst 

def get_bounds(date, horizon):
    date = datetime.strptime(date, '%Y%m%d')
    if horizon == '15days':
        date1 = date 
        date2 = date + relativedelta(days=14)
    if horizon == '30days':
        date1 = date 
        date2 = date + relativedelta(days=29)
    if horizon == '45days':
        date1 = date 
        date2 = date + relativedelta(days=44)
    if horizon == '2ndfortnite':
        date1 = date + relativedelta(days=15)
        date2 = date + relativedelta(days=29)
    if horizon == '3ndfortnite':
        date1 = date + relativedelta(days=30)
        date2 = date + relativedelta(days=44)

    return date1, date2

horizons = ['15days', '30days', '45days', '2ndfortnite', '3ndfortnite'] 

dates = get_dates(2018)

pr_f_cor_mat = np.zeros((17,5))
pr_f_obs_mat = np.zeros((17,5))
pr_h_cor_mat = np.zeros((17,20,5))
pr_h_obs_mat = np.zeros((17,20,5))
'''
for ih,horizon in enumerate(horizons):
    
    pr_f_cor_mat[:,ih], pr_f_obs_mat[:,ih], pr_h_cor_mat[:,:,ih], pr_h_obs_mat[:,:,ih] = remove_bias_flow(dates, horizon)
np.save('pr_f_cor_mat.npy',pr_f_cor_mat)
np.save('pr_f_obs_mat.npy',pr_f_obs_mat) 
np.save('pr_h_cor_mat.npy',pr_h_cor_mat)
np.save('pr_h_obs_mat.npy',pr_h_obs_mat)
'''

pr_f_cor_mat = np.load('pr_f_cor_mat.npy')
pr_h_cor_mat = np.load('pr_h_cor_mat.npy')
pr_f_obs_mat = np.load('pr_f_obs_mat.npy')
pr_h_obs_mat = np.load('pr_h_obs_mat.npy')

print(pr_f_cor_mat)

#idef flow_metrics(pr_f_cor_mat, pr_h_cor_mat, pr_f_obs_mat, pr_h_obs_mat, dates ):
#    np.coerf_pearson
         
#    for horiz in horizons:
#    date = datetime.strptime(date, '%Y%m%d')
#    if horiz == '15days':
#    date1 = date 
#    date2 = date + relativedelta('14days')
    
  
args = arguments()
exp = args.exp
#method = args.method
#read_s2s(exp, method)

exit()

