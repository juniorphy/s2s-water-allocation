# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
#mpl.use('Agg')
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from glob import glob
import os
from gamma_correction import gamma_correction as gc
from scipy.stats import pearsonr as correl

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
        
            obs = pd.read_csv('vazao_posto_iguatu_36160000.csv', header=None,index_col=0) 
            pr_h_obs.append(obs[dateh1:dateh2].mean().values[0])
            #print(obs[dateh1:dateh2])
            
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
            pr_h_cor.append(hind_y)         # mudei aqui
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

        pr_f_cor_m[id,:] = pr_f  # mudei aqui
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
    if horizon == '2ndfortnight':
        date1 = date + relativedelta(days=15)
        date2 = date + relativedelta(days=29)
    if horizon == '3ndfortnight':
        date1 = date + relativedelta(days=30)
        date2 = date + relativedelta(days=44)
    if horizon == '30daysAfter15':
        date1 = date + relativedelta(days=15)
        date2 = date + relativedelta(days=44)
    return date1, date2

horizons = ['15days', '30days', '45days', '30daysAfter15','2ndfortnight', '3ndfortnight'] 

dates = get_dates(2018)

pr_f_cor_mat = np.zeros((17,6))
pr_f_obs_mat = np.zeros((17,6))
pr_h_cor_mat = np.zeros((17,20,6))
pr_h_obs_mat = np.zeros((17,20,6))

for ih,horizon in enumerate(horizons):
    
    pr_f_cor_mat[:,ih], pr_f_obs_mat[:,ih], pr_h_cor_mat[:,:,ih], pr_h_obs_mat[:,:,ih] = remove_bias_flow(dates, horizon)

np.save('q_f_cor_mat_raw.npy',pr_f_cor_mat)
np.save('q_f_obs_mat_raw.npy',pr_f_obs_mat) 
np.save('q_h_cor_mat_raw.npy',pr_h_cor_mat)
np.save('q_h_obs_mat_raw.npy',pr_h_obs_mat)

q_f_cor_mat = np.load('q_f_cor_mat_raw.npy') 
q_h_cor_mat = np.load('q_h_cor_mat_raw.npy') 
q_f_obs_mat = np.load('q_f_obs_mat_raw.npy') 
q_h_obs_mat = np.load('q_h_obs_mat_raw.npy') 



correl_hind = np.full((17,6), np.nan)

q_f_cor_mat[np.isnan(q_f_obs_mat)] = np.nan
q_h_cor_mat[np.isnan(q_h_obs_mat)] = np.nan

def flow_metrics(q_f_cor_mat, q_h_cor_mat, q_f_obs_mat, q_h_obs_mat, dates, horizons):
   # for ih,hor in enumerate(horizons):
    for id, date in enumerate(dates):
       for ih,hor in enumerate(horizons):
     
           r = q_h_cor_mat[id,:,ih] ; r = r[~np.isnan(r)] #; print(r)
           s = q_h_obs_mat[id,:,ih] ; s = s[~np.isnan(s)] #; print(s) ; 
           #input()
           correl_hind[id, ih] = correl(r,s)[0]
           #print(date, hor,correl(r,s)[0])
           #print(len(r))
           #print(len(s))
            
    return correl_hind

correl_hind = flow_metrics(q_f_cor_mat, q_h_cor_mat, q_f_obs_mat, q_h_obs_mat, dates, horizons)
dd = []
date_dt = []
for d in dates:
    day=d[-2:]
    mon=d[-4:-2]
    dd.append('{}/{}'.format(mon,day))
    date_dt.append(datetime.strptime(d,'%Y%m%d'))

fig, ax = plt.subplots(dpi=100,figsize=(2,1.))
df = pd.DataFrame(data=correl_hind, index=date_dt,columns=horizons)
df.to_csv('correl_RAW_matrix_hindcast.txt', sep=' ')
#dt =i pd.Datetimeindex(dates, 
df.iloc[:,0].plot(color='k',marker='X',markersize=11.)
df.iloc[:,1].plot(color='k',marker='>',markersize=11.)
df.iloc[:,2].plot(color='k',marker='*',markersize=11.)
df.iloc[:,3].plot(color='k',marker='^',markersize=11.)
#df.iloc[:,4].plot(color='k',marker='o',markersize=11.)
#df.iloc[:,5].plot(color='k',marker='D',markersize=11.)
ax.legend(bbox_to_anchor=(1.0, 0.5 ))
#ax.legend('center left', bbox_to_anchor(1,0.5))
ax.set_title('RAW DATA correlation stremflow Hindcast ECMWF-smap vs Obs')
ax.set_xticks(dates)
ax.set_xticklabels(dd,rotation=45) 
ax.grid(True)
ax.set_xlabel('Runs [month/day]')
#plt.savefig('correl_hind_9817.png')
plt.show()


#fist fig 1b
         
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

