# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
#mpl.use('Agg')
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from glob import glob
import os
from gamma_correction import gamma_correction as gc
from scipy.stats import pearsonr as correl
from statsmodels.distributions.empirical_distribution import ECDF

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
    pr_f_raw_m = np.full((17,1), np.nan)
    pr_h_cor_m = np.full((17,20), np.nan)
    pr_h_obs_m = np.copy(pr_h_cor_m)
    pr_f_obs_m = np.copy(pr_f_cor_m)
    pr_h_raw_m = np.copy(pr_h_cor_m)   
    for id, date in enumerate(dates):
        date1,date2 = get_bounds(date, horizon)
        pr_h_cor = []
        pr_f_cor = []
        pr_h = []
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
        pr_f_raw_m[id] = pr_f
        #reading obs fcst file
        pr_f_obs = get_obs(datetime(2018,mh1,md1),datetime(2018,mh2,md2))
        pr_f_obs_m[id, :] = pr_f_obs
        #reading hindcast and obs hind for correcting fcst
        pr_h, pr_h_obs = read_hind_and_obs('fcst',date, date1,date2,2018) 
        pr_h_raw_m[id,:] = pr_h
        pr_f_cor.append(gc(pr_h_obs, pr_h, pr_f))

        pr_f_cor_m[id,:] = pr_f_cor[0]
        pr_h_cor_m[id,:] = np.array(pr_h_cor)
        pr_h_obs_m[id,:] = pr_h_obs
        
    # vou insegir o ecdf aqui.
    a = np.reshape(pr_h_raw_m, (1,17*20))
    b = np.reshape(pr_h_obs_m, (1,17*20))
    #print(a[0])
    #print(b[0])
    
    #ecdfh = ECDF(a[0])
    #print(ecdfh.x, ecdfh.y)
    #ecdfo = ECDF(b)
    #plt.title('{} - all runs'.format(horizon))
    #plt.plot(ecdfh.x, ecdfh.y,label='model')
    #plt.plot(ecdfo.x, ecdfo.y,label='obs')
    #plt.show()
    #input()
    print(pr_f_raw_m)       
    return np.squeeze(pr_f_cor_m), np.squeeze(pr_f_obs_m), pr_h_cor_m, pr_h_obs_m,np.squeeze(pr_h_raw_m), np.squeeze(pr_f_raw_m)

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

horizons = ['15days', '30days', '45days', '30daysAfter15' ] #'2ndfortnight', '3ndfortnight'] 

dates = get_dates(2018)
'''
pr_f_cor_mat = np.zeros((17,4))
pr_f_raw_mat = np.zeros((17,4))
pr_f_obs_mat = np.zeros((17,4))
pr_h_cor_mat = np.zeros((17,20,4))
pr_h_obs_mat = np.zeros((17,20,4))
pr_h_raw_mat = np.copy(pr_h_cor_mat)

for ih,horizon in enumerate(horizons):
    pr_f_cor_mat[:,ih], pr_f_obs_mat[:,ih], pr_h_cor_mat[:,:,ih], pr_h_obs_mat[:,:,ih], pr_h_raw_mat[:,:, ih], pr_f_raw_mat[:,ih] = remove_bias_flow(dates, horizon)

np.save('q_f_cor_mat.npy',pr_f_cor_mat)
np.save('q_f_raw_mat.npy',pr_f_raw_mat)
np.save('q_f_obs_mat.npy',pr_f_obs_mat) 
np.save('q_h_cor_mat.npy',pr_h_cor_mat)
np.save('q_h_obs_mat.npy',pr_h_obs_mat)
np.save('q_h_raw_mat.npy',pr_h_raw_mat)
'''


q_f_cor_mat = np.load('q_f_cor_mat.npy') 
q_h_cor_mat = np.load('q_h_cor_mat.npy') 
q_f_obs_mat = np.load('q_f_obs_mat.npy') 
q_h_obs_mat = np.load('q_h_obs_mat.npy')
q_h_raw_mat = np.load('q_h_raw_mat.npy')
q_f_raw_mat = np.load('q_f_raw_mat.npy')

#criando arquivo dirceu vazao 15, 50 45 dias brutos e dados de chuva para todos os 17 datas de initialização do modelo
print(q_h_raw_mat.shape)
print(q_h_obs_mat.shape)
'''
for ih, hon in enumerate(horizons):
    idx=pd.Index(dates,name='dates')
    df_q_h_raw = pd.DataFrame(data=q_h_raw_mat[:,:,ih], index=idx, columns=range(1998,2018))
    df_q_h_raw.to_csv('data/hind/flow_hind_9817_{}.txt'.format(hon))
    
    df_q_h_raw = pd.DataFrame(data=q_h_obs_mat[:,:,ih], index=idx, columns=range(1998,2018))
    df_q_h_raw.to_csv('data/obs/flow_obs_9817_{}.txt'.format(hon))
    
    df = pd.DataFrame(data=q_f_raw_mat[:,ih], index=idx, columns=[2018])
    df.to_csv('data/fcst/flow_fcst_2018_{}.txt'.format(hon))
    
    df = pd.DataFrame(data=q_f_obs_mat[:,ih], index=idx, columns=[2018])
    df.to_csv('data/obs/flow_obs_2018_{}.txt'.format(hon))
''' 

correl_hind = np.full((17,4), np.nan)
correl_global = np.zeros((4,))
bias = np.full((17,4), np.nan)
q_h_cor_mat[np.isnan(q_h_obs_mat)] = np.nan
q_f_raw_mat[np.isnan(q_f_obs_mat)] = np.nan
q_h_raw_mat[np.isnan(q_h_obs_mat)] = np.nan


def flow_metrics(q_f_cor_mat, q_h_cor_mat, q_f_obs_mat, q_h_obs_mat, dates, horizons, q_h_raw_mat):
    for ih,hor in enumerate(horizons):
        p = []
        q = []

        for id, date in enumerate(dates):
       #for ih,hor in enumerate(horizons):

        
            r = q_h_raw_mat[id,:,ih] ; r = r[~np.isnan(r)] #; print(r)
            s = q_h_obs_mat[id,:,ih] ; s = s[~np.isnan(s)] #; print(s) 
            correl_hind[id, ih] = correl(r,s)[0]
            bias[id, ih] = np.mean(r-s)
            for yy in range(len(s)):
                p.append(r[yy])
                q.append(s[yy])
        p = np.array(p)
       
        q = np.array(q)

        correl_global[ih] = correl(p,q)[0]
        np.savetxt('ts_model_to_global_correl_{0}.txt'.format(hor), p)  
        np.savetxt('ts_obs_to_global_correl_{0}.txt'.format(hor),q)  
        print(correl(p,q)[0])
    return correl_hind,bias,correl_global


def plot_fcst_data(q_f_raw_mat, q_f_obs_mat, dates, horizons):
    dd = []
    date_dt = []
    for d in dates:
        day=d[-2:]
        mon=d[-4:-2]
        dd.append('{}/{}'.format(mon,day))
        date_dt.append(datetime.strptime(d,'%Y%m%d'))
    
    #dfh = pd.DataFrame(data=q_f_cor_mat,index=date_dt,columns=horizons)
    #dfo = pd.DataFrame(data=q_f_obs_mat,index=date_dt,columns=horizons)
    for f,hor in enumerate(horizons):
        fig, ax = plt.subplots(dpi=200,figsize=(7,4))
        qf = q_f_raw_mat[:,f] 
        qo = q_f_obs_mat[:,f]
        plt.title('2018 forecast raw data and obs - {}'.format(hor),fontsize=18)
        plt.plot(dates, qf, 'b-o',label='fcst raw data')
        plt.plot(dates,qo,'k->',label='obs')
        ax.set_xticklabels(dd,rotation=45)
        ax.set_ylim([0,110])
        plt.xlabel('Issue model run dates')
        ax.legend()
        plt.savefig('fcst_{}.png'.format(hor))
        

#plot_fcst_data(q_f_raw_mat, q_f_obs_mat, dates,horizons)
    
correl_hind, bias_hind, correl_global = flow_metrics(q_f_cor_mat, q_h_cor_mat, q_f_obs_mat, q_h_obs_mat, dates, horizons, q_h_raw_mat)

dd = []
date_dt = []
correl_global = np.tile(correl_global,(17,1))
#print(dates)
for d in dates:
    dt=datetime.strptime(d,'%Y%m%d')
    date_dt.append(datetime.strptime(d,'%Y%m%d'))
    dt_str = dt.strftime('%d%b')
    #day=datetime.strftime('%d')
    #mon=d[-4:-2]
    dd.append('{}'.format(dt_str))
labs = ['15 days mean', '30 days mean', '45 days mean']   

fig, ax = plt.subplots(dpi=150,figsize=(25,8.))
df = pd.DataFrame(data=correl_hind, index=date_dt,columns=horizons)
dfc = pd.DataFrame(data=correl_global, index=date_dt)
df.to_csv('correl_matrix_hindcast.txt', sep=' ')
#dt =i pd.Datetimeindex(dates 
df.iloc[2:,0].plot(linestyle='-',color='black',linewidth=3,label=labs[0])
df.iloc[2:,1].plot(linestyle='--',color='black',linewidth=3,label=labs[1])
df.iloc[2:,2].plot(linestyle='dotted',color='black',linewidth=3,label=labs[2])
#dfc.iloc[2:,0].plot(linestyle='-',color='gray',linewidth=1,label='aa')
#dfc.iloc[2:,1].plot(linestyle='--',color='gray',linewidth=1,label='a')
#dfc.iloc[2:,2].plot(linestyle='dotted',color='gray',linewidth=1)
#ax.text(0.5,0.5,'aaaaa',transform=ax.transAxes,fontsize=20)
plt.axhline(correl_global[0,0],linestyle='-', color='gray', linewidth=1)#,label='correl global 15days')
plt.axhline(correl_global[0,1],linestyle='--', color='gray', linewidth=1)#,label='correl global 15days')
plt.axhline(correl_global[0,2],linestyle='dotted', color='gray', linewidth=1)
#plt.axvline(date_dt[5],linestyle='-', color='gray', linewidth=1,label='kkkkk')

# plt.plot(date_dt[2:], correl_global[2:,2],linestyle='-', color='gray',linewidth=1)
#df.iloc[:,3].plot(color='k',marker='^',markersize=11.)
#df.iloc[:,4].plot(color='k',marker='o',markersize=11.)
#df.iloc[:,5].plot(color='k',marker='D',markersize=11.)
ax.legend(bbox_to_anchor=(1.0, 0.3 ),fontsize=19)
#ax.legend('center left', bbox_to_anchor(1,0.5))
ax.set_ylim([0.34,1.])
ax.set_title('b) Flow prediction perfomance (1998-2017)', fontsize=25)
ax.set_xticks(date_dt[2:])
ax.set_xticklabels(dd[2:],rotation=0,fontsize=19)
ax.tick_params(axis="y", labelsize=19)

ax.set_xlabel('Initialization date',fontsize=22)
ax.set_ylabel('Correlation',fontsize=22)
#ax.grid(True, linestyle='--', color='gray')
plt.savefig('correl_hind_9817.png')
exit()

fig, ax = plt.subplots(dpi=120,figsize=(18.,5))
df = pd.DataFrame(data=bias_hind, index=date_dt,columns=horizons)
df.to_csv('bias_matrix_hindcast.txt', sep=' ')
#dt =i pd.Datetimeindex(dates,
df.iloc[2:,0].plot(color='k',marker='X',markersize=11.)
df.iloc[2:,1].plot(color='k',marker='>',markersize=11.)
df.iloc[2:,2].plot(color='k',marker='*',markersize=11.)
#df.iloc[2:,3].plot(color='k',marker='^',markersize=11.)
#df.iloc[:,4].plot(color='k',marker='o',markersize=11.)
#df.iloc[:,5].plot(color='k',marker='D',markersize=11.)
ax.legend(bbox_to_anchor=(1.0, 0.5 ))
#ax.legend('center left', bbox_to_anchor(1,0.5))
ax.set_title('Bias stremflow Hindcast ECMWF-smap vs Obs')
ax.set_xticks(dates)
ax.set_xticklabels(dd,rotation=45)
ax.grid(True)
ax.set_xlabel('Runs [month/day]')
plt.savefig('bias_hind_9817.png')
  
args = arguments()
exp = args.exp
#method = args.method
#read_s2s(exp, method)

exit()

