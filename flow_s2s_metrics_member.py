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
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

def closest(fcst_target, clim_list):
    aux = []    
    for ii in range(0, clim_list.shape[0]):
        aux.append(abs(fcst_target-clim_list[ii]))
    min_value = np.nanmin(aux)
    idx_min = np.where(aux == min_value)[-1][-1]

    return idx_min


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="fcst or hind")
    parser.add_argument("--method", type=str, help="ensemble or member")
    return parser.parse_args()

def read_hind_and_obs(exp, date,date1,date2, year=None):
    mf=date.month
    df=date.day
    mh1=date1.month
    md1=date1.day
    mh2=date2.month
    md2=date2.day
    pr_h_mm = np.full((20,11),np.nan)
    pr_h_y_mm = np.full((11,1),np.nan)
    if exp == 'fcst':
        pr_h = []
        pr_h_obs = []
        pr_h_y = []     
        prh_dct = {}   
        for iy,y in enumerate(range(1998,2018)):
            if y == year:
                continue
           
            datef = datetime(y,mf,df).strftime('%Y%m%d')
            dateh1 = datetime(y,mh1,md1).strftime('%Y-%m-%d')
            dateh2 = datetime(y,mh2,md2).strftime('%Y-%m-%d')
            fin = glob('data/{0}/qflow/{1}/*{2}*member*'.format('hind',y,datef))
            
            pr_dummy = pd.read_csv(fin[0],header=None, sep=' ', index_col=0)
            for im in range(11):
                pr_h_mm[iy,im] = pr_dummy.iloc[:,im][dateh1:dateh2].mean()
                #print(pr_dummy.iloc[:,im][dateh1:dateh2].mean())#.values[0]
                 
            #print(np.mean(pr_h_mm[iy,:]))
            #input() #.mean().values[0])
      

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
            fin = glob('data/{0}/qflow/{1}/*{2}*member*'.format('hind',year,datef))
            #print('data/{0}/qflow/{1}/*{2}*ensemble*'.format('hind',year,datef))
            #print(fin)
            
            pr  = pd.read_csv(fin[0],header=None, sep=' ', index_col=0)
            for im in range(11):
            
                pr_h_y_mm[im] = pr.iloc[:,im][dateh1:dateh2].mean() #.values[0]
    #pr_h = np.array(pr_h)
    pr_h_obs = np.array(pr_h_obs)

    if year == 2018:
        return pr_h_mm, pr_h_obs
    else:   
        return pr_h_mm ,pr_h_obs, pr_h_y_mm

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
    pr_f_cor_m_mm = np.full((17,51), np.nan)
    pr_h_cor_m_mm = np.full((17,20,11), np.nan)
    pr_h_raw_m_mm = np.full((17,20,11), np.nan)
    pr_h_obs_m = np.full((17,20),np.nan)
    pr_f_obs_m = np.full((17,1),np.nan)
    pr_f_raw_m_mm = np.full((17,51),np.nan)
       
    for id, date in enumerate(dates):
        date1,date2 = get_bounds(date, horizon)
        date = datetime.strptime(date, '%Y%m%d')
        mf=date.month
        df=date.day
        mh1=date1.month
        md1=date1.day
        mh2=date2.month
        md2=date2.day  
        print(date.strftime('%m%d'), horizon)
        
        ## correction of hindcast 
        #for iy, yy in enumerate(range(1998,2018)):
        #    hind, obs, hind_y = read_hind_and_obs('fcst', date, date1,date2,yy)
        #    hind = np.delete(hind, iy, axis=0)
        #    for im in range(11):
        #        cor = gc(obs,hind,hind_y[im])
                
        #        pr_h_cor_m_mm[id,iy,im] = cor
                #pr_h_raw_m_mm[id,iy,im] = hind_y[im]

        # correcting forecasst   
        # reading fcst data
        exp = 'fcst'
        fin = glob('data/{0}/qflow/{1}/*{2}*member*'.format(exp,2018,'2018{0:02d}{1:02d}'.format(mf,df)))
        pr = pd.read_csv(fin[0],header=None, sep=' ', index_col=0)

        for im in range(51):
            pr_f = pr.iloc[:,im][date1.strftime('%Y-%m-%d'):date2.strftime('%Y-%m-%d')].mean()
            pr_f_raw_m_mm[id,im] = pr_f
            pr_h_mm, pr_h_obs = read_hind_and_obs('fcst',date, date1,date2,2018)
            pr_f_cor_m_mm[id,im] = gc(pr_h_obs, pr_h_mm, pr_f)
        #reading obs fcst file
        pr_f_obs = get_obs(datetime(2018,mh1,md1),datetime(2018,mh2,md2))
        pr_f_obs_m[id, :] = pr_f_obs
        #print(np.mean(pr_f_raw_m_mm[id,0:11]))
        #input()
        pr_h_raw_m_mm[id,:] = pr_h_mm
        #pr_h_cor_m[id,:] = np.array(pr_h_cor)
        pr_h_obs_m[id,:] = pr_h_obs

        
    return np.squeeze(pr_f_cor_m_mm), np.squeeze(pr_f_obs_m), pr_h_cor_m_mm, pr_h_obs_m, pr_f_raw_m_mm, pr_h_raw_m_mm

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

horizons = ['15days', '30days', '45days', '30daysAfter15'] #,'2ndfortnight', '3ndfortnight'] 

dates = get_dates(2018)

pr_f_cor_mm= np.zeros((17,51,4))
pr_f_raw_mm= np.zeros((17,51,4))
pr_f_obs_mat = np.zeros((17,4))
pr_h_cor_mm = np.zeros((17,20,11,4))
pr_h_raw_mm = np.zeros((17,20,11,4))
pr_h_obs_mat = np.zeros((17,20,4))
'''
for ih,horizon in enumerate(horizons):
    
    pr_f_cor_mm[:,:,ih], pr_f_obs_mat[:,ih], pr_h_cor_mm[:,:,:,ih], pr_h_obs_mat[:,:,ih],pr_f_raw_mm[:,:,ih], pr_h_raw_mm[:,:,:,ih] = remove_bias_flow(dates, horizon)

np.save('q_f_cor_mat_mm.npy',pr_f_cor_mm)
np.save('q_f_obs_mat_mm.npy',pr_f_obs_mat) 
np.save('q_f_raw_mat_mm.npy',pr_f_raw_mm)
np.save('q_h_cor_mat_mm.npy',pr_h_cor_mm)
np.save('q_h_raw_mat_mm.npy',pr_h_raw_mm)
np.save('q_h_obs_mat_mm.npy',pr_h_obs_mat)
'''
q_f_cor_mm = np.load('q_f_cor_mat_mm.npy') 
q_h_cor_mm = np.load('q_h_cor_mat_mm.npy') 
q_f_obs_mat = np.load('q_f_obs_mat_mm.npy') 
q_h_obs_mat = np.load('q_h_obs_mat_mm.npy') 
q_f_raw_mm =  np.load('q_f_raw_mat_mm.npy')
q_h_raw_mm =  np.load('q_h_raw_mat_mm.npy')
'''
q_f_cor_mat = np.load('q_f_cor_mat.npy')
q_h_cor_mat = np.load('q_h_cor_mat.npy')
q_f_obs_mat = np.load('q_f_obs_mat.npy')
q_h_obs_mat = np.load('q_h_obs_mat.npy')
q_h_raw_mat = np.load('q_h_raw_mat.npy')
q_f_raw_mat = np.load('q_f_raw_mat.npy')
'''


# Criando arquivos caio 
# arquivo com 20 anos e 11 members
# 15 dias, 30 e 45 dias

#for id, date in enumerate(dates[0]):
#    for iy, yy in enumerate(range(1998,2018)):
#        idx=pd.Index(dates,name='dates')
#        df_q_h_raw = pd.DataFrame(data=q_h_raw_mm[8,:,:,2], index=range(1998,2018))
#        df_q_h_raw.to_csv('data/hind/flow_hind_{}_{}_member.txt'.format('45days','0301'))
    
#        df_q_h_raw = pd.DataFrame(data=q_h_obs_mat[8,:,2], index=range(1998,2018))
#        df_q_h_raw.to_csv('data/obs/flow_obs_{}_{}_member.txt'.format('45days','0301'))

# testando empirical cdf bias remove

q_f_cor_em = np.zeros((17,51,4))
for ih in range(3):
    for id in range(17):

        for ii in range(51):

           
            obs_fcst = q_f_obs_mat[id,ih]
            hind = q_h_raw_mm[id, :,:,ih]
            #hind = np.mean(q_h_raw_mm[id, :,:,ih], axis=1)
            hind = np.reshape(hind, (220,))
            #print(hind)
            #exit()
            fcst = np.squeeze(q_f_raw_mm[id,ii,ih])
            #fcst = np.mean(fcst)
             
            obs = np.squeeze(q_h_obs_mat[id,:,ih])
            obs = obs[~np.isnan(obs)]

            ehcdf = ECDF(np.sort(hind))
            eocdf = ECDF(np.sort(obs))

            prob = np.interp(fcst,ehcdf.x,ehcdf.y)
            #print(prob)
            fcst_cor_np = np.interp(prob, eocdf.y, eocdf.x)
             
            ehcdfintp = interp1d(ehcdf.x, ehcdf.y,'linear', bounds_error=False)
            eocdfintp = interp1d(eocdf.y, eocdf.x,'linear')
            
            prob =  ehcdfintp(fcst)
            fcst_cor = eocdfintp(prob)
     
    #        print('date', dates[id], 'membro ', ii+1)
            q_f_cor_em[id, ii, ih] = fcst_cor_np
     #       print(fcst, fcst_cor_np, obs_fcst)
#dealing with -inf
#q_f_cor_em[~np.isfinite(q_f_cor_em)] = np.nan

'''            
             print('date', dates[id], 'membro ', ii+1)    
            obs_fcst = q_f_obs_mat[id,ih]

            hind = q_h_raw_mm[id, :,:,ih]
            hind = np.reshape(hind, (220,))
            fcst = np.squeeze(q_f_raw_mm[id,:,ih])
            obs = np.squeeze(q_h_obs_mat[id,:,ih])
            ehcdf = ECDF(hind)
            eocdf = ECDF(obs)

            id_fcst = closest(fcst[ii],ehcdf.x)
        #print(id_fcst, fcst[ii], ehcdf.x[id_fcst])        
            prob = ehcdf.y[id_fcst]
        #print(prob)
            id_obs = closest(prob, eocdf.y)
        #print(eocdf.y[id_obs])
            fcst_cor = eocdf.x[id_obs]
        #print(fcst[ii], fcst_cor, obs_fcst)
        #print()
            q_f_cor_em[id, ii, ih] = fcst_cor    
    #print(obs)
    #input()
'''



dd = []
date_dt = []
for d in dates:
    day=d[-2:]
    mon=d[-4:-2]
    dd.append('{}/{}'.format(mon,day))
    date_dt.append(datetime.strptime(d,'%Y%m%d'))

plt.figure(dpi=100,figsize=(20,10))
obs_fcst = q_f_obs_mat[:,2]	
plt.plot(range(15), obs_fcst[2:], linestyle='-',marker='o', markersize=10, color='k', label='Obs Streamflow')
#plt.plot(range(15), np.nanmean(q_f_cor_em[2:,:,1],axis=1), linestyle='--',marker='>', markersize=10, color='b', label='FCST MEAN')

bp=plt.boxplot(q_f_cor_em[2:,:,2].T, positions=range(15),showfliers=True, widths=[0.3]*15)

plt.setp(bp[   'boxes'], linewidth=2)
plt.setp(bp[ 'medians'], linewidth=2)
plt.setp(bp[    'caps'], linewidth=2)
plt.setp(bp['whiskers'], linewidth=2)

plt.ylim(0,400)
plt.legend(bbox_to_anchor=(1.0, 0.8))
plt.title('FCST BIAS REMOVED interp - ECMWF-SMAP vs Obs - 45days',fontsize=15)
plt.xticks(range(15),dates[2:],rotation=25)
#plt.xticklabels(dates,rotation=45)
plt.grid(True,linestyle='--',color='c',linewidth=0.5)
plt.xlabel('Initialization date')
plt.show()
exit()

correl_hind = np.full((17,4), np.nan)
bias = np.full((17,4), np.nan)
q_f_cor_mm[np.isnan(q_f_obs_mat)] = np.nan
q_h_cor_mm[np.isnan(q_h_obs_mat)] = np.nan
q_h_raw_mm[np.isnan(q_h_obs_mat)] = np.nan

 

def flow_metrics(q_f_cor_mm, q_h_raw_mm, q_f_obs_mat, q_h_obs_mat, dates, horizons):
   # for ih,hor in enumerate(horizons):
    for id, date in enumerate(dates):
       for ih,hor in enumerate(horizons):
     
           r = q_h_cor_mat[id,:,:,ih] ; r = r[~np.isnan(r)] #; print(r)
           s = q_h_obs_mat[id,:,:,ih] ; s = s[~np.isnan(s)] #; print(s) ; 
           #input()
           correl_hind[id, ih] = correl(r,s)[0]
           bias[id, ih] = np.mean(r-s)
           
#           print(date, hor,correl(r,s)[0])
#           print(len(r))
#           print(len(s))
            
    return correl_hind,bias

def plot_fcst_data(q_f_cor_mat, q_f_obs_mat, dates, horizons):
    dd = []
    date_dt = []
    for d in dates:
        day=d[-2:]
        mon=d[-4:-2]
        dd.append('{}/{}'.format(mon,day))
        date_dt.append(datetime.strptime(d,'%Y%m%d'))
        
    
    dfh = pd.DataFrame(data=q_f_cor_mat,index=date_dt,columns=horizons)
    dfo = pd.DataFrame(data=q_f_obs_mat,index=date_dt,columns=horizons)
    #for f,hor in enumerate(horizons):
        #fig, ax = plt.subplots(dpi=100,figsize=(2,.8))
        #qf = q_f_cor_mat[:,f]
        #qo = q_f_obs_mat[:,f]
        #plt.title('2018 forecast and obs',fontsize=18)
        #plt.plot(dates, qf, 'b-o',label='bias removed fcst')
        #plt.plot(dates,qo,'k->',label='obs')
        #ax.set_xticklabels(dd,rotation=45)
        #plt.xlabel('Issue model run dates')
        #ax.legend()
        #plt.savefig('savefig.png')

        

plot_fcst_data(q_f_cor_mat, q_f_obs_mat, dates,horizons)
exit()    

correl_hind, bias_hind = flow_metrics(q_f_cor_mat, q_h_cor_mat, q_f_obs_mat, q_h_obs_mat, dates, horizons)
dd = []
date_dt = []
for d in dates:
    day=d[-2:]
    mon=d[-4:-2]
    dd.append('{}/{}'.format(mon,day))
    date_dt.append(datetime.strptime(d,'%Y%m%d'))

fig, ax = plt.subplots(dpi=100,figsize=(2,1.))
df = pd.DataFrame(data=correl_hind, index=date_dt,columns=horizons)
df.to_csv('correl_matrix_hindcast.txt', sep=' ')
#dt =i pd.Datetimeindex(dates, 
df.iloc[:,0].plot(color='k',marker='X',markersize=11.)
df.iloc[:,1].plot(color='k',marker='>',markersize=11.)
df.iloc[:,2].plot(color='k',marker='*',markersize=11.)
df.iloc[:,3].plot(color='k',marker='^',markersize=11.)
#df.iloc[:,4].plot(color='k',marker='o',markersize=11.)
#df.iloc[:,5].plot(color='k',marker='D',markersize=11.)
ax.legend(bbox_to_anchor=(1.0, 0.5 ))
#ax.legend('center left', bbox_to_anchor(1,0.5))
ax.set_title('correlation stremflow Hindcast ECMWF-smap vs Obs')
ax.set_xticks(dates)
ax.set_xticklabels(dd,rotation=45) 
ax.grid(True)
ax.set_xlabel('Runs [month/day]')
plt.savefig('correl_hind_9817.png')



fig, ax = plt.subplots(dpi=100,figsize=(2,1.))
df = pd.DataFrame(data=bias_hind, index=date_dt,columns=horizons)
df.to_csv('bias_matrix_hindcast.txt', sep=' ')
#dt =i pd.Datetimeindex(dates,
df.iloc[:,0].plot(color='k',marker='X',markersize=11.)
df.iloc[:,1].plot(color='k',marker='>',markersize=11.)
df.iloc[:,2].plot(color='k',marker='*',markersize=11.)
df.iloc[:,3].plot(color='k',marker='^',markersize=11.)
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

