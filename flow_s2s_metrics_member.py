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
np.set_printoptions(precision=3, suppress=True)
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import arange, array, exp

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, list(xs))))

    return ufunclike

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

#-------------------------------------------------------------------------#

# Empirical cdf bias remove

q_f_cor_em = np.zeros((17,51,4))
for ih in range(3):
    for id in range(17):

        for ii in range(51):
           
            obs_fcst = q_f_obs_mat[id,ih]
            hind = q_h_raw_mm[id, :,:,ih]
            hind = np.reshape(hind, (220,))

            fcst = np.squeeze(q_f_raw_mm[id,ii,ih])
             
            obs = np.squeeze(q_h_obs_mat[id,:,ih])
            obs = obs[~np.isnan(obs)]

            ehcdf = ECDF(np.sort(hind))
            eocdf = ECDF(np.sort(obs))

            prob = np.interp(fcst,ehcdf.x,ehcdf.y)
          
            fcst_cor_np = np.interp(prob, eocdf.y, eocdf.x)
             
            ehcdfintp = interp1d(ehcdf.x, ehcdf.y,'linear', bounds_error=False)
            eocdfintp = interp1d(eocdf.y, eocdf.x,'linear',bounds_error=False)
            
            prob =  ehcdfintp(fcst)
            fcst_cor = eocdfintp(prob)
     
    #        print('date', dates[id], 'membro ', ii+1)
            q_f_cor_em[id, ii, ih] = fcst_cor_np
     #       print(fcst, fcst_cor_np, obs_fcst)
#dealing with -inf

# bias removing cross validation
print('q_h_raw_mm ', q_h_raw_mm.shape)
#print('q_h_obs_mat', q_h_obs_mat.shape)

for ih in range(4):
    for id in range(17):
        for iy in range(20):

            hind_d = np.copy(q_h_raw_mm)
            obs_d = np.copy(q_h_obs_mat)
            obs_d = obs_d[id,:,ih]
            hind_d=hind_d[id,:,:,ih]

            shind = hind_d[iy,:]
            sobs = obs_d[iy]
            hind_d = np.delete(hind_d, iy, axis=0)
            obs_dd  = np.delete(obs_d, iy,axis=0)
            hind_d = np.reshape(hind_d, (19*11,))
            obs_dd = obs_d[~np.isnan(obs_d)]

            ehcdf = ECDF(hind_d)
            eocdf = ECDF(obs_dd)
            ehcdfintp = interp1d(ehcdf.x, ehcdf.y,kind='linear',bounds_error=False)
            eocdfintp = interp1d(eocdf.y, eocdf.x,kind='linear',bounds_error=False)
            #extrap1d
            
            #ehcdfextp = extrap1d(ehcdfintp)
            #eocdfextp = extrap1d(eocdfintp)
            
            for im in range(11):
                h_y = shind[im]
                prob = ehcdfintp(h_y)
                perc = eocdfintp(prob)
                #print(perc)
                # prob = np.interp(h_y, ehcdf.x, ehcdf.y)
                
                # perc  = np.interp(prob, eocdf.y, eocdf.x)
                
                if ~np.isfinite(prob):
                    if np.isnan(prob):
                        #print(prob)
                        #print(h_y,np.max(hind_d),prob,perc,sobs)
                        h_y = np.max(hind_d)#-10*np.max(hind_d)/100.
                        prob =  ehcdfintp(h_y)
                        perc = eocdfintp(prob)
                        #print(h_y,prob,perc,sobs)
                        #input()
                if ~np.isfinite(perc):
                    if np.isnan(perc):
                        print(perc)
                        #print(h_y,np.max(hind_d),prob,perc,sobs)
                        #perc = np.max(obs_d)
                        #print(h_y,prob,perc,sobs)
                        #input()
                    else:
                        #print(perc)
                        #print(h_y,prob,perc,sobs)
                        perc = 0.0
                        #print(h_y,prob,perc,sobs)
                        #input()
                #print(h_y,prob,perc)
                    #input()
                #     if np.isnan(perc):
                #         perc = np.nanmax(obs_d)
                #     else:
                #         perc = 0.0
                # print(prob, perc)
                                
                q_h_cor_mm[id,iy,im,ih]=perc
                np.set_printoptions(precision=3, suppress=True)
                #print('ano =', 1998+iy, 'member =', im+1)
                #print('prob =',prob,' hind_raw =',h_y, ' hind_cor = ', perc, 'obs = ',sobs)     

#print(q_h_cor_mm[np.isnan(q_h_cor_mm)]
#$np.save('q_h_cor_mm.npy',q_h_cor_mm) 
q_h_cor_mm = np.load('q_h_cor_mm.npy')
#print(q_h_cor_mm.shape)
#exit()
p33 = np.zeros((17,4))
p66 = np.copy(p33)
p50 = np.copy(p33)
p80 = np.copy(p33)

for id in range(17):
    for ih in range(4):
        #print(np.squeeze(q_h_obs_mat[id,:,ih]))
        pobs = np.squeeze(q_h_obs_mat[id,:,ih])
        p33[id, ih] = np.percentile(pobs[~np.isnan(pobs)], 33.33)   #abaixo
        p66[id, ih]  = np.percentile(pobs[~np.isnan(pobs)], 66.66)  #acima
        p50[id, ih]  = np.percentile(pobs[~np.isnan(pobs)], 50)     #abaixo
        p80[id, ih]  = np.percentile(pobs[~np.isnan(pobs)], 80)     #acima
        

#zero 1 

toroc = np.full((17,4,20,3,2),np.nan)

for ih in range(4):
    for id in range(0,17):
        for iy in range(20):
            
            obs = q_h_obs_mat[id, iy, ih]
            if np.isnan(obs):
                continue
            else:
                if obs >= p80[id,ih]:
                    bin = [ 0 , 0, 1]
                elif obs <= p50[id,ih]:
                    bin = [ 1, 0, 0 ]
                else:
                    bin = [ 0,1,0]
                toroc[id,ih,iy,:,1] = bin

                membs = q_h_cor_mm[id, iy, :,ih]
                #print(membs)
                #print(p50[id,ih],p80[id,ih])
                ib = len(np.where (membs <= p50[id, ih])[0])/11.
                #print(np.where (membs <= p50[id, ih])[0])
                #print(np.where (membs >= p80[id, ih])[0])
                ia = len(np.where (membs >= p80[id, ih])[0])/11.
                inn = 1. - (ia + ib)
                #print(ib,inn,ia)
                probs = [ib, inn, ia]
                #print(probs)
                #input()
                toroc[id,ih,iy,:,0] = probs

roc_area = np.full((17,4,3), np.nan)
roc_area_global = np.full((4,3), np.nan)
cats = ['below', 'normal', 'above']

for ic, cat in enumerate(cats):
    
    for ih in range(4):
        p_bing = toroc[:,ih,:,ic,:]
        p_bing = np.reshape(p_bing,(17*20,2))

        p_bing = p_bing[np.argsort(p_bing[:,0])]
        prob = np.unique(p_bing[:,0])
        proba = prob[~np.isnan(prob)][::-1]
            
        hr = np.zeros((len(proba)+2,))
        far = np.copy(hr)
        hr[0] = 0
        hr[-1] = 1
        far[0]=0
        far[-1]=1
        for ip in range(1,len(proba)+1):
            f = p_bing[p_bing[:,0] >= proba[ip-1],:]
            a = np.sum(f[:,1] ==1) #hit
            b = np.sum(f[:,1] ==0) #false alarm
            g = p_bing[p_bing[:,0] < proba[ip-1],:]
            c = np.sum(g[:,1] ==1) #
            d = np.sum(g[:,1] ==0) #
            hr[ip] = (a /(a+c))
            far[ip] =( b/(b+d))
        rocarea = np.trapz(hr,far)
        roc_area_global[ih, ic] = rocarea
        
        # for dates
        for id in range(17):
            p_bin = toroc[id,ih,:,ic,:]
            p_bin = p_bin[np.argsort(p_bin[:,0])]
            
            prob = np.unique(p_bin[:,0])
            proba = prob[~np.isnan(prob)][::-1]
            
            hr = np.zeros((len(proba)+2,))
            far = np.copy(hr)
            hr[0] = 0
            hr[-1] = 1
            far[0]=0
            far[-1]=1
            for ip in range(1,len(proba)+1):
                f = p_bin[p_bin[:,0] >= proba[ip-1],:]
                a = np.sum(f[:,1] ==1) #hit
                b = np.sum(f[:,1] ==0) #false alarm
                g = p_bin[p_bin[:,0] < proba[ip-1],:]
                c = np.sum(g[:,1] ==1) #
                d = np.sum(g[:,1] ==0) #
                hr[ip] = (a /(a+c))
                far[ip] =( b/(b+d))
            
            #plt.plot(far,hr,'-',color='black')
            #plt.plot([0,1],[0,1],'--',color='blue')
            #plt.xlim([0.,1.])
            #plt.xlim([0.,1.])
            #plt.xlabel('False alarm rate')
            #plt.ylabel('Hit rate')
            rocarea = np.trapz(hr,far)
            roc_area[id, ih, ic] = rocarea

            #plt.title('ROC curve for date 18Jan - 15 days mean\n ROC AREA {0:2.2f} above 80th percentile '.format(rocarea))
            
#np.save('roc_area_mat.npy',roc_area)
#roc_area = np.load('roc_area_mat.npy')

#exit()

q_f_cor_em[~np.isfinite(q_f_cor_em)] = 0.0

for id in range(17):
    for ih in range(4):
        #print(np.squeeze(q_h_obs_mat[id,:,ih]))
        pobs = np.squeeze(q_h_obs_mat[id,:,ih])
        p33[id, ih] = np.percentile(pobs[~np.isnan(pobs)], 33.33)
        p66[id, ih]  = np.percentile(pobs[~np.isnan(pobs)], 66.66)
       
labelh = ['15 day mean', '30 day mean', '45 day mean']
panel = ['d' , 'c', 'f']
dd = []
date_dt = []
for d in dates:
    dt=datetime.strptime(d,'%Y%m%d')
    date_dt.append(datetime.strptime(d,'%Y%m%d'))
    dt_str = dt.strftime('%d%b')
    #day=datetime.strftime('%d')
    #mon=d[-4:-2]
    dd.append('{}'.format(dt_str))

for ih,hor in enumerate(horizons[0:3]):
    plt.figure(dpi=160,figsize=(25,8))
    obs_fcst = q_f_obs_mat[:,ih]	
    plt.plot(range(15), obs_fcst[2:], linestyle='-',marker='o',linewidth='2', markersize=8, color='k', label='Obs flow in 2018')
    #plt.plot(range(15), np.nanmean(q_f_cor_em[2:,:,1],axis=1), linestyle='--',marker='>', markersize=10, color='b', label='FCST MEAN')
    plt.plot(range(15),p50[2:,ih],linestyle='--', linewidth = 2, color='gray', label='Climatological (1998-2017) 50th and 80th percentiles' )
    plt.plot(range(15),p80[2:,ih],linestyle='--', linewidth = 2, color='gray') #, label='66th percentile' )
    
    bp=plt.boxplot(q_f_cor_em[2:,:,ih].T, positions=range(15),showfliers=True, widths=[0.3]*15)

    plt.setp(bp[   'boxes'], linewidth=2)
    plt.setp(bp[ 'medians'], linewidth=2)
    plt.setp(bp[    'caps'], linewidth=2)
    plt.setp(bp['whiskers'], linewidth=2)

    plt.ylim(0,250)
    plt.xlim(-1.0,15.)
    plt.legend(bbox_to_anchor=(0.38, 1.0),fontsize=17,loc='upper right')
    #plt.title('{1}) Flow Forecast for 2018 ({0})'.format(labelh[ih], panel[ih]),fontsize=30)
    #plt.title('{1})'.format(labelh[ih], panel[ih]), loc='left',fontsize=32)
    plt.text(-2, 255, '{0})'.format(panel[ih]), fontsize=32)
    plt.xticks(range(15),dd[2:],rotation=0, fontsize=23)
    plt.tick_params(axis="y", labelsize=23)

    #plt.xticklabels(dates,rotation=45)
    plt.grid(True,linestyle='--',color='c',linewidth=0.5)
    plt.xlabel('Initialization date', fontsize=25)
    plt.ylabel('flow (m\u00b3/s)',fontsize=25)
    plt.savefig('fcst_bias_removed_2018_{}.png'.format(hor))
    plt.close()


plt.plot(range(51), q_f_cor_em[9,:,0],'-o',color='black',label='bias removed')
plt.plot(range(51), q_f_raw_mm[9,:,0], '->',color='blue',label='raw')
plt.legend()

#q_f_cor_mm[np.isnan(q_f_obs_mat)] = np.nan
#q_h_cor_mm[np.isnan(q_h_obs_mat)] = np.nan
#q_h_raw_mm[np.isnan(q_h_obs_mat)] = np.nan


# def flow_metrics(q_f_cor_mm, q_h_raw_mm, q_f_obs_mat, q_h_obs_mat, dates, horizons):
#    # for ih,hor in enumerate(horizons):
#     for id, date in enumerate(dates):
#        for ih,hor in enumerate(horizons):
     
#            r = q_h_cor_mat[id,:,:,ih] ; r = r[~np.isnan(r)] #; print(r)
#            s = q_h_obs_mat[id,:,:,ih] ; s = s[~np.isnan(s)] #; print(s) ; 
#            #input()
#            correl_hind[id, ih] = correl(r,s)[0]
#            bias[id, ih] = np.mean(r-s)
           
# #           print(date, hor,correl(r,s)[0])
# #           print(len(r))
# #           print(len(s))
            
#     return correl_hind,bias

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

#plot_fcst_data(q_f_cor_mat, q_f_obs_mat, dates,horizons)
    
'''
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
'''

cats = ['below 50th', 'between 50-80th', 'above 80th']
fign = ['50th', '50-80th', '80th']
labs = ['15 day mean', '30 day mean', '45 day mean']      
dd = []
date_dt = []
for d in dates:
    dt=datetime.strptime(d,'%Y%m%d')
    date_dt.append(datetime.strptime(d,'%Y%m%d'))
    dt_str = dt.strftime('%d%b')
    #day=datetime.strftime('%d')
    #mon=d[-4:-2]
    dd.append('{}'.format(dt_str))

for ic, cat in enumerate(cats):
   
    fig, ax = plt.subplots(dpi=150,figsize=(25,8.))
    df = pd.DataFrame(data=roc_area[:,:,ic], index=date_dt,columns=horizons)
    #dfc = pd.DataFrame(data=correl_global, index=date_dt)
    df.iloc[2:,0].plot(linestyle='-',color='black',linewidth=4,label=labs[0])
    df.iloc[2:,1].plot(linestyle='--',color='black',linewidth=4,label=labs[1])
    df.iloc[2:,2].plot(linestyle='dotted',color='black',linewidth=4,label=labs[2])
    #ax.text(0.5,0.5,'aaaaa',transform=ax.transAxes,fontsize=20)
    plt.axhline(roc_area_global[0,ic],linestyle='-', color='darkgray', linewidth=4)#,label='correl global 15days')
    plt.axhline(roc_area_global[1,ic],linestyle='--', color='darkgray', linewidth=4)#,label='correl global 15days')
    plt.axhline(roc_area_global[2,ic],linestyle='dotted', color='darkgray', linewidth=4)
    #plt.axvline(date_dt[5],linestyle='-', color='gray', linewidth=1,label='kkkkk')
    ax.legend(bbox_to_anchor=(1.0, 0.3 ),fontsize=19)
    #ax.legend('center left', bbox_to_anchor(1,0.5))
    ax.set_ylim([0.34,1.])
    ax.set_title('c) ROC Area (1998-2017) for flow {0} percentile'.format(cat), fontsize=30)
    ax.set_xticks(date_dt[2:])
    ax.set_xticklabels(dd[2:],rotation=0,fontsize=23)
    ax.tick_params(axis="y", labelsize=23)

    ax.set_xlabel('Initialization date',fontsize=25)
    ax.set_ylabel('Roc Area',fontsize=25)
    #ax.grid(True, linestyle='--', color='gray')
    plt.savefig('roc_area_hind_9817_{}.png'.format(fign[ic]))
