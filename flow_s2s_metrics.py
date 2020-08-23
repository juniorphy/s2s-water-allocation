# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from smapm import smapm
from datetime import datetime
#from f_pr_pet import main_pr, main_pet
#from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
#from hidropy.utils.hidropy_utils import create_path, read_thiessen_obs, lsname
#from hidropy.utils.write_flow import write_flow
from pfct.hidro.smapd import smapd
from basins_smap_parameters_daily import smap_param_day
from calendar import monthrange
from glob import glob
import os

#f = smapd(parms, pr, pet)


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


def read_s2s(exp, method):
    if exp == 'hind':
        year = range(1998,2018)
    else:
        year =  ['2018']
    for y in year:
        f_ls = np.sort(glob('data/{0}/ascii/{1}/*{2}*.txt'.format(exp, y, method)))
        for fin in f_ls:
            try:
                os.mkdir('data/{0}/qflow/{1}'.format(exp,y))
            except OSError as error:
                pass

            qcook = [ ]
            date = fin.split('/')[-1].split('_')[7]
            print(exp, method, y, date,fin)
            if method == 'member':
                flag = 1
                if exp == 'hind':
                    ic = 11
                if exp == 'fcst':
                    ic = 51
            else:
                flag = 0
                ic = 1
            for ii in range(ic):
                print('processing member ', ii+1)
                
                pr_s2s = pd.read_csv(fin,header=None, sep=' ',index_col=0, usecols=[0,ii+1])
                #exit()
                pr_s2s.rename(columns= {ii+1:'pr'}, inplace=True)
                pet_s2s = pet_clim_s2s(date)
            
                pr_back = read_obs(date)
              
                pr_back.index.set_names(['date'],inplace=True)
                pr_back.columns = ['pr']
                pet_back = pet_clim_back(date)
            #concatenate
                pr=pd.concat([pr_back,pr_s2s],axis=0)
                pet=pd.concat([pet_back,pet_s2s], axis=0)
                #print(pr.iloc[-60:])
            # Calculando vazao smap
                param = smap_param_day['oros_dirceu']
                #if flag == 1:
                data=smapd(param, pr.values, pet.values)
                qcook.append(data)
                #else:
                #   data=smapd(param, pr.values, pet.values)
                #   qcook.append(data)
            data = np.array(qcook).T
            qcal = pd.DataFrame(data=data, index=pet.index)
            
            
            qcal.loc[date:].to_csv('data/{1}/qflow/{2}/flow_daily_s2s_ecmwf_hind9817_{1}_{0}_{3}_46days.txt'.format(date,exp,y,method), sep=' ', header=None)                
    #def fcst_metrics()
   
def remove_bias_flow():

    #reading hind 9817
    a = np.sort(glob('data/hind/qflow/1997/*ensemble*.txt'))
    print(a)
    date = a[0].split('/')[-1].split('_')[7]
          
    print(date)
       #for y in range(1997,2018):
            
  
args = arguments()
exp = args.exp
#method = args.method
#read_s2s(exp, method)

remove_bias_flow()


#print(len(pet_clim_s2s('20200101')))
#print(pet_clim_back('20200101'))

exit() 
    
#    fin = 'data/{0}/{y}'.format()
#    pd.read_csv('')
#    pr_daily_s2s_ecmwf_hind9817_hind_ensemble_19980104_thiessen_46days.txt


exit()

obs_dir = os.environ["OBS_DIR"]
args = arguments()
'''
# Configuracoes iniciais
for basin in smap_param_mon:
    cod_basin = int(basin.split("-")[0])
    name_basin = basin.split("-")[1]
    
    date_serie_i_basin = '19740101' #data inicial da simulacao
    date_serie_f_basin = args.date #data final da simulacao
    list_dates_f = pd.date_range(start=date_serie_i_basin, end=date_serie_f_basin, freq='M')
    path_dados = "{0}/smap_runoff/data".format(obs_dir)

    try:
        name_q_teste = "{0}/vazao_teste/qvaz_cogerh_{1}_{2}-{3}.txt".format(path_dados, dict_datas["{0}-{1}".format(cod_basin, name_basin)], cod_basin, name_basin)
        q_teste = np.loadtxt(name_q_teste)
        q_teste[np.where(q_teste == -999.)] = np.nan 
        print "input     -->",name_q_teste
        list_dates_q_teste = pd.date_range(start=dict_datas["{0}-{1}".format(cod_basin, name_basin)][0:6]+'01', end=dict_datas["{0}-{1}".format(cod_basin, name_basin)][7:]+'31', freq='M')
    except:
        pass
        #print "fail      -->",name_q_teste


    # Coletando informacoes de pr e pet
    pr = main_pr(cod_basin, name_basin, date_serie_i_basin, date_serie_f_basin)
    pet = main_pet(cod_basin, name_basin, date_serie_i_basin, date_serie_f_basin)

    # Calculando vazao smap
    param = smap_param_mon["{0}-{1}".format(cod_basin, name_basin)]
    qcal = smapm(param, pr, pet)
'''
