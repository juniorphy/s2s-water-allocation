# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from smapm import smapm
from datetime import datetime
from f_pr_pet import main_pr, main_pet
from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
from hidropy.utils.hidropy_utils import create_path, read_thiessen_obs, lsname
from hidropy.utils.write_flow import write_flow
from hidropy.utils.smapm import smapm
from hidropy.utils.basins_smap_parameters_monthly import smap_param_mon
from calendar import monthrange



f = smapm(parms, pr, pet)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date in formar YYYYMMDD")
    return parser.parse_args()


# Parametros
smap_param_mon = {
'194-castanhao': [  931.  ,  3.979,  0.,  0.,  30.,  0.,  14926.  ], # Parametros calculados por Robson - Dados de vazao para calculo da vazao disponibilizados no Portal Hidro
'88-pacoti'    : [ 1227.  ,  4.9  ,  0.,  0.,  30.,  0.,    873.92],
'9-oros'       : [  601.  ,  3.92 ,  0.,  0.,  30.,  0.,   9568.09],
'2-banabuiu'   : [  961.  ,  2.95 ,  0.,  0.,  30.,  0.,  14249.  ],
'0-prg'        : [ 1999.98,  3.78 ,  0.,  0.,  30.,  0.,   4742.01]
 }

#~ dict_datas = {
#~ '194-castanhao': "200201_201512",
#~ '88-pacoti'    : "200201_201707",
#~ '9-oros'       : "198601_201512",
#~ '2-banabuiu'   : "198601_201512",
#~ '0-prg'        : "200201_201707"
#~ }

obs_dir = os.environ["OBS_DIR"]
args = arguments()

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

    # Salvando vazao smap
    path_runoff = "{0}/runoff".format(path_dados)
    if not os.path.exists(path_runoff):
        os.makedirs(path_runoff)
    name_f_out = "{0}/vazao_smap_{1}_{2}_{3}.asc".format(path_runoff, name_basin, date_serie_i_basin[:6], date_serie_f_basin[:6])
    f_out = open(name_f_out, 'w')
    for index, date in enumerate(list_dates_f):
        f_out.write("{0} {1} {2:.3f}\n".format(date.year, date.month, qcal[index]))
    f_out.close()
    print "done      -->", name_f_out
    # Plotando resposta
    #~ plt.plot_date(list_dates_f, qcal, fmt='k-o', xdate=True, linewidth=4)
    #~ try:
        #~ plt.plot_date(list_dates_q_teste, q_teste, fmt='r-x', xdate=True, linewidth=2)
        #~ plt.legend(['SMAP', 'Referencia'])
        #~ plt.ylabel('Q (m3/s)', fontsize=15)
        #~ plt.title('Vazao Mensal SMAP {0} - {1} a {2}'.format(name_basin.upper(), date_serie_i_basin[:6], date_serie_f_basin[:6]), fontsize=20)
    #~ except:
        #~ plt.ylabel('Q (m3/s)', fontsize=15)
        #~ plt.title('Vazao Mensal SMAP {0} - {1} a {2}'.format(name_basin.upper(), date_serie_i_basin[:6], date_serie_f_basin[:6]), fontsize=20)

    #~ try:
        #~ name_fig = "/home/funceme/smap/dados/figuras/vazao_smap_{0}_{1}_{2}.png".format(name_basin, date_serie_i_basin[:6], date_serie_f_basin[:6])
        #~ plt.savefig(name_fig)
        #~ print "done      -->", name_fig
    #~ except:
        #~ print "fail      -->", name_fig
    #~ plt.show ()
