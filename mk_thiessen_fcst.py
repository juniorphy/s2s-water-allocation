import numpy as np
import pandas as pd
from pfct.Thiessen import thiessen
import os 
import glob 
import netCDF4 as nc4
from datetime import datetime 
from mpl_toolkits.basemap import shiftgrid

bas = ['oros'] #, 'castanhao', 'banabuiu'] 

start_date = "2015-01-01"
end_date = "2015-01-03"

dates = pd.date_range(start="2018-01-04", end="2018-04-30", freq="7D")
dates_dt = dates.to_pydatetime()


for bb in bas:
    nbas = glob.glob('/dados/s2s-water-allocation/reservatorio/*{0}*'.format(bb))[0]
    #print(nbas)
    for dd in dates:
        mm = dd.month
        yy = dd.year

        date = dd.strftime("%Y%m%d")
        fname = "/dados/s2s-water-allocation/data/fcst/ECMWF/{0}/{1:02d}/pr_daily_s2s_ecmwf_hind9817_fcst_{2}.nc".format(yy, mm, date)
        print('in = ', fname.split())
        ncid = nc4.Dataset(fname)
        lon = ncid.variables['longitude'][:]
        lat= ncid.variables['latitude'][:]
        pr = ncid.variables['pr'][:]
        #pr = np.mean(pr, axis=1)
        pr[pr < 0.] = 0.
        #print(pr.shape)
        nmemb = len(ncid.variables['member'][:])
        
        pr_mean = np.zeros((46,nmemb))

        for memb in range(nmemb):
            pr_cook = np.squeeze(pr[:,memb, ...])

            pr_thi = thiessen(pr_cook,lat,lon, nbas, pf=-1, usenc=True)
            pr_mean[:,memb] = pr_thi
        pr_ens = np.mean(pr_mean, axis=1)
        data_index = pd.date_range(date, periods=46,freq='1D')
        #print(len(data_index))
        #exit()

        df = pd.DataFrame(data=pr_ens, index=data_index, columns=[ 'pr'])
        #print(df)
        fout = 'data/fcst/ascii/{0}/pr_daily_s2s_ecmwf_hind9817_fcst_ensemble_{1}_thiessen_46days.txt'.format(yy, date)
        print('out = ', fout , '\n')
        #np.savetxt(fout,pr_ens, fmt='%6.2f')
        df.to_csv(fout, header=False, sep=' ')#, float_format='%6.2f')


        df = pd.DataFrame(data=pr_mean, index=data_index)
        #print(df)
        #input()
        fout = 'data/fcst/ascii/{0}/pr_daily_s2s_ecmwf_hind9817_fcst_member_{1}_thiessen_46days.txt'.format(yy, date)
        print('out = ', fout , '\n')
        #np.savetxt(fout,pr_ens, fmt='%6.2f')
        df.to_csv(fout, header=False, sep=' ') #, float_format='%6.2f')

    

  
