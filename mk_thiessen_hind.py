import numpy as np
import pandas as pd
from pfct.Thiessen import thiessen
import os 
import glob 
import netCDF4 as nc4
from datetime import datetime, timedelta 
from mpl_toolkits.basemap import shiftgrid
from calendar import isleap
bas = ['oros'] #, 'castanhao', 'banabuiu'] 

dates_fc = pd.date_range('2018-01-04','2018-04-30', freq='7D')

dates_fc = dates_fc.to_pydatetime()

dates_hd = []
for ano in range(1998,2018):
    dates = pd.date_range(start="{}-01-04".format(ano), end="{}-04-30".format(ano), freq="7D")
    dates_hd.append(dates)
#dates = [item for sublist in dates_hd for item in sublist]
#print(dates_hd[0])
    dates_dt = dates.to_pydatetime()
    c = 0

    for bb in bas:
        nbas = glob.glob('/dados/s2s-water-allocation/reservatorio/*{0}*'.format(bb))[0]
        print(nbas)
        for dd in dates_dt:
            d=dd.day
            mm = dd.month
            yy = dd.year

            mf = dates_fc[c].month
            df = dates_fc[c].day
            date = '{0}{1:02d}{2:02d}'.format(yy,mf,df)

            c += 1
            if c-1 == len(dates_fc):
                c=0

            #date = dd.strftime("%Y%m%d")
            fname = "/dados/s2s-water-allocation/data/hind/ECMWF/{0}/{3:02d}/pr_daily_s2s_ecmwf_hind9817_2018{3:02d}{4:02d}_{2}.nc".format(yy, mm, date,mf,df)
            print('in = ', fname)
            
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
                pr_mean[:, memb] = pr_thi
            pr_ens = np.mean(pr_mean, axis=1)
            data_index = pd.date_range(date, periods=46,freq='1D')
            df = pd.DataFrame(data=pr_ens, index=data_index, columns=[ 'pr'])

        #try: 
        #    os.mkdir('data/hind/ascii/{}'.format(yy)) 
        #except OSError as error: 
        #    print('pasta existe')  
        
            fout = 'data/hind/ascii/{0}/pr_daily_s2s_ecmwf_hind9817_hind_ensemble_{1}_thiessen_46days.txt'.format(yy, date)
            print('out = ', fout , '\n')
            df.to_csv(fout, header=False, sep=" ") #, float_format='%6.2f')
            
            df = pd.DataFrame(data=pr_mean, index=data_index)

        #try:
        #    os.mkdir('data/hind/ascii/{}'.format(yy))
        #except OSError as error:
        #    print('pasta existe')

            fout = 'data/hind/ascii/{0}/pr_daily_s2s_ecmwf_hind9817_hind_member_{1}_thiessen_46days.txt'.format(yy, date)
            print('out = ', fout , '\n')
            df.to_csv(fout, header=False, sep=" ")#, float_format='%6.2f')


        #np.savetxt(fout,pr_ens, fmt='%6.2f')
    


    

    

