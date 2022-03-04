
from pfct.Thiessen import thiessen
import numpy as np
import pandas as pd
import netCDF4 as nc4

nbas = '/home/junior/OneDrive/36160000_iguatu_ana.asc'
nbas = '/home/junior/OneDrive/36160000.asc'
nbas = '/home/junior/Downloads/Telegram_Desktop/bacia_oros2_incremental/Nova_Bacia_Increm_Oros2.asc'

def pr_thiessen(nbas):
	data = np.loadtxt('pr_daily_funceme_19730101_20210628.asc')

	pr = data[:, 3:]
	lon = data[:, 1]
	lat = data[:, 2]
	dates = pd.date_range('19730101','20210628')
	dates = dates.to_pydatetime()

	pr_thi = thiessen(pr,lat,lon, nbas,sep=',', pf=0.0,pf_step=0.3,pf_max=0.9).T

	df = pd.DataFrame(data=pr_thi,index=dates)

	print(pr[-100:])

	return df 
  
# def pet_thiessen(nbas):
# 	thi = []
# 	for y in range(1973,2021):
# 		for m in range(1,13):
# 			print('http://opendap-ng.funceme.br:8001/dados-obs/inmet-ana-sinda/kriging/daily/pet/{0}/pet-daily-inmet-sinda-{0}{1:02d}-0p5.nc'.format(y,m))
      
# 			if y == 2020 and m == 9:
# 				break
# 			else:
# 				ndid = nc4.Dataset('http://opendap-ng.funceme.br:8001/dados-obs/inmet-ana-sinda/kriging/daily/pet/{0}/pet-daily-inmet-sinda-{0}{1:02d}-0p5.nc'.format(y,m))

# 				pet = ndid.variables['pet'][:]
# 				lat = ndid.variables['lat'][:]
# 				lon = ndid.variables['lon'][:]
# 				time = ndid.variables['time']
# 				pet_thi = thiessen(pet,lat,lon, nbas, sep=',',pf=0.0,pf_step=0.3,pf_max=0.9,usenc=True).T
# 				for ii in range(len(pet_thi)):
# 					thi.append(pet_thi[ii][0])
# 				print(nc4.num2date(time[:],time.units)[-1].strftime('%Y%m%d'))
  
# 	return np.array(thi)
# #thi = pet_thiessen(nbas)
#dates=pd.date_range('19730101','20200823')
#df = pd.DataFrame(data=thi, index=dates)
#df.to_csv('pet_daily_thiessen_bac_iguatu_36160000.csv', header=False, sep=' ')

pr = pr_thiessen(nbas)
pr.to_csv('pr_daily_thiessen_bac_iguatu_36160000_19730101_20210628.csv', header=False, sep=' ')
#print(thi)

