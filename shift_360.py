from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

import numpy as np

import netCDF4 as nc4

nc = nc4.Dataset('data.nc')

lon = nc.variables['X'][:]
tp = nc.variables['tp'][:]
lat = nc.variables['Y'][:]
print(lat,lon)
#lon[lon > 180 ] = lon 

tp, lon = shiftgrid(180., tp, lon, start=False)
xy = np.full((len(lon)*len(lat), 2), np.nan)

c=0
for ii in range(len(lon)):
    for jj in range(len(lat)):
    
        xy[c,:] = [lon[ii],lat[jj] ] 
        c +=1 
    

np.savetxt('xy.txt', xy, delimiter=',')


