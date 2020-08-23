import numpy as np

import pandas as pd
from calendar import monthrange

dd = pd.read_csv('vazoes_C_36160000.csv',usecols=range(4,35),sep=';')
array = dd.values

#inverse raw order
array = array[::-1,:]
print(array.shape)

dates = pd.date_range('19730101','20200531',freq='M')
dates = dates.to_pydatetime()
data = []
for line in range(array.shape[0]):
    m = dates[line].month
    y = dates[line].year
    a, daysm = monthrange(y,m)

    for column in range(daysm):
        print(m, y )
        data.append(array[line, column])
data = np.array(data)
print(data.shape)
print(len(dates))

dates = pd.date_range('19730101','20200531',freq='D')
dd = pd.DataFrame(data=data, index=dates)

dd.to_csv('vazao_posto_iguatu_36160000.csv',header=False,na_rep='nan')


