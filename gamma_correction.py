# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss


def gamma_correction(data1, data2, data3):
    """
    :param data1: Observation values array.                   Ex: data1.shape --> (30) --> [ 90.5, 135.6, ...]
    :param data2: Historical  values array.                   Ex: data2.shape --> (30) --> [111.5,  86.7, ...]
    :param data3: Historical or Future scenario values array. Ex: data3.shape --> (30) --> [111.5,  86.7, ...]
    
    Return a array similar to data3, but with values adjusted by gamma distribution.
    
    
    Some definitions (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html)
    
    fit(data, a, loc=0, scale=1) --> Parameter estimates for generic data.
    cdf(   x, a, loc=0, scale=1) --> Cumulative distribution function.
    ppf(   q, a, loc=0, scale=1) --> Percent point function (inverse of cdf — percentiles).
    
    """
    data1 = data1[~np.isnan(data1)]
    data1[data1<0.01] = 0.01
    data2[data2<0.01] = 0.01
    if data3 < 0.001: 
        data3  = 0.001
    if data2.ndim == 2:
        data2 = np.reshape(data2,(data2.shape[0]*data2.shape[1],1))
    #print(data1,data2)

    alpha_d1, loc_d1, beta_d1 = ss.gamma.fit(data1, floc=0.)
    alpha_d2, loc_d2, beta_d2 = ss.gamma.fit(data2, floc=0.)
  
    d3_adjusted = np.full(data3.shape, np.nan)
           
    #for i, d3 in enumerate(data3):
    d3_probability = ss.gamma.cdf(data3, alpha_d2, scale=beta_d2)
    d3_adjusted = ss.gamma.ppf(d3_probability, alpha_d1, scale=beta_d1)
            
    d3_adjusted = np.where(d3_adjusted == np.inf, np.nan, d3_adjusted)

    return d3_adjusted

##def empirical_cdf(data1, data2, data3);
#    if data2.ndim == 2:
#    data2 = np.reshape(data2,(data2.shape[0]*data2.shape[1],1))


        
