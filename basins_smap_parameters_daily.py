# -*- coding: utf-8 -*-
"""
Dictionary of subbasins monthly parameters for smap model.

Dictionary format:
    smap_param_day = {'macro_micro_basin':[str, k2t, crec, ai, capcc, kkt, tuin, ebin, area], ... }

Parameters description:
    Str:   soil saturation capacity (mm).                    [Min=1000 Max=3000]  
    K2t:   surface runoff recession constant(days)           [Min=0    Max=5   ]
    Crec:  underground reservoir recharge coefficient (%)    [Min=0    Max=1   ]
    Ai:    initial abstraction(mm)                           [Min=0    Max=10  ]
    Capcc: field capacity (%)                                [Min=0    Max=0   ]
    Kkt:   base flow recession constant (days)               [Min=5    Max=5   ]
    Tuin:  initial moisture content (dimensionless).         [Min=0    Max=100 ]
    Ebin:  initial basic flow (m3 s-1).                      [Min=5    Max=1000]
    Area:  basin area (km2).

Usage:
    from hidropy.utils.basins_smap_parameters_monthly import smap_param_mon
    print smap_param_day['oros']

    if you are using smap calling from another script
    smap_model_routine(input_parameter = smap_param_mon['oros'])
"""

# Parametros TCC Walter Xavier

smap_param_day = {
'oros_walter':     [ 292.61,   3.94,   0.,  5.71,  71.66, 0., 0.,  0.,  14926.] #,
#'oros_vinicius' : [ ] # 
 }
























