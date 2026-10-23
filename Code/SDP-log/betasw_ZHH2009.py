#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PML python implementation of pure-water backscattering in Kramer et al. 2022.
from https://github.com/sashajane19/Rrs_pigments.


"""

import numpy as np

def RI_RIdirSal(Tc, S, wl):
    
    ''' Compute nsw (absolute refractive index of seawater) and dndS
    (partial derivative of seawater refractive index w.r.t. salinity) '''
    
    # RI of air - Ciddor (1996, Applied Optics)
    nair = 1.0 + (5792105.0 / (238.0185 - 1.0 / (wl / 1e3)**2) + 
                   167917.0 / (57.362 - 1.0 / (wl / 1e3)**2)) / 1e8

    # RI of seawater- Quan and Fry (1994, Applied Optics)
    n0 = 1.31405; n1 = 1.779e-4 ; n2 = -1.05e-6 ; n3 = 1.6e-8 ; n4 = -2.02e-6 ;
    n5 = 15.868; n6 = 0.01155;  n7 = -0.00423;  n8 = -4382 ; n9 = 1.1455e6;
    nsw = (n0 + (n1 + n2 * Tc + n3 * Tc**2) * S + 
             n4 * Tc**2 + (n5 + n6 * S + n7 * Tc) / wl + 
             n8 / wl**2 + n9 / wl**3) * nair  # pure seawater


    dnsw_dS = (n1 + n2 * Tc + n3 * Tc**2 + n6 / wl) * nair
    
    return nsw, dnsw_dS

def BetaT(Tc, S):
    
    ''' Compute Isothermal compressibility from Millero 1980'''

    # pure water secant bulk Millero (1980, Deep-sea Research)
    kw = 19652.21 + 148.4206 * Tc - 2.327105 * Tc**2 + 1.360477e-2 * Tc**3 - 5.155288e-5 * Tc**4
    Btw_cal = 1.0 / kw

    # seawater secant bulk
    a0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc**2 - 6.167e-5 * Tc**3
    b0 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc**2

    Ks = kw + a0 * S + b0 * S**1.5

    # calculate seawater isothermal compressibility from the secant bulk
    IsoComp = 1.0 / Ks * 1e-5  # unit is Pascals
    
    return IsoComp

def density_seawater(Tc, S):
    
    ''' Density of water and seawater, unit is Kg/m^3, from UNESCO,38,1981'''
  
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    # density for pure water 
    rhow = (b0 + b1 * Tc + b2 * Tc**2 + b3 * Tc**3 + 
                  b4 * Tc**4 + b5 * Tc**5)
    
    # density for pure seawater
    rhosw = (rhow + ((a0 + a1 * Tc + a2 * Tc**2 + a3 * Tc**3 + 
                                 a4 * Tc**4) * S + 
                                (a5 + a6 * Tc + a7 * Tc**2) * S**1.5 + 
                                a8 * S**2))
    return rhosw


def  WaterAct_dir_Sal(Tc, S):
    ''' Compute partial derivative of natural logarithm of `water activity' wrt 
    to salinity. Water activity data of seawater is from Millero and Leung (1976,American
    % Journal of Science,276,1035-1077). '''

    dlogaw_dS = (-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc**2 + 
               2.40808e-11 * Tc**3) + \
              1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 
                     2.08919e-9 * Tc**2 - 1.39872e-11 * Tc**3) * S**0.5 + \
              2 * (-2.31065e-6 - 1.37674e-9 * Tc - 
                    1.93316e-11 * Tc**2) * S
    return dlogaw_dS


def RI_dir_density(nsw):
    'Density derivative of refractive index wrt to density from PMH model'
    dnsw_drho = (nsw**2 - 1) * (1 + 2 / 3 * (nsw**2 + 2) * 
                                             (nsw / 3 - 1 / 3 / nsw)**2)
    return dnsw_drho


def compute_betasw_ZHH2009(wl, Tc, S, delta=0.039):
    
    ''' Based Xiaodong Zhang, Lianbo Hu, and Ming-Xia He (2009), Scattering by pure
     seawater: Effect of salinity, Optics Express, Vol. 17, No. 7, 5698-5710 
    
    Input: 
    % wl (nm): wavelength
    % Tc: temperture in degree Celsius, must be a scalar
    % S: salinity, must be scalar
    % delta: depolarization ratio, if not provided, default = 0.039 will be used
    
    # In general, function of scattering angle (theta). Here we compute for 90 deg (i.e. backscatter)
    
    Output:
        
    % bw: total scattering coefficient for purewater'
    '''

    # values of the constants
    Na = 6.0221417930e23   # Avogadro's constant
    Kbz = 1.3806503e-23    # Boltzmann constant
    Tk = Tc + 273.15       # Absolute temperature (K)
    M0 = 18e-3             # Molecular weigth of water in kg/mo

    # RI and salinity derivative for seawater
    nsw, dnsw_dS = RI_RIdirSal(Tc, S, wl)
     
    # Isothermal compressibility 
    isoComp= BetaT(Tc, S)

    # Density of seawater    
    rhosw = density_seawater(Tc, S)
    
    # derivative of natural logarithm of water activit  wrt.alinity
    dlogaw_dS = WaterAct_dir_Sal(Tc, S)

    # density derivative of refractive index from PMH model
    dnsw_drho = RI_dir_density(nsw)
 
    # volume scattering at 90 degree due to the density fluctuation
    beta_df = (np.pi**2/2)*((wl*1e-9)**(-4))*Kbz*Tk*isoComp*dnsw_drho **2*(6+6*delta)/(6-7*delta)
    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S*M0*dnsw_dS**2/rhosw/(-dlogaw_dS)/Na;    
    beta_cf = (2*np.pi**2)*((wl*1e-9)**(-4))*nsw**2*(flu_con)*(6 + 6*delta)/(6 - 7*delta)

    # sum of density and concentration terms
    beta90sw = beta_df + beta_cf #
    bsw = 8*np.pi/3*beta90sw*(2 + delta)/(1 + delta)
    
    # this is an approximate check using simpler model for bbw (no explcit S and T dependece)
    # as employed in Groestch et al. 2017, OE.
    # b1 = 0.00111
    # wl_ref_water = 500
    # S_water = -4.32
    # bb_water = b1 * (wl / wl_ref_water)**S_water
    # plt.figure()
    # plt.plot(2*(bsw.T))
    # plt.plot(bb_water)
    
    return bsw
 
  
