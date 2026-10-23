#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PML implementation of delta Rrs residul model in Kramer et al. 2022, applied to AMT data
(original matlab repo here: https://github.com/sashajane19/Rrs_pigments).             

We also consider a modified form of the bio-optical model where a Tchla based 
prediction of hyperspectral particulate absorption ap is used in place of the 
original phytoplankton absoprtion ap model in Kramer 2022.

Tom Jordan: tjor@pml.ac.uk.

"""

# imports from external python modules
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import lmfit as lm
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter 

# imports from other scripts in codebase
from betasw_ZHH2009 import compute_betasw_ZHH2009
from gsm_optimization import  gsm_invert
from gsm_optimization import  gsm_invert_cdomfree


def wl_interp(spec, spec_wl, wl_out, method ='pw_lin'):
   
    """
    Performs wavelength interpolation on 2D (dims: time X wavelength) spectral
    matrices - cubic or piecewise linear (recommended default) are given as options. 
      
    """  
    
    if method == 'pw_lin': # piecewise linear 
        spec_interp = np.nan*np.ones([len(spec), len(wl_out)])
        for i in range(len(spec)):
            spec_interp[i,:] = np.interp(wl_out, spec_wl, spec[i,:])  # interpolate to c
   
    elif method == 'cubic': # cubic spline
        spec_interp = np.nan*np.ones([len(spec), len(wl_out)])
        for i in range(len(spec)):
            spline_i = CubicSpline(spec_wl, spec[i,:])  # interpolate to c
            spec_interp[i,:] = spline_i(wl_out)

    return spec_interp


def L2chlor_a(Rrs443, Rrs488, Rrs547, Rrs555, Rrs667):
    
    """ Use weighted MODIS Aqua bands to calculate chlorophyll concentration
    using oc3m blended algorithm with CI (Hu et al. 2012)  -  this code  was taken 
    from HyperCP radiometry code base: https://github.com/nasa/HyperCP"""

    thresh = [0.15, 0.20]
    a0 = 0.2424
    a1 = -2.7423
    a2 = 1.8017
    a3 = 0.0015
    a4 = -1.2280

    ci1 = -0.4909
    ci2 = 191.6590
    
    if Rrs443 > Rrs488:
        Rrsblue = Rrs443
    else:
        Rrsblue = Rrs488

    log10chl = a0 + a1 * (np.log10(Rrsblue / Rrs547)) \
        + a2 * (np.log10(Rrsblue / Rrs547))**2 \
            + a3 * (np.log10(Rrsblue / Rrs547))**3 \
                + a4 * (np.log10(Rrsblue / Rrs547))**4

    oc3m = np.power(10, log10chl)

    CI = Rrs555 - ( Rrs443 + (555 - 443)/(667 - 443) * \
        (Rrs667 -Rrs443) )
        
    ChlCI = 10** (ci1 + ci2*CI)

    if ChlCI <= thresh[0]:
        chlor_a = ChlCI
    elif ChlCI > thresh[1]:
        chlor_a = oc3m
    else:
        chlor_a = oc3m * (ChlCI-thresh[0]) / (thresh[1]-thresh[0]) +\
            ChlCI * (thresh[1]-ChlCI) / (thresh[1]-thresh[0])

    return chlor_a


def salt_water_iops(T, S, wl):
    
    """
    Loads pure water IOP components following Kramer et al. 2022
    
    Input:
        
        T: Temperature
        S: Salinity
        wl: wavelength grid 
  
    Output:
        
        asw: salt water spectral absorption coefficient
        bsw: salt water spectral backscatter coefficient
    
    All matrices have dims time/sample no. by wavelength
  
    """
    #
    path_root = str(Path(os.getcwd()).parents[1]) + '/'
    path_data = path_root + 'Data/Input/'
    
    # salt water absorption
    asw_data = scipy.io.loadmat(path_data + 'Model_params/asw_all.mat') # defined on [350,700] nm
    asw = asw_data['asw_all'][50:, 1]  # asw on [400,700] nm
    
    # salt water backscattering
    bsw = np.nan*np.ones((len(T), len(wl)))  
    for i in range(len(T)):
        bsw[i, :] = compute_betasw_ZHH2009(wl, T[i], S[i])
    bbsw = 0.5 * bsw
    
    return  asw, bbsw


def non_water_iops(Rrs, rrs, wl):
    
    """
    Estimates non-water input IOP components from Case 1 rrs/Rrs algorithms. 
    This will potentially change if Case-2 waters are considered
    
    Input:
        
        Rrs: Above-water spectral reflectance
        rrs: In-water spectral reflectance
        wl: wavelength grid 
  
    Output:
        
        adgstar: combined anap-acdom spectral absorption coeffcieint
        bbpstar: particualte spectral backscattering coefficient
        A, B: spectral coefficients in aph power-law model
        
    All matrices have dims time/sample no. X wavelength
  
    """
    
    path_root = str(Path(os.getcwd()).parents[1]) + '/'
    path_data = path_root + 'Data/Input/'
    
    # compute adg slope and adg offsetr (note - this is referred to as cdom in the original matlab codebase
    # but it is a combined detrital-cdom exponential model)
    if Rrs.ndim == 2:
        adg_s =  - (0.01447 + 0.00033 * Rrs[:,90] / Rrs[:,155])
        adgstar= np.exp(adg_s[:, np.newaxis] * (wl - 443))
        
        # compute bbp slope and bbp as estimated from rrs
        bbp_s = 2.0 * (1. - 1.2 * np.exp(-0.9 * rrs[ :,40] / rrs[:,155]))
        bbpstar = (443/wl[:, np.newaxis])**bbp_s
        bbpstar = bbpstar.T
    elif Rrs.ndim == 1:
        adg_s =  - (0.01447 + 0.00033 * Rrs[90] / Rrs[155])
        adgstar= np.exp(adg_s[:, np.newaxis] * (wl - 443))
        
        # compute bbp slope and bbp as estimated from rrs
        bbp_s = 2.0 * (1. - 1.2 * np.exp(-0.9 * rrs[ :,40] / rrs[:,155]))
        bbpstar = (443/wl[:, np.newaxis])**bbp_s
        bbpstar = bbpstar.T
        
    # From Kramer 2022 code base
    ab_coeffs = scipy.io.loadmat(path_data + 'Model_params/AB_coeffs.mat')
    A = ab_coeffs['A'][50:].ravel()  # [400,700] nm
    B = ab_coeffs['B'][50:].ravel()  # [400,700] nm
    
    return adgstar, bbpstar, A, B
    

def non_water_iops_cdomfree(Rrs, rrs, wl):
    
    """
    Estimates non-water input IOP components from Case 1 rrs/Rrs alogorithms. 
    Considers ap model as an input  rather than  aph.
    
    ag rather than adg is therefore the `free' absorption cpt. 
    Hence we call the function `cdom_free''
    
    Input:
        
        Rrs: Above-water spectral reflectance
        rrs: In-water spectral reflectance
        wl: wavelength grid 
  
    Output:
        
        bbpstar: particualte spectral backscattering coeffcient
        A, B: spectral coefficients in ap power-law model
        
    All matrices have dims time/sample no. X wavelength
  
    """

    path_root = str(Path(os.getcwd()).parents[1]) + '/'
    path_data = path_root + 'Data/Input/'

    # acdom is now done within the optimization
  
    # compute bbp slope and bbp as estimated from rrs
    if Rrs.ndim == 2:
        bbp_s = 2.0 * (1. - 1.2 * np.exp(-0.9 * rrs[ :,40] / rrs[:,155]))
        bbpstar = (443/wl[:, np.newaxis])**bbp_s
        bbpstar = bbpstar.T
    elif Rrs.ndim == 1:
        bbp_s = 2.0 * (1. - 1.2 * np.exp(-0.9 * rrs[40] / rrs[155]))
        bbpstar = (443/wl)**bbp_s
        bbpstar = bbpstar.T
    
    # ap model from Chase 2017 
    # ab_coeffs = pd.read_csv('Chase_AB.csv')
    # wl_coeffs = ab_coeffs['Wavelength (nm)'].values
    # A_coeffs = ab_coeffs['Achl (m-1)'].values
    # B_coeffs = ab_coeffs['Bchl (unitless)']
       
    # ap model from AMT
    ab_coeffs = pd.read_csv(path_data + 'Model_params/AB_fits_AMT.csv')
    wl_coeffs = ab_coeffs['wl'].values
    A_coeffs = ab_coeffs['A3'].values
    B_coeffs = ab_coeffs['B3'].values

    wl_out = np.arange(400,701,1)
 
    spline_A = CubicSpline(wl_coeffs, A_coeffs, bc_type="natural")
    A = spline_A(wl_out) 
    # A = savgol_filter(A,window_size, 3)

    spline_B = CubicSpline(wl_coeffs, B_coeffs, bc_type="natural")  # interpolate to c
    B = spline_B(wl_out)
    # B = savgol_filter(B,window_size, 3)
  
    return bbpstar, A, B


def deltaRrs_opt_aph_input(rrs, Rrs, Rrs_nonsmooth, asw, bbsw, adgstar, bbpstar, A, B):
       
    """ Version of optimization using aph (phytoplankton absorption) as model input. 
    
    Input:
   
        rrs: In-water spectral reflectance
        Rrs: Above-water spectral reflectance with spectral smoothing
        Rrs_nonsmooth: Above-water spectral reflectance without spectral smoothing
        asw: salt water spectral absorption coefficient
        bsw: salt water spectral backscatter coefficient
        bbpstar: particualte spectral backscattering coeffcient
        adgstar: combined anap-acdom spectral absorption coeffcieint
        bbp: particualte spectral backscattering coefficient
        A, B: spectral coefficients in aph power-law model
        
        Rrs_nonsmooth is used as input to OC3 algorithm (which is used as inital guess for Chl)
        
    Output:
        
        opt_params =  [chl_opt, chl_oc3,  resid]: optimzation parameters of interest
        rrs_opt: optimized in-water rrs
        Rrs_opt: optimized above-water rrs
        Delta_Rrs = (Rrs - Rrs_opt)
        
    """
    
    chl_opt = [] # list of optimized chl
    chl_oc3 =[] # list oc3 chl
    resid = [] # list of optimzation residuals (sum of Euclidian distance between rrs_opt and rrs)
    rrs_opt = [] # list of optimized spectra
   
    if rrs.ndim == 2:
        Ntot = len(rrs)
    elif rrs.ndim == 1:
        Ntot = 1  
   
    for i in range(Ntot):
        # intialize linear model
        params = lm.Parameters()
        
        # OC3  - use to initalize Chl
        chl_oc3_i = L2chlor_a(Rrs_nonsmooth[i,43], Rrs_nonsmooth[i,89], Rrs_nonsmooth[i,148], Rrs_nonsmooth[i,156], Rrs_nonsmooth[i,268]) 
    
        #(Name,  Value,  Vary,   Min,  Max,  Expr)
        params.add_many(
         ('T_Chl_a',  chl_oc3_i , True, 0.01, 10, None),           
         ('adg_443', 0.01, True, 0.001, 0.05, None), 
         ('bbp_443', 0.0029, True, 0.0001, 0.004 , None), 
         )
       
        # fixed optimization function arguments: asw, bbsw, adgstar, bbpstar, A, B, rrs
        fnc_args = [asw, bbsw[i,:], adgstar[i,:], bbpstar[i,:], A, B, rrs[i,:]]
        fnc_args = tuple(fnc_args)
      
        # optimization routine - calls gsm inversion function
        chl_opt_i,  adg_opt_i, resid_i, rrs_opt_i = gsm_invert(params, fnc_args)
        
        # append outputs
        chl_opt.append(chl_opt_i)
        chl_oc3.append(chl_oc3_i)
        resid.append(resid_i)
        rrs_opt.append(rrs_opt_i)
    
    # stack scalar outputs
    chl_opt = np.array(chl_opt)
    chl_oc3 = np.array(chl_oc3)
    resid = np.array(resid)
    opt_params =  [chl_opt, chl_oc3,  resid]
    
    # stack vector outputs
    rrs_opt = np.array(rrs_opt)
    Rrs_opt = (0.52*rrs_opt)/(1-(1.7*rrs_opt))
    Delta_Rrs = (Rrs - Rrs_opt)
    
    return opt_params, rrs_opt, Rrs_opt, Delta_Rrs


def deltaRrs_opt_ap_input(rrs, Rrs, Rrs_nonsmooth, asw, bbsw, bbpstar, A, B):
       
    ''' Version of optimization using ap (particulate absorption) as model input.
    A, B are now the power-law coefficients for ap power-law. See previous functon
    for inputs/outputs
    '''

    if rrs.ndim == 2:  
       chl_opt = []  # list of optimized chl
       chl_oc3 =[]   # list oc chl -input
       S_opt = []   # list of optimized CDOM slope
       ag_opt = []  # list of ag offset
       resid = []    # list of optimization residuals (sum of Euclidian distance between rrs_opt and rrs)
       rrs_opt = []  # list of optimized spectra
   
       Ntot = len(rrs)
       for i in range(Ntot):
       
           params = lm.Parameters()
           
           # OC3  - use as prior to calculate chl in optimisation
           chl_oc3_i = L2chlor_a(Rrs_nonsmooth[i,43], Rrs_nonsmooth[i,89], Rrs_nonsmooth[i,148], Rrs_nonsmooth[i,156], Rrs_nonsmooth[i,268]) 
             
           #(Name,  Value,  Vary,   Min,  Max,  Expr)
           params.add_many(
               ('T_Chl_a',  chl_oc3_i , True, 0.01, 10, None),           
               ('ag_443', 0.005, True, 0.001, 0.025, None), 
               ('bbp_443', 0.0029, True, 0.0001, 0.004 , None), 
               ('Sg_443', -0.015, True, -0.020 , -0.012, None) # range based on Dall Olmo 2017
               
           )
         
           # fixed optimization function arguments: asw, bbsw, adgstar, bbpstar, A, B, rrs
           fnc_args = [asw, bbsw[i,:], bbpstar[i,:], A, B, rrs[i,:]]
           fnc_args = tuple(fnc_args)
        
           # opt
           chl_opt_i, S_opt_i, ag_opt_i, resid_i, rrs_opt_i = gsm_invert_cdomfree(params, fnc_args)
     
           # append outputs
           chl_opt.append(chl_opt_i)
           chl_oc3.append(chl_oc3_i)
           S_opt.append(S_opt_i)
           ag_opt.append(ag_opt_i)
           resid.append(resid_i)
           rrs_opt.append(rrs_opt_i)
       
       # stack scalar outputs
       chl_opt = np.array(chl_opt)
       chl_oc3 = np.array(chl_oc3)
       resid = np.array(resid)
       S_opt = np.array(S_opt)
       ag_opt = np.array(ag_opt)
       opt_params =  [chl_opt, chl_oc3, resid, S_opt,ag_opt]
       
       # stack vector outputs
       rrs_opt = np.array(rrs_opt)
       Rrs_opt = (0.52*rrs_opt)/(1-(1.7*rrs_opt))
       Delta_Rrs = (Rrs - Rrs_opt)
       
    elif rrs.ndim ==1:
           
       params = lm.Parameters()
        
       # OC3  - use to calculate chl
       chl_oc3 = L2chlor_a(Rrs_nonsmooth[43], Rrs_nonsmooth[89], Rrs_nonsmooth[148], Rrs_nonsmooth[156], Rrs_nonsmooth[268]) 
          
       #(Name,  Value,  Vary,   Min,  Max,  Expr)
       params.add_many(
            ('T_Chl_a',  chl_oc3_i , True, 0.01, 10, None),           
            ('ag_443', 0.005, True, 0.001, 0.025, None), 
            ('bbp_443', 0.0029, True, 0.0001, 0.004 , None), 
            ('Sg_443', -0.015, True, -0.020 , -0.012, None) # range broadly based on Dall Olmo 2017
        )
      
       # fixed optimization function arguments: asw, bbsw, adgstar, bbpstar, A, B, rrs
       fnc_args = [asw, bbsw, bbpstar, A, B, rrs]
       fnc_args = tuple(fnc_args)
     
       # gsm optimization
       chl_opt, S_opt, ag_opt, resid, rrs_opt,  = gsm_invert_cdomfree(params, fnc_args)
  
       # stack vector outputs
       rrs_opt = np.array(rrs_opt)
       Rrs_opt = (0.52*rrs_opt)/(1-(1.7*rrs_opt)).ravel()
       Delta_Rrs = (Rrs - Rrs_opt).ravel()
       opt_params =  [chl_opt, chl_oc3, resid, S_opt, ag_opt]

    return opt_params, rrs_opt, Rrs_opt, Delta_Rrs


def plot_output_rrs():

    plt.figure(figsize=(25,16))
    
    plt.rcParams.update({'font.size': 18})
    plt.subplot(3,1,1)
    plt.plot(wl,Rrs.T)
    plt.ylabel('Rrs (observed) [sr$^{-1}$]')
    plt.xlabel('Wavelength [nm]')
    
    plt.subplot(3,1,2)
    plt.plot(wl,Rrs_opt.T)
    plt.ylabel('Rrs (model-optimised) [sr$^{-1}$]')
    plt.xlabel('Wavelength [nm]')

    plt.subplot(3,1,3)
    plt.plot(wl,Rrs.T- Rrs_opt.T)
    plt.ylabel('$\delta$ Rrs [sr$^{-1}$]')
    plt.xlabel('Wavelength [nm]')
    
    return

def plot_chl_corr():
        
    plt.figure(figsize=(9,9))
    mod = scipy.stats.linregress(np.log10(chl_hplc),np.log10(chl_opt)) #- no                                            
    r_sq = np.round(1000*mod.rvalue**2)/1000
    x = np.arange(0.01,3,0.001)
    y = 10**(np.log10(x)*mod.slope + mod.intercept)

    plt.title('R_squared = ' + str(r_sq))
    plt.rcParams.update({'font.size': 18})
    plt.plot(x,x,color='red',linestyle='dashed',label='1:1')
    plt.plot(x,y,color='blue',linestyle='dotted',label ='Linear model')
    plt.scatter(chl_hplc, chl_opt)
    plt.xlim(0.08,8)
    plt.ylim(0.08,8)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    # plt.grid('on', ls='--') 
    plt.xlabel('Chl HPLC [mg m$^{-3}]$')
    plt.ylabel('Chl optimized [mg m$^{-3}]$')
    plt.legend(fontsize=14)
    plt.axis('equal')
    
    return

if __name__ == '__main__':

    #####################################
    # Assign paths
    #####################################

    # local path
    path_root = str(Path(os.getcwd()).parents[1]) + '/'
    path_code =  os.getcwd()
    
    # input/output data
    path_data = path_root + 'Data/Input/'
    path_output = path_root + 'Data/Output/'    
    path_figs = path_root + 'Figs/'

    # select modes of bio-optical model 
    mode = 'aph_input' # aph power-law input as in Kramer 22
    #mode = 'ap_input' # using ap power-law from AMT 
    
    #######################################
    # Load co-located AMT HPLC/Rrs data    
    #######################################
    
    data = pd.read_csv(path_data + '/AMT_Rrs_HPLC.csv')[:210]
    Rrs = np.array(data.iloc[:,-301:].values) # above-water Rrs
    wl = np.arange(400,701,1) # wavelength range
    chl_hplc = data['Tchla'].values # total chl-a
    T = data['Temp'].values # temperature
    S = data['Sal'].values # salinity
 
    # smoothing
    poly_order = 3 # 3rd order polynomial as default
    window_size = 20 # 20 nm kernel as default - see optimization results in paper
    Rrs_nonsmooth = Rrs # make a copy of `non-smoothed' Rrs
    Rrs = savgol_filter(Rrs, window_size, poly_order,  axis=1)
 
    # Convert Rrs to rrs - uses Lee 2002 parameterisation
    rrs = Rrs / (0.52 + 1.7 * Rrs)

    ###############################################################
    # Configure IOP inputs, and perform optimizations for delta_Rrs
    ##############################################################
 
    # salt water IOPs used in model optimization
    asw, bbsw = salt_water_iops(T, S, wl)
   
    if mode == 'aph_input':
       # non water IOPs used in model optimization for aph input as in Kramer 2022
       adgstar, bbpstar, A, B = non_water_iops(Rrs, rrs, wl)
     
       # delta_Rrs optimization
       opt_params, rrs_opt, Rrs_opt, Delta_Rrs = deltaRrs_opt_aph_input(rrs, Rrs, Rrs_nonsmooth, asw, bbsw, adgstar, bbpstar, A, B)
       chl_opt, chl_oc3, resid = opt_params[0], opt_params[1], opt_params[2]
   
    elif mode == 'ap_input':
       # non water IOPs used in model optimization for ap input
       # Includes  A,B from ap powr-law model. CDOM is now solved for in optimization
       bbpstar, A, B = non_water_iops_cdomfree(Rrs, rrs, wl)
       
       # delta_Rrs optimization
       opt_params, rrs_opt, Rrs_opt, Delta_Rrs = deltaRrs_opt_ap_input(rrs, Rrs, Rrs_nonsmooth, asw, bbsw, bbpstar,A, B)
       chl_opt, chl_oc3, resid, S_opt, ag_opt = opt_params[0], opt_params[1], opt_params[2], opt_params[3], opt_params[4]   

    #########################################
    # Example plots and output files
    #########################################
    
    plot_output_rrs()
    plot_chl_corr()

    # output files
    df_Delta_Rrs= pd.DataFrame(Delta_Rrs,columns =wl) # delta Rrs
    fname = path_output  + '/Delta_Rrs_AMT_' + mode +'.csv'
    df_Delta_Rrs.to_csv(fname,index=False)
           
    df_Rrs_opt = pd.DataFrame(Rrs_opt,columns =wl) # optimzed Rrs
    fname = path_output  + '/Rrs_opt_AMT_' + mode + '.csv'
    df_Rrs_opt.to_csv(fname,index=False)
       
    df_OC_CHL = pd.DataFrame(chl_oc3)  # CHL from OC3
    fname = path_output  + '/chl_oc3.csv'
    df_OC_CHL.to_csv(fname,index=False)
   
    df_opt_aph_CHL = pd.DataFrame(chl_opt) # Optimized Chl
    fname = path_output  + '/chl_opt_' + str(mode) + '.csv'
    df_opt_aph_CHL.to_csv(fname,index=False)
   

