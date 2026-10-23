#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a basic workflow to model pigment concentrations along a continuous cruise transect

Transect plots are produced from AMT30 as an example

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler

import sys 
sys.path.append('../SDP-log')
sys.path.append('../Data/Output/')
import glob

from SDP_hyperRrs import salt_water_iops
from SDP_hyperRrs import non_water_iops
from SDP_hyperRrs import non_water_iops_cdomfree
from SDP_hyperRrs import deltaRrs_opt_aph_input
from SDP_hyperRrs import deltaRrs_opt_ap_input
from SDP_hyperRrs import wl_interp


from scipy.signal import savgol_filter 
import SB_support as sbs
import datetime

def calc_pigments(d2D_Rrs, alpha_mean, beta_mean, pig_mdl):   
    ''' This derives pigments from trained SDP coefficients'''    
    
    scaler = StandardScaler()
    Dspec_norm = scaler.fit_transform(d2D_Rrs)
    
    Cpig = []
    for i in range(len(pig_mdl)):
       Cpig_i = 10**(np.matmul(Dspec_norm, beta_mean[i,:]) + alpha_mean[i])
       Cpig.append(Cpig_i)
    Cpig = np.stack(Cpig)
    Cpig = Cpig.T
 
    return Cpig 

def load_seabass(file_path):
     ' Loads Rrs seabass file as a dataframe'
  
     data = sbs.readSB(file_path)
  
     # Convert data dictionary to DataFrame
     df = pd.DataFrame(data.data,columns=data.data.keys()) 
    
     # index by TS
     timestamp = []
     for i in range(0, len(df)):
         date_i = str(df['date'].iloc[i])[0:4] + '-' + str(df['date'].iloc[i])[4:6]   + '-' + str(df['date'].iloc[i])[6:8]
         time_i = str(df['time'].iloc[i])
         timestamp.append(datetime.datetime.strptime(date_i + ':'+  time_i, '%Y-%m-%d:%H:%M:%S'))
     timestamp = np.array(timestamp,dtype='datetime64[ns]')
     df['timestamp'] = timestamp
     df = df.set_index('timestamp')
     
     
     cruise_i = file_path.split('/')[7]
     df['time']=cruise_i
     df =df.rename(columns={"time": "cruise"})

     return df
 
    
def load_seabass_hplc(file_path):
     ' Loads hplc seabass file as a dataframe'

     data = sbs.readSB(file_path)
  
     # Convert data dictionary to DataFrame
     df = pd.DataFrame(data.data, columns = data.data.keys()) 
     #  breakpoint()
     # index by TS
     timestamp = []
     for i in range(0, len(df)):
         date_i = str(df['year'].iloc[i]) + '-' + str(df['month'].iloc[i]) + '-' + str(df['day'].iloc[i])
         time_i = str(df['time'].iloc[i])
         timestamp.append(datetime.datetime.strptime(date_i + ':'+  time_i, '%Y-%m-%d:%H:%M:%S'))
     timestamp = np.array(timestamp,dtype='datetime64[ns]')
     df['timestamp'] = timestamp
     df = df.set_index('timestamp')
     
     cruise_i = file_path.split('/')[7]
     df['time']=cruise_i
     df = df.rename(columns={"time": "cruise"})

     return df

def wl_interp(spec, spec_wl, wl_out, method ='pw_lin'):
   
    """
    Performs wavelength interpolation on 2D (dims: time X wavelength) spectra
    matrices - cubic or piecewise linear are give as options
      
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

def get_wlpython(data_python):
    
   wl_python = []
   wl_keys = data_python.keys()[10:146]
   for i in range(len(wl_keys)):
     wl_python.append(float(wl_keys[i][3:]))
   wl_python = np.stack(wl_python)
   
   return wl_python


if __name__ == '__main__':
    
   #
  # path_root = '/mnt/d/Phytoplankton_algorithms/Hyperspectral_pigments/'
  
   path_rrs= '/mnt/d/Phytoplankton_algorithms/Hyperspectral_pigments/Data/SeaBASS_Output/SeaBASS_Output/'
 
   path_root = str(Path(os.getcwd()).parents[1]) + '/'
   path_code =  os.getcwd()
   path_data = path_root + 'Data/'    
   path_pigments =  path_root + '/Data/Output/'
   sys.path.append(path_data)
   mode = 'aph_input'
  
    # AMT_ full_cruise    
   rrs_files = sorted(glob.glob(path_rrs +'/AMT*/*_HSASRrs_n*'))
   
   data = load_seabass(rrs_files[-2]) # AMT 30
   meta = data.iloc[:,:10]
   wl = np.arange(400,701,1)
   Rrs = np.array(data.iloc[:,10:146])
   wl_data = get_wlpython(data)
   Rrs = wl_interp(Rrs, wl_data, wl, 'pw_lin')  
   
   T = meta['sal'].values
   S = meta['bincount'].values 
   lat = meta['lat'].values
    
   
   # Wavelength interpolation/smoothing of Rrs
   poly_order = 3 # 3rd order polynomial as default
   window_size = 20 # 
   Rrs_nonsmooth = Rrs # make a copy of non-smoothe Rrs (used in Chl algorithm)
   Rrs = savgol_filter(Rrs, window_size, poly_order,  axis=1)
   
   # Convert Rrs to rrs - uses Lee 2002 parameterisation
   rrs = Rrs / (0.52 + 1.7 * Rrs)
      
   # Delta_Rrs optimization

   # salt water IOPs used in model optimization
   asw, bbsw = salt_water_iops(T, S, wl)
    
   if mode == 'aph_input':
        # non water IOPs used in model optimization for aph input aqs in Kramer 2022
        # Includes adg, A, B from aph powr-law model
        adgstar, bbpstar, A, B = non_water_iops(Rrs, rrs, wl)
      
        # delta_Rrs optimization
        opt_params, rrs_opt, Rrs_opt, Delta_Rrs = deltaRrs_opt_aph_input(rrs, Rrs, Rrs_nonsmooth, asw, bbsw, adgstar, bbpstar, A, B)
        chl_opt, chl_oc3, resid = opt_params[0], opt_params[1], opt_params[2]
    
   elif mode == 'ap_input':
        # non water IOPs used in model optimization for aph input aqs in Kramer 2022
        # Includes  A,B from ap powr-law model. No CDOM as this is now solved for in optimization
        bbpstar, A, B = non_water_iops_cdomfree(Rrs, rrs, wl)
       
        # delta_Rrs optimization
      
        opt_params, rrs_opt, Rrs_opt, Delta_Rrs = deltaRrs_opt_ap_input(rrs, Rrs, Rrs_nonsmooth,asw, bbsw, bbpstar, A, B)
        chl_opt, chl_oc3, resid, S_opt = opt_params[0], opt_params[1], opt_params[2], opt_params[3]    


   dD_Rrs =  pd.DataFrame(np.gradient(np.array(Delta_Rrs), axis=1),columns = wl.astype(str)) # 1st order derivative using second order accurate central differences
   d2D_Rrs = pd.DataFrame(np.gradient(np.array(dD_Rrs), axis=1),columns = wl.astype(str)) # end order derivative using second order accurate central differences
  
   pig_mdl= ['Tchla','Tchlb','ABcaro','But',
                    'Hex', 'Allo', 'Diadino','Diato',
                    'Fuco', 'Perid','Zea', 'DVchla',
                    'Chlc12','Chlc3', 'Neo', 'Viola']
           
   pig_mdl_SB = ['tot_chl_a','tot_chl_b', 'alpha-beta-car' , 'but-fuco',
                 'hex-fuco', 'allo', 'diadino', 'diato',
                 'fuco', 'perid', 'zea' , 'dv_chl_a',
                 'chl_c1c2','chl_c3', 'neo', 'viola']
               

   # calculated pigments
   beta_mean = np.load(path_pigments + 'beta_mean_aph_log.npy')
   alpha_mean = np.load(path_pigments + 'alpha_mean_aph_log.npy')
   Cpig = calc_pigments(d2D_Rrs, alpha_mean, beta_mean, pig_mdl)

   # load HPLC (all data)
   path_hplc = '/mnt/d/Phytoplankton_algorithms/Hyperspectral_pigments/Data/SeaBASS_Rrs'
   sb_files = glob.glob(path_hplc +'/AMT*/*HPLC*.sb*')
   df_hplc_list = [load_seabass_hplc(file) for file in   sb_files]
   df_hplc = df_hplc_list[-2]# AMT 30
   df_hplc[df_hplc['depth']=='underway'] = 5
   df_hplc = df_hplc[df_hplc.depth < 10]
   
   # load HPLC (match-ups)
   df_hplc_m = pd.read_csv(path_data + 'Input/AMT_Rrs_HPLC.csv')[0:210]
   times = []
   for i in range(0, len(df_hplc_m)):
       str_i = str(df_hplc_m['DateTime'].iloc[i])
       str_i = str(df_hplc_m['DateTime'].iloc[i])
       times.append(datetime.datetime.strptime(str_i, '%m/%d/%Y %H:%M'))
   times = np.array(times,dtype='datetime64[ns]')
   df_hplc_m['times'] = times
   df_hplc_m = df_hplc_m.set_index('times')
   df_hplc_m = df_hplc_m[df_hplc_m['Cruise']=='AMT30']
   
 
   
   fig = plt.figure(figsize=(16,18))
   
   pig_mdl= ['Tchla','Tchlb','ABcaro','But',
                    'Hex', 'Allo', 'Diadino','Diato',
                    'Fuco', 'Perid','Zea', 'DVchla',
                    'Chlc12','Chlc3', 'Neo', 'Viola']
   
   # Fucoxanthin (Diatoms)Peridinin (Dinoflagellates)Nanophytoplankton Markers:19'-hexanoyloxyfucoxanthin (Haptophytes / Prymnesiophytes)19'-butanoyloxyfucoxanthin (Chrysophytes)Alloxanthin (Cryptophytes)Picophytoplankton Markers:Zeaxanthin (Cyanobacteria)Total Chlorophyll b (Chlorophytes) 
   fig = plt.figure(figsize=(22,18))
   index=[0,1,3,4,8,10,5,9]
   for i in range(0,8):
       #if pig_mdl[i] in  pig_mdl_select:
           k = index[i]
           plt.subplot(4,2,i+1)
           plt.title(str(pig_mdl[k]))
           plt.rcParams.update({'font.size': 21})    
         
           plt.scatter(lat, Cpig[:,k],alpha=0.3,s=15,color='tab:blue') 
           plt.scatter(lat,np.ones(len(Cpig[:,k]))*10, label='Rrs algorithm',s=50,color='tab:blue') 
           plt.scatter(df_hplc['lat'],df_hplc[pig_mdl_SB[k]],label='HPLC: not in training',s=50,color='red',edgecolors='black')
           plt.scatter(df_hplc_m['Lat'],df_hplc_m[pig_mdl[k]],label='HPLC: in training',s=50,color='yellow',edgecolors='black')
           plt.plot(np.arange(-60,60,1),0.001*np.ones(120),label='Detection limit',color='grey',linestyle='dashed') 
           plt.yscale('log')
           if i<6:
               plt.ylim(0.0005,1)
             
           else:
              plt.ylim(0.0001,0.05)
           if i == 0: 
             plt.legend(fontsize=18)     
           plt.xlim(-50,50)
   fig.supylabel('Pigment concentration [mg m$^{-3}$]')
   fig.supxlabel('Latitude [deg]') 
   plt.tight_layout()
   
   filename  = '/mnt/d/Phytoplankton_algorithms/Hyperspectral_pigments/Figs/Transect1.pdf'
   plt.savefig(filename,dpi=600,bbox_inches='tight')
   fig = plt.figure(figsize=(22,18))
   #index=[1,3,4,5,8,9,10,11]
   index=[2,6,7,11,12,13,14,15]
   for i in range(0,8):
       #if pig_mdl[i] in  pig_mdl_select:
           k = index[i]
           plt.subplot(4,2,i+1)
           plt.title(str(pig_mdl[k]))
           plt.rcParams.update({'font.size': 21})    
         
           plt.scatter(lat, Cpig[:,k],alpha=0.3,s=15,color='tab:blue') 
           plt.scatter(lat,np.ones(len(Cpig[:,k]))*10, label='Rrs algorithm',s=50,color='tab:blue') 
           plt.scatter(df_hplc['lat'],df_hplc[pig_mdl_SB[k]],label='HPLC: not in training',s=50,color='red',edgecolors='black')
           plt.scatter(df_hplc_m['Lat'],df_hplc_m[pig_mdl[k]],label='HPLC: in training',s=50,color='yellow',edgecolors='black')
           plt.plot(np.arange(-60,60,1),0.001*np.ones(120),label='Detection limit',color='grey',linestyle='dashed') 
           plt.yscale('log')
           if i<6:
               plt.ylim(0.0005,1)
           else:
              plt.ylim(0.0001,0.05)
           if i == 0: 
             plt.legend(fontsize=18)     
           plt.xlim(-50,50)
   fig.supylabel('Pigment concentration [mg m$^{-3}$]')
   fig.supxlabel('Latitude [deg]') 
   plt.tight_layout()
     
   filename  = '/mnt/d/Phytoplankton_algorithms/Hyperspectral_pigments/Figs/Transect2.pdf'
   plt.savefig(filename,dpi=600,bbox_inches='tight')

   # Plot for 3 pigments
   plt.figure()
   index=[4,8,5]
   plt.rcParams.update({'font.size': 30})    
   for i in range(0,3):
        #if pig_mdl[i] in  pig_mdl_select:
            k = index[i]
            plt.subplot(3,1,i+1)
            plt.title(str(pig_mdl[k]))
      
          
            plt.scatter(lat, Cpig[:,k],alpha=0.3,s=15,color='tab:blue') 
            plt.scatter(lat,np.ones(len(Cpig[:,k]))*10, label='Rrs algorithm',s=50,color='tab:blue') 
            plt.scatter(df_hplc['lat'],df_hplc[pig_mdl_SB[k]],label='HPLC: not in training',s=50,color='red',edgecolors='black')
            plt.scatter(df_hplc_m['Lat'],df_hplc_m[pig_mdl[k]],label='HPLC: in training',s=50,color='yellow',edgecolors='black')
            plt.plot(np.arange(-60,60,1),0.001*np.ones(120),label='Detection limit',color='grey',linestyle='dashed') 
            plt.yscale('log')
            if i<2:
                plt.ylim(0.0005,1)
              
            else:
               plt.ylim(0.0001,0.05)
            if i == 2: 
              plt.legend(fontsize=28,   loc=9)     
            plt.xlim(-50,50)
   fig.supylabel('Pigment concentration [mg m$^{-3}$]')
   fig.supxlabel('Latitude [deg]') 
 
   plt.tight_layout()
     
   filename  = '/mnt/d/Phytoplankton_algorithms/Hyperspectral_pigments/Figs/Transect_highlight.png'
   plt.savefig(filename,dpi=900,bbox_inches='tight')

  