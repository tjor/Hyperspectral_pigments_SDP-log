#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional plots for AMT Rrs/SDP paper

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy

import scipy
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler


def plot_aph_models():
    '''
     A and B parameters in aphy/aph models. Also includes Chase 2017
     This plot is not in paper.
    ''' 
    
    # Case 1: aph Kramer 2022
    ab_coeffs = scipy.io.loadmat(path_data +'Model_params/AB_coeffs.mat')
    A1 = ab_coeffs['A'][50:].ravel()  # [400,700] nm
    B1 = ab_coeffs['B'][50:].ravel()  # [400,700] nm
    
   
    # Case 2: ap Chase 2017 - not used in paper
    ab_coeffs = pd.read_csv(path_data +'Model_params/Chase_AB.csv')
    wl_coeffs = ab_coeffs['Wavelength (nm)'].values
    A_coeffs =  ab_coeffs['Achl (m-1)'].values
    B_coeffs =  ab_coeffs['Bchl (unitless)']

    wl_out = np.arange(400,701,1)
 
    #uses cubic spline for interpolation as in Kramer et al. 2022
    spline_A = CubicSpline(wl_coeffs, A_coeffs, bc_type="natural")  # interpolate to c
    A2 = spline_A(wl_out)

    spline_B = CubicSpline(wl_coeffs, B_coeffs, bc_type="natural")  # interpolate to c
    B2 = spline_B(wl_out)
    
   
    # Case 3: ap AMT
    ab_coeffs = pd.read_csv(path_data +'Model_params/AB_fits_AMT.csv')
    wl_coeffs = ab_coeffs['wl'].values
    A3_coeffs = ab_coeffs['A3'].values
    B3_coeffs = ab_coeffs['B3'].values

    wl_out = np.arange(400,701,1)
    
    #uses cubic spline for interpolation as in Kramer et al. 2022
    spline_A = CubicSpline(wl_coeffs, A3_coeffs, bc_type="natural")  # interpolate to c
    A3 = spline_A(wl_out)

    spline_B = CubicSpline(wl_coeffs, B3_coeffs, bc_type="natural")  # interpolate to c
    B3 = spline_B(wl_out)
     
    plt.figure(figsize=(16,16))
    plt.rcParams.update({'font.size': 18})
    
    plt.subplot(2,2,1)
    plt.title('CHL = ' + str(1) + ' mg m$^{-3}$')
    plt.plot(wl_out, A1*1**B1, label='KR22: aph')
    plt.plot(wl_out, A2*1**B2,label='CH17: ap')
    plt.plot(wl_out, A3*1**B3,label='AMT: ap')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('aph or ap (CHL prediction) [m$^{-1}$]')
    plt.legend()
    
    plt.subplot(2,2,2)
    plt.title('CHL = ' + str(0.5) + ' mg m$^{-3}$')
    plt.plot(wl_out, A1*0.5**B1)
    plt.plot(wl_out, A2*0.5**B2)
    plt.plot(wl_out, A3*0.5**B3)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('aph or ap (CHL prediction) [m$^{-1}$]')
    
    plt.subplot(2,2,3)
    plt.title('CHL = ' + str(0.1) + ' mg m$^{-3}$')
    plt.plot(wl_out, A1*0.1**B1)
    plt.plot(wl_out, A2*0.1**B2)
    plt.plot(wl_out, A3*0.1**B3)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('aph or ap (CHL prediction) [m$^{-1}$]')
    
    plt.subplot(2,2,4)
    plt.title('CHL = ' + str(0.05) + ' mg m$^{-3}$')
    plt.plot(wl_out, A1*0.05**B1)
    plt.plot(wl_out, A2*0.05**B2)
    plt.plot(wl_out, A3*0.05**B3)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('aph or ap (CHL prediction) [m$^{-1}$]')
    
    plt.tight_layout()
    
    return A1, B1, A3, B3

def plot_CHL_matchesmap():
    '''
     Plots map of CHL match-ups for AMT
    ''' 
      
    plt.figure(figsize=(15,10))
    extent = [-65,10,-60, 60]
    #  request = cimgt.GoogleTiles(style='satellite')
    #request = cimgt.OSM()
      
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.add_image(request, 7)
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.set_extent(extent, ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    #  gl.top_labels_top = gl.right_labels = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter =  LONGITUDE_FORMATTER
    gl.yformatter =  LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14,  'rotation':0}
    gl.ylabel_style = {'size': 14,  'rotation': 0}

    from cartopy.io.shapereader import Reader
    from cartopy.feature import ShapelyFeature
    
    fname = path_data + '/longhurst_v4_2010/Longhurst_world_v4_2010.shp'
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(),facecolor='none',edgecolor='grey',linestyle='--')
    ax.add_feature(shape_feature)
     
    
    plt.text(.37, .92,   'NADR', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.12, .80,   'NASW', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.59, .78,   'NASE', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.20, .65,   'NATR', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.70, .49,   'ETRA', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.28, .53,   'WTRA', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.39, .33,   'SATL', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.48, .15,   'SSTC', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)
    plt.text(.45, .09,   'SANT', ha='left', va='top', color='grey', transform=ax.transAxes,fontsize=16)    
      
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    
    sc = ax.scatter(data['Lon'],data['Lat'],c=chl_hplc, s=25,norm="log",cmap="rainbow", linewidth=0.5, edgecolor='k', vmin=0.01,vmax=1)
    cbar = plt.colorbar(sc,extend='both', cax=ax.inset_axes((0.70, 0.03, 0.05, 0.4)))
    cbar.minorticks_on()
    cbar.set_label('Tchla [mg m$^{-3}$]',fontsize=13)
    cbar.ax.tick_params(labelsize=13) 
     
    filename  =  path_figs + '/'  + 'CHLmatches_map.png'
    plt.savefig(filename,dpi=900)   
   
    return

def plot_pig_dists():
    
    '''
     Plots pigments
    ''' 
      
        
    bar_colors =[]
    for i in range(16):
        bar_colors.append(plt.cm.tab20c(i))
       
    pigkeys = ['Tchla','Tchlb','ABcaro','But',
                    'Hex', 'Allo', 'Diadino','Diato',
                    'Fuco', 'Perid','Zea', 'DVchla',
                    'Chlc12','Chlc3', 'Neo', 'Viola']
           

    fig= plt.figure(figsize=(16,16))
    plt.rc('font', size=18)
    for i in range(len(pigkeys)):
            plt.subplot(4,4,i+1)
            data_i = data[pigkeys[i]].values[data[pigkeys[i]].values>0]
            data_i =   data_i[data_i>=0.001]
            N_i = np.sum(data_i >= 0)
            
            plt.title(str(pigkeys[i]))
        
            bins = np.arange(0.001,5,0.25)
            hist, bins = np.histogram(data_i, bins=bins)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            plt.hist(data_i, bins=logbins,color=bar_colors[i],edgecolor='black',label='N = ' + str(len(data_i)))
            plt.xscale('log')
            # plt.ylabel('Frequency')
            # plt.xlabel('Concentration  [mg m$^{-3}]$')
            plt.xlim(0.0006,4)
            label= 'N = ' + str(len(data_i))
            plt.legend()
    fig.supylabel('Frequency')
    fig.supxlabel('Pigment concentration [mg m$^{-3}]$') 
    plt.tight_layout() 
    
    plt.tight_layout()
    filename  =  path_figs + '/'  + 'Pig_concs.pdf'
    plt.savefig(filename,dpi=600)   
    
        
        
def plot_gsm_input_results():
    
     '''
      Plots bio-optical optimization results
     ''' 
    
     wl = np.arange(400,701,1)
     chl_opt_aph = pd.read_csv(path_output  + '/chl_opt_' + 'aph_input' + '.csv').values.ravel()
     chl_opt_ap = pd.read_csv(path_output  + '/chl_opt_' + 'ap_input' + '.csv').values.ravel()
 
     plt.figure(figsize=(16,12))
     plt.rcParams.update({'font.size': 20})
    
     plt.subplot(2,2,1)
     plt.text(.05, .95,  'A', ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)    
     plt.plot(wl, A1, label='KR22: $A_{chl}^{ph}(\lambda)$',color='tab:blue')
     plt.plot(wl, A3,label='AMT: $A_{chl}^{p}(\lambda)$',linestyle='dashed',color='tab:green')
     plt.xlabel('Wavelength [nm]')
     plt.ylabel('$A_{chl}^{ph}(\lambda)$ or $A_{chl}^{p}(\lambda)$  [m$^{-1}$]')
     plt.xlim(400,700)    
     plt.ylim(0,0.06)
     plt.legend()
    
     plt.subplot(2,2,2)
     plt.text(.05, .95,  'B', ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)  
     plt.plot(wl, B1, label='KR22: $B_{chl}^{ph}(\lambda)$',color='tab:blue')
     plt.plot(wl, B3,label='AMT: $B_{chl}^{p}(\lambda)$',linestyle='dashed',color='tab:green')
     plt.xlabel('Wavelength [nm]')
     plt.ylabel('$B_{chl}^{ph}(\lambda)$ or $B_{chl}^{p}(\lambda)$')
     plt.xlim(400,700)
     plt.ylim(0.5,1.2)
     plt.legend()
    
 
     plt.subplot(2,2,3)
     plt.text(.05, .95,  'C', ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)  
       
     mod = scipy.stats.linregress(np.log10(chl_hplc),np.log10( chl_opt_aph)) #- no                                            
     r_sq = np.round(1000*mod.rvalue**2)/1000
     x = np.arange(0.01,3,0.001)
     y = 10**(np.log10(x)*mod.slope + mod.intercept)
     
     mod_lin = scipy.stats.linregress(chl_hplc,chl_opt_aph) # no                                            
     r_sq_lin = np.round(1000*mod_lin.rvalue**2)/1000
    
     mape = 100*np.nanmean(abs(chl_opt_aph - chl_hplc)/chl_hplc)
     mpe = 100*np.nanmean((chl_opt_aph -chl_hplc)/(chl_hplc))
    
     # medape = 100*np.nanmedian(abs(chl_opt_aph - chl_hplc)/chl_hplc)
     # medpe = 100*np.nanmedian((chl_opt_aph -chl_hplc)/(chl_hplc))
    
     label_metrics_aph = ['N = '  + str(len(chl_hplc)) + '\n' +
             'rsq_log = ' + str(r_sq)[0:5] + '\n'
             'rsq_lin = ' + str(r_sq_lin)[0:5] + '\n'
             'MAPE = '    + str(mape)[0:5] + ' %' +'\n'
             'MPE = '    + str(mpe)[0:5] + ' %' +'\n'
              # 'MedAPE = '    + str(medape)[0:5] + ' %' +'\n'
              # 'MedPE = '    + str(medpe)[0:5] + ' %'
             ][0]
    
     plt.plot(x,x,color='grey',linestyle='dotted')
     plt.plot(x,y,color='black',linestyle='dashed')
     
     plt.scatter(chl_hplc, chl_opt_aph,alpha=0.3,label = label_metrics_aph,color='tab:blue')
  
     plt.gca().set_yscale('log')
     plt.gca().set_xscale('log')
     plt.xlabel('HPLC Tchla [mg m$^{-3}]$')
     plt.ylabel('Model-optimised Tchla: \n $a_{p}(\lambda)$ input [mg m$^{-3}]$')
     plt.legend(fontsize=16)
     plt.axis('equal')
    
 
     plt.subplot(2,2,4)
     plt.text(.05, .95,  'D', ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)  
       
     mod = scipy.stats.linregress(np.log10(chl_hplc),np.log10( chl_opt_ap)) #- no                                            
     r_sq = np.round(1000*mod.rvalue**2)/1000
     x = np.arange(0.01,3,0.001)
     y = 10**(np.log10(x)*mod.slope + mod.intercept)
    
     mod_lin = scipy.stats.linregress(chl_hplc,chl_opt_ap) # no                                            
     r_sq_lin = np.round(1000*mod_lin.rvalue**2)/1000
    
     mape = 100*np.nanmean(abs(chl_opt_ap - chl_hplc)/chl_hplc)
     mpe = 100*np.nanmean((chl_opt_ap -chl_hplc)/(chl_hplc))
    
     #medape = 100*np.nanmedian(abs(chl_opt_ap - chl_hplc)/chl_hplc)
     #medpe = 100*np.nanmedian((chl_opt_ap -chl_hplc)/(chl_hplc))
    
     label_metrics = ['N = '  + str(len(chl_hplc)) + '\n' +
             'rsq_log = ' + str(r_sq)[0:5] + '\n'
             'rsq_lin = ' + str(r_sq_lin)[0:5] + '\n'
             'MAPE = '    + str(mape)[0:5] + ' %' +'\n'
             'MPE = '    + str(mpe)[0:5] + ' %' +'\n'
              # 'MedAPE = '    + str(medape)[0:5] + ' %' +'\n'
              # 'MedPE = '    + str(medpe)[0:5] + ' %'
             ][0]
    
     plt.plot(x,x,color='grey',linestyle='dotted')
     plt.plot(x,y,color='black',linestyle='dashed')
     plt.scatter(chl_hplc, chl_opt_ap,alpha=0.3,label = label_metrics,color='tab:green')
     plt.xlim(0.01,1)
     plt.ylim(0.01,1)
     plt.gca().set_yscale('log')
     plt.gca().set_xscale('log')
     # plt.grid('on', ls='--')  
     plt.xlabel('HPLC Tchla [mg m$^{-3}]$')
     plt.ylabel('Model-optimised Tchla: \n $a_{p}(\lambda)$ input [mg m$^{-3}]$')
     plt.legend(fontsize=16)
     plt.axis('equal')
     
     plt.tight_layout()
     filename  =  path_figs + '/'  + 'GSM_absmodesl.png'
     plt.savefig(filename,dpi=900)   
     
     
     
def plot_delta_Rrs():
    
    '''
      Compares delta_Rrs for aph and ap inputs
    ''' 

    D_Rrs_aph = np.array(pd.read_csv(path_output + '/Delta_Rrs_AMT_aph_input.csv'))  
    D_Rrs_ap = np.array(pd.read_csv(path_output + '/Delta_Rrs_AMT_ap_input.csv'))  
     
    wl =np.arange(400,701,1)
    colors = cm.rainbow(np.linspace(0, 1,101))
    color_index = np.round(100*(np.log10(chl_hplc) +2)/2).astype(int)
    color_index[color_index > 100] = 100
    order_i =  (color_index).argsort()
    color_index = color_index[order_i]    
      
    data = pd.read_csv(path_data + '/AMT_Rrs_HPLC.csv')[:210]
    Rrs = np.array(data.iloc[:,-301:].values) #
    Rrs = Rrs[order_i]
    D_Rrs_aph = D_Rrs_aph[order_i]
    D_Rrs_ap = D_Rrs_ap[order_i]
    
    
    plt.figure(figsize=(18,22))
    plt.rcParams.update({'font.size': 30})
    plt.subplot(4,1,1)
    plt.text(.95, .95,  'B', ha='left', va='top', transform=plt.gca().transAxes,fontsize=30)  
    for i in range(len(Rrs)):
      plt.plot(wl, Rrs[i,:], color=colors[color_index[i]],linewidth=1, alpha=0.4)
    #  plt.xlabel('Wavelength [nm]')
    plt.ylabel('Measured $R_{rs}(\lambda)$ [sr$^{-1}$] \n \n ')
    plt.xlim(400,700)
  
    plt.subplot(4,1,4)
    plt.text(.95, .95, 'E', va='top', transform=plt.gca().transAxes,fontsize=30)  
    for i in range(len(Rrs)):
      plt.plot(wl, D_Rrs_ap[i,:]-D_Rrs_aph[i,:], color=colors[color_index[i]],linewidth=1, alpha=0.6)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Difference in $\delta R_{rs}(\lambda)$  [sr$^{-1}$]')
    plt.xlim(400,700)
    plt.ylim(-0.0003,0.0003)
   
    plt.subplot(4,1,2)
    plt.text(.95, .95,  'C', ha='left', va='top', transform=plt.gca().transAxes,fontsize=30)  
    for i in range(len(Rrs)):
      plt.plot(wl,  D_Rrs_aph [i,:], color=colors[color_index[i]],linewidth=1, alpha=0.6)
    #plt.xlabel('Wavelength [nm]')
    plt.ylabel('$\delta R_{rs}(\lambda)$ [sr$^{-1}$]: \n $a_{ph}(\lambda)$ input ')
    plt.xlim(400,700)
    plt.ylim(-0.0008,0.0008)
   
    plt.subplot(4,1,3)
    plt.text(.95, .95,  'D', ha='left', va='top', transform=plt.gca().transAxes,fontsize=30)  
    for i in range(len(Rrs)):
      plt.plot(wl,  D_Rrs_ap [i,:], color=colors[color_index[i]],linewidth=1, alpha=0.6)
    #plt.xlabel('Wavelength [nm]')
    plt.ylabel('$\delta R_{rs}(\lambda)$ [sr$^{-1}$]: \n $a_{p}(\lambda)$ input ')
    plt.xlim(400,700)
    plt.ylim(-0.0008,0.0008)
   
    plt.tight_layout()
    filename  =  path_figs + '/'  + 'delta_Rrs.png'
    plt.savefig(filename,dpi=900)   
    
    return
   
    
def plot_delta_Rrs_single():
    
    '''
      Plots delta_Rrs for aph model alone
    ''' 
    
    D_Rrs_aph = np.array(pd.read_csv(path_output + '/Delta_Rrs_AMT_aph_input.csv'))  
    D_Rrs_ap = np.array(pd.read_csv(path_output + '/Delta_Rrs_AMT_ap_input.csv'))  
     
    wl =np.arange(400,701,1)
    colors = cm.rainbow(np.linspace(0, 1,101))
    color_index = np.round(100*(np.log10(chl_hplc) +2)/2).astype(int)
    color_index[color_index > 100] = 100
    order_i =  (color_index).argsort()
    color_index = color_index[order_i]    
      
    data = pd.read_csv(path_data + 'Rrs/AMT_HPLC_Rrs.csv')[:210]
    Rrs = np.array(data.iloc[:,-301:].values) #
    Rrs = Rrs[order_i]
    D_Rrs_aph = D_Rrs_aph[order_i]
    D_Rrs_ap = D_Rrs_ap[order_i]
    
    plt.figure(figsize=(18,6))
   
    
  #  plt.text(.95, .95,  'C', ha='left', va='top', transform=plt.gca().transAxes,fontsize=30)  
    for i in range(len(Rrs)):
      plt.plot(wl,  D_Rrs_aph [i,:], color=colors[color_index[i]],linewidth=1, alpha=0.6)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('$\delta R_{rs}(\lambda)$ [sr$^{-1}$]: \n $a_{ph}(\lambda)$ input ')
    plt.xlim(400,700)
    plt.ylim(-0.0008,0.0008)

    
    plt.tight_layout()
    filename  =  path_figs + '/'  + 'delta_Rrs_single.png'
    plt.savefig(filename,dpi=900)   
   


def summary_coeffweights(Dspec,beta_mean_ap,beta_mean_aph):
     
      """
      This plots the fractional contribution of each beta wl bin to the predicted pigment
      concentration. 
      """
      
      wl = np.arange(400,701,1)
      fig = plt.figure(figsize=(22,18))
      plt.rcParams.update({'font.size': 22})
      bar_colors = []
     
      for i in range(16):
          bar_colors.append(plt.cm.tab20c(i))
              
                    
      c = [0,4,8,12]   # corresponds to  pigkeys = ['Tchlb','Hex','Zea','Chlc12']
      
      # aph
      for i in range(len(pigkeys)):
     
          # model predicted pigments
          scaler = StandardScaler()
          Dspec_norm_i = scaler.fit_transform(Dspec)
        
          # loop over `projection vectors' (interpretted as fractional contribution
          # to concentration each wavelength bin)
          proj_vectors = []
          # breakpoint(
          for n in range(len((Dspec))): 
            
              proj_n  = np.abs(np.multiply(Dspec_norm_i[n], beta_mean_aph[i,:]))
              proj_n = 100*proj_n/np.sum(proj_n)
              proj_vectors.append(proj_n)
    
          proj_vectors = np.array(proj_vectors)
          mean_proj_vector = np.mean(np.stack(proj_vectors),axis=0)
      
          plt.subplot(4,2,int(2*i+1))
          plt.title(str(pigkeys[i]) + ': $a_{ph}(\lambda)$ input')
          plt.plot(wl, mean_proj_vector,color=bar_colors[c[i]])
          plt.xlim(400,700)
          plt.ylim(0,1.7)

      # ap
      for i in range(len(pigkeys)):
          # model predicted pigments
     
          scaler = StandardScaler()
          Dspec_norm_i = scaler.fit_transform(Dspec)
        
          # loop over `projection vectors' (interpretted as fractional contribution
          # to concentration each wavelength bin)
          proj_vectors = []
         # breakpoint()
        

          for n in range(len((Dspec))): 
            
              proj_n  = np.abs(np.multiply(Dspec_norm_i[n], beta_mean_ap[i,:]))
              proj_n = 100*proj_n/np.sum(proj_n)
              proj_vectors.append(proj_n)
    
          proj_vectors = np.array(proj_vectors)
    
          mean_proj_vector = np.mean(np.stack(proj_vectors),axis=0)
      
          plt.subplot(4,2,int(2*i+2))
          plt.title(str(pigkeys[i]) + ': $a_{p}(\lambda)$ input')
          plt.plot(wl, mean_proj_vector,color=bar_colors[c[i]])
          plt.xlim(400,700)
          plt.ylim(0,1.7)
       
      fig.supylabel('Wavelength coefficient weighting [%]')
      fig.supxlabel('Wavelength [nm]')
      fig.tight_layout()
      
      plt.tight_layout()
      filename  =  path_figs + '/'  + 'wl_coeffs.pdf'
      plt.savefig(filename,dpi=600)   
      
      return 
  
    
def pig_ratios():
           
    """
     Pigment ratio boxplot
    """
        
    fig = plt.figure(figsize=(14,18))
    plt.rcParams.update({'font.size': 20})
    df_hplc = data 
    
    pig_mdl = ['Tchla','Tchlb','ABcaro','But',
               'Hex', 'Allo', 'Diadino','Diato',
               'Fuco', 'Perid','Zea', 'DVchla',
               'Chlc12','Chlc3', 'Neo', 'Viola']
    
    bar_colors =[]
    for i in range(16):
        bar_colors.append(plt.cm.tab20c(i))         
    ratios = [] 
    plt.subplot(3,1,1)
    plt.text(.95, .95, 'A', va='top', transform=plt.gca().transAxes,fontsize=22)  
    for i in range(1,len(pig_mdl)):  
        Cchl = df_hplc[pig_mdl[0]] # Tchla
        Cpig_i = df_hplc[pig_mdl[i]]
        rat_i = Cpig_i/Cchl
        rat_i = rat_i[rat_i>0]
        ratios.append(rat_i)

    bp = plt.boxplot(ratios ,showfliers=True,patch_artist=True, medianprops=dict(color='black'), whis=[10,90]) 
    plt.xlim(0.5,15.5)
    plt.ylim(0,1)
    for i in range(15):
        bp['boxes'][i].set_facecolor(bar_colors[i+1])
          
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],pig_mdl[1:16])
    plt.xticks(rotation = 90)
    plt.ylabel('AccPig/Tchla')
    plt.grid(visible=None, which='major', axis='y')
    
    resid_abs = []
    plt.subplot(3,1,2)
    plt.text(.95, .95, 'B', va='top', transform=plt.gca().transAxes,fontsize=22) 
    df_hplc = data 
    conc_space ='log'
    beta_mean= np.load(path_output + 'beta_mean_aph_log.npy')
    alpha_mean= np.load(path_output + 'alpha_mean_aph_log.npy')
    for i in range(1,len(pig_mdl)):     
           
        Cchl = df_hplc[pig_mdl[0]] # Tchla
        Cpig_i = df_hplc[pig_mdl[i]]

        scaler = StandardScaler()
        Dspec_norm_i = scaler.fit_transform(Dspec)
     
        if conc_space =='log':
            Cchl_pred = 10**(np.matmul(Dspec_norm_i, beta_mean[0,:]) + alpha_mean[0])
            Cpig_pred_i = 10**(np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])
        elif conc_space =='lin':
            Cchl_pred = (np.matmul(Dspec_norm_i, beta_mean[0,:]) + alpha_mean[0])
            Cpig_pred_i = (np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])

        # values above detection limit
        Cchl_pred = Cchl_pred[Cpig_i >= 0.001]
        Cchl = Cchl[Cpig_i >= 0.001]
        Cpig_pred_i = Cpig_pred_i[Cpig_i >= 0.001]
        Cpig_i = Cpig_i[Cpig_i >= 0.001]
        
        if conc_space =='lin':
            Cchl_pred = Cchl_pred[Cpig_pred_i > 0] 
            Cchl = Cchl[Cpig_pred_i > 0] 
            Cpig_i = Cpig_i[Cpig_pred_i > 0] 
            Cpig_pred_i = Cpig_pred_i[Cpig_pred_i > 0] 
        
        Cpig_rat_i = Cpig_i/Cchl
        Cpig_pred_rat_i = Cpig_pred_i/Cchl_pred  
        
        resid_abs.append(Cpig_pred_rat_i - Cpig_rat_i)
        

    plt.plot(np.arange(0,18,1), np.zeros(18),linestyle ='dashed',color='gray')
    bp = plt.boxplot(resid_abs ,showfliers=True,patch_artist=True, medianprops=dict(color='black'), whis=[10,90]) 
    plt.xlim(0.5,15.5)
    plt.ylim(-0.15, 0.15)
    for i in range(15):
        bp['boxes'][i].set_facecolor(bar_colors[i+1])
          
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],pig_mdl[1:16])
    plt.xticks(rotation = 90)
    plt.ylabel('(AccPig/Tchla)$_{m}$ - AccPig/Tchla: \n logarithmic model')
    plt.grid(visible=None, which='major', axis='y')
    # linear
    resid_abs = []
    plt.subplot(3,1,3)
    plt.text(.95, .95, 'C', va='top', transform=plt.gca().transAxes,fontsize=22) 
    df_hplc = data 
    conc_space ='lin'
    beta_mean= np.load(path_pigments + 'beta_mean_aph_lin.npy')
    alpha_mean= np.load(path_pigments + 'alpha_mean_aph_lin.npy')
    for i in range(1,len(pig_mdl)):     
           
        Cchl = df_hplc[pig_mdl[0]] # Tchla
        Cpig_i = df_hplc[pig_mdl[i]]

        scaler = StandardScaler()
        Dspec_norm_i = scaler.fit_transform(Dspec)
     
        if conc_space =='log':
            Cchl_pred = 10**(np.matmul(Dspec_norm_i, beta_mean[0,:]) + alpha_mean[0])
            Cpig_pred_i = 10**(np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])
        elif conc_space =='lin':
            Cchl_pred = (np.matmul(Dspec_norm_i, beta_mean[0,:]) + alpha_mean[0])
            Cpig_pred_i = (np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])

        # values above detection limit
        Cchl_pred = Cchl_pred[Cpig_i >= 0.001]
        Cchl = Cchl[Cpig_i >= 0.001]
        Cpig_pred_i = Cpig_pred_i[Cpig_i >= 0.001]
        Cpig_i = Cpig_i[Cpig_i >= 0.001]
        
        if conc_space =='lin':
            Cchl_pred = Cchl_pred[Cpig_pred_i > 0] 
            Cchl = Cchl[Cpig_pred_i > 0] 
            Cpig_i = Cpig_i[Cpig_pred_i > 0] 
            Cpig_pred_i = Cpig_pred_i[Cpig_pred_i > 0] 
        
        Cpig_rat_i = Cpig_i/Cchl
        Cpig_pred_rat_i = Cpig_pred_i/Cchl_pred  
        
        resid_abs.append(Cpig_pred_rat_i - Cpig_rat_i)
          

    plt.plot(np.arange(0,18,1), np.zeros(18),linestyle ='dashed',color='gray')
    bp = plt.boxplot(resid_abs ,showfliers=True,patch_artist=True, medianprops=dict(color='black'), whis=[10,90]) 
    plt.xlim(0.5,15.5)
    plt.ylim(-0.15, 0.15)
    for i in range(15):
          bp['boxes'][i].set_facecolor(bar_colors[i+1])
            
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],pig_mdl[1:16])
    plt.xticks(rotation = 90)
    plt.ylabel('(AccPig/Tchla)$_{m}$ - AccPig/Tchla: \n linear model')
    
    plt.grid(visible=None, which='major', axis='y')
    
    plt.tight_layout()
    filename  =  path_figs + '/'  + 'AccPig_RATIOS.pdf'
    plt.savefig(filename,dpi=900)   
  
    return 
  
    
  
    def skew_plot():
        
        """
         Ilustrative example of log/linear space distributions
        """
        
        bar_colors =[]
        for i in range(16):
            bar_colors.append(plt.cm.tab20c(i))
           
        pigkeys = ['Tchla','Tchlb','ABcaro','But',
                        'Hex', 'Allo', 'Diadino','Diato',
                        'Fuco', 'Perid','Zea', 'DVchla',
                        'Chlc12','Chlc3', 'Neo', 'Viola']
               
    
        i= 10# ZEA
     
        data_i = data[pigkeys[i]].values[data[pigkeys[i]].values>0]
        data_i =   data_i[data_i>=0.001]
        N_i = np.sum(data_i >= 0)
        
        plt.figure(figsize=(14,6))
        plt.rcParams.update({'font.size': 22})
        plt.subplot(1,2,1)
        plt.title(str(pigkeys[i]) + ': Linear conc. scale')
        bins = np.arange(0.005,1,0.01)
        hist, bins = np.histogram(data_i, bins=bins)
        #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(data_i, bins=bins,color=bar_colors[i],edgecolor='black',label='N = ' + str(len(data_i)))
       # plt.xscale('log')
        plt.ylabel('Frequency')
        plt.xlabel('Concentration  [mg m$^{-3}]$')
        plt.xlim(0,0.2)
        label= 'N = ' + str(len(data_i))
        plt.legend()
        
      
        plt.subplot(1,2,2)
        plt.title(str(pigkeys[i]) + ': Log conc. scale')
        bins = np.arange(0.01,1,0.05)
        hist, bins = np.histogram(data_i, bins=bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(data_i, bins=logbins,color=bar_colors[i],edgecolor='black',label='N = ' + str(len(data_i)))
        plt.xscale('log')
        plt.ylabel('Frequency')
        plt.xlabel('Concentration  [mg m$^{-3}]$')
        plt.xlim(0.007,0.7)
        label= 'N = ' + str(len(data_i))
        plt.legend()
        
        plt.tight_layout()
        filename  =  path_figs + '/'  + 'Lin_log.png'
        plt.savefig(filename,dpi=900)  
        
        return
    

if __name__ == '__main__':
    
    path_root = str(Path(os.getcwd()).parents[1]) + '/'
    path_code =  os.getcwd()
    
    # input/output data
    path_data = path_root + 'Data/Input/'
    path_output = path_root + 'Data/Output/'    
    path_figs = path_root + 'Figs/'
    path_pigments = path_root + 'Data/Output/'
    
    # Delta Rrs
    D_Rrs = pd.read_csv(path_output + '/Delta_Rrs_AMT_aph_input.csv')
    dD_Rrs = pd.DataFrame(np.gradient(np.array(D_Rrs), axis = 1),columns=D_Rrs.keys()) # 1st derivative using second order-accurate central differences
    d2D_Rrs = pd.DataFrame(np.gradient(np.array(dD_Rrs), axis = 1),columns=D_Rrs.keys()) # 2nd derivative using second order accurate central differences
    Dspec = d2D_Rrs
    
    # spectral coeffeicients
    beta_mean_ap = np.load(path_output + 'beta_mean_ap_log.npy')
    beta_mean_aph = np.load(path_output + 'beta_mean_aph_log.npy')
   
    # Rrs and Chl 
    data = pd.read_csv(path_data + '/AMT_Rrs_HPLC.csv')[:210]
    Rrs = np.array(data.iloc[:,-301:].values) #
    chl_hplc = data['Tchla'].values

    # coefficients for ap and aph models
    A1, B1, A3, B3 = plot_aph_models()

    # pigment distributions
    plot_pig_dists()
    # GSM optimsation results
    plot_gsm_input_results()
    
    #Match-up map
    #plot_CHL_matchesmap()
    
    # Delta Rrs comparisons
    plot_delta_Rrs()
    
    # coeffcient weights
    pigkeys = ['Tchlb','Hex','Zea','Chlc12']
    summary_coeffweights(Dspec,beta_mean_ap,beta_mean_aph)
    
    #pigment ratios
    pig_ratios()

 
  
    
    
    
    
    
    
    
    
    
    
    
   
    
  
