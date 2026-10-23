#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to co-locate above-water Rrs and HPLC from AMT cruises, and proudce 
data figures for paper submission

"""

import pandas as pd
import numpy as np
import os
#import xarray as xr
import datetime
import glob
import cartopy.geodesic as gd
from pathlib import Path

# matplotlib/cartopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import matplotlib.patches as mpatches
#import matplotlib.colors as cl
import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy

# use for longhurst
#from xml.dom.minidom import *
import SB_support as sbs

import dierssen_qwip_functions as qf

def load_seabass(file_path):
     ''' Wrapper to load SeaBASS Rrs & structure as dataframe'''

     # read_data returns header, comments, missing, and data (as ordered dict)  
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
     
     cruise_i = file_path.split('/')[9]
     df['time']=cruise_i
     df =df.rename(columns={"time": "cruise"})

     return df
 
    
def load_seabass_hplc(file_path):
    
     ''' Wrapper to load SeaBASS HPLC & structure as dataframe'''
     # read_data returns header, comments, missing, and data (as ordered dict)
     data = sbs.readSB(file_path)
  
     # Convert data dictionary to DataFrame
     df = pd.DataFrame(data.data, columns = data.data.keys()) 
   
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
     df['time'] = cruise_i
     df = df.rename(columns={"time": "cruise"})
    
     return df
     
    
def wl_interp(spec, spec_wl, wl_out, method ='pw_lin'):
   
    """
    Performs wavelength interpolation on 2D (dims: time X wavelength) spectrla
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
    
    """
    Gets Rrs wavelength as np array for HyperCP-processed data
      
    """  
    
    wl_python = []
    wl_keys = data_python.keys()[10:146]
    for i in range(len(wl_keys)):
        wl_python.append(float(wl_keys[i][3:]))
    wl_python = np.stack(wl_python)
   
    return wl_python


def run_qwip_calculations(df, spectral_resolution, wavelengths=None, sensor_coeffs=None):
   
    """
    Run the QWIP calculation for a given dataframe. From: https://github.com/mowerther/Rrs_quality_assurance
    Used here for AVW.

    : params df: input dataframe
    : params spectral_resolution: the spectral_resolution either 'hyper' or 'multi'
    : params wavelengths (optional): multispectral wavelengths - needs to be defined if spectral_resolution == 'multi'
     params sensor_coeffs (optional): multispectral sensor coeffs - needs to be defined if spectral_resolution == 'multi'
    """

    # Extract visible Rrs from input DF and calculate the apparent visible wavelength (AVW)
    rrs_vis = qf.extract_columns(df, spectral_resolution, wavelengths)
    avw = qf.calculate_avw(rrs_vis, spectral_resolution, wavelengths, sensor_coeffs)
    
    # calculate NDI / QCI
    ndi,index_492,index_665 = qf.calculate_ndi(rrs_vis.to_numpy(), wavelengths)
    
    # calculate QWIP and scoring
    qwip_score, fit1 = qf.calculate_qwip_score(ndi, avw)
    qwip_flag = qf.flag_qwip_score(qwip_score)
    
    # calc classification indices
    ind_600a, ind_500a, ind_400a = qf.calculate_classification_indices(df, wavelengths)

    # add QWIP values to df
    df['ndi'] = ndi
    df['index_492'] = index_492
    df['index_665'] = index_665
    df['qwip_score'] = qwip_score
    df['qwip_flag'] = qwip_flag
    df['ind_400a'] = ind_400a
    df['ind_500a'] = ind_500a
    df['ind_600a'] = ind_600a
   
    print(f'QWIP {spectral_resolution} spectral calculation complete.')

    return df, fit1, avw, ndi, ind_400a, ind_500a, ind_600a, index_492,index_665
 
    
def spatial_temporal_coloc(match_len = 1500, match_temp = 2):
     
      """
      Function for spatial temporal co-location.
      
      Input:
          match_len = spatial tolerance (in m)
          match_temp = temporal tolerance (in hr)
      
      Output:
    
          co-located Rrs and HPLC matrices 
      
      """

      i_keep = [] # index of spectra to keep in average
      rrs_av = [] # averaged rrs, rrs_unc, key meta
      

      #  First stack Rrs into `point matrix' for spatial co-location
      rrs_points =[]
      for i in range(len(df_rrs_tot)):
           rrs_points.append((df_rrs_tot['lon'].iloc[i], df_rrs_tot['lat'].iloc[i])) 
                  
      # pivot around each pigment sample and apply spatial/temporal location criteria in loop
      geo = gd.Geodesic()
      for i in range(len(df_hplc)):
           # compute distances from hplc: rrs
           hplc_point = (df_hplc['lon'].iloc[i], df_hplc['lat'].iloc[i])
           hplc_time = df_hplc.index[i]
          
           # apply temporal co-location
           d = geo.inverse(points=hplc_point, endpoints=rrs_points)[:,0]
           n_spectra = len(d[d<match_len]) # n spectra is # satisfying temporal match
           # def nanaverage(A,weights,axis):
           # return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

           # apply spatial co-location
           if n_spectra > 0:
               spatial_matches = df_rrs_tot[d < match_len]
               delta_secs = (spatial_matches.index - hplc_time).total_seconds()
               spatial_matches = spatial_matches[delta_secs < 60*60*match_temp]
               
               # weighted average of rrs spectral (uses uncertaninty weighted on 400-600 nm)
               if len(spatial_matches) > 1:
                   unc_weight = 1/(np.nanmean(spatial_matches.iloc[:,311:511].values,axis=1)) 
                   # print(unc_weight)
                   spatial_matches = spatial_matches[np.isnan(unc_weight)==0]
                   unc_weight= unc_weight[np.isnan(unc_weight)==0]
                   wmean_av = np.average(spatial_matches.iloc[:,4:], axis=0, weights=unc_weight)
       
                   i_keep.append(i)
                   rrs_av.append(wmean_av)
       
               elif len(spatial_matches) == 1:
                   wmean_av = np.nanmean(spatial_matches.iloc[:,4:], axis=0)
                   i_keep.append(i)
                   rrs_av.append(wmean_av)
       
      # stack into dataframe
      df_hplc_m = df_hplc.iloc[np.stack(i_keep),:]
      rrs_av = np.stack(rrs_av)
      rrs_keys = list(df_rrs_tot.keys()[4:])

      A = pd.DataFrame(np.array([df_hplc_m['cruise'].values,df_hplc_m['lat'].values,df_hplc_m['lon'].values]).T,  columns=['cruise','lat','lon'],index=df_hplc_m.index)
      B = pd.DataFrame(rrs_av, columns=rrs_keys,index=df_hplc_m.index)
      df_rrs_tot_m = pd.concat([A,B],axis=1)

      return  df_rrs_tot_m, df_hplc_m   
   
    
# Coverage map functions
def plot_coverage():
    
       plt.figure(figsize=(15,10))
       extent = [-65,10,-60, 60]
       
       #plt.subplot(1,2,1)
        #  request = cimgt.GoogleTiles(style='satellite')
        #request = cimgt.OSM()
          
       ax = plt.axes(projection=ccrs.PlateCarree())
        #ax.add_image(request, 7)
       ax.coastlines()
       ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
       ax.add_feature(cartopy.feature.LAND, edgecolor='black')
       ax.set_extent(extent, ccrs.PlateCarree())
       gl = ax.gridlines(draw_labels=True)
       gl.top_labels = False
       gl.right_labels = False
       gl.xformatter =  LONGITUDE_FORMATTER
       gl.yformatter =  LATITUDE_FORMATTER
       gl.xlabel_style = {'size': 14,  'rotation': 0}
       gl.ylabel_style = {'size': 14,  'rotation': 0}
        
       lon_formatter = LongitudeFormatter(zero_direction_label=True)
       lat_formatter = LatitudeFormatter()
       ax.xaxis.set_major_formatter(lon_formatter)
       ax.tick_params(labelsize=14)
           
       labels = ['AMT 23 (2013) \n N = ' + str(sum(meta.cruise=='AMT23')),
                 'AMT 25 (2015) \n N = ' + str(sum(meta.cruise=='AMT25')),
                 'AMT 26 (2016) \n N = ' + str(sum(meta.cruise=='AMT26')),
                 'AMT 27 (2017) \n N = ' + str(sum(meta.cruise=='AMT27')),
                 'AMT 28 (2018) \n N = ' + str(sum(meta.cruise=='AMT28')),
                 'AMT 29 (2019) \n N = ' + str(sum(meta.cruise=='AMT29')),
                 'AMT 30 (2023) \n N = ' + str(sum(meta.cruise=='AMT30')),
                 'AMT 31 (2024) \n N = ' + str(sum(meta.cruise=='AMT31'))]
                
       
       ax.scatter(meta['lon'][meta['cruise']=='AMT23'],meta['lat'][meta['cruise']=='AMT23'], s=12, color='blue', label= labels[0])
       ax.scatter(meta['lon'][meta['cruise']=='AMT25'],meta['lat'][meta['cruise']=='AMT25'], s=12, color='red', label= labels[1])
       ax.scatter(meta['lon'][meta['cruise']=='AMT26'],meta['lat'][meta['cruise']=='AMT26'], s=12, color='green', label= labels[2])                
       ax.scatter(meta['lon'][meta['cruise']=='AMT27'],meta['lat'][meta['cruise']=='AMT27'], s=12, color='yellow', label= labels[3])
       ax.scatter(meta['lon'][meta['cruise']=='AMT28'],meta['lat'][meta['cruise']=='AMT28'], s=12, color='pink', label= labels[4])
       ax.scatter(meta['lon'][meta['cruise']=='AMT29'],meta['lat'][meta['cruise']=='AMT29'], s=12, color='purple',  label =labels[5])
       ax.scatter(meta['lon'][meta['cruise']=='AMT30'],meta['lat'][meta['cruise']=='AMT30'], s=12, color='brown', label= labels[6])
       ax.scatter(meta['lon'][meta['cruise']=='AMT31'],meta['lat'][meta['cruise']=='AMT31'], s=12, color='black', label= labels[7])
       
       plt.legend(markerscale=2,loc=4,fontsize=11)   
       
       from cartopy.io.shapereader import Reader
       from cartopy.feature import ShapelyFeature
        
       fname = path_data + 'longhurst_v4_2010/Longhurst_world_v4_2010.shp'
       shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                        ccrs.PlateCarree(),facecolor='none',edgecolor='gray',linestyle='--')
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
   
        
       filename  =  path_figs + '/'  + 'Rrs_AMT_coverage.png'
       plt.savefig(filename,dpi=900)   

       plt.figure(figsize=(15,10))
       extent = [-65,10,-60, 60]
                    
   
         #  request = cimgt.GoogleTiles(style='satellite')
         #request = cimgt.OSM()
           
       ax2 = plt.axes(projection=ccrs.PlateCarree())
         #ax.add_image(request, 7)
       ax2.coastlines()
       ax2.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
       ax2.add_feature(cartopy.feature.LAND, edgecolor='black')
       ax2.set_extent(extent, ccrs.PlateCarree())
       gl2 = ax2.gridlines(draw_labels=True)
       gl2.top_labels = False
       gl2.right_labels = False
       gl2.xformatter =  LONGITUDE_FORMATTER
       gl2.yformatter =  LATITUDE_FORMATTER
       gl2.xlabel_style = {'size': 14,  'rotation': 0}        
       gl2.ylabel_style = {'size': 14,  'rotation': 0}
         
       lon_formatter = LongitudeFormatter(zero_direction_label=True)
       lat_formatter = LatitudeFormatter()
       ax2.xaxis.set_major_formatter(lon_formatter)
       ax2.tick_params(labelsize=14)
            
       labels = ['AMT 23 (2013) \n N = ' + str(sum(df_hplc.cruise=='AMT23')),
                  'AMT 25 (2015) \n N = ' + str(sum(df_hplc.cruise=='AMT25')),
                  'AMT 26 (2016) \n N = ' + str(sum(df_hplc.cruise=='AMT26')),
                  'AMT 27 (2017) \n N = ' + str(sum(df_hplc.cruise=='AMT27')),
                  'AMT 28 (2018) \n N = ' + str(sum(df_hplc.cruise=='AMT28')),
                  'AMT 29 (2019) \n N = ' + str(sum(df_hplc.cruise=='AMT29')),
                  'AMT 30 (2023) \n N = ' + str(sum(df_hplc.cruise=='AMT30')),
                  'AMT 31 (2024) \n N = ' + str(sum(df_hplc.cruise=='AMT31'))]
                 
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT23'],df_hplc['lat'][df_hplc['cruise']=='AMT23'], s=12, color='blue', label= labels[0])
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT25'],df_hplc['lat'][df_hplc['cruise']=='AMT25'], s=12, color='red', label= labels[1])
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT26'],df_hplc['lat'][df_hplc['cruise']=='AMT26'], s=12, color='green', label= labels[2])                
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT27'],df_hplc['lat'][df_hplc['cruise']=='AMT27'], s=12, color='yellow', label= labels[3])
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT28'],df_hplc['lat'][df_hplc['cruise']=='AMT28'], s=12, color='pink', label= labels[4])
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT29'],df_hplc['lat'][df_hplc['cruise']=='AMT29'], s=12, color='purple',  label =labels[5])        
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT30'],df_hplc['lat'][df_hplc['cruise']=='AMT30'], s=12, color='brown', label= labels[6])
       ax2.scatter(df_hplc['lon'][df_hplc['cruise']=='AMT31'],df_hplc['lat'][df_hplc['cruise']=='AMT31'], s=12, color='black', label= labels[7])
        
       plt.legend(markerscale=2,loc=4,fontsize=11)   
         
       
       fname = path_data + 'longhurst_v4_2010/Longhurst_world_v4_2010.shp'
       shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                         ccrs.PlateCarree(),facecolor='none',edgecolor='gray',linestyle='--')
       ax2.add_feature(shape_feature)
          
       
       plt.text(.37, .92,   'NADR', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.12, .80,   'NASW', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.59, .78,   'NASE', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.20, .65,   'NATR', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.70, .49,   'ETRA', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.28, .53,   'WTRA', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.39, .33,   'SATL', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)
       plt.text(.48, .15,   'SSTC', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)        
       plt.text(.45, .09,   'SANT', ha='left', va='top', color='grey', transform=ax2.transAxes,fontsize=16)    
         
      
       filename  =  path_figs + '/'  + 'pigs_AMT_coverage.pdf'
       plt.savefig(filename,dpi=600)   
       
       return
   
    

def plot_unc():

       rrs_unc[rrs_unc < 1] = np.nan  # uncertainties < 1% in octave data will be unphysical
       rrs_unc_sr =rrs_unc*rrs/100
      
       plt.rcParams.update({'font.size': 16})
        
       fig = plt.figure(figsize=(12,5))
       plt.subplot(1,2,2)
       plt.plot(wl,np.nanmedian(np.abs(rrs_unc),axis=0),label ='Median',color='black')   
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc),75,axis=0),label='Quartiles',color='black',linestyle='dashed')
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc),25,axis=0),color='black',linestyle='dashed')
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc),10,axis=0),color='black',linestyle='dotted')
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc),90,axis=0),label='10th, 90th percentiles',color='black',linestyle='dotted')
       plt.gca().fill_between(wl, np.nanpercentile(np.abs(rrs_unc),25,axis=0), np.nanpercentile(np.abs(rrs_unc),75,axis=0), alpha=0.2)
      # plt.grid()
       #plt.legend(loc=6)
       plt.ylim(0,60)
       plt.xlim(400,700)
       plt.ylabel('$R_{rs}$ uncertainty [%]')
       plt.xlabel('Wavelength [nm]')
       plt.text(.05, .95,  'B', ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)  
      
       plt.subplot(1,2,1)
       plt.plot(wl,np.nanmedian(np.abs(rrs_unc_sr),axis=0),label ='Median',color='black')   
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc_sr),75,axis=0),label='Quartiles',color='black',linestyle='dashed')
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc_sr),25,axis=0),color='black',linestyle='dashed')
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc_sr),10,axis=0),color='black',linestyle='dotted')
       plt.plot(wl,np.nanpercentile(np.abs(rrs_unc_sr),90,axis=0),label='10th, 90th percentiles',color='black',linestyle='dotted')
       plt.gca().fill_between(wl, np.nanpercentile(np.abs(rrs_unc_sr),25,axis=0), np.nanpercentile(np.abs(rrs_unc_sr),75,axis=0), alpha=0.2)
   #    plt.grid()
       #plt.ylim(0,80)
       plt.xlim(400,700)
       plt.legend()
       plt.ylabel('$R_{rs}$ uncertainty [sr$^{-1}$]')
       plt.xlabel('Wavelength [nm]')
       plt.text(.05, .95,  'A', ha='left', va='top', transform=plt.gca().transAxes,fontsize=20)  
       
       plt.tight_layout()
       
       filename  =  path_figs + '/'  + 'Rrs_unc.pdf'
       plt.savefig(filename,dpi=600)   
       
       return

def plot_AVW_map():
   
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
        
      fname = path_data + 'longhurst_v4_2010/Longhurst_world_v4_2010.shp'
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
      sc = ax.scatter(meta['lon'],meta['lat'],c=avw, s=10,cmap="terrain",vmin=450,vmax=500,alpha=1)
      cbar = plt.colorbar(sc,extend='both', cax=ax.inset_axes((0.72, 0.03, 0.05, 0.4)))
      cbar.minorticks_on()
      cbar.ax.tick_params(labelsize=13)
      cbar.set_label('AVW [nm]',fontsize=13)
      
   
      #  meta1 =  meta.iloc[np.where(np.logical_and(avw > 445, avw < 455))]
      # meta2 =  meta.iloc[np.where(np.logical_and(avw > 455, avw < 465))]
      # meta3 =  meta.iloc[np.where(np.logical_and(avw > 465, avw < 475))]
      # meta4 =  meta.iloc[np.where(np.logical_and(avw > 475, avw < 485))]
      # meta5 =  meta.iloc[np.where(np.logical_and(avw > 485, avw < 495))]
      # meta6 =  meta.iloc[np.where(np.logical_and(avw > 495, avw < 505))]
   
      #ax.scatter(meta1['lon'],meta1['lat'])
      #ax.scatter(meta2['lon'],meta2['lat'])
      #ax.scatter(meta3['lon'],meta3['lat'])
      #ax.scatter(meta4['lon'],meta4['lat'])
      #ax.scatter(meta5['lon'],meta5['lat'])
      #ax.scatter(meta6['lon'],meta6['lat'])
          
      filename  =  path_figs + '/'  + 'AVW_map.png'
      plt.savefig(filename,dpi=900)   
     
      return

def plot_CHL_matchesmap():
       
    
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
      
    fname = path_data + 'longhurst_v4_2010/Longhurst_world_v4_2010.shp'
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(),facecolor='none', edgecolor='grey',linestyle='--')
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
    sc = ax.scatter(df_rrs_tot_m['lon'],df_rrs_tot_m['lat'],c=chl_m, s=25,norm="log",cmap="rainbow", linewidth=0.5, edgecolor='k', vmin=0.01,vmax=1)
    cbar = plt.colorbar(sc,extend='both', cax=ax.inset_axes((0.70, 0.03, 0.05, 0.4)))
    cbar.minorticks_on()
    cbar.set_label('Chl-a [mg m$^{-3}$]',fontsize=16)
      
       
      #  meta1 =  meta.iloc[np.where(np.logical_and(avw > 445, avw < 455))]
      # meta2 =  meta.iloc[np.where(np.logical_and(avw > 455, avw < 465))]
      # meta3 =  meta.iloc[np.where(np.logical_and(avw > 465, avw < 475))]
      # meta4 =  meta.iloc[np.where(np.logical_and(avw > 475, avw < 485))]
      # meta5 =  meta.iloc[np.where(np.logical_and(avw > 485, avw < 495))]
      # meta6 =  meta.iloc[np.where(np.logical_and(avw > 495, avw < 505))]
       
      #ax.scatter(meta1['lon'],meta1['lat'])
      #ax.scatter(meta2['lon'],meta2['lat'])
      #ax.scatter(meta3['lon'],meta3['lat'])
      #ax.scatter(meta4['lon'],meta4['lat'])
      #ax.scatter(meta5['lon'],meta5['lat'])
      #ax.scatter(meta6['lon'],meta6['lat'])
      
    filename  =  path_figs + '/'  + 'CHLmatches_map.pdf'
    plt.savefig(filename,dpi=600)   
   
    return


def plot_rrs_by_avw():

    rrs_1 =  rrs[np.where(np.logical_and(avw > 445, avw < 455))]
    rrs_2 =  rrs[np.where(np.logical_and(avw > 455, avw < 465))]
    rrs_3 =  rrs[np.where(np.logical_and(avw > 465, avw < 475))]
    rrs_4 =  rrs[np.where(np.logical_and(avw > 475, avw < 485))]
    rrs_5 =  rrs[np.where(np.logical_and(avw > 485, avw < 495))]
    rrs_6 =  rrs[np.where(np.logical_and(avw > 495, avw < 505))]
     
    colors= cm.terrain(np.linspace(0, 1,6))
    
    rrs_bin = [rrs_1, rrs_2, rrs_3, rrs_4, rrs_5, rrs_6] 
    letter =  ['B', 'C','D', 'E', 'F', 'G'] 
    title =  ['445 < AVW < 455 nm, N = ' + str(len(rrs_1)),
              '455 < AVW < 465 nm, N = ' + str(len(rrs_2)),
              '465 < AVW < 475 nm, N = ' + str(len(rrs_3)),
              '475 < AVW < 485 nm, N = ' + str(len(rrs_4)),
              '485 < AVW < 495 nm, N = ' + str(len(rrs_5)),
              '495 < AVW < 505 nm, N = ' + str(len(rrs_6))]
    
    fig = plt.figure(figsize=(14,16))   

    plt.rcParams.update({'font.size': 20})
    for i in range(len(rrs_bin)):
        plt.subplot(3,2,i+1)
        plt.title(title[i])
        plt.gca().fill_between(wl, np.nanpercentile(rrs_bin[i],25,axis=0), np.nanpercentile(np.abs(rrs_bin[i]),75,axis=0), color=colors[i],alpha=0.4)
        plt.plot(wl,np.nanmedian(rrs_bin[i],axis=0),label ='Median',color='black')   
        plt.plot(wl,np.nanpercentile(rrs_bin[i],25,axis=0),color='black',linestyle='dashed')
        plt.plot(wl,np.nanpercentile(rrs_bin[i],75,axis=0),color='black',linestyle='dashed', label='Quartiles')
        plt.plot(wl,np.nanpercentile(rrs_bin[i],10,axis=0),color='black',linestyle='dotted')
        plt.plot(wl,np.nanpercentile(rrs_bin[i],90,axis=0),label='10th, 90th percentiles',color='black',linestyle='dotted')
     
        if i==0:
            plt.legend(framealpha=0)
    
     
        plt.text(.90, .95,  letter[i], ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)  
        plt.ylim(0,0.020)
        plt.xlim(400,700)
        
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('$R_{rs}(\lambda)$ [sr$^{-1}$]')
    plt.tight_layout()

    filename  =  path_figs + '/'  + 'rrs_by_AVW.png'
    plt.savefig(filename,dpi=900)   
    return 


def plot_match_ups():
            
       # breakpoint()
       plt.figure(figsize=(15,10))
       extent = [-65,10,-60, 60]
       
       # plt.subplot(1,2,1)
       #  request = cimgt.GoogleTiles(style='satellite')
       # request = cimgt.OSM()
          
       ax = plt.axes(projection=ccrs.PlateCarree())
        #ax.add_image(request, 7)
       ax.coastlines()
       ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
       ax.add_feature(cartopy.feature.LAND, edgecolor='black')
       ax.set_extent(extent, ccrs.PlateCarree())
       gl = ax.gridlines(draw_labels=True)
       gl.top_labels = False
       gl.right_labels = False
       gl.xformatter =  LONGITUDE_FORMATTER
       gl.yformatter =  LATITUDE_FORMATTER
       gl.xlabel_style = {'size': 14,  'rotation': 0}
       gl.ylabel_style = {'size': 14,  'rotation': 0}
        
       lon_formatter = LongitudeFormatter(zero_direction_label=True)
       lat_formatter = LatitudeFormatter()
       ax.xaxis.set_major_formatter(lon_formatter)
       ax.tick_params(labelsize=14)
           
       labels = ['AMT 23 (2013) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT23')),
                 'AMT 25 (2015) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT25')),
                 'AMT 26 (2016) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT26')),
                 'AMT 27 (2017) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT27')),
                 'AMT 28 (2018) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT28')),
                 'AMT 29 (2019) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT29')),
                 'AMT 30 (2023) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT30')),
                 'AMT 31 (2024) \n N = ' + str(sum(df_rrs_tot_m.cruise=='AMT31'))]
                
       
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT23'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT23'], s=20, color='blue', label= labels[0])
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT25'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT25'], s=20, color='red', label= labels[1])
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT26'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT26'], s=20, color='green', label= labels[2])                
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT27'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT27'], s=20, color='yellow', label= labels[3])
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT28'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT28'], s=20, color='pink', label= labels[4])
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT29'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT29'], s=20, color='purple',  label =labels[5])
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT30'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT30'], s=20, color='brown', label= labels[6])
       ax.scatter(df_rrs_tot_m['lon'][df_rrs_tot_m['cruise']=='AMT31'],df_rrs_tot_m['lat'][df_rrs_tot_m['cruise']=='AMT31'], s=20, color='black', label= labels[7])
       
       plt.legend(markerscale=2,loc=4,fontsize=11)   
       
       from cartopy.io.shapereader import Reader
       from cartopy.feature import ShapelyFeature
        
       fname = '/mnt/d/ESSD_paper/Tests_and_plots/longhurst_v4_2010/Longhurst_world_v4_2010.shp'
       shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                        ccrs.PlateCarree(),facecolor='none',edgecolor='gray',linestyle='--')
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
      
      
       filename  =  path_figs + '/'  + 'matchups_AMT_coverage.png'
       plt.savefig(filename,dpi=600)   
       
       return
   

if __name__ == '__main__':
    
   
   path_root = str(Path(os.getcwd()).parents[1]) + '/'
   path_code =  os.getcwd()
    
   # input/output data
   path_data = path_root + 'Data/'
   path_output = path_root + 'Data/Output/'    
   path_figs = path_root + 'Figs/Paper/'
   path_rrs =  path_data+  '/SeaBASS_Output/SeaBASS_Output/'
   
   # These are PML AMR BRDF-corrected Rrs files  - available from https://seabass.gsfc.nasa.gov/archive/PML/AMT/
   rrs_files = glob.glob(path_rrs +'/AMT*/*_HSASRrs_MO02*')
   df_list = [load_seabass(file) for file in   rrs_files]
   wl = np.arange(400,701,1)
    
   # exctract octave-processed data (Lin et al) and interpolate to 1 nm grid
   data_octave = pd.concat(df_list[0:6], axis=0, ignore_index=False)
   meta_octave = data_octave.iloc[:,:10]

   rrs_octave = np.array(data_octave.iloc[:,10:161])
   rrs_unc_octave = np.array(data_octave.iloc[:,161:])
   wl_octave = np.arange(400,702,2)
   
   rrs_octave = wl_interp(rrs_octave, wl_octave, wl, 'pw_lin')
   rrs_unc_octave = wl_interp(rrs_unc_octave, wl_octave, wl, 'pw_lin')

   
   # extract python-processed data (HyperCP) and interpolate to 1 nm grid
   data_python = pd.concat(df_list[6:8], axis=0, ignore_index=False)
   meta_python = data_python.iloc[:,:10]
   
   rrs_python = np.array(data_python.iloc[:,10:146])
   rrs_unc_python = np.array(data_python.iloc[:,146:])
   
   wl_python = get_wlpython(data_python)
   rrs_python = wl_interp(rrs_python, wl_python, wl, 'pw_lin')
   rrs_unc_python = wl_interp(rrs_unc_python, wl_python, wl, 'pw_lin')
  
   # stack into single arrays abd data frames
   meta = pd.concat([meta_octave, meta_python])
   rrs = np.concatenate([rrs_octave, rrs_python])
   df_rrs = pd.DataFrame(rrs, columns='rrs_' + wl.astype(str),index=meta.index)
   rrs_unc = np.concatenate([rrs_unc_octave, rrs_unc_python])
   df_rrs_unc = pd.DataFrame(rrs_unc, columns='rrs_unc_' + wl.astype(str),index=meta.index)
   df_rrs_tot = pd.concat([meta,df_rrs, df_rrs_unc],axis=1,ignore_index=False)
   
   # breakpoint()
   # HPLC
   # These are PML AMT HPLC files  - available from https://seabass.gsfc.nasa.gov/archive/PML/AMT/
   path_hplc=path_rrs
   sb_files = glob.glob(path_hplc +'/AMT*/*HPLC*.sb*')
   df_hplc_list = [load_seabass_hplc(file) for file in   sb_files]
   df_hplc = pd.concat(df_hplc_list, axis=0, ignore_index=False).drop_duplicates()
   df_hplc = df_hplc[df_hplc.depth < 10]
   df_hplc[df_hplc['depth']=='underway']= 5

   # Alternative match-up protocol based purely on time (not recommended)
   # match_win = 20
   # delta_t = datetime.timedelta(hours=0, minutes=match_win, seconds=0)
   # df_rrs_tot.iloc[:,10:] = df_rrs_tot.iloc[:,10:].rolling(delta_t).median()    
   # df_rrs_tot_m = df_rrs_tot.reindex(df_hplc.index,method='nearest', tolerance = str(match_win)+'min') # resam
   # df_hplc_m = df_hplc[np.isnan(df_rrs_tot_m['lat']) == 0] # match-up dataset for hplc
   # df_rrs_tot_m = df_rrs_tot_m[np.isnan(df_rrs_tot_m['lat']) == 0] # match-up data set for rrs 
   # df_rrs_tot_m['date'] = np.arange(1,len(df_rrs_tot_m)+1,1) 
   # df_rrs_tot_m = df_rrs_tot_m.rename(columns={"date": "sample_no"})   
   # breakpoint()
   
   # match-ups within 2000 m and +/- 3hr
   df_rrs_tot_m, df_hplc_m  = spatial_temporal_coloc(match_len = 2000, match_temp = 3)
   rrs_m = np.array(df_rrs_tot_m.iloc[:,9:310])
   rrs_unc_m = np.array(df_rrs_tot_m.iloc[:,310:])

   # compute AVW                       
   avw = run_qwip_calculations(pd.DataFrame(rrs, columns=np.arange(400, 701).astype('str')), 'hyper', wavelengths=np.arange(400, 701))[2].flatten()
   avw_m = run_qwip_calculations(pd.DataFrame(rrs_m, columns=np.arange(400, 701).astype('str')), 'hyper', wavelengths=np.arange(400, 701))[2].flatten()

   # filtering of 2 anonalous green spectra in coastal waters
   df_hplc_m = df_hplc_m[avw_m <515]
   df_rrs_tot_m = df_rrs_tot_m[avw_m <515]
   rrs_m = rrs_m[avw_m <515]
   rrs_unc_m = rrs_unc_m [avw_m <515]
   avw_m = avw_m[avw_m <515]
   chl_m = df_hplc_m['tot_chl_a'].values
 
   
   # clean-up of hplc data frame
   df_hplc_m = df_hplc_m.rename(columns={"sample": "original_id"})
   df_hplc_m['year'] = np.arange(1,len(df_rrs_tot_m)+1,1) 
   df_hplc_m =  df_hplc_m.rename(columns={"year": "sample_no"})
   
   cols2drop = ['month', 'day','hplc_gsfc_id','volfilt',
                'water_depth','ppc_tcar',  'bottle','ppc_tpg', 'psc_tcar',
                'psp_tpg', 'tacc_tchla', 'tchla_tpg', 'tchl_tcar']
   df_hplc_m =  df_hplc_m.drop(cols2drop , axis=1)
   
   def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df
   
   df_hplc_m = df_column_switch(df_hplc_m, "sample_no", "original_id")
   df_hplc_m = df_hplc_m.replace(-8888,0)

   # Trim for single salinty
   df_hplc_m = df_hplc_m[df_rrs_tot_m['sal'] > 0]
   df_rrs_tot_m =  df_rrs_tot_m[df_rrs_tot_m['sal'] > 0]
   
  
   # breakpoint()
   df_rrs_tot_m.to_csv('rrs_matches_AMT.csv')
   df_hplc_m.to_csv('hplc_matches_AMT.csv')
   # plt.ylim(0,20)
   # Data/Output/Rrs is Output from delta_rrs optimization
   # Data/Input/Rrs is Input for delta_rrs optimization

   # load Delta  Rrs as pandas dataframe - rows == timestamps, wavelengths = columns
   # breakpoint()
   colors = cm.rainbow(np.linspace(0, 1,101))
   color_index = np.round(100*(np.log10(chl_m) +2)/2).astype(int)
   color_index[color_index > 100] = 100
   order_i =  (color_index).argsort()
   color_index =  color_index[order_i]
   plot_AVW_map()
   plot_rrs_by_avw()
 

   
  

