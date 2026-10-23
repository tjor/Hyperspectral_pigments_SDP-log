#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to combine Rrs and HPLC matches & homogenize to NASA names

@author: tjor
"""


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':

   
   data_dir ='..'

   Rrs_data = pd.read_csv(data_dir + 'rrs_matches_AMT.csv')
   Pigment_data = pd.read_csv(data_dir + 'hplc_matches_AMT.csv')

   
   # filtering data for when pigments T, S not present
   Pigment_data = Pigment_data[Rrs_data['sal'] > 0]
   Rrs_data = Rrs_data[Rrs_data['sal'] > 0]
  
   #
   L = len(Rrs_data)
   
   # Extra fields
   sample_ID  = np.arange(1, L+1)
   
   # intialize metadata
   new_data = pd.DataFrame(data=None, index=sample_ID, columns=None, dtype=None)
   new_data['Sample_ID'] =   sample_ID 
   new_data['Original_ID'] = Pigment_data['original_id'].values
   new_data['Cruise'] = Pigment_data['cruise'].values
   new_data['Original_ID'][new_data['Cruise']=='AMT31']= Pigment_data['station']
   new_data['Data_link'] = ["https://seabass.gsfc.nasa.gov/archive/PML/AMT/" for i in range(L)]
   new_data['HPLC_lab'] = np.zeros(len(sample_ID))
   new_data['HPLC_lab'][new_data['Cruise']=='AMT23'] ='PML'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT25'] ='PML'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT26'] ='PML'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT27'] ='DHI'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT28'] ='DHI'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT29'] ='NASA_GSFC'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT30'] ='NASA_GSFC'
   new_data['HPLC_lab'][new_data['Cruise']=='AMT31'] ='FURG'
   new_data['Rad_method'] = ["Above-water" for i in range(L)]
   new_data['DateTime'] = Pigment_data['timestamp'].values
   new_data['Lat'] = Pigment_data['lat'].values
   new_data['Lon'] = Pigment_data['lon'].values
   new_data['Temp'] = Rrs_data['wt'].values
   new_data['Sal'] = Rrs_data['sal'].values

   
   print(Pigment_data.keys())
   # primary pigs 
   new_data['Tchla'] = Pigment_data['tot_chl_a'].values
   new_data['Tchlb'] = Pigment_data['tot_chl_b'].values  
   new_data['Tchlc'] = Pigment_data['tot_chl_c'].values   
   new_data['ABcaro'] = Pigment_data['alpha-beta-car'].values  
   new_data['But'] = Pigment_data['but-fuco'].values    
   new_data['Hex'] = Pigment_data['hex-fuco'].values   
   new_data['Allo'] = Pigment_data['allo'].values   
   new_data['Allo'][np.isnan(new_data['Allo'])==True] = 0  
   new_data['Diadino'] = Pigment_data['diadino'].values  
   new_data['Diato'] = Pigment_data['diato'].values   
   new_data['Fuco'] = Pigment_data['fuco'].values   
   new_data['Perid'] = Pigment_data['perid'].values   
   new_data['Zea'] = Pigment_data['zea'].values   
      
   # secondary/tertiary pigs 
   new_data['MVchla'] = Pigment_data['mv_chl_a'].values
   new_data['MVchla'].iloc[0:57] = Pigment_data['chl_a'].iloc[0:57].values # PML labs
   new_data['DVchla'] = Pigment_data['dv_chl_a'].values
   new_data['DVchla'][np.isnan(new_data['DVchla'])==True].values = 0  
   # new_data['Chllide'] = data['Chllide'] Remove degregation pigs
   new_data['MVchlb'] = Pigment_data['mv_chl_b'].values
   new_data['DVchlb'] = Pigment_data['dv_chl_b'].values
   new_data['DVchlb'].iloc[100:150][np.isnan(new_data['DVchlb']).iloc[100:150]==True] = 0  
   new_data['MVchlb'] = new_data['Tchlb'].values - new_data['DVchlb'].values
   new_data['Chlc12'] = Pigment_data['chl_c1c2']
   new_data['Chlc12'].values[new_data['Cruise']=='AMT23'] = Pigment_data['tot_chl_c'].values[new_data['Cruise']=='AMT23'] - Pigment_data['chl_c3'].values[new_data['Cruise']=='AMT23']
   new_data['Chlc3'] = Pigment_data['chl_c3']
   new_data['Lut'] = Pigment_data['lut'].values
   new_data['Lut'][np.isnan(new_data['Lut'])==True] = 0  
   new_data['Neo'] = Pigment_data['neo'].values
   new_data['Neo'][np.isnan(new_data['Neo'])==True] = 0  
   new_data['Viola'] = Pigment_data['viola']
   new_data['Viola'][np.isnan(new_data['Viola'])==True] = 0  
   # new_data['Phytin'] = data['Phytin']
   # new_data['Phide'] = data['Phide']
   new_data['Pras'] =  Pigment_data['pras']
   new_data['Pras'][np.isnan(new_data['Pras'])==True] = 0  
   
   # pigment sums
   new_data['Psc'] = new_data['But'] + new_data['Fuco'] + new_data['Hex'] + new_data['Perid'] 
   new_data['Ppc'] = new_data['Allo'] + new_data['ABcaro'] + new_data['Diato'] + new_data['Diadino'] + new_data['Zea']
   
   for i in range(400,701,1):
       new_data['Rrs_' + str(i)] =Rrs_data['rrs_' + str(i)]
   
    
   new_data.to_csv('AMT_HPLC_Rrs.csv', index=False)


   PD = new_data
   fig =plt.figure(figsize=(25,20))
   plt.rc('font', size=22)
   pigkeys = PD.keys()[11:11+24]
   for i in range(len(pigkeys)):
         plt.subplot(5,5,i+1)
         data_i = PD[pigkeys[i]][PD[pigkeys[i]].values>0]
         plt.title(str(pigkeys[i]) + ':  N(Cpig>0)=' + str(len(data_i)))
         N_i = np.sum(data_i > 0)
         bins = np.arange(0.0001,10,0.5)
         hist, bins = np.histogram(data_i, bins=bins)
         logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
         plt.hist(data_i, bins=logbins,edgecolor='black',label='$N$' + str(np.sum(~np.isnan(data_i))))
         plt.xscale('log')
         #plt.ylabel('Frequency')
         #plt.xlabel('Concentration  [mg m$^{-3}]$')
         plt.xlim(0.0001,10)
     
   fig.supylabel('Frequency')
   fig.supxlabel('Concentration  [mg m$^{-3}]$') 
   fig.tight_layout()    
     
   

   
