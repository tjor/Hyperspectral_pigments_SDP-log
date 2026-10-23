#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script is the PML python implementation for training a reconfigured version 
of the SDP - Spectral Derivative Pigments - algorithm (Kramer et al. 2022, RSE)

This script was orginally based on the public MATLAB implemenation in Kramer et al. 2022:
https://github.com/sashajane19/Rrs_pigments. It combines the basic workflow of 
Kramer_Rrs_pigments.m and rrsModelTrain.m model in a single python script.

A key development from the MATLAB code is that the statistical model can be trainined in 
logarthmic concentration space

As in Kramer et al. 2022, various metrics can be used to optimize the statistical model.
We recommend that the mean percentage error (proportional to absolute error in log-space)
works best a single metric. 

MLR (multivariate linear regression) is used to refer to the principal component regression. 
We use the sklearn library for the regression model. Standard-scaler (`z-score normalization') 
is used for input Rrs spectra.

The key outputs are ensemble-averaged regression coeffcients (alpha, beta), 
which can then applied to hyperpsectral Rrs data outside the training 
(e.g. radiometry systems on other cruises, or PACE-OCI Rrs).

Tom Jordan: tjor@pml.ac.uk.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,median_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
   
def rrsModelTrain(Dspec, Cpig, n_perm, max_pcs, n_splits, mdl_pick_metric,pigmdl,conc_space='log'):
    
    """
    Trains and tests model, using k-folding of the input spectral data. 
    
    Workflow is based on https://github.com/sashajane19/Rrs_pigments but PCA 
    and optimization routines now split into functions.
    
    The function is applied to each pigment in turn, but the 
    seeding of the shuffles ensures fold consistency between pigments.                       
    
    Inputs:
    
        Dspec: 2D array (dims = no. of observations X wl). Spectra to train/test the model
        (e.g. Delta_Rrs or deriatives, which have been obtained from bio-optical optimizaton)
       
        Cpig: concentration of target pigment [mg/m^3].
    
        n_perm: number of cross-validations 
        Note - there are k-folds within each permutation
    
        max_pcs: the maximum number of principal cpts that can
        be used in model training. The model will test/use all principal components up
        to and including this number
        
        n_splits: the number of k-folds 
    
        mdl_pick_metric: optimization statistic - MPE (signed % error) is recommended 
        to producean unbiased regression line in log concentration space 
        (overall, works better than r^2 or MAPE)
        
    Outputs:
        
        pig_MLR_fit_params: summary table for performance metrics
        pig_alpha: intercept: dims = (no. of pigments, 2). 
        pig_beta: wl coefficients: dims = (no. of pigments, 2*wavelength)
        
        pig_alpha and pig_beta store
        
    """ 

    # Initalize lists for MLR regression modl
    all_MLR_fit_params = []
    all_alpha = []
    all_beta = []
    
    for n in range(n_perm): # master loop over sets of folds. n is the set/permutation index 
    
        # k-folds with shuffle. loop index `n' seeds a different random state 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=n)  
       
        # initalize MLR parameter lists for nth permuation
        alpha_n = [] 
        beta_wl_n = []
        for k, (train_index, test_index) in enumerate(kf.split(Dspec)): # loop over folds. k is the fold index
            # print(f"Fold {i}:") # optional output to print to screen
            # print(f"  Training dataset index: {train_index}")
            # print(f"  Test dataset index: {test_index}")
            
            # Divide spectra and pigments into train and test sets for each k-fold 
            Dspec_train = Dspec.values[train_index]
            Dspec_test = Dspec.values[test_index]
          
            Cpig_train = Cpig[train_index]
            Cpig_test = Cpig[test_index] 

            # note we are now at line 142 in the Matlab version of rrsModelTrain in https://github.com/sashajane19/Rrs_pigments. 
            
            # z-score normalization prior to PCA - (this is standard scalar option in sci-kit learn)
            scaler = StandardScaler()
            Dspec_train_norm = scaler.fit_transform(Dspec_train)
            Dspec_test_norm = scaler.fit_transform(Dspec_test)
 
            # perform PCA on training data (EOFs == wl coefficients, AFs == representation in principal coordinates)
            EOFs_train, AFs_train = PCA_decomp(Dspec_train_norm, max_pcs)
        
            # optimize MLR model w.r.t. number of principal components - MLR optimze is a function in this Python script,
            opt_index = MLR_optimize(AFs_train, EOFs_train, Cpig_train, Dspec_test_norm, Cpig_test, mdl_pick_metric,conc_space)
            
            # derived ML parameters for the kth fold of the nth permutation with the optimized no. of cpts
            # alpha (intercept), beta_wl (wavelength coeffs)
            alpha_kn, beta_wl_kn = MLR(AFs_train, EOFs_train, Cpig_train, Dspec_test_norm, Cpig_test, opt_index,conc_space)
            alpha_n.append(alpha_kn)
            beta_wl_n.append(beta_wl_kn)
        
        # mean (k-fold average) of MLR model parameters for the nth permutation (drop n index for convenience)
        alpha_mean = np.mean(np.array(alpha_n))
        beta_wl_mean = np.mean(np.array(beta_wl_n),axis=0)
        alpha_std = np.std(np.array(alpha_n))
        beta_wl_std = np.std(np.array(beta_wl_n),axis=0)
        
        # Predicted pigments using mean MLR parameters - the log is inverted here, so this is in linear units
        if conc_space == 'log':
            Cpig_pred = 10**(np.matmul(Dspec_test_norm, beta_wl_mean) + alpha_mean)
        elif  conc_space == 'lin':
            Cpig_pred = (np.matmul(Dspec_test_norm, beta_wl_mean) + alpha_mean)
     
        # mask for predicted values above detection limit 
        if pig_mdl == 'DVchla': # we can get better results for DVchla with a a higher `lower floor'
            Cpig_pred=  Cpig_pred[Cpig_test >= 0.001]
            Cpig_test = Cpig_test[Cpig_test >= 0.001]
        else:
            Cpig_pred =  Cpig_pred[Cpig_test >= 0.001]
            Cpig_test = Cpig_test[Cpig_test >= 0.001]
            
       # breakpoint()
        # addtional mask - this is check for physical solutions
        if conc_space == 'lin':
            Cpig_test = Cpig_test[Cpig_pred > 0.00]
            Cpig_pred = Cpig_pred[Cpig_pred > 0.00]
        
        # fit metrics in linear (concentration) space using mean over folds - in form `metric(True, Pred)' for in-built sk learn functions
        N = int(len(Cpig_test))
        mean_ae = mean_absolute_error(Cpig_test, Cpig_pred)         
        rsq = r2_score(Cpig_test, Cpig_pred)
        med_ae = median_absolute_error(Cpig_test, Cpig_pred)   
        mean_ae_pct = 100*mean_absolute_percentage_error(Cpig_test, Cpig_pred)   
        mean_e_pct = 100*np.mean((Cpig_pred-Cpig_test)/Cpig_test)   
        med_ae_pct = 100*np.median(abs(Cpig_pred-Cpig_test)/Cpig_test)   
        med_e_pct = 100*np.median((Cpig_pred-Cpig_test)/Cpig_test)      
        
        # fit metrics in log (concentration) space using mean over folds - in form `metric(True, Pred)' fo
        rsq_log = r2_score(np.log10(Cpig_test), np.log10(Cpig_pred))
       
        # append to summary lists
        all_MLR_fit_params.append([mean_ae, rsq, med_ae, mean_ae_pct, mean_e_pct, med_ae_pct, med_e_pct,  rsq_log])
        all_alpha.append(np.array([alpha_mean, alpha_std]).ravel())
        all_beta.append(np.array([beta_wl_mean, beta_wl_std]).ravel())
  
    # summary lists. The mean, std amd median fit statistics are permutations 
    pig_MLR_fit_params = np.array([np.nanmean(all_MLR_fit_params,axis=0), np.nanstd(all_MLR_fit_params,axis=0)]).ravel() 
    pig_MLR_fit_params = np.append(pig_MLR_fit_params,N)
    pig_alpha = np.mean(all_alpha,axis = 0) 
    pig_beta = np.mean(all_beta, axis = 0)
            
    return  pig_MLR_fit_params, pig_alpha, pig_beta


def PCA_decomp(Dspec_train_norm, max_pcs=30):
    
    ''' 
    
     Peforms PCA decomposition of spectral residual (delta_Rrs or delta_Rrs spectral derivatives)
     for training data
     
     Inputs: 
         
         Dspec_train_norm: set of training spectra
         max_pcs: number of maximum principal components for optimization
     
     Outputs: 
         
         EOFs_train: (empirical orthogonal functions) - these are wl coeffs for each 
         principal component: dims = wl * max pcs.
         
         AFs train: representations of spectral data in the principal component space
         dims = no. of samples *  max_pcs
        
    '''
    
    pca = PCA(n_components = max_pcs, whiten=False)
    pca.fit(Dspec_train_norm)  
    EOFs_train = np.transpose(pca.components_) #  wavelength  coeffs for each PC  - dims: wl * max pcs
    AFs_train = pca.fit_transform(Dspec_train_norm) # spectral data in the principal component space
    
    # plt.figure()
    # plt.plot(EOFs_train[:,0:3])
    
    # plt.figure()
    # plt.plot(AFs_train[0:3,:])
    
    return EOFs_train, AFs_train


def MLR_optimize(AFs_train, EOFs_train, Cpig_train, Dspec_test_norm, Cpig_test, mdl_pick_metric='mpe',conc_space='log'):
                     
    '''    
        Optimizes multivariate linear regresion w.r.t. number of principal cpts.

        Inputs:  
           
          EOFs_train: (empirical orthogonal functions) - these are wl coeffs for each 
          principal component: dims = wl * max pcs.
          
          AFs train: representations of spectral data in the principal component space
          dims = no. of samples *  max_pcs
        
          Cpig_train: pigment concentrations for training set: dims = no. of samples 
          
          Dspec_test_norm: normalized hyperspectral input (e.g. delta Rrs or derivatives)
          dims = no. of samples 

          Cpig_test: pigment concentrations for test set: dims = no. of samples 
         
          mdl_pick_metric: optimization metric to derive no of principal components
          
          conc_space: logarithmic or linear statistical concenrtation models
        
        Outputs: 
            
          opt_index = the index of the optimal number of components 
          (n_cpts = opt_index + 1 due to indexing from 0)
        
        
    '''
     
    opt_metric = [] # list for optimization metric
    for l  in range(0,len(AFs_train.T)): # n_cpts == l+1 
      
      # MLR model 
      # log-transformation ensures that predicted Cpig > 0. Change from standard Kramer 22 method?
      #weights = compute_weights(Cpig_train)
      if conc_space=='log':
          reg_l = LinearRegression().fit(AFs_train[:,0:l+1], np.log10(Cpig_train))
      elif conc_space=='lin':
          reg_l = LinearRegression().fit(AFs_train[:,0:l+1], Cpig_train)
    
      # MLR coeffs and intercept (note: `no intercept' is sklearn default for coef_)
      beta_l = reg_l.coef_ 
      alpha_l = reg_l.intercept_ 
    
      # transform beta coeffs into wl-space 
      beta_l_wl = np.matmul(EOFs_train[:,0:l+1], beta_l) # matrix product

      # Predicted linear pigment concentrations in wl-space  for metric calculations  
      if conc_space=='log':
          Cpig_pred = 10**(np.matmul(Dspec_test_norm, beta_l_wl) + alpha_l)
      elif conc_space=='lin':
          Cpig_pred = (np.matmul(Dspec_test_norm, beta_l_wl) + alpha_l)
  
      if pig_mdl == 'DVchla':
         Cpig_pred_l=  Cpig_pred[Cpig_test >= 0.001]
         Cpig_test_l = Cpig_test[Cpig_test >= 0.001]
      else:
         Cpig_pred_l =  Cpig_pred[Cpig_test >= 0.001]
         Cpig_test_l = Cpig_test[Cpig_test >= 0.001]
  
      # for linear model, we first filter for physical predictions (removes negatives)
      if conc_space =='lin':
          Cpig_test_l = Cpig_test_l[Cpig_pred_l > 0] 
          Cpig_pred_l = Cpig_pred_l[Cpig_pred_l > 0] 

      # calculate optimization metrics
      if mdl_pick_metric =='mae':
          mae_log = mean_absolute_error(np.log10(Cpig_test_l), np.log10(Cpig_pred_l))         
          opt_metric.append(mae_log)
      elif mdl_pick_metric =='mpe':   
          mpe = 100*np.mean((Cpig_test_l-Cpig_pred_l)/Cpig_pred_l) 
          opt_metric.append(mpe)
      elif mdl_pick_metric =='mape':
          mape = 100*mean_absolute_percentage_error(Cpig_test_l, Cpig_pred_l)   
          opt_metric.append(mape)
      elif mdl_pick_metric == 'rsq_log':
          rsq_log = r2_score(np.log10(Cpig_test_l), np.log10(Cpig_pred_l))
          opt_metric.append(1/rsq_log) # inverse, so that we can use minimzation syntax
      elif mdl_pick_metric == 'rsq':
          rsq = r2_score(Cpig_test_l, Cpig_pred_l)
          opt_metric.append(1/rsq) #  inverse, so that we can use minimzation syntax
    
    opt_index = int(np.nanargmin(np.array(opt_metric))) # find minimum l index for opt no of cpts
  
    return opt_index


def MLR(AFs_train, EOFs_train, Cpig_train, Dspec_test_norm, Cpig_test, opt_index,conc_space='log'):
    
    '''    
        Multivariate linear regression model, using optimal number of princpal components
        (which has already been derived from MLR_optimize function)
        
        Inputs:  
            
           EOFs_train: (empirical orthogonal functions) - these are wl coeffs for each 
           principal component: dims = wl * max pcs.
           
           AFs train: representations of spectral data in the principal component space
           dims = no. of samples *  max_pcs
         
           Cpig_train: pigment concentrations for training set: dims = no. of samples 
           
           Dspec_test_norm: normalized hyperspectral input (e.g. delta Rrs or derivatives)
           dims = no. of samples 

           Cpig_test: pigment concentrations for test set: dims = no. of samples 
          
           opt_index: index for optimal number of princial components
           
           conc_space: logarithmic or linear statistical concenrtation models
        
        Outputs: 
           
           (alpha, beta_wl): Intercept and coefficients in wl space for instance of MLR model 
          
    ''' 

    # MLR model 
    # weights = compute_weights(Cpig_train)
    # log-transformation ensures that predicted Cpig > 0. Change from standard Kramer 22 method
    if conc_space == 'log':     
        reg = LinearRegression().fit(AFs_train[:,0:opt_index +1 ], np.log10(Cpig_train))
    elif conc_space == 'lin':
        reg = LinearRegression().fit(AFs_train[:,0:opt_index +1 ], Cpig_train)

   # breakpoint()
    # MLR coeffs and intercept (note: `no intercept' is sklearn default for coef_)
    beta = reg.coef_ 
    alpha = reg.intercept_ 
  
    # transform beta coeffs into wl-space (alpha already is)
    beta_wl = np.matmul(EOFs_train[:,0: opt_index +1 ], beta) # matrix product
    
    return alpha, beta_wl


def adjust_inputdata(df_hplc, pig_mdl, conc_model = 'log'):
    
    ''' Adjusts input data for cases there are pigment concentrations
      not recorded (nan), or (if training in log-space)
      pigment concentraions below detection limit'''
      
 
    for i in range(len(pig_mdl)):
        pig_i = df_hplc[pig_mdl[i]].values  
       
        # standardise lower detection limit for log and linear models (0.001)
        if pig_mdl[i] =='DVchla':
            pig_i[pig_i<0.001] = 0 
        else:
            pig_i[pig_i<0.001] = 0 # standardises a few odd samples to below standard dection limit
      
        # creates sythetic data below detection limit for cases of zero when training in log-space
        if conc_model=='log':
            if pig_mdl[i] =='DVchla':
                upper_lim = 0.0007
                lower_lim = 0.0001
            else:   
                upper_lim = 0.0007
                lower_lim = 0.0001
           
            for j in range(len(pig_i)):
                    if pig_i[j] == 0:
                        pig_i[j] = 10**np.random.uniform(np.log10(lower_lim),np.log10(upper_lim))
             
            df_hplc[pig_mdl[i]] = pig_i
      
    return df_hplc

   
def round_sf(num, n=3):
    
    ''' Rounds print output to 3-SF for plot displays'''
    formatstr = '%.'+str(n)+'g'
    return float(formatstr % num)


def summary_table_and_scatter(df_hplc, alpha_mean, beta_mean):
     
      """
       Scatter plots for predicted (modelled) versus observed pigments using
       mean alpha and beta. This follows `reconstrcuted scatter plots in Kramer 2022 
       where all data is used in the scattter (combination of `test and train')
       
       
      """ 
    
      fig = plt.figure(figsize=(16,16))
      plt.rcParams.update({'font.size': 18})
      
      # scatter plot
      bar_colors =[]
      for i in range(16):
          bar_colors.append(plt.cm.tab20c(i))
      
      for i in range(len(pig_mdl)):          
          Cpig_i = df_hplc[pig_mdl[i]]
  
          scaler = StandardScaler()
          Dspec_norm_i = scaler.fit_transform(Dspec)
                
          if conc_space =='log':
              Cpig_pred_i = 10**(np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])
          elif conc_space =='lin':
              Cpig_pred_i = (np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])

          # pigs below dection (BD) limit of training data
          Cpig_pred_i_BD = Cpig_pred_i[Cpig_i < 0.001]
          Cpig_i_BD = Cpig_i[Cpig_i < 0.001]
          
          # values above detection limit
          Cpig_pred_i = Cpig_pred_i[Cpig_i >= 0.001]
          Cpig_i = Cpig_i[Cpig_i >= 0.001]
          
          # additional masking for linear model -remobves negatives
          if conc_space =='lin':
              Cpig_i = Cpig_i[Cpig_pred_i> 0] 
              Cpig_pred_i = Cpig_pred_i[Cpig_pred_i> 0] 
          
          # scatter plots
          plt.subplot(4,4,i+1)
          
          # fit linear model for regression in log-log space
          x = np.arange(0.0001,4,0.00001)
          mod = scipy.stats.linregress(np.log10(Cpig_i), np.log10(Cpig_pred_i))
          y = 10**(np.log10(x)*mod.slope +mod.intercept)
          
          mod_lin = scipy.stats.linregress(Cpig_i, Cpig_pred_i)
          
          mape = np.nanmean(100*abs(Cpig_pred_i - Cpig_i)/Cpig_i)
          mpe = np.nanmean(100*(Cpig_pred_i - Cpig_i)/Cpig_i)

          plt.title(str(pig_mdl[i]))
          if i==0:
              plt.plot(x,x,color='gray',linestyle='dotted',label='1:1')
              plt.plot(x,y,color='black',linestyle='dashed',label='OLS')
          else :
              plt.plot(x,x,color='gray',linestyle='dotted')
              plt.plot(x,y,color='black',linestyle='dashed')
        
          label = ['N = '  + str(int(len(Cpig_i))) + '\n' +
                   'S_log = ' + str(round_sf(mod.slope)) + ' $\pm$ ' + str(round_sf(2*mod.stderr)) + '\n'
                   'rsq_log = ' + str(round_sf((mod.rvalue)**2)) + '\n'
                   'rsq_lin = ' + str(round_sf((mod_lin.rvalue)**2)) + '\n'
                   'MAPE = '    + str(round_sf(mape)) + ' %' +'\n' +
                   'MPE = '    + str(round_sf(mpe)) + ' %'
                    ][0]
    
          plt.scatter(Cpig_i_BD, Cpig_pred_i_BD, alpha=0.5,label='Simulated',color='black',marker=".",edgecolors='black')
          plt.scatter(Cpig_i, Cpig_pred_i, alpha=0.5,color=bar_colors[i],edgecolors='black')
          
          if i==0:
              plt.ylim(0.01,4)
              plt.xlim(0.01,4)
          else:
             plt.ylim(0.0001,0.4)
             plt.xlim(0.0001,0.4)
             
          if i==0:
              plt.legend(loc=4,fontsize=12)
          plt.gca().set_yscale('log')
          plt.gca().set_xscale('log')
          plt.gca().set_aspect('equal')
          # plt.legend(fontsize=12,framealpha=0.6,markerscale=0)
          plt.text(.01, .99, label, ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
     
      fig.supylabel('Predicted pigment concentration [mg m$^{-3}$]')
      fig.supxlabel('Measured pigment concentration [mg m$^{-3}$]')
     
      fig.tight_layout() 
      filename  =  path_figs + '/'  + file_handle  +'pigs_scatter.pdf'
      plt.savefig(filename,dpi=600,bbox_inches='tight')
     
      # residual plots
      fig = plt.figure(figsize=(18,16))
      plt.rcParams.update({'font.size': 18})
      
      for i in range(len(pig_mdl)):          
          Cpig_i = df_hplc[pig_mdl[i]]
          scaler = StandardScaler()
          Dspec_norm_i = scaler.fit_transform(Dspec)  
          if conc_space =='log':
              Cpig_pred_i = 10**(np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])
          elif conc_space =='lin':
              Cpig_pred_i = (np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])

        
          # values above detection limit
          Cpig_pred_i = Cpig_pred_i[Cpig_i >= 0.001]
          Cpig_i = Cpig_i[Cpig_i >= 0.001]
            
          # addtional masking for linear model         
          if conc_space =='lin':
              Cpig_i = Cpig_i[Cpig_pred_i> 0] 
              Cpig_pred_i = Cpig_pred_i[Cpig_pred_i> 0] 
          

          mape = np.nanmean(100*abs(Cpig_pred_i - Cpig_i)/Cpig_i)
          mpe = np.nanmean(100*(Cpig_pred_i - Cpig_i)/Cpig_i)
        
          label = ['MAPE = '    + str(round_sf(mape)) + ' %' +'\n' +
                   'MPE = '    + str(round_sf(mpe)) + ' %'][0]
         
          plt.subplot(4,4,i+1)
          
          plt.title(str(pig_mdl[i]))
          res_i = 100*(Cpig_pred_i - Cpig_i)/Cpig_i
          plt.plot(x,np.zeros(len(x)),color='grey',linestyle='dotted')
          plt.grid(visible=None, which='major', axis='y')
          plt.scatter(Cpig_i,  res_i ,label=label, color=bar_colors[i],alpha=0.35,edgecolors='black')
          plt.legend()
        
          if i==0:
              plt.xlim(0.01,3)
          else:
              plt.xlim(0.0001,0.3)
              
          plt.ylim(-150,150)
          plt.gca().set_xscale('log')
   
      fig.supylabel('Percentage residual [%]')
      fig.supxlabel('Measured pigment concentration [mg m$^{-3}$]')
      
      fig.tight_layout()
      
      filename  =  path_figs + '/'  + file_handle + 'pigs_resid.pdf'
      plt.savefig(filename,dpi=600,bbox_inches='tight')

      return 




def summary_table_and_scatter_2b2(df_hplc, alpha_mean, beta_mean):
     
      """
       Scatter plots for predicted (modelled) versus observed pigments using
       mean alpha and beta. This follows scatter plots in Kramer 2022 where all data 
       is used in the scattter (combination of `test and train')
       
       
       Note: Tot_Chl_b is used as DV and MV_Chl_b were not discriminated
       
      """ 
    
      fig = plt.figure(figsize=(10,10))
      plt.rcParams.update({'font.size': 18})
      # breakpoint()
      
      bar_colors =[]
      for i in range(16):
          bar_colors.append(plt.cm.tab20c(i))
      
      j=0
      for i in range(len(pig_mdl)):          
          Cpig_i = df_hplc[pig_mdl[i]]
  
          scaler = StandardScaler()
          Dspec_norm_i = scaler.fit_transform(Dspec)
          # if pig_mdl[i] == 'DVchla':
          #   Cpig_pred_i = 10**(np.matmul(Dspec_norm_i, beta_mean[0,:]) + alpha_mean[0]) - 10**(np.matmul(Dspec_norm_i, beta_mean[11,:]) + alpha_mean[11]) 
                
        #  breakpoint()          
          if conc_space =='log':
              Cpig_pred_i = 10**(np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])
          elif conc_space =='lin':
              Cpig_pred_i = (np.matmul(Dspec_norm_i, beta_mean[i,:]) + alpha_mean[i])

          # pigs below dection (BD) limit of training data
          Cpig_pred_i_BD = Cpig_pred_i[Cpig_i < 0.001]
          Cpig_i_BD = Cpig_i[Cpig_i < 0.001]
          
          # values above detection limit
          Cpig_pred_i = Cpig_pred_i[Cpig_i >= 0.001]
          Cpig_i = Cpig_i[Cpig_i >= 0.001]
          
          if conc_space =='lin':
              Cpig_i = Cpig_i[Cpig_pred_i> 0] 
              Cpig_pred_i = Cpig_pred_i[Cpig_pred_i> 0] 
          
          if i == 1 or  i == 4  or i == 8 or  i == 15:
              # scatter plots
              j=j+1
              plt.subplot(2,2,j)
              
              # fit linear model for regression line log-log space
              x = np.arange(0.0001,4,0.00001)
              mod = scipy.stats.linregress(np.log10(Cpig_i), np.log10(Cpig_pred_i))
              y = 10**(np.log10(x)*mod.slope +mod.intercept)
              
              mod_lin = scipy.stats.linregress(Cpig_i, Cpig_pred_i)
              
              mape = np.nanmean(100*abs(Cpig_pred_i - Cpig_i)/Cpig_i)
              mpe = np.nanmean(100*(Cpig_pred_i - Cpig_i)/Cpig_i)
    
              plt.title(str(pig_mdl[i]))
              if i==1:
                  plt.plot(x,x,color='gray',linestyle='dotted',label='1:1')
                  plt.plot(x,y,color='black',linestyle='dashed',label='OLS')
              else :
                  plt.plot(x,x,color='gray',linestyle='dotted')
                  plt.plot(x,y,color='black',linestyle='dashed')
            
              label = ['N = '  + str(int(len(Cpig_i))) + '\n' +
                       'S_log = ' + str(round_sf(mod.slope)) + ' $\pm$ ' + str(round_sf(2*mod.stderr)) + '\n'
                       'rsq_log = ' + str(round_sf((mod.rvalue)**2)) + '\n'
                       'rsq_lin = ' + str(round_sf((mod_lin.rvalue)**2)) + '\n'
                       'MAPE = '    + str(round_sf(mape)) + ' %' +'\n' +
                       'MPE = '    + str(round_sf(mpe)) + ' %'
                        ][0]
        
              plt.scatter(Cpig_i_BD, Cpig_pred_i_BD, alpha=0.5,label='Simulated',color='black',marker=".",edgecolors='black')
              plt.scatter(Cpig_i, Cpig_pred_i, alpha=0.5,color=bar_colors[i],edgecolors='black')
              
              if i==0:
                  plt.ylim(0.01,4)
                  plt.xlim(0.01,4)
              else:
                 plt.ylim(0.0001,0.4)
                 plt.xlim(0.0001,0.4)
                 
              if i==1:
                  plt.legend(loc=4,fontsize=14)
              plt.gca().set_yscale('log')
              plt.gca().set_xscale('log')
              plt.gca().set_aspect('equal')
              # plt.legend(fontsize=12,framealpha=0.6,markerscale=0)
              plt.text(.01, .99, label, ha='left', va='top', transform=plt.gca().transAxes,fontsize=14)
         
      fig.supylabel('Predicted pigment concentration [mg m$^{-3}$]')
      fig.supxlabel('Measured pigment concentration [mg m$^{-3}$]')
     
      fig.tight_layout() 
      
      filename  =  path_figs + '/'  + file_handle  +'pigs_scatter_2b2.png'
      plt.savefig(filename,dpi=900,bbox_inches='tight')
     
    
      return 


def metric_plot_V2(fitparams,filename):
      
      pig_lab = pig_mdl
      
      N = fitparams['N_test'].values

      rsq_log = fitparams['rsq_log'].values
      rsq = fitparams['rsq'].values
      ae = fitparams[ 'mean_ae'].values 
      ae_pct = fitparams['mean_ae_pct'].values 
      e_pct = fitparams[ 'mean_e_pct'].values 
      
      # standard error on the mean - 100 is default
      rsq_log_std = fitparams['std_rsq_log'].values/np.sqrt(n_permutations)
      rsq_std = fitparams['std_rsq'].values/np.sqrt(n_permutations)
      ae_std = fitparams['std_mean_ae'].values/np.sqrt(n_permutations)
      ae_pct_std = fitparams['std_mean_ae_pct']/np.sqrt(n_permutations) 
      e_pct_std = fitparams['std_e_pct'].values/np.sqrt(n_permutations)
 
      plt.figure(figsize = (18,18))     
      plt.rcParams.update({'font.size': 20})
       
      bar_colors =[]
      for i in range(16):
          bar_colors.append(plt.cm.tab20c(i))
        
      plt.subplot(3,2,1)    
      plt.text(.03, .97, 'A', ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)
      plt.grid(visible=None, which='major', axis='y')
      plt.bar(pig_lab, N, color = bar_colors,edgecolor='black')
      plt.xticks(rotation=90)
      plt.gca().set_xticks([])
      plt.ylabel('$N_{test}$')
      plt.ylim(0,60)
 
      plt.subplot(3,2,2)   
      bar_colors_mask = []
      for i in range(len(pig_lab)):
          if rsq_log[i] <= 0:
                bar_colors_mask.append('white')
                rsq_log[i]=0
                rsq_log_std[i]=0
          else:
                bar_colors_mask.append(bar_colors[i])
      plt.text(.03, .97, 'B', ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)
      plt.grid(visible=None, which='major', axis='y')
      plt.bar(pig_lab,rsq_log,color=bar_colors,edgecolor='black')
      plt.errorbar(pig_lab, rsq_log, yerr=rsq_log_std, fmt="o", color="k")
      plt.xticks(rotation=90)
      plt.gca().set_xticks([])
      plt.ylabel('r$^2$ (log)')
      plt.ylim(0,1)
      
      bar_colors_mask = []
      for i in range(len(pig_lab)):
          if rsq_log[i] <= 0:
                bar_colors_mask.append('white')
                rsq[i]=0
                rsq_std[i]=0
          else:
                bar_colors_mask.append(bar_colors[i])
                
      plt.subplot(3,2,3)    
      plt.text(.03, .97, 'C', ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)
      plt.grid(visible=None, which='major', axis='y')
      plt.grid(visible=None, which='major', axis='y')
      plt.bar(pig_lab,rsq,color=bar_colors_mask,edgecolor='black')
      plt.errorbar(pig_lab,rsq, yerr=rsq_std, fmt="o", color="k")
      plt.xticks(rotation=90)
      plt.ylabel('r$^2$ (lin)')
      plt.gca().set_xticks([])
      plt.grid(visible=None, which='major', axis='y')
      plt.ylim(0,1)
    
      plt.subplot(3,2,4)    
      plt.text(.03, .97, 'D', ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)
      plt.grid(visible=None, which='major', axis='y')      #  for i in range(len(pig_lab)):
      plt.bar(pig_lab,ae,color =bar_colors,edgecolor='black')
      plt.errorbar(pig_lab,ae, yerr=ae_std, fmt="o", color="k")
      plt.xticks(rotation=90)
      plt.gca().set_xticks([])
      plt.ylabel('Mean absolute error [mg m$^{-3}$]')
      plt.ylim(0,0.08)
    
      plt.subplot(3,2,5)   
      ae_pct_max=100
      bar_colors_mask = []
      for i in range(len(pig_lab)):
          if abs(ae_pct[i]) > ae_pct_max:
                bar_colors_mask.append('white')
                ae_pct_std[i]=0
          else:
                bar_colors_mask.append(bar_colors[i])   
      plt.grid(visible=None, which='major', axis='y')
      plt.text(.03, .97, 'E', ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)
      plt.bar(pig_lab,ae_pct,color=bar_colors_mask,edgecolor='black')
      plt.errorbar(pig_lab,ae_pct, yerr=ae_pct_std, fmt="o", color="k")
      plt.xticks(rotation=90)
      plt.ylabel('Mean absolute percentage error [%]')
      plt.ylim(0, 100)
     
      plt.subplot(3,2,6)    
      bar_colors_mask = []
      for i in range(len(pig_lab)):
          if abs(e_pct[i]) > ae_pct_max:
                bar_colors_mask.append('white')
                e_pct_std[i]=0
          else:
                bar_colors_mask.append(bar_colors[i])         
      plt.text(.03, .97, 'F', ha='left', va='top', transform=plt.gca().transAxes,fontsize=24)
      plt.grid(visible=None, which='major', axis='y')
      plt.bar(pig_lab,e_pct,color =bar_colors_mask,edgecolor='black')
      plt.errorbar(pig_lab,e_pct, yerr=e_pct_std, fmt="o", color="k")
      plt.xticks(rotation=90)
      plt.ylabel('Mean percentage error [%]')
      plt.ylim(-100, 100)
      
      plt.tight_layout()
      filename  =  path_figs + '/'  + file_handle + '_metrics' + '.pdf'
      plt.savefig(filename,dpi=600,bbox_inches='tight')

      return
  



if __name__ == '__main__':

    # spectral inputs
    path_root = str(Path(os.getcwd()).parents[1]) + '/'
    path_code =  os.getcwd()
    path_data = path_root + 'Data/'
    path_output = path_data + 'Output/' 
    path_figs = path_root + 'Figs/'
    

    # intialiaze algorithm  training settings/config
    n_permutations = 100 # number of independent model validations to do 
    max_pcs = 30 # max number of spectral principal cpts to incorporate into the model
    mdl_pick_metric = 'mpe' # pick the metric by which to evaluate pigment model optimization. (mae, mpe, mape, rq,rsq_log)
    n_splits = 4; # number of k-folds within each permutation (`k' in Kramer et al. 2022)
    conc_space = 'lin' # concentration model: lin or log
    abs_model = 'aph' # absorption model: aph or ap
    file_handle = abs_model + '_' + conc_space
    
    # load Delta Rrs as pandas dataframe - rows == timestamps, wavelengths = columns
    D_Rrs = pd.read_csv(path_data + 'Output/Delta_Rrs_AMT_' + abs_model + '_input.csv')
    dD_Rrs = pd.DataFrame(np.gradient(np.array(D_Rrs), axis = 1),columns=D_Rrs.keys()) # 1st derivative using second order-accurate central differences
    d2D_Rrs = pd.DataFrame(np.gradient(np.array(dD_Rrs), axis = 1),columns=D_Rrs.keys()) # 2nd derivative using second order accurate central differences
    
    # choose a spectral input. 2nd derivative has best overall performance 
    Dspec = d2D_Rrs

    # load pigments 
    df_hplc = pd.read_csv(path_data + 'Input/AMT_Rrs_HPLC.csv')[0:210]
    
    # define set of pigments for training
    pig_mdl = ['Tchla','Tchlb','ABcaro','But',
               'Hex', 'Allo', 'Diadino','Diato',
               'Fuco', 'Perid','Zea', 'DVchla',
               'Chlc12','Chlc3', 'Neo', 'Viola']
    
    # adjust input data 
    df_hplc = adjust_inputdata(df_hplc, pig_mdl, conc_space) 

    # fit
    summary_fit_params = []  # fit metrics
    summary_alpha = [] # intercept
    summary_beta = [] # wl coefficient
    for i in range(len(pig_mdl)): 
 
        # Observed pigment conc
        Cpig_i = df_hplc[pig_mdl[i]].values
        
        # model training for ith pigment
        print('Training model for ' + str(pig_mdl[i]) + ' : N = ' + str(len(Cpig_i)))
        fit_params_i, alpha_i, beta_i = rrsModelTrain(Dspec, Cpig_i, n_permutations, max_pcs, n_splits, mdl_pick_metric,pig_mdl[i], conc_space)
   
        # append to summary lists
        summary_fit_params.append(fit_params_i)
        summary_alpha.append(alpha_i)
        summary_beta.append(beta_i)
            
    
    stat_keys = ['mean_ae',  'rsq',  'med_ae' , 'mean_ae_pct', 'mean_e_pct' , 'med_ae_pct', 'med_e_pct' ,'rsq_log',  
                 'std_mean_ae', 'std_rsq',  'std_med_ae'   ,  'std_mean_ae_pct', 'std_e_pct' , 'std_med_ae_pct', 'std_med_e_pct',
                 'std_rsq_log',    'N_test']

    
    fitparams = pd.DataFrame(np.stack(summary_fit_params), columns=stat_keys)   
    # Output plots/tables ect

    # convert summary alpha and beta to np arrays, take mean, and geneate summary scatter plots  
    alpha_mean = np.stack(summary_alpha)[:,0] # extracts mean alpha, beta for each pigment
    beta_mean = np.stack(summary_beta)[:,0: len(Dspec.T)] # first 301 elements are  beta
    summary_table_and_scatter(df_hplc, alpha_mean, beta_mean)

  
    # summary metric plots 
    metric_plot_V2(fitparams, file_handle)
  
    # output alpha and beta as np. arrays 
    np.save(path_output + 'alpha_mean' + file_handle, alpha_mean)
    np.save(path_output  + 'beta_mean' + file_handle, beta_mean)
    
    


    
   