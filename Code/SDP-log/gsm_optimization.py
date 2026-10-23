#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PML implementation of bio-optical model (GSM) optimization routines in
Kramer et al. 2022 from https://github.com/sashajane19/Rrs_pigments.

gsm_optimization includes gsm_cost.m and gsm_inversion.m functions 
are in a single script

We conisder a new formulation of the GSM model where ap is an input rather than aph
This means that the CDOM slope becomes a degree of freedom

Tom Jordan: tjor@pml.ac.uk.

"""

import numpy as np
import lmfit as lm
import time

def gsm_cost(params, asw, bbsw, adgstar, bbpstar, A, B, rrs):
             
           
    """
    
    Bio-optical model cost function
    (model based on Maritorena et al. (2002); specific to Case 1  waters)
    Note: aph parameterization is different from Maritorena et al. (2002)
    
    
    Input:
        
        params: 3 component vector of the bio-optical variables which are free parametrs:
       'T_Chl_a, adg_442, bbp_443
       
        asw, bbsw, adgstar, bbpstar, A, B, are all fixed
       
        rrs: observed in-water rrs (transformed from above-water Rrs)
        
 
    Output:
        
        cost metric used in optimization

    """
   
    
    p = params.valuesdict()    
    g = [0.0949, 0.0794] # constants in eq. 2 Gordon et al., 1988
    
    aph = A*p['T_Chl_a']**B # phytoplankton absorption
    adg = p['adg_443']*adgstar # combined nap:cdom absorption
    a_tot = asw + aph + adg # total absorption
    
    bbp =  p['bbp_443']*bbpstar # particualate bakscattering
    bb_tot = bbsw + bbp # total backscatter
    
    x = bb_tot/(a_tot + bb_tot); # model for rrs (in-water reflectance)
    rrs_mod = (g[0] + g[1]*x)*x 
    
    cost = sum((rrs-rrs_mod)**2)

    return cost 


def gsm_cost_ap(params, A, B, ap, wl):
             
           
    """
    
    Bio-optical cost function for ap (AC-S input) - not used in SDP
 
    Input:
        
        params: 2 component vector of the bio-optical variables which are free parameters:
       'T_Chl_a, anap_440
        
       A, B, are fixed
       
    Output:
        
        ap_mod: modelled ap

    """

    p = params.valuesdict()    
     
    ap_mod = A*p['T_Chl_a']**B # phytoplankton absorption
    #anap = p['CNAP_400']*np.exp(-p['SNAP_400']*(wl-400))
    
 
    cost = sum((ap-ap_mod)**2)
    
    return cost


def gsm_cost_cdomfree(params, asw, bbsw, bbpstar, A, B, rrs):
                 
    """
    
    Bio-optical cost function for revised version of bi-optical model 
    (cdom slope free parameter and ap input)
                                                                  
        params: 2 component vector of the bio-optical variables which are free parameters:
       'T_Chl_a, anap_440
        
       A, B, are fixed
 
    Output:

        cost metric used in optimization

    """
    wl = np.arange(400, 701, 1)
    
    p = params.valuesdict()    
    g = [0.0949, 0.0794] # constants in eq. 2 Gordon et al., 1988
    
    
    ap = A*p['T_Chl_a']**B # particulate absorption
    
    ag_s = p['Sg_443'] # CDOM slope - now an optimization param
    agstar= np.exp(ag_s*(wl - 443))
    ag = p['ag_443']* agstar
    
    a_tot = asw + ap + ag # total absorption (note p and g terms rather than ph and dg)
    
    bbp =  p['bbp_443']*bbpstar # particulate backsattering
    bb_tot = bbsw + bbp # total backscatter
    
    x = bb_tot/(a_tot + bb_tot); # model for rrs (in-water reflectance)
    rrs_mod = (g[0] + g[1]*x)*x 
    
    cost = sum((rrs-rrs_mod)**2)

    return cost 


def gsm_reconstruct(params, asw, bbsw, adgstar, bbpstar, A,B, rrs):
             
    """
    Bio-optical model cost function
    (model based on Maritorena et al. (2002); specific to Case 1  waters)
    Note: aph parameterization is different from Maritorena et al. (2002)
    
    
    Input:
        
        params: 3 component vector of the bio-optical variables which are free parametrs:
        'T_Chl_a, adg_442, bbp_443
        rrs: observed in-water rrs (transformed from above-water Rrs)

    Output:
        
        rrs_mod: modelled rrs (in-water Rrs)

    """

    p = params.valuesdict()    
    g = [0.0949, 0.0794] # constants in eq. 2 Gordon et al., 1988

   # print(p['T_Chl_a'])
    
    aph = A*p['T_Chl_a']**B # phytoplankton absorption
    adg = p['adg_443']*adgstar # combined nap:cdom absorption
    a_tot = asw + aph + adg # total absorption
    
    bbp =  p['bbp_443']*bbpstar # particualate bakscattering
    bb_tot = bbsw + bbp # total backscatter
    
    x = bb_tot/(a_tot + bb_tot); # model for rrs (in-water reflectance)
    rrs_mod = (g[0] + g[1]*x)*x

    return rrs_mod


def gsm_reconstruct_ap(params, A, B, ap, wl):
             
           
    """
    
    Bio-optical cost function for AC-S ap: not used in SDP 
 
    Input:
        
        params: 2 component vector of the bio-optical variables which are free parameters:
       'T_Chl_a, anap_440
        
       A, B, are fixed
       
    Output:
        
        ap_mod: modelled ap

    """

    p = params.valuesdict()    
     
    ap_mod = A*p['T_Chl_a']**B # phytoplankton absorption
    #anap = p['CNAP_400']*np.exp(-p['SNAP_400']*(wl-400))
    #ap_mod = ap #+ anap # modelled ap aph = A*p['T_Chl_a']**B 
 
    return ap_mod


def gsm_reconstruct_cdomfree(params, asw, bbsw, bbpstar, A, B, rrs):
                 
    """
    
    Bio-optical cost function for cdom free version of inversion
 
        params: 2 component vector of the bio-optical variables which are free parameters:
       'T_Chl_a, anap_440
        
       A, B, are fixed
 
    Output:

        cost metric used in optimization

    """
    wl = np.arange(400, 701, 1)
    
    p = params.valuesdict()    
    g = [0.0949, 0.0794] # constants in eq. 2 Gordon et al., 1988
    
    
    ap = A*p['T_Chl_a']**B # particulate absorption
    
    ag_s = p['Sg_443'] # CDOM slope - now an optimization param
    agstar= np.exp(ag_s*(wl - 443))
    ag = p['ag_443']* agstar
    
    a_tot = asw + ap + ag # total absorption (note p and g terms rather than ph and dg)
    
    bbp =  p['bbp_443']*bbpstar # particulate backsattering
    bb_tot = bbsw + bbp # total backscatter
    
    x = bb_tot/(a_tot + bb_tot); # model for rrs (in-water reflectance)
    rrs_mod = (g[0] + g[1]*x)*x 

    return rrs_mod


def gsm_invert(params_input, fnc_args):
   
    """
    Bio-optical model optimization, using lmfit opimization (which is built on sci-py)
    
    Numerical (lm) options are currently hard-coded
    
    fnc_args is a tuple for the non-varying spectral vectors
    
    # asw = fnc_args[0]
    # bbsw = fnc_args[1]
    # adgstar = fnc_args[2]
    # bbpstar = fnc_args[3]
    # A = fnc_args[4]
    # B = fnc_args[5]
    # rrs = fnc_args[6]

    """
    lm_options= {'disp': True, 'gtol': 1e-9, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20}
    # options = {'xatol': 1e-9, 'fatol': 1e-9, 'maxfun': 2000, 'maxiter': 2000} - from Kramer MATLAB
  
    # optimization loop is within lm.minimize
    #start_time = time.time()
    model_opt = lm.minimize(gsm_cost, params = params_input, args = fnc_args , method='’leastsq', options=lm_options)  
    #model_opt = lm.minimize(gsm_cost, params = params_input, args = fnc_args , method='cg', options=lm_options)  

    params_opt = model_opt.params
    Chl_opt = model_opt.params['T_Chl_a']
    adg_opt = model_opt.params['adg_443']
    resid = model_opt.residual
    
    # print("--- %s seconds ---" % (time.time() - start_time))

    
    rrs_opt = gsm_reconstruct(params_opt, fnc_args[0], fnc_args[1],fnc_args[2] ,fnc_args[3] ,fnc_args[4], fnc_args[5], fnc_args[6])
 
    return  Chl_opt, adg_opt, resid, rrs_opt


def gsm_invert_cdomfree(params_input, fnc_args):
   
    """
    Bio-optical model optimization, using lmfit opimization (which is built on sci-py)
    
    Numerical (lm) options are currently hard-coded
    
    fnc_args is a tuple for the non-varying spectral vectors
    
    # asw = fnc_args[0]
    # bbsw = fnc_args[1]
    # adgstar = fnc_args[2]
    # bbpstar = fnc_args[3]
    # A = fnc_args[4]
    # B = fnc_args[5]
    # rrs = fnc_args[6]

    """
    lm_options= {'disp': True, 'gtol': 1e-9, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20}
    # options = {'xatol': 1e-9, 'fatol': 1e-9, 'maxfun': 2000, 'maxiter': 2000} - from Kramer MATLAB
  
    # optimization loop is within lm.minimize
    start_time = time.time()
    model_opt = lm.minimize(gsm_cost_cdomfree, params = params_input, args = fnc_args , method='’leastsq', options=lm_options)  
    #model_opt = lm.minimize(gsm_cost, params = params_input, args = fnc_args , method='cg', options=lm_options)  

    params_opt = model_opt.params
 
    Chl_opt = model_opt.params['T_Chl_a']
    S_opt = model_opt.params['Sg_443']
    ag_opt = model_opt.params['ag_443']
    resid = model_opt.residual
    #print("--- %s seconds ---" % (time.time() - start_time))

    
    rrs_opt = gsm_reconstruct_cdomfree(params_opt, fnc_args[0], fnc_args[1],fnc_args[2] ,fnc_args[3] , fnc_args[4], fnc_args[5])
 
    return  Chl_opt, S_opt, ag_opt, resid, rrs_opt


def gsm_invert_ap(params_input, fnc_args):
   
    """
    Bio-optical model optimization, using lmfit opimization (which is built on sci-py)
    
    Numerical (lm) options are currently hard-coded
    
    fnc_args is a tuple for the non-varying spectral vectors
   
    # A = fnc_args[0]
    # B = fnc_args[1]
    # ap = fnc_args[2]
    # wl = fnc_args [3]

    """
    
    lm_options= {'disp': True, 'gtol': 1e-9, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20}
    # options = {'xatol': 1e-9, 'fatol': 1e-9, 'maxfun': 2000, 'maxiter': 2000} - from Kramer MATLAB
  
    # optimization loop is within lm.minimize
    start_time = time.time()
    model_opt = lm.minimize(gsm_cost_ap, params = params_input, args = fnc_args , method='’leastsq', options=lm_options)  
    #model_opt = lm.minimize(gsm_cost, params = params_input, args = fnc_args , method='cg', options=lm_options)  

    params_opt = model_opt.params
    chl_opt = model_opt.params['T_Chl_a']
    resid = model_opt.residual
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    ap_opt = gsm_reconstruct_ap(params_opt, fnc_args[0], fnc_args[1],fnc_args[2],fnc_args[3])
 
    return  chl_opt, resid, ap_opt
