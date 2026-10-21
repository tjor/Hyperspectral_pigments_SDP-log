# Hyperspectral_pigments_SDP-log

This codebase is the Plymouth Marine Laboratory (PML) version of `Spectral Derivative Pigments' (SDP) algorithm applied to above-water reflectance data from the Atlantic Meridional Transect.

The code is based on the original MATLAB implementation by Dr. Sasha Kramer, University of Boston: https://github.com/sashajane19/Rrs_pigments, as applied in Kramer et al. 2022. Where possible, we note which scripts are similar.

Kramer, S.J., Siegel, D.A., Maritorena, S., Catlett, D., 2022. Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on global scales. Remote Sensing of Environment 270,1233, https://archimer.ifremer.fr/doc/00743/85550/ 

# Key developments

Key developments in the PML Python version are:

  (i)	A new hyperspectral particulate absorption model configuration is provided, which leads to a reformulation of the bio-optical   optimisation. The default phytoplankton model/ bio-optical model in Kramer et al. 2022 is also included.
  
  (ii)	The statistical (principal component regression) model can now be trained in either logarithmic or linear space. The logarithmic version uses synthetic data to model pigment concentrations below the harmonised detection limit.
  
  (iii)	A Savitzky-Golay filter is used to reduce noise and optimise algorithm performance.

# Example training data and co-location

Co-located Rrs and HPLC pigments from 8 AMT cruises (AMT 23, AMT 25-31) are provided from in ..\Data\Input\AMT_Rrs_HPLC.csv. The Rrs spectra have been oversampled to 1 nm on the interval [400,700] nm. 

The co-location/formatting scripts are ..\Code\Addtional\Rrs_collocate.py and ..\Code\Addtional\HPLC_Rrs_combo.py. These use the AMT Rrs and HPLC NASA SeaBASS files as an input, which are available here: https://seabass.gsfc.nasa.gov/archive/PML/AMT/.

# Running the code

  (i) To train the algrorithm, the user has to first run ..Code\SDP-log\SDP_hyperRrs.py, which derives the resiudal delta_Rrs for each co-located HPLC: Rrs record. 

  (ii) The user then has to train the statistical model ..Code\SDP-log\SDP_pigments_train.py, which derives performance metrics and algorithm coefficeints that can be used on non-trained Rrs

 (iii) An example of applying the coeffiecients to non-trained Rrs (continuous Rrs data from AMT30) ...Code\Addtional\Transect_pigments_AMT30.py


