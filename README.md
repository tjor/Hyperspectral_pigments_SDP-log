# Hyperspectral_pigments_SDP-log

This codebase is the Plymouth Marine Laboratory (PML) version of `Spectral Derivative Pigments' (SDP) algorithm applied to above-water reflectance data from the Atlantic Meridional Transect.

The code is based on the original MATLAB implementation by Dr. Sasha Kramer, University of Boston: https://github.com/sashajane19/Rrs_pigments, as applied in Kramer et al. 2022. Where possible, we note which scripts are similar.

Key developments in the PML Python version are:

  (i)	A new hyperspectral particulate absorption model configuration is provided, which leads to a reformulation of the bio-optical   optimisation. The default phytoplankton model/ bio-optical model in Kramer et al. 2022 is also included.
  
  (ii)	The statistical (principal component regression) model can now be trained in either logarithmic or linear space. The logarithmic version uses synthetic data to model pigment concentrations below the harmonised detection limit.
  
  (iii)	A Savitzky-Golay filter is used to reduce noise and optimise algorithm performance.

Kramer, S.J., Siegel, D.A., Maritorena, S., Catlett, D., 2022. Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on global scales. Remote Sensing of Environment 270,1233, https://archimer.ifremer.fr/doc/00743/85550/ 
