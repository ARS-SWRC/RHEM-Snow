# RHEM-Snow Demo

In this directory, there are two notebooks, RunRHEMSnow_CligenSites.ipynb and RunRhemSnow_Testing.ipynb.  RunRHEMSnow_CligenSites.ipynb runs the demonstrates how to run RHEM-Snow in various configurations (using Cligen or observed forcing variable, and at various sites in Arizona and Idaho).  RunRHEMSnow_CligenSites.ipynb is more of an advanced test, where RHEM-Snow is run with Cligen forcing data, model outputs (rain + snowmelt) are disaggregated into 4 minute timesteps, and output files for Kineros2 (K2) are generated (note that K2 performs the overland flow runoff and erosion estimates in RHEM).

The Scripts directory contains scripts for preprocessing forcing data, running RHEM-Snow, and outputting model data (such as displaying results or saving disaggregated net water input values).  The purpose of each script is explained in their header lines as well as in see section 6 of the technical documentation.

The Data directory (note that upon download, the three zipped files in this directory need to be unzipped first) contains example forcing and validation that are used in the RHEM-Snow examples. 
