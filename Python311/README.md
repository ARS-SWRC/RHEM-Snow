# RHEM-Snow (coupled with KINEROS2)

This example shows how to run RHEM-Snow, both coupled with Kineros2, as well as as a standalone function.  The standalone version is written purely in python, while the coupled version is written in python (RHEM-Snow) and Fortran (Kineros2).  For the coupled model, there is a compiled fortran program (k2_snow_v2.exe) which runs the RHEM-Snow python codes (snow.py), and has the python program pass the necissary information about rainfall plus snowmelt.  

## Running the Demo
There are batch files for running both the coupled and standalone models (demo_coupledmodel.bat and 
demo_standalonemodel.bat).  The batch files contain examples of how to run k2_snow_v2.exe and snow.py (which are
in the ModelCodes directory:

k2_snow_v2 <Kineros PAR File> <KINEROS Output Directory> <KINEROS CN> <CLIGEN stm file> <RHEM-Snow Output Directory> <Soil Type> <Slope> <Aspect>
where <Kineros PAR File> is the hillslope parameter file for Kineros2
      <KINEROS Output Directory> is the output directory where the Kineros output files are written
	  <KINEROS CN> is the curve number used for the kineros simulation
	  <CLIGEN stm file> is the CLIGEN storm file
	  <RHEM-Snow Output Directory> is the output directory where RHEM-snow output files are written
	  <Soil Type> is the soil texture used by RHEM-Snow (note that this is only important for continuous simulation of soil moisture)
	  <Slope> is the Slope angle (in degrees)
	  <Aspect> is the Aspect (degrees from north, measured clockwise)
	  
python snow.py <CLIGEN stm file> <RHEM-Snow Output Directory> <Soil Type> <Slope> <Aspect>
where <CLIGEN stm file> is the CLIGEN storm file
	  <RHEM-Snow Output Directory> is the output directory where RHEM-snow output files are written
	  <Soil Type> is the soil texture used by RHEM-Snow (note that this is only important for continuous simulation of soil moisture)
	  <Slope> is the Slope angle (in degrees)
	  <Aspect> is the Aspect (degrees from north, measured clockwise)


## Model Codes: 
demo_coupledmodel.bat - bat file for running the coupled model
demo_standalonemodel.bat - bat file for running the standalone model (RHEM-Snow only)
snow.py - Rhem-snow python codes
k2_snow_v2.exe - executable for running the combined model
modpaths.txt - required paths to tell k2_snow_v2.exe where the python installation is located

## Model Input Files:
SiteSpecificParameters.csv - contains site specific parameters for RHEM-Snow.  Currently, the only site specific parameter is 
the rain-snow threshold
wy485055.stm - the CLIGEN storm file that contains the meteorological forcing data
data_input_id_17159_485055_201156019020701R2.PAR - the hillslope parameter file for Kineros2

## Model Ouput files:
Kineros2 will output two files, with the name of the par file .csv (which contains annual statistics for runoff and 
erosion) and _events.csv (which contains information about runoff and erosion for each individual event). RHEM-Snow 
also optionally generates a csv file that is outputted contains daily statistics about the mass balance (Rainfall 
Off Snow, Rainfall On Snow, Snowfall, SWE, Sublimation, Snowmelt, Net Water Input), as well as information about 
soil ice, saturation, and maximum rainfall/snowmelt intensity). Note that if the <RHEM-Snow Output Directory> is set 
to "None", then no RHEM-Snow output will be generated.

In this example, the kineros outputs are saved in Output/data_input_id_17159_485055_201156019020701R2.csv and 
Output/data_input_id_17159_485055_201156019020701R2_events.csv and the mass balance file from RHEM-Snow is
Output/wy485055_table.csv.

## Documentation Files:
Documentation.docx - RHEM-Snow Documentation
Readme.txt - this file

## Notes
Important: This version of RHEM-Snow only works with python 3.11, and the python installation paths need to be 
set in modpaths.txt

RHEM-Snow Requires the following python modules: sys, os, numpy, datetime, scipy, copy, time.  Most packages are standard 
but numpy and scipy might need to be installed separately.  This version of RHEM-Snow was tested with numpy v1.25.2 and 
scipy v1.11.2.  Different versions are likely to give the same results but to ensure consistency, it is recommended that a 
user renames the existing output files and runs the demo (double clicks demo_coupledmodel.bat and demo_standalonemodel.bat) 
and verifies that the o files generated on the user's machine are the same.
