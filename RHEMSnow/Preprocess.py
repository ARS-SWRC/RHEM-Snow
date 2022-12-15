import numpy as np
from datetime import datetime, timedelta
import scipy.io as sio

# Patrick Broxton [broxtopd@arizona.edu] - December 2022

def default_model_pars(nlocs):
    # Function to populate RHEM-Snow Parameters with their default values
    # [these can be replaced by site level parameters in the .par or .mat
    # files, or in the code running the model]
    # 
    # model_pars: structure containing model parameter values

    model_pars = {}

    # Spatial Parameters [note, these parameters, especially latitude and
    # elevation are generally overwritten with actual station values]
    model_pars['latitude'] = np.ones(nlocs) * 45.                   # Latitude [degrees]
    model_pars['elevation'] = np.ones(nlocs) * 1000.                # Elevation [meters]
    model_pars['slope'] = np.ones(nlocs) * 0.                       # Slope [degrees]
    model_pars['aspect'] = np.ones(nlocs) * 0.                      # Aspect [degrees from north, clockwise]
    model_pars['lai'] = np.ones(nlocs) * 0.                         # Leaf Area Index [m2/m2]

    # Initial Conditions
    model_pars['swe_i'] = np.ones(nlocs) * 0.                       # Initial snow water equivalent [mm]
    model_pars['swe_age_a_i'] = np.ones(nlocs) * 0.                 # Initial soil moisture [cm3/cm3]
    model_pars['density_i'] = np.ones(nlocs) * 0.1                  # Initial snow density [g/cm3]
    model_pars['cansnowstor_i'] = np.ones(nlocs) * 0.               # Initial canopy intercepted snow storage [mm]
    model_pars['Tm_i'] = np.ones(nlocs) * 0.                        # Initial snowpack temperature [C]
    # model_pars['albedosnow_i'] = np.ones(nlocs) * 0.5
    model_pars['T_soil_i'] = np.ones(nlocs) * 20.                   # Initial soil layer temperature [C]
    model_pars['ice_fraction_soil_i'] = np.ones(nlocs) * 0.         # Initial soil ice fraction [cm3/cm3]
    model_pars['sm_i'] = np.ones(nlocs) * 0.3                       # Initial soil moisture [cm3/cm3]
    model_pars['q_vadose_i'] = np.ones(nlocs) * 0.5                 # Initial water content in the vadose zone (below the top soil layer) [mm]
    model_pars['q_phreatic_i'] = np.ones(nlocs) * 0.1               # Initial water content in the preatic zone [mm]
    
    # Forcing Parameters 
    model_pars['srad_mult'] = np.ones(nlocs) * 1.                   # Shortwave Radiation Multiplier [-]
    model_pars['lrad_mult'] = np.ones(nlocs) * 1                    # Longwave Radiation Multiplier [-]
    model_pars['snow_mult'] = np.ones(nlocs) * 1.                   # Snowfall Multiplier [-]
    model_pars['temp_adj'] = np.ones(nlocs) * 0.                    # Adjustment to air temperature [C]
    model_pars['RainThresh'] = np.ones(nlocs) * 0.                  # Rain / Snow Transition Temperature [C] (at which rain and snow make up equal parts)
    model_pars['RainThresh_dh'] = np.ones(nlocs) * 2.               # Range of temperatures over which the rain/snow transition occurs [C]
    model_pars['CloudTransmission'] = np.ones(nlocs) * 0.2        	# Fraction of potential solar radiation to be considered cloudy
    model_pars['use_tdew_ppm'] = False                              # Flag to specify whether to use dewpoint temperature instead of air temperature when computing the rain-snow transition

    # Snow Albedo Parameters
    model_pars['albedo_snow_reset'] = np.ones(nlocs) * 10.          # Size of daily snowfall to reset the snowpack surface age [mm]
    model_pars['albedo_decay'] = np.ones(nlocs) * 0.05              # Snow albedo decay rate [fraction/day]
    model_pars['albedo_i'] = np.ones(nlocs) * 0.85                  # Albedo of fresh snowpack [-]
    model_pars['minalbedo'] = np.ones(nlocs) * 0.4                  # Minimum snowpack albedo [-]
    
    # Snow Density Parameters
    model_pars['density_min'] = np.ones(nlocs) * 0.1                # Density of new snow [g/cm3]
    model_pars['density_max'] = np.ones(nlocs) * 0.5                # Maximum snow density [g/cm3]
    model_pars['apar'] = np.ones(nlocs) * 0.02                      # Snow densification rate due to age [Fraction / day]
    model_pars['dpar'] = np.ones(nlocs) * 0.002                     # Snow densfication rate due to overburdin [Fraction / cm [SWE]]
    model_pars['rpar'] = np.ones(nlocs) * 0.05                      # Snow densfication rate due to liquid in snowpack [Fraction when isothermal snowpack]

    # Snow Interception Parameters
    model_pars['melt_drip_par'] = np.ones(nlocs) * 0.5              # Melt Drip Rate [mm/day per deg-C above freezing]
    model_pars['snow_unload_par'] = np.ones(nlocs) * 0.3            # Fraction of canopy snow that unloads each day [-]
    model_pars['canopy_sub_mult'] = np.ones(nlocs) * 50.            # Canopy sublimation multiplier applied to potential sublimation rate [which is computed for snowpack surface] [-]

    # Miscellaneous Snowpack Parameters
    model_pars['Ch'] = np.ones(nlocs) * 1                           # Multiplier applied to sensible heating equation
    model_pars['ground_sub_mult'] = np.ones(nlocs) * 1              # Ground sublimation multiplier applied to potential sublimation rate [-]     	
    model_pars['sroughness'] = np.ones(nlocs) * 5E-5                # Snow surface roughness length [m]
    model_pars['windlevel'] = np.ones(nlocs) * 10.                  # Height of windspeed measurement [m]
    model_pars['fstab'] = np.ones(nlocs) * 0.                       # Stability parameter [-] (0-1; 1:  totally on Richardson number corrections; 0: assumes neutral atmospheric stability)
    model_pars['kappa_snow'] = np.ones(nlocs) * 0.1                	# Snow thermal conductivity [W/m/K]
    model_pars['kappa_soil'] = np.ones(nlocs) * 0.5              	# Soil thermal conductivity [W/m/K]
    model_pars['tempdampdepth'] = np.ones(nlocs) * 1.               # Temperature at damping depth underneath a snowpack [C]
    model_pars['dampdepth'] = np.ones(nlocs) * 1.                   # Damping depth [m]
    model_pars['albedo_0'] = np.ones(nlocs) * 0.2                   # Albedo of snow-free ground
    model_pars['groundveght'] = np.ones(nlocs) * 0.05               # Ground Vegetation height [m]

    model_pars['PET_Mult'] = np.ones(nlocs) * 1.0                   # Potential evapotranspirataion multiplier [-]
    
    model_pars['max_infil_mult'] = np.ones(nlocs) * 0.7             # Fraction of incoming water that becomes infiltration excess when the soil moisture is above the level specified by sm_max_infil [-]
    model_pars['sm_max_infil'] = np.ones(nlocs) * 0.4               # Soil moisture content below which infiltration excess runoff is minimized [-]
    model_pars['sm_min_infil'] = np.ones(nlocs) * 0.2               # Soil moisture content above which infiltration excess runoff is maximized [-]
    model_pars['H'] = np.ones(nlocs) * 250.                         # Thickness of surface soil layer [mm]
    
    model_pars['ssat'] = np.ones(nlocs) * 0.451                     # Saturated water content [-]
    model_pars['psi_s'] = np.ones(nlocs) * 146.                     # Soil air entry pressure [cm]
    model_pars['g'] = np.ones(nlocs) * 20.                          # g Parameter for calculating residual soil moisture
    model_pars['b_soil'] = np.ones(nlocs) * 5.39                    # Pore size distribution index [-]
    model_pars['k_soil'] = np.ones(nlocs) * 600.48                  # Saturated hydraulic conductivity [mm/day]
    model_pars['wp'] = np.ones(nlocs) * 0.12                        # Wilting Point [-]
    model_pars['cmc'] = np.ones(nlocs) * 0.30                       # Critical Moisture Content [-]

    model_pars['coef_vadose'] = np.ones(nlocs) * 0.1                # Vadose Zone Reservoir Decay Parameter (multiplier) [-]
    model_pars['coef_vadose_exp'] = np.ones(nlocs) * 1              # Vadose Zone Reservoir Decay Parameter (exponent) [-]
    model_pars['coef_vadose2phreatic'] = np.ones(nlocs) * 0.02      # Leakage Rate between Vadose and Phreatic Reservoirs [-]
    model_pars['coef_phreatic'] = np.ones(nlocs) * 0.001            # Phreatic Zone Decay Parameter (multiplier) [-]   
    model_pars['coef_phreatic_exp'] = np.ones(nlocs) * 1            # Phreatic Zone Decay Parameter (exponent) [-]

    return(model_pars)

def get_soil_pars(model_pars):
                        # Soil Table (Clapp and Hornberger 1978, G, WP, and FC from kineros 2 manual)
                        #                   MCF     b       psi_s   Log_psi psi_f   ths     Ks      S`          G       WP      FC
                        #                                   cm              cm      cm3/cm3 cm/min  cm/min^1/2  cm
    SoilTable = data = [['Sand',            0.03,   4.05,   12.1,   3.50,   4.66,   0.395,  1.056,  1.52,       5.,     0.08,   0.21],
                        ['Loamy sand',      0.06,   4.38,   9.0,    1.78,   2.38,   0.410,  0.938,  1.04,       7.,     0.13,   0.29],
                        ['Sandy loam',      0.09,   4.9,    21.8,   7.18,   9.52,   0.435,  0.208,  1.03,       13.,    0.21,   0.46],
                        ['Silt loam',       0.14,   5.3,    78.6,   56.6,   75.3,   0.485,  0.0432, 1.26,       11.,    0.25,   0.58],
                        ['Loam',            0.19,   5.39,   47.8,   14.6,   20.0,   0.451,  0.0417, 0.693,      20.,    0.27,   0.66],
                        ['Sandy clay loam', 0.28,   7.12,   29.9,   8.63,   11.7,   0.420,  0.0378, 0.488,      26.,    0.37,   0.64],
                        ['Silty clay loam', 0.34,   7.75,   35.6,   14.6,   19.7,   0.477,  0.0102, 0.310,      26.,    0.42,   0.69],
                        ['Clay loam',       0.34,   8.52,   63.0,   36.1,   48.1,   0.476,  0.0147, 0.537,      35.,    0.44,   0.78],
                        ['Sandy clay',      0.43,   10.4,   15.3,   6.16,   8.18,   0.426,  0.0130, 0.223,      30.,    0.56,   0.79],
                        ['Silty clay',      0.49,   10.4,   49.0,   17.4,   23.0,   0.492,  0.0062, 0.242,      38.,    0.52,   0.81],
                        ['clay',            0.63,   11.4,   40.5,   18.6,   24.3,   0.482,  0.0077, 0.268,      41.,    0.57,   0.83]]
                        
    SoilTable = np.array(SoilTable)
    
    for i in range(len(model_pars['Soil'])):
        table_loc = np.where(SoilTable[:,0] == model_pars['Soil'][i])
        model_pars['ssat'][i] = float(SoilTable[table_loc[0][0],6])
        model_pars['b_soil'][i] = float(SoilTable[table_loc[0][0],2])
        model_pars['k_soil'][i] = float(SoilTable[table_loc[0][0],7]) * 1440 * 10
        model_pars['psi_s'][i] = float(SoilTable[table_loc[0][0],4]) * 10
        model_pars['g'][i] = float(SoilTable[table_loc[0][0],9])
        model_pars['wp'][i] = float(SoilTable[table_loc[0][0],10]) * float(SoilTable[table_loc[0][0],6])
        model_pars['cmc'][i] = float(SoilTable[table_loc[0][0],11]) * float(SoilTable[table_loc[0][0],6])
        
    return(model_pars)

def solarradiation(doys,L,slop,asp):
    # PUPROSE: Calculate solar radiation for a digital elevation model (DEM)
    #          over one year for clear sky conditions in W/m2
    # -------------------------------------------------------------------
    # USAGE: srad = solarradiation(doys,L,slop,asp)
    # where: doys are the days to calculate solar radiation for
    #        L is the latitude
    #        slop is the slope in degrees
    #        asp in the aspect in degrees from north, clockwise
    #
    #       srad is the solar radiation in W/m2 over one year per grid cell
    #
    # EXAMPLE:
    #       srad = solarradiation(peaks(50)*100,54.9:-0.1:50,1000,0.2);
    #       - calculates the solar radiation for an example 50x50 peak surface.
    #
    # See also: GRADIENT, CART2POL
    #
    # Note: Follows the approach of Kumar et al 1997. Calculates clear sky
    #       radiation corrected for the incident angle (selfshading) plus
    #       diffuse and reflected radiation. Insolation is depending on time of year (and day), 
    #       latitude, elevation, slope and aspect. 
    #       Relief shading is not considered.
    #
    # Reference: Kumar, L, Skidmore AK and Knowles E 1997: Modelling topographic variation in solar radiation in 
    #            a GIS environment. Int.J.Geogr.Info.Sys. 11(5), 475-497
    #
    #
    # Felix Hebeler, Dept. of Geography, University Zurich, May 2008.

    # Modified for use with RHEM-Snow by Patrick Broxton (to accept already
    # given slope and aspect, to separate solar irradiance for each day of
    # year, separately) by Patrick Broxton (broxtopd@arizona.edu) - March 2010

    # PDB: get aspect into the expected convention
    asp2 = asp - 180
    asp = asp + 180
    asp[asp > 360] = asp2[asp > 360]
    # parameters
    r = 0.20            # ground reflectance coefficient (more sensible to give as input)
    n = 1;              # timestep of calculation over sunshine hours: 1=hourly, 0.5=30min, 2=2hours etc
    tau_a = 365         # length of the year in days
    S0 = 1367           # solar constant W m^-2   default 1367
    dr= 0.0174532925    # degree to radians conversion factor
    L=L*dr              # convert to radians
    fcirc = 360 * dr    # 360 degrees in radians 
    
    srad = np.ones([len(doys), len(L)]) * np.nan
    day_length = np.ones([len(doys), len(L)]) * np.nan

    ## some setup calculations
    sinL = np.sin(L)
    cosL = np.cos(L)
    tanL = np.tan(L)
    sinSlop = np.sin(slop*2*np.pi/360)
    cosSlop = np.cos(slop*2*np.pi/360)
    cosSlop2 = cosSlop * cosSlop
    sinSlop2 = sinSlop * sinSlop
    sinAsp = np.sin(asp*2*np.pi/360)
    cosAsp = np.cos(asp*2*np.pi/360)
    term1 = (sinL * cosSlop - cosL * sinSlop * cosAsp)
    term2 = (cosL * cosSlop + sinL * sinSlop * cosAsp)
    term3 = sinSlop * sinAsp
    
    ## loop over year
    for d in range(1, 366+1):
        #display(['Calculating melt for day ',num2str(d)])  
        # clear sky solar radiation
        I0 = S0 * (1 + 0.0344 * np.cos(fcirc*d/tau_a))  # extraterrestrial rad per day
        # sun declination dS
        dS = 23.45 * dr * np.sin(fcirc * ( (284+d)/tau_a ) ) #in radians, correct/verified
        # angle at sunrise/sunset
        hsr = np.arccos(-tanL * np.tan(dS)).real  # angle at sunrise
        # this only works for latitudes up to 66.5 deg N! Workaround:
        # hsr[hsr<-1)=acos(-1);
        # hsr[hsr>1)=acos(1);
        It_0 = 12 * (1 + hsr/np.pi) - 12 * (1 - hsr/np.pi)              # calc daylength
        It = np.round(12 * (1 + hsr/np.pi) - 12 * (1 - hsr/np.pi))      # calc daylength
        It[np.isnan(It)] = 0
        ##  daily loop
        I = 0
        for t in np.arange(1, np.max(It[:]) + 1, n):  # loop over sunshine hours
            # if accounting for shading should be included, calc hillshade here
            # hourangle of sun hs  
            hs = hsr - (np.pi * t / It)               # hs(t)
            #solar angle and azimuth
            sinAlpha = sinL * np.sin(dS) + cosL * np.cos(dS) * np.cos(hs)   # solar altitude angle
            # correction  using atmospheric transmissivity taub_b
            M = np.sqrt(1229 + ((614 * sinAlpha))**2) - 614 * sinAlpha      # Air mass ratio
            tau_b = 0.56 * (np.exp(-0.65 * M) + np.exp(-0.095 * M))
            tau_d = 0.271 - 0.294 * tau_b   # radiation diffusion coefficient for diffuse insolation
            tau_r = 0.271 + 0.706 * tau_b   # reflectance transmitivity
            # correct for local incident angle
            cos_i = (np.sin(dS) * term1) + (np.cos(dS) * np.cos(hs) * term2) + (np.cos(dS) * term3 * np.sin(hs))
            Is = I0 * tau_b # potential incoming shortwave radiation at surface normal (equator)
            # R = potential clear sky solar radiation W m2
            R = Is * cos_i
            R[R < 0] = 0        # kick out negative values
            Id = I0 * tau_d * cosSlop2 / 2 *sinAlpha        #diffuse radiation;
            Ir = I0 * r * tau_r * sinSlop2 / 2 * sinAlpha # reflectance
            R = R + Id + Ir
            R[R < 0] = 0
            I = I + R * It_0 / It       # solar radiation per day (sunshine hours)
            # PDB - Correct for rounding error when discretizing into hours

        I = I/24
        #  PDB add up radiation part melt for every day
        NHours = It_0
        srad[doys == d, :] = np.tile(I, [np.sum(doys == d), 1])
        day_length[doys == d, :] = np.tile(NHours, [np.sum(doys == d), 1])

    return srad, day_length

def get_forcing_cligen(forcing_files,model_pars):

    # Function to get cligen forcing data from one or more cligen files, prepare 
    # the data for RHEM-snow, and get their associated site-specific parameter 
    # values from parameter files [if any]
    #
    # Inputs
    #   forcing_files is a list of files to read forcing data from [each will be
    #   treated as a separate location]
    #   model_pars: structure with all of the model parameters 
    # Outputs
    #   TS_vec: an array of matlab timestamps that the data is valid for
    #   forcing_data: structure with all of the model parameters

    # Load Cligen Data

    i = 0

    non_blank_count = 0
    with open(forcing_files[0]) as file:
        for line in file:
            if line.strip():
                non_blank_count += 1
    nrows = non_blank_count-15
    nlocs = len(forcing_files)
    
    day = np.ones([nrows, nlocs]) * np.nan
    mon = np.ones([nrows, nlocs]) * np.nan
    year = np.ones([nrows, nlocs]) * np.nan
    prcp = np.ones([nrows, nlocs]) * np.nan
    stmdur = np.ones([nrows, nlocs]) * np.nan
    timep = np.ones([nrows, nlocs]) * np.nan
    ip = np.ones([nrows, nlocs]) * np.nan
    tmax = np.ones([nrows, nlocs]) * np.nan
    tmin = np.ones([nrows, nlocs]) * np.nan
    srad = np.ones([nrows, nlocs]) * np.nan
    wind = np.ones([nrows, nlocs]) * np.nan
    tdpt = np.ones([nrows, nlocs]) * np.nan
    
    for cligen_file in forcing_files:
    
        print('Reading data from ' + forcing_files[i])

        f = open(cligen_file)
        for t in range(5):
            tline = f.readline()
        f.close
        
        fields = tline.split()
        model_pars['latitude'][i] = fields[0]
        model_pars['elevation'][i] = fields[2]

        cligen_data = np.loadtxt(open(cligen_file, 'rb'), skiprows=15)
        day[:, i] = cligen_data[:, 0]     # day of simulation
        mon[:, i] = cligen_data[:, 1]     # month of simulation
        year[:, i] = cligen_data[:, 2]    # year of simulation
        prcp[:, i] = cligen_data[:, 3]    # daily precipitation amount [mm of water]
        stmdur[:, i] = cligen_data[:, 4]  # duration of precipitation [hr]
        timep[:, i] = cligen_data[:, 5]   # ratio of time to rainfall peak / rainfall duration
        ip[:, i] = cligen_data[:, 6]      # ratio of maximum rainfall intensity / average rainfall intensity
        tmax[:, i] = cligen_data[:, 7]    # maximum daily temperature [degrees C]
        tmin[:, i] = cligen_data[:, 8]    # minimum daily temperature [degrees C]
        srad[:, i] = cligen_data[:, 9]    # daily solar radiation [langleys/day] - real
        wind[:, i] = cligen_data[:, 10]   # wind speed
        tdpt[:, i] = cligen_data[:, 12]   # dew point temperature [degrees C]
        i = i+1
    
    print('Processing forcing data')
    
    # Fix problem where leap days appear on 100, 200, and 300th year
    locs = np.logical_and(np.mod(year,400) > 0, np.logical_and(np.mod(year,100) == 0, np.logical_and(mon == 2, day == 29)))
    day[locs] = 28
    
    TS_vec = []
    for i in range(len(day)):
        TS_vec.append(datetime(int(year[i, 0]),int(mon[i, 0]),int(day[i, 0])))
    doys = []
    for TS in TS_vec:
        doys.append(TS.timetuple().tm_yday)
    TS_vec = np.array(TS_vec)
    doys = np.array(doys)

    # Humidity Conversions
    tmean = (tmax + tmin) / 2       # Average daily temperature (degrees C)
    esat = 0.6108 * np.exp(17.27 * tmean / (237.3 + tmean)) * 1000    # Saturated vapor pressure (Pa)            
    vapp = 0.6108 * np.exp(17.27 * tdpt / (237.3 + tdpt)) * 1000;     # Vapor pressure (Pa)
    rh = vapp / esat * 100         # Relative Humidity

    tmean = tmean + model_pars['temp_adj']
    esat = 0.6108 * np.exp(17.27 * tmean / (237.3 + tmean)) * 1000  # Saturated vapor pressure (Pa)
    vapp = esat * rh/100


    # Partition Rainfall and Snowfall
    # Use a rainfall threshold (smooth function from T+1 to T-1)
    dh = 1
    
    rainthresh_tmax = model_pars['RainThresh'] + model_pars['RainThresh_dh']/2
    rainthresh_tmin = model_pars['RainThresh'] - model_pars['RainThresh_dh']/2
    dx = rainthresh_tmax - rainthresh_tmin
    if model_pars['use_tdew_ppm']:
        T = tdpt
    else:
        T = tmean
    
    f_s = np.zeros(T.shape)
    for i in range(len(T[:,0])):
        T_i = T[i,:]
        fs = 1 - ((dh/dx) * (T[i,:]-rainthresh_tmin) - (dh * np.sin((2*np.pi/dx) * (T[i,:]-rainthresh_tmin))) / (2*np.pi))
        fs[fs < 0] = 0
        fs[fs > 1] = 1
        f_s[i,:] = fs

    rainfall = prcp * (1-f_s)  # daily rainfall amount (mm of water) 
    snowfall = prcp * f_s      # daily snowfall amount (mm of water) 

    # Apply the snowfall multiplier if specified
    if len(model_pars['snow_mult']) > 1:
        for i in range(len(snowfall[0,:])):
            snowfall[:,i] = snowfall[:,i] * model_pars['snow_mult'][i]
    else:
        snowfall = snowfall * model_pars['snow_mult'][0]

    # Potential solar radiation and solar forcing index
    srad = srad * 0.484583         # Convert forcing solar radiation to W/m2
    # Compute potential solar forcing on flat vs inclined surface (for
    # correction on differently oriented slopes)
    R0, day_length = solarradiation(doys, model_pars['latitude'], model_pars['slope']*0, model_pars['aspect'])
    Rs, dummy = solarradiation(doys, model_pars['latitude'], model_pars['slope'], model_pars['aspect'])
    SFI = Rs / R0
    
    # Correct for min and max values based on observed solar data
    T_summer = np.amax(srad[np.logical_and(doys > 150, doys < 200), :], axis=0) / np.amax(R0[np.logical_and(doys > 150, doys < 200), :], axis=0)
    T_winter = np.amax(srad[np.logical_or(doys > 350, doys < 10), :], axis=0) / np.amax(R0[np.logical_or(doys > 350, doys < 10), :], axis=0)
    frac = np.ones(R0.shape) * np.nan
    for i in range(len(R0[0, :])):
        frac[:, i] = (R0[:, i] - np.min(R0[:, i])) / (np.max(R0[:, i]) - np.min(R0[:, i]))

    T_ti = frac * T_summer + (1-frac) * T_winter
    R0 = R0 * T_ti

    # Incoming Longwave Radiation

    # Compute the longwave radiation input by first, computing 
    # cloud fraction (compare observed and potential solar radiation)
    CF = np.ones(R0.shape) * np.nan
    for i in range(len(srad[0, :])):
        cf = (1 - (srad[:, i] / R0[:, i])) * (1 - model_pars['CloudTransmission'][i])
        cf[cf < 0] = 0
        cf[cf > 1] = 1
        CF[:, i] = cf

    CF = np.maximum(np.minimum(1,prcp/25.4), CF)
    # Then, calculate incoming longwave radiation
    Eacls = 1.08 * (1 - np.exp(-(vapp/100)**(tmean/2016)))
    Ea = CF + (1-CF) * Eacls
    lrad = Ea * 5.67E-8 * (tmean + 273.15)**4

    # Apply multiplier to shortwave radiation if specified
    for i in range(len(srad[0, :])):
        srad[:, i] = srad[:, i] * SFI[:, i] * model_pars['srad_mult'][i]

    # Apply longwave radiation multiplier (if specified)
    for i in range(len(lrad[0, :])):
        lrad[:, i] = lrad[:, i] * model_pars['lrad_mult'][i]
            
    
    ## Put data in output structure
    forcing_data = {}
    forcing_data['day'] = day
    forcing_data['mon'] = mon
    forcing_data['year'] = year
    forcing_data['tmean'] = tmean
    forcing_data['wind'] = wind
    forcing_data['srad'] = srad
    forcing_data['day_length'] = day_length
    forcing_data['lrad'] = lrad
    forcing_data['vapp'] = vapp
    forcing_data['rh'] = rh
    forcing_data['rainfall'] = rainfall
    forcing_data['snowfall'] = snowfall
    forcing_data['stmdur'] = stmdur
    forcing_data['timep'] = timep
    forcing_data['ip'] = ip

    rad = forcing_data['srad'] / 1000000 * 3600 * 24    # MJ/day/m2
    latent_heat_flux = 2.26                             # MJ/kg  (2260 kj/g)
    rho = 1000                                          # Water density in kg/m3
    # Potential evapotranspiration is computed in m/day
    forcing_data['PET'] = rad / latent_heat_flux / rho * ((forcing_data['tmean']+5) / 100)
    locs = forcing_data['tmean'] < -5                   # mm/day
    forcing_data['PET'][locs] = 0
    forcing_data['PET'] = forcing_data['PET'] * 1000   # PET is transformed in mm
    
    return TS_vec, forcing_data

def date_range(StartDate, EndDate):
    start = datetime.strptime(StartDate, '%d-%b-%Y')
    end = datetime.strptime(EndDate, '%d-%b-%Y')
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days

def get_forcing_mat(forcing_file, model_pars, use_forcing_partitioned_precip, use_forcing_lrad):
    # Function to get cligen observed forcing data from a .mat file, prepare
    # the data for RHEM-snow, and get their associated site-specific parameter
    # values from parameter files (if any).
    #
    # Inputs
    #   forcing_file is the path to a file to read forcing data from
    #   model_pars: structure with all of the model parameters
    #   use_forcing_partitioned_precip: Flag to use pre-partitioned precip data
    #       (If false, uses a temperature threshold instead)
    #   use_forcing_lrad: Flag to use longwave radiation from the forcing data
    #       (If false, model estimates longwave radiation)
    # Outputs
    #   TS_vec: an array of matlab timestamps that the data is valid for
    #   forcing_data: structure with all of the model parameters
    #   modelpars: structure with all of the model parameters (included as an
    #   output because under some circumstances, this function will modify
    #   the modelpars structure)
    #
    # Patrick Broxton (broxtopd@arizona.edu) - December 2022

    # Load Cligen Data
    md = sio.loadmat(forcing_file)

    tmax = md['tmax']
    tmin = md['tmin']
    srad = md['srad']
    wind = md['wind']
    tdpt = md['tdpt']

    print('Processing forcing data')

    TS_vec = date_range(md['StartDate'][0], md['EndDate'][0])
    day = []
    mon = []
    year = []
    doys = []
    for TS in TS_vec:
        doys.append(TS.timetuple().tm_yday)
        day.append(TS.day)
        mon.append(TS.month)
        year.append(TS.year)

    TS_vec = np.array(TS_vec)
    doys = np.array(doys)

    # Humidity Conversions
    tmean = (tmax + tmin) / 2 # Average daily temperature (degrees C)
    esat = 0.6108 * np.exp(17.27 * tmean / (237.3 + tmean)) * 1000  # Saturated vapor pressure (Pa)
    vapp = 0.6108 * np.exp(17.27 * tdpt / (237.3 + tdpt)) * 1000;  # Vapor pressure (Pa)
    rh = vapp / esat * 100  # Relative Humidity
    
    tmean = tmean + model_pars['temp_adj']
    esat = 0.6108 * np.exp(17.27 * tmean / (237.3 + tmean)) * 1000  # Saturated vapor pressure (Pa)
    vapp = esat * rh/100

    # Partition Rainfall and Snowfall
    # Use a rainfall threshold (smooth function from T+1 to T-1)
    if ~use_forcing_partitioned_precip:
        dh = 1
        rainthresh_tmax = model_pars['RainThresh'] + model_pars['RainThresh_dh'] / 2
        rainthresh_tmin = model_pars['RainThresh'] - model_pars['RainThresh_dh'] / 2
        dx = rainthresh_tmax - rainthresh_tmin
        T = tdpt
        f_s = np.zeros(T.shape)
        for i in range(len(T[:, 0])):
            T_i = T[i, :]
            fs = 1 - ((dh / dx) * (T[i, :] - rainthresh_tmin) - (
                        dh * np.sin((2 * np.pi / dx) * (T[i, :] - rainthresh_tmin))) / (2 * np.pi))
            fs[fs < 0] = 0
            fs[fs > 1] = 1
            f_s[i, :] = fs

        prcp = md['prcp']
        rainfall = prcp * (1 - f_s)  # daily rainfall amount (mm of water)
        snowfall = prcp * f_s  # daily snowfall amount (mm of water)
    else:
        snowfall = md['snowfall']
        rainfall = md['rainfall']

    # Apply the snowfall multiplier if specified
    if len(model_pars['snow_mult']) > 1:
        for i in range(len(snowfall[0, :])):
            snowfall[:, i] = snowfall[:, i] * model_pars['snow_mult'][i]
    else:
        snowfall = snowfall * model_pars['snow_mult'][0]

    # Potential solar radiation and solar forcing index
    srad = srad * 0.484583  # Convert forcing solar radiation to W/m2
    # Compute potential solar forcing on flat vs inclined surface (for
    # correction on differently oriented slopes)
    R0, day_length = solarradiation(doys, model_pars['latitude'], model_pars['slope'] * 0, model_pars['aspect'])
    Rs, dummy = solarradiation(doys, model_pars['latitude'], model_pars['slope'], model_pars['aspect'])
    SFI = Rs / R0

    # Correct for min and max values based on observed solar data
    T_summer = np.amax(srad[np.logical_and(doys > 150, doys < 200), :], axis=0) / np.amax(
        R0[np.logical_and(doys > 150, doys < 200), :], axis=0)
    T_winter = np.amax(srad[np.logical_or(doys > 350, doys < 10), :], axis=0) / np.amax(
        R0[np.logical_or(doys > 350, doys < 10), :], axis=0)
    frac = np.ones(R0.shape) * np.nan
    for i in range(len(R0[0, :])):
        frac[:, i] = (R0[:, i] - np.min(R0[:, i])) / (np.max(R0[:, i]) - np.min(R0[:, i]))

    T_ti = frac * T_summer + (1 - frac) * T_winter
    R0 = R0 * T_ti

    # Incoming Longwave Radiation

    # Compute the longwave radiation input by first, computing
    # cloud fraction (compare observed and potential solar radiation)
    if ~use_forcing_lrad:
        CF = np.ones(R0.shape) * np.nan
        for i in range(len(srad[0, :])):
            cf = (1 - (srad[:, i] / R0[:, i])) * (1 - model_pars['CloudTransmission'][i])
            cf[cf < 0] = 0
            cf[cf > 1] = 1
            CF[:, i] = cf

        CF = np.maximum(np.minimum(1,prcp/20), CF)
        
        # Then, calculate incoming longwave radiation
        Eacls = 1.08 * (1 - np.exp(-(vapp / 100) ** (tmean / 2016)))
        Ea = CF + (1 - CF) * Eacls
        lrad = Ea * 5.67E-8 * (tmean + 273.15) ** 4

    else:
        lrad = md['lrad'] * 0.484583

    # Apply multiplier to shortwave radiation if specified
    for i in range(len(srad[0, :])):
        srad[:, i] = srad[:, i] * SFI[:, i] * model_pars['srad_mult'][i]

    # Apply longwave radiation multiplier (if specified)
    for i in range(len(lrad[0, :])):
        lrad[:, i] = lrad[:, i] * model_pars['lrad_mult'][i] 
        
    ## Put data in output structure
    forcing_data = {}
    forcing_data['day'] = day
    forcing_data['mon'] = mon
    forcing_data['year'] = year
    forcing_data['tmean'] = tmean
    forcing_data['wind'] = wind
    forcing_data['srad'] = srad
    forcing_data['day_length'] = day_length
    forcing_data['lrad'] = lrad
    forcing_data['vapp'] = vapp
    forcing_data['rh'] = rh
    forcing_data['rainfall'] = rainfall
    forcing_data['snowfall'] = snowfall

    rad = forcing_data['srad'] / 1000000 * 3600 * 24  # MJ/day/m2
    latent_heat_flux = 2.26  # MJ/kg  (2260 kj/g)
    rho = 1000  # Water density in kg/m3
    # Potential evapotranspiration is computed in m/day
    forcing_data['PET'] = rad / latent_heat_flux / rho * ((forcing_data['tmean'] + 5) / 100)
    locs = forcing_data['tmean'] < -5  # mm/day
    forcing_data['PET'][locs] = 0
    forcing_data['PET'] = forcing_data['PET'] * 1000  # PET is transformed in mm

    return TS_vec, forcing_data