import numpy as np
import scipy.special as sp_spec
import scipy.optimize as sp_optim
import copy
import time

np.seterr(all="ignore")

# @profile
def run_model(TS_vec, forcing_data, model_pars):
    print('Running RHEM-Snow')
    # Model Constants
    modelconst = {}
    modelconst['karman'] = 0.41  # Von karman constant
    modelconst['subheat'] = 2.85e6  # Latent heat of sublimation [J/kg]
    modelconst['fusheat'] = 3.34e5  # Latent heat of fusion [J/kg]
    modelconst['specheat_a'] = 1008  # Specific heat of air [J/kg-K]
    modelconst['specheat_w'] = 4181  # Specific heat of water [J/kg-K]
    modelconst['specheat_i'] = 2050  # Specific heat of ice [J/kg-K]
    modelconst['Rd'] = 287  # Dry gas constant [J kg-1 K-1]
    modelconst['g'] = 9.81  # Gravity at earth's surface [m/s2]
    modelconst['I0'] = 1388  # Solar constant [W/m2]
    modelconst['emiss_snow'] = 0.99  # Snow surface emmissivity [-]
    modelconst['sigma'] = 5.67E-8  # Stefan-Boltzmann constant [W/m2-k4]
    modelconst['M2MM'] = 1000  # Conversion factor between millimeters and meters [mm/m]
    modelconst['TS'] = 86400  # Model Timestep [s]
    modelconst['DAY'] = 86400  # Conversion factor between seconds and days [s/day]
    modelconst['K'] = 273.15  # Conversion factor between celcius and kelvin
    modelconst['rhoi'] = 931  # density of ice   [kg/m3]
    modelconst['rhoa'] = 1.229  # density of air   [kg/m3]
    modelconst['rhow'] = 1000  # density of water [kg/m3]
    modelconst['P0'] = 101325  # Standard Sea level pressure [Pa]
    modelconst['L'] = 6.5E-3  # Standard Lapse Rate [K/m]
    modelconst['rhos'] = 1300  # density of soil   (kg/m3)
    modelconst['specheat_s'] = 1480  # specific heat of soil [J/kg-K]

    # Initialization

    #  Size of state (for simultaneous execution on multiple cells)
    sz = forcing_data['rainfall'][1, :].shape

    # Initialize snow states to zero
    state = {}

    state['swe'] = np.ones(sz) * model_pars['swe_i']  # SWE [mm]
    state['cansnowstor'] = np.ones(sz) * model_pars['cansnowstor_i']  # Canopy snow storage [mm]
    state['swe_age_a'] = np.ones(sz) * model_pars['swe_age_a_i']  # Age of snowpack surface [day]
    Tm = np.ones(sz) * model_pars['Tm_i']  # Internal snowpack temperature [C]
    state['cc'] = Tm / ((np.maximum(1, state['swe']) / modelconst['M2MM']) * modelconst['rhow'] * modelconst['specheat_i'])  # Cold Content [J/m2]
    state['density'] = np.ones(sz) * model_pars['density_i']  # Snow Density [g/cm3]
    state['sm_stor'] = np.ones(sz) * model_pars['H'] * model_pars['sm_i']  # Amount of water in soil [mm]
    
    state['Q_soil'] = np.ones(sz) * model_pars['T_soil_i'] * ((state['sm_stor'] / 1000 * modelconst['specheat_w'] * modelconst['rhow']) + ((model_pars['H'] - state['sm_stor']) / 1000 * modelconst['specheat_s'] * modelconst['rhos']))  # Energy content of soil water
    state['ice_fraction_soil'] = np.ones(sz) * model_pars['ice_fraction_soil_i']  # Percent soil ice
    state['x_vadose'] = model_pars['q_vadose_i'] / model_pars['coef_vadose']  # Water in Vadose Zone
    state['x_phreatic'] = model_pars['q_phreatic_i'] / model_pars['coef_phreatic']  # Water in Phreatic Zone

    depth = state['swe'] / state['density']
    depth_p = depth
    sm_sat = model_pars['ssat'] * model_pars['H']  # Saturated soil water content
    sr = (0.1 + 0.3/(1.+(1./(model_pars['g']/1000.))**4.)**.25)
    sm_res = sr * model_pars['H'] * np.ones(sz)
    cc_p = state['cc']  # Previous cold content
    # Tm = state['cc']/((np.maximum(1, state['swe']) / modelconst['M2MM']) * modelconst['rhow'] * modelconst['specheat_i'])

    # Initialize the model output variables based on the size of the forcing data
    model_output = {}
    model_output['swe'] = np.zeros(forcing_data['tmean'].shape)  # Snow Water Equivalent [mm]
    model_output['depth'] = np.zeros(forcing_data['tmean'].shape)  # Snow Depth [mm]
    model_output['density'] = np.zeros(forcing_data['tmean'].shape)  # Snow Density [g/cm3]
    model_output['rain_on_snow'] = np.zeros(forcing_data['tmean'].shape)  # Rain on Snow [mm/day]
    model_output['snowpack_sublimation'] = np.zeros(forcing_data['tmean'].shape)  # Sublimation (from snowpack) [mm/day]
    model_output['tsfall'] = np.zeros(forcing_data['tmean'].shape)  # Snow throughfall (below canopy) [mm/day]
    model_output['snow_unload'] = np.zeros(forcing_data['tmean'].shape)  # Snow unloading (from canopy) [mm/day]
    model_output['melt_drip'] = np.zeros(forcing_data['tmean'].shape)  # Melt Drip (from canopy) [mm/day]
    model_output['canopy_sublimation'] = np.zeros(forcing_data['tmean'].shape)  # Sublimation (from canopy) [mm/day]
    model_output['canopy_snow_storage'] = np.zeros(forcing_data['tmean'].shape)  # Canopy Snow Storage [mm]
    model_output['melt'] = np.zeros(forcing_data['tmean'].shape)  # Snowmelt [mm/day]
    model_output['albedo'] = np.zeros(forcing_data['tmean'].shape)  # Surface albedo [-]
    model_output['Tm'] = np.zeros(forcing_data['tmean'].shape)  # Integrated snowpack temperature [C]
    model_output['Ts'] = np.zeros(forcing_data['tmean'].shape)  # Surface temperature [C]
    model_output['Qsn'] = np.zeros(forcing_data['tmean'].shape)  # Net shortwave radiation [W/m2]
    model_output['Qle'] = np.zeros(forcing_data['tmean'].shape)  # Outgoing Longwave Radiation [W/m2]
    model_output['Qn'] = np.zeros(forcing_data['tmean'].shape)  # Net Radiation [W/m2]
    model_output['Qn_snow'] = np.zeros(forcing_data['tmean'].shape)  # Net Radiation over snowpack [W/m2]
    model_output['Qh'] = np.zeros(forcing_data['tmean'].shape)  # Sensible heat [W/m2]
    model_output['Qg'] = np.zeros(forcing_data['tmean'].shape)  # Ground heat [W/m2]
    model_output['Qe'] = np.zeros(forcing_data['tmean'].shape)  # Latent heat [W/m2]
    model_output['Qp'] = np.zeros(forcing_data['tmean'].shape)  # Heat from precip [W/m2]
    model_output['Qm'] = np.zeros(forcing_data['tmean'].shape)  # Melt heat [W/m2]
    model_output['Q'] = np.zeros(forcing_data['tmean'].shape)  # Cold Content [J/m2]
    model_output['T_soil'] = np.zeros(forcing_data['tmean'].shape)
    model_output['ice_fraction_soil'] = np.zeros(forcing_data['tmean'].shape)
    model_output['ET'] = np.zeros(forcing_data['tmean'].shape)
    model_output['SMC'] = np.zeros(forcing_data['tmean'].shape)
    model_output['infil_runoff'] = np.zeros(forcing_data['tmean'].shape)
    model_output['sat_runoff'] = np.zeros(forcing_data['tmean'].shape)
    model_output['perc'] = np.zeros(forcing_data['tmean'].shape)
    model_output['caprise'] = np.zeros(forcing_data['tmean'].shape)
    model_output['infiltration'] = np.zeros(forcing_data['tmean'].shape)
    model_output['x_vadose'] = np.zeros(forcing_data['tmean'].shape)
    model_output['x_phreatic'] = np.zeros(forcing_data['tmean'].shape)
    model_output['q_vadose'] = np.zeros(forcing_data['tmean'].shape)
    model_output['q_phreatic'] = np.zeros(forcing_data['tmean'].shape)

    NDays = len(TS_vec)
    
    tmean_all = copy.deepcopy(forcing_data['tmean'])
    wind_all = copy.deepcopy(forcing_data['wind'])
    srad_all = copy.deepcopy(forcing_data['srad'])
    lrad_all = copy.deepcopy(forcing_data['lrad'])
    vapp_all = copy.deepcopy(forcing_data['vapp'])
    rainfall_all = copy.deepcopy(forcing_data['rainfall'])
    snowfall_all = copy.deepcopy(forcing_data['snowfall'])
    PET_all = copy.deepcopy(forcing_data['PET'])

    # st = time.time()
    # c = 0
    # print(tmean_all.shape)
    
    
    ## Main Time Loop
    for TS in range(NDays):

        # c = c+1
        # Extract forcing data for a particular day
        airt = tmean_all[TS, :]        # Mean temperature [C]
        wind = wind_all[TS, :]          # Wind speed [m/s]
        srad = srad_all[TS, :]          # Solar radiation [W/m2]
        lrad = lrad_all[TS, :]          # Incoming longwave radiation [W/m2]
        vapp = vapp_all[TS, :]          # Vapor Pressure [pa]
        rainfall = rainfall_all[TS, :]  # Rainfall [mm]
        snowfall = snowfall_all[TS, :]  # Snowfall [mm]
        PET = PET_all[TS, :]            # Potential Evaporation [mm]
        # doy = forcing_data['doys'][TS]            # Potential Evaporation [mm]
        
        svapp = 0.6108 * np.exp(17.27 * airt / (237.3 + airt)) * 1000
        rh = vapp/svapp*100
        rh[rh > 100] = 100
        
        swe_p = state['swe']  # Previous SWE

        # Albedo

        # Increment age of snow surface
        state['swe_age_a'][state['swe'] == 0] = 0
        state['swe_age_a'] = state['swe_age_a'] + (modelconst['TS'] / modelconst['DAY'])

        # Reset the age of the snowpack based on the size of the storm
        state['swe_age_a'] = state['swe_age_a'] * np.maximum(0, (model_pars['albedo_snow_reset'] - snowfall) / model_pars['albedo_snow_reset'])

        # Compute albedo based on the snowpack age
        depth = state['swe'] / state['density']
        
        albedosnow = np.maximum(model_pars['minalbedo'], model_pars['albedo_i'] - (state['swe_age_a'] * model_pars['albedo_decay']))
        snowfrac = np.minimum(1, depth / model_pars['groundveght'] * 100)
        albedo = snowfrac * albedosnow + (1 - snowfrac) * model_pars['albedo_0']

        ## Net Radiation

        # Net shortwave radiation
        Qsn = srad * (1 - albedo)

        # Incoming longwave radiation (given as forcing variable)
        Qli = lrad

        Ts_nosnow = airt + 0.0192 * (Qsn + Qli) - 0.0428 * rh - 4.1790
        Ts_snow = np.minimum(0, airt + 0.0258 * (Qsn + Qli) + 0.0648 * rh - 14.5601)
        Ts = Ts_snow * snowfrac + Ts_nosnow * (1 - snowfrac)

        # Computed based on snow temperature
        Qle = modelconst['emiss_snow'] * modelconst['sigma'] * (Ts + modelconst['K']) ** 4

        # Net Radiation
        Qn = Qsn + Qli - Qle

        # Architectural resistance

        # Resistance for equilibrium conditions
        k0 = modelconst['karman'] ** 2 * wind / (np.log(model_pars['windlevel'] / model_pars['sroughness'])) ** 2

        # Richardson number (for stability correction)
        Ri = modelconst['g'] * (airt - Ts) * model_pars['windlevel'] / (wind **2 * airt)
        ka = k0 * 1
        ka[Ri > 0] = ka[Ri > 0] / (1 + 10 * Ri[Ri > 0])
        ka[Ri < 0] = ka[Ri < 0] * (1 - 10 * Ri[Ri < 0])
        ka = k0 + model_pars['fstab'] * (ka - k0)
        ka[np.isnan(ka)] = 0
        # model_pars['fstab']: Stability parameter (0-1, where 0 means no richardson number adjustment and 1 means full richardson number adjustment

        # ka = k0  # For now, do not apply stability corrections

        ## Sensible heat flux

        # Estimate how air density changes with altitude by assuming a station
        # pressure based on elevation
        pres = modelconst['P0'] * np.exp(-modelconst['g'] * model_pars['elevation'] / (modelconst['Rd'] * (airt + modelconst['K'] + modelconst['L'] * model_pars['elevation'])))
        rhoa = pres / (modelconst['Rd'] * (airt + modelconst['K']))

        Qh = ka * rhoa * modelconst['specheat_a'] * (airt - Ts) * model_pars['Ch']
        ## Latent heat flux and Sublimation

        # Saturated vapor pressure
        svapp = 0.6108 * np.exp(17.27 * Ts_nosnow / (237.3 + Ts_nosnow)) * 1000

        # Sublimation heat
        Qe = k0 * (modelconst['subheat'] * 0.622) / (modelconst['Rd'] * (airt + modelconst['K'])) * (vapp - svapp)
        Qe[Qe > 0] = 0  # Limit to mass losses

        # Convert to actual sublimation amount, later we will need to adjust if SWE is not enough
        sublimation_potential = -(Qe / (modelconst['subheat'] * modelconst['rhow'])) * modelconst['TS'] * modelconst['M2MM']

        # Canopy Snow Interception

        # Canopy Storage capacity
        cansnowstorcap = 4.4 * model_pars['lai']  # model_pars['lai']: Leaf area index (mm/mm)

        # Snow that is caught in canopy

        L = 0.7 * (cansnowstorcap - state['cansnowstor']) * (1 - np.exp(-snowfall / (cansnowstorcap + 1E-6)))
        L[np.isnan(L)] = 0  # Throughfall is snow that falls through canopy
        tsfall = snowfall - L
        state['cansnowstor'] = state['cansnowstor'] + L

        # Snow Drip Rate
        melt_drip = np.maximum(0, model_pars['melt_drip_par'] * airt)
        # model_pars['melt_drip_par']: Melt Drip Rate (mm/day / deg-C above freezing

        # Snow Unloading from the canopy
        snow_unload = np.maximum(model_pars['snow_unload_par'] * state['cansnowstor'] * modelconst['TS'] / modelconst['DAY'], 0)  # Snow Unload Rate
        # model_pars['snow_unload_par']:

        # Canopy sublimation
        acsub = np.maximum(0, sublimation_potential) * model_pars['canopy_sub_mult']
        # model_pars['canopy_sub_mult']: Canopy sublimation multiplier applied to potential sublimation rate [-]

        r = 5E-4
        a = 0.9
        C_e = 0.01 * (state['cansnowstor'] / (cansnowstorcap + 1E-6)) ** 0.4    # Calculate Canopy Sublimation
        C_e[np.isnan(C_e)] = 0
        m = modelconst['rhoi'] * 4/3 * np.pi * r ** 3                       # Mass of Ice Sphere (kg)
        rho_v = 0.622 * vapp / (287 * (airt + modelconst['K']))        # Water Vapor Density (kg/m3)
        S_p = np.pi * r ** 2 * (1-a) * srad                                 # Radiation absorbed by partical (W/m2)
        D = 2.06E-5 * ((airt + modelconst['K'])/modelconst['K']) ** 1.75 # Diffusivity of air (m2/s)
        nu = 1.3E-5                                                   # Kinematic viscosity of air (m2/s)
        a_flow = 0.9 * model_pars['lai']                               # Canopy flow index
        u_c = wind * np.exp(-a_flow * (1-0.6))                           # ventilation velocity
        Re = 2 * r * u_c/nu                                           # Reynolds number
        Sh = 1.79 + 0.606 * Re ** 0.5                                 # Sherwood number
        Nu = Sh                                                       # Nusset number
        M = 18.01E-3                                                  # Molecular weight of water(kg/mol)
        k_t = 0.024                                                   # Thermal conductivity of air (W/m2-K)
        R_const = 8314                                                # Universal gas constant (J/mol-K)
    
        omega = (1 / (k_t * (airt + modelconst['K']) * Nu)) * ((modelconst['subheat'] * M) / (R_const * (airt + modelconst['K'])) -1)        # Ventilation factor
        dmdt = (2*np.pi*r*((vapp/svapp)-1) - S_p * omega) / (modelconst['subheat'] * omega + 1 / (D * rho_v * Sh))
        psi_s = dmdt/m
        # Sublimation rate loss coefficient
        acsub = np.real(np.maximum(0,-C_e * state['cansnowstor'] * psi_s * modelconst['TS'])*model_pars['canopy_sub_mult'])
        acsub[np.isnan(acsub)] = 0

        # Figure out total ablation demand, and scale based on how much snow is
        # available (to avoid over-emptying of canopy snow storage)
        
        p_canopy_abl = acsub + melt_drip + snow_unload
        
                                              
        multiplier = state['cansnowstor'] / (np.maximum(1E-6,p_canopy_abl))
        multiplier[multiplier > 1] = 1                                                                              
        melt_drip = melt_drip * multiplier
        snow_unload = snow_unload * multiplier
        acsub = acsub * multiplier

        # Revised canopy snow storage
        state['cansnowstor'] = np.maximum(0, state['cansnowstor'] - acsub - melt_drip - snow_unload)  # Subtract evaporated snow from the canopy

        # If canopy snow storage is exceeded (e.g. due to deposition), then unload
        snow_unload = snow_unload + np.maximum(0, state['cansnowstor'] - cansnowstorcap)

        # Add all solid and liquid precipitation to SWE
        rain_on_snow = copy.deepcopy(rainfall)
        rain_on_snow[np.logical_and(state['swe'] == 0, snowfall == 0)] = 0
        
        state['swe'] = state['swe'] + tsfall + rain_on_snow + melt_drip + snow_unload

        # Heat from precip

        Qp_r = ((rainfall + melt_drip) / (modelconst['M2MM'] * modelconst['TS'])) * (modelconst['fusheat'] * modelconst['rhow'] + modelconst['specheat_w'] * modelconst['rhow'] * np.maximum(0, airt))
        Qp_s = ((tsfall + snow_unload) / (modelconst['M2MM'] * modelconst['TS'])) * (modelconst['specheat_i'] * modelconst['rhow'] * np.minimum(0, airt))
        Qp = Qp_s + Qp_r
        
        # Ground Heat
        Qg = model_pars['kappa_snow'] * (model_pars['tempdampdepth'] - Tm) / (model_pars['H'] / modelconst['M2MM'] + (swe_p / state['density']) / (2 * modelconst['M2MM']))

        # Adjust if not enough SWE
        sublimation_potential = sublimation_potential * model_pars['ground_sub_mult']
        sublimation = np.minimum(state['swe'], sublimation_potential)
        state['swe'] = state['swe'] - sublimation
        
        # Recompute sublimation heat and melt heat
        Qe = -sublimation * (modelconst['subheat'] * modelconst['rhow']) / (modelconst['TS'] * modelconst['M2MM'])
        
        # Melt

        # Potential melt based on known energy inputs
        pdq = -np.maximum(0, state['cc'] + (Qn + Qe + Qh + Qp + Qg) * modelconst['TS'])
        melt_potential = -pdq / (modelconst['rhow'] * modelconst['fusheat']) * modelconst['M2MM']

        # Adjust if not enough SWE
                                                      
        melt = np.minimum(state['swe'], melt_potential)    
        min_melt = 0.2*(np.maximum(-5,airt)+5) ** 2 + rainfall # Lower bound approx 2x 75th%ile melt in Broxton et al., 2016, fig.11
        melt = np.minimum(min_melt,melt)                                                                                        
        state['swe'] = state['swe'] - melt

        # Recompute sublimation heat and melt heat
                                                                                                                  
        Qm = (-melt / modelconst['M2MM']) * (modelconst['rhow'] * modelconst['fusheat']) / modelconst['TS']

        # Frozen soil model
        ice_soil_0 = state['ice_fraction_soil'] * model_pars['H']
        T_soil = state['Q_soil'] / ((state['sm_stor'] / 1000 * modelconst['specheat_w'] * modelconst['rhow']) + ((model_pars['H'] - state['sm_stor']) / 1000 * modelconst['specheat_s'] * modelconst['rhos']))
        g_abv = model_pars['kappa_soil'] / (model_pars['H'] / 2 / 1000) * (Ts - T_soil) * modelconst['TS']
        g_blw = model_pars['kappa_soil'] / (model_pars['dampdepth'] - (model_pars['H'] / 2 / 1000)) * (model_pars['tempdampdepth'] - T_soil) * modelconst['TS']
        g_abv_snow = -Qg * modelconst['TS']
        g_abv[state['swe'] > 0] = 0
        g_abv_snow[state['swe'] <= 0] = 0
        g_abv = g_abv + g_abv_snow
        p_dQ = (g_abv + g_blw)
        gtlocs = state['Q_soil'] > 0
        ltlocs = state['Q_soil'] < 0
        Q_soil_gtlocs = np.maximum(0, state['Q_soil'] + p_dQ)
        Q_soil_ltlocs = np.minimum(0, state['Q_soil'] + p_dQ)
        Q_soil_gtlocs[gtlocs == 0] = 0
        Q_soil_ltlocs[ltlocs == 0] = 0
        Q_soil = Q_soil_gtlocs + Q_soil_ltlocs
        residual = state['Q_soil'] + p_dQ - Q_soil
        soil_melt = (residual / (modelconst['rhow'] * modelconst['fusheat'])) * 1000

        p_ice_soil = state['sm_stor']
        ice_soil_0 = ice_soil_0 - soil_melt
        ice_soil = np.maximum(0, np.minimum(p_ice_soil, ice_soil_0))
        residual = ice_soil - ice_soil_0
        Q_soil = Q_soil + residual * (modelconst['rhow'] * modelconst['fusheat']) / 1000

        state['Q_soil'] = Q_soil
        state['ice_fraction_soil'] = ice_soil / (model_pars['H'])
        T_soil = state['Q_soil'] / (((state['sm_stor'] - ice_soil) / 1000 * modelconst['specheat_w'] * modelconst['rhow']) + ((ice_soil) / 1000 * modelconst['specheat_i'] * modelconst['rhoi']) + ((model_pars['H'] - state['sm_stor']) / 1000 * modelconst['specheat_s'] * modelconst['rhos']))

        # Snow Density

        # Reduce density based on new snow depth relative to old snow depth
        depth_p = swe_p / state['density']
        new_depth = np.maximum(1E-6, snowfall / model_pars['density_min'])
        # model_pars['density_min']: Density of fresh snowfall [g/cm3]

        new_frac = new_depth / (depth_p + new_depth)
        density_p = state['density']
        state['density'] = (1 - new_frac) * density_p + new_frac * model_pars['density_min']

        # Densify snowpack based on age, overburdin, and warm snowpacks
        state['density'] = state['density'] + ((model_pars['density_max'] - density_p) * model_pars['apar'] * modelconst['TS'] / modelconst['DAY'])
        state['density'] = state['density'] + ((model_pars['density_max'] - density_p) * model_pars['dpar'] * state['swe'] / 10 * modelconst['TS'] / modelconst['DAY'])
                       
                                               
        state['density'] = state['density'] + ((model_pars['density_max'] - density_p) * model_pars['rpar'] * (state['cc'] == 0) * modelconst['TS'] / modelconst['DAY'])
        # model_pars['rpar']: Snow densfication rate due to liquid in snowpack [Fraction when isothermal snowpack]
        if len(model_pars['density_min']) > 1:
            state['density'][state['density'] > model_pars['density_max']] = model_pars['density_max'][state['density'] > model_pars['density_max']]
        else:
            state['density'][state['density'] > model_pars['density_max']] = model_pars['density_max']

        if len(model_pars['density_min']) > 1:
            state['density'][state['density'] < model_pars['density_min']] = model_pars['density_min'][state['density'] < model_pars['density_min']]
        else:
            state['density'][state['density'] < model_pars['density_min']] = model_pars['density_min']

        depth = state['swe'] / state['density']
        density = state['density'] * 1
        density[state['swe'] == 0] = np.nan

        # Compute the energy balance
        
        Qn_snow = Qn * 1
        Qn_snow[state['swe'] == 0] = 0

        Tm_min = np.minimum(0,Ts)
        cc_min = Tm_min * (state['swe'] / modelconst['M2MM']) * modelconst['rhow'] * modelconst['specheat_i']
        cc_i = cc_p + (Qn_snow + Qh + Qe + Qp + Qm + Qg) * modelconst['TS']
        state['cc'] = np.minimum(0,np.maximum(cc_min,cc_i))
        Tm = state['cc'] / ((state['swe'] / modelconst['M2MM']) * modelconst['rhow'] * modelconst['specheat_i'])
        Tm[state['swe'] == 0] = 0

        # Clean up, prepare for next iteration, and fill output structure
        
        # Make sure that when SWE is zero, heats are not reported
        state['cc'][state['swe'] == 0] = 0
        state['cc'][state['cc'] > 0] = 0
        Qe[state['swe'] == 0] = 0
        Qp[state['swe'] == 0] = 0
        Qm[state['swe'] == 0] = 0
        Qg[state['swe'] == 0] = 0
        Qh[state['swe'] == 0] = 0
        density[state['swe'] == 0] = 0

        # Compute energy imbalance caused by above acounting and add to sensible heat term
        imbal = ((state['cc'] - cc_p) - (Qn_snow + Qh + Qe + Qp + Qm + Qg) * modelconst['TS']) / modelconst['TS']
        Qh = Qh + imbal
        cc_p = state['cc']

        # Compute Infiltration excess runoff
        net_input = rainfall - rain_on_snow + melt
        infil_runoff = net_input * model_pars['max_infil_mult'] * (np.maximum(0, (state['sm_stor'] / model_pars['H']) - model_pars['sm_min_infil']) / (model_pars['sm_max_infil'] - model_pars['sm_min_infil']))
        net_input = net_input - infil_runoff
        state['sm_stor'] = state['sm_stor'] + net_input

        # Compute actual ET
        et = PET * np.minimum(1,((np.maximum(0,state['sm_stor'] / model_pars['H'])) - model_pars['wp']) / (model_pars['cmc'] - model_pars['wp']))

        # Saturation excess runoff, if any
        state['sm_stor'] = np.maximum(0, state['sm_stor'] - et)
        sat_runoff = np.maximum(0, state['sm_stor'] - sm_sat)
        state['sm_stor'] = state['sm_stor'] - sat_runoff
        
        # Compute percolation out of the top soil layer
        
        perc = np.minimum(state['sm_stor'] / 3, model_pars['k_soil'] * (np.maximum(0, (state['sm_stor']-sm_res) / (sm_sat-sm_res))) ** (2 * model_pars['b_soil'] + 3)) * modelconst['TS'] / modelconst['DAY']
        
        bw = 1-(state['sm_stor']-sm_res) / (sm_sat-sm_res)
        bw[bw < 0] = 0
        bw[bw > 1] = 1
        beta = 2 + 3/model_pars['b_soil']
        alpha = 1 + (3/2) / (beta -1);
        caprise = bw * model_pars['k_soil'] * alpha * (model_pars['psi_s'] / (model_pars['H']) ** beta ) * modelconst['TS'] / modelconst['DAY']
        locs = caprise > state['x_vadose']
        caprise[locs] = state['x_vadose'][locs]
        
        state['sm_stor'] = state['sm_stor'] - perc + caprise
        state['x_vadose'] = state['x_vadose'] + perc - caprise
        q_vadose = model_pars['coef_vadose'] * (state['x_vadose'] ** model_pars['coef_vadose_exp']) * modelconst['TS'] / modelconst['DAY']
        state['x_vadose'] = state['x_vadose'] - q_vadose
        vadose_2_phreatic = model_pars['coef_vadose2phreatic'] * state['x_vadose'] * modelconst['TS'] / modelconst['DAY']
        state['x_vadose'] = state['x_vadose'] - vadose_2_phreatic

        state['x_phreatic'] = state['x_phreatic'] + vadose_2_phreatic
        q_phreatic = model_pars['coef_phreatic'] * (state['x_phreatic'] ** model_pars['coef_phreatic_exp']) * modelconst['TS'] / modelconst['DAY']
        state['x_phreatic'] = state['x_phreatic'] - q_phreatic

        model_output['x_vadose'][TS, :] = state['x_vadose']
        model_output['x_phreatic'][TS, :] = state['x_phreatic']
        model_output['q_vadose'][TS, :] = q_vadose
        model_output['q_phreatic'][TS, :] = q_phreatic

        infiltration = net_input - infil_runoff - sat_runoff

        model_output['ET'][TS, :] = et
        model_output['SMC'][TS, :] = state['sm_stor'] / model_pars['H'] * 100
        model_output['infil_runoff'][TS, :] = infil_runoff
        model_output['sat_runoff'][TS, :] = sat_runoff
        model_output['perc'][TS, :] = perc
        model_output['infiltration'][TS, :] = infiltration
        model_output['caprise'] [TS,:] = caprise

        model_output['swe'][TS, :] = state['swe']  # Snow Water Equivalent [mm]
        model_output['depth'][TS, :] = depth  # Snow Depth [mm]
        model_output['density'][TS, :] = density  # Snow Density [g/cm3]
        model_output['rain_on_snow'][TS, :] = rain_on_snow  # Rain on Snow [mm/day]
        model_output['snowpack_sublimation'][TS, :] = sublimation  # Sublimation (from snowpack) [mm/day]
        model_output['tsfall'][TS, :] = tsfall  # Snow throughfall (below canopy) [mm/day]
        model_output['snow_unload'][TS, :] = snow_unload  # Snow unloading (from canopy) [mm/day]
        model_output['melt_drip'][TS, :] = melt_drip  # Melt Drip (from canopy) [mm/day]
        model_output['canopy_sublimation'][TS, :] = acsub  # Sublimation (from canopy) [mm/day]
        model_output['canopy_snow_storage'][TS, :] = state['cansnowstor']  # Canopy Snow Storage [mm]
        model_output['melt'][TS, :] = melt  # Snowmelt [mm/day]
        model_output['albedo'][TS, :] = albedo  # Surface albedo [-]
        model_output['Tm'][TS, :] = Tm  # Integrated snowpack temperature [C]
        model_output['Ts'][TS, :] = Ts  # Surface temperature [C]
        model_output['Qsn'][TS, :] = Qsn  # Net shortwave radiation [W/m2]
        model_output['Qle'][TS,] = Qle  # Outgoing Longwave Radiation [W/m2]
        model_output['Qn'][TS, :] = Qn  # Net Radiation [W/m2]
        model_output['Qn_snow'][TS, :] = Qn_snow  # Net Radiation over snowpack [W/m2]
        model_output['Qh'][TS, :] = Qh  # Sensible and ground heat [W/m2]
        model_output['Qg'][TS, :] = Qg  # Ground heat [W/m2]
        model_output['Qe'][TS, :] = Qe  # Latent heat [W/m2]
        model_output['Qp'][TS, :] = Qp  # Heat from precip [W/m2]
        model_output['Qm'][TS, :] = Qm  # Melt heat [W/m2]
        model_output['Q'][TS, :] = state['cc']  # Cold Content [J/m2]
        model_output['T_soil'][TS, :] = T_soil
        model_output['ice_fraction_soil'][TS, :] = state['ice_fraction_soil'] * 100
        
        # print(time.time()-st)
        # sys.exit()

    # print((time.time()-st)/c)
    
    return model_output


def f(u, *a):
    return 1 - np.exp(-u) - a * u


def eqroot(a, b):
    u = sp_optim.fsolve(f, b, args=a)
    return u


def get_ts_data(forcing_data, model_output, TS_increment):

    # Function to disaggregate net water input from RHEM-Snow
    #
    # Inputs
    #   forcing_data: structure will all of the forcing data that will be used
    #   model_output: structure containing all model outputs
    #   TS_increment: the desired timestep (fraction of a day)
    # Outputs
    #   AllTSRainfall: Disaggregated rainfall timeseries
    #   AllTSMelt: Disaggregated snowmelt timeseries
    #
    # Patrick Broxton (broxtopd@arizona.edu) - December 2022

    # Separate rainfall falling with no snowpack with that falling on snow
    rainfall_g = forcing_data['rainfall'] - model_output['rain_on_snow']
    # Since rain-on-snow is counted as precipitation on snow in the model,
    # separate out the melt that would be caused by that rainfall (assume that
    # is the lesser of the amount of snowmelt or the amount of rain_on_snow)
    rain_on_snow_g = np.minimum(model_output['rain_on_snow'], model_output['melt'])
    # Treat the rest as melt (which will have a diurnal cycle)
    melt_g = model_output['melt'] - rain_on_snow_g
    # Daylight hourshours
    day_length = copy.deepcopy(forcing_data['day_length'])

    # Set stmdur, timep, ip, and day_length to zero if the required rainfall or
    # snowmelt components are also zero
    stmdur = copy.deepcopy(forcing_data['stmdur'])
    timep = copy.deepcopy(forcing_data['timep'])
    ip = copy.deepcopy(forcing_data['ip'])
    stmdur[np.logical_and(rainfall_g == 0, rain_on_snow_g == 0)] = 0
    timep[np.logical_and(rainfall_g == 0, rain_on_snow_g == 0)] = 0
    ip[np.logical_and(rainfall_g == 0, rain_on_snow_g == 0)] = 0
    day_length[melt_g == 0] = 0

    # The intent is to use the existing disaggregation method (based on Storm
    # Duration, time to peak/storm duration, and maximum intensity/average
    # intensity) to both rain off snow and rain on snow (though there could be
    # different treatment if desired).  Melt, which is not directly caused by a
    # rainfall event should have a diurnal cycle.

    AllTSRainfall = np.zeros([len(melt_g[:, 0]) * int(1 / TS_increment), len(melt_g[0, :])])
    AllTSMelt = np.zeros([len(melt_g[:, 0]) * int(1 / TS_increment), len(melt_g[0, :])])
    print('Getting ts data')
    for loc in range(len(melt_g[1, :])):
        
        TSRainfall = np.zeros([len(melt_g[:, 0]) * int(1 / TS_increment), 1])
        TSMelt = np.zeros([len(melt_g[:, 0]) * int(1 / TS_increment), 1])

        for dy in range(len(melt_g[:, 0])):

            melt = melt_g[dy, loc]
            dlen = day_length[dy, loc]

            if melt > 0:
                t = np.arange(TS_increment, 1 + TS_increment, TS_increment)
                alpha = np.maximum(3, 50 - 3 * dlen)  # For now, tie alpha parameter to the day length
                beta = alpha * 0.75  # This keeps maximum temperature approximately 2-3 in the afternoon

                # Based on Webb et al., 2017 - Defining Diurnal Pattern of Snowmelt using a
                # beta distribution function

                FDM = t ** (alpha - 1) * (1 - t) ** (beta - 1) * sp_spec.gamma(alpha + beta) / (
                        sp_spec.gamma(alpha) * sp_spec.gamma(beta))
                FDM = FDM / sum(FDM)  # Rescale so that it sums to one
                # Compute ts melt as the product of the rescaled FDM and the daily melt
                Melt_TS = FDM * melt

                Melt_TS[Melt_TS < np.minimum(0.001, np.max(Melt_TS) / 3)] = 0
                Melt_TS = Melt_TS * melt / np.sum(Melt_TS)

                TSMelt[int(1 / TS_increment) * dy: int(1 / TS_increment) * (dy + 1), 0] = np.transpose(Melt_TS)

            stmdur_day = float(stmdur[dy, loc])
            timep_day = float(timep[dy, loc])
            if timep_day < 0.01:
                timep_day = 0.01
            if timep_day > 0.99:
                timep_day = 0.99
            ip_day = float(ip[dy, loc])
            if ip_day > 60:
                ip_day = 60
            p = np.sum(rainfall_g[dy, loc]) + rain_on_snow_g[dy, loc]

            if stmdur_day > 0 and p > 1E-1:
                # Run double exponential assuming 20 increments
                u = eqroot(1 / ip_day, ip_day)
                b = u / timep_day
                a = ip_day * np.exp(-u)
                d = u / (1 - timep_day)

                ninten = 20
                deltfq = 1 / ninten
                fqx = 0

                timedl = np.ones([ninten + 1, 1]) * np.nan
                intdl = np.ones([ninten + 1, 1]) * np.nan
                timedl[0] = 0
                i1 = 0
                for i in range(ninten):
                    i1 = i + 1
                    if i <= ninten - 1:
                        fqx = fqx + deltfq
                        if fqx < timep_day:
                            timedl[i1] = (1.0 / b) * np.log(1.0 + (b / a) * fqx)
                        else:
                            if 1.0 - (d / ip_day) * (fqx - timep_day) > 0:
                                timedl[i1] = timep_day - (1.0 / d) * np.log(1.0 - (d / ip_day) * (fqx - timep_day))
                            else:
                                timedl[i1] = 0

                        intdl[i] = np.maximum(0, deltfq / (timedl[i1] - timedl[i]))

                timedl[i1] = 1
                intdl[i1] = 0

                timem = np.real(timedl * stmdur_day * 60)  # Minutes
                intsty = np.real(intdl * p / stmdur_day)  # mm/hr

                # Interpolate intensity to mm/timestep
                t_ = np.arange(0, np.max(timem), 1)  # Regular interval in minutes
                i_ = np.interp(t_, timem[:, 0], intsty[:, 0] / 60)
                a_ = np.cumsum(i_)
                t = np.arange(0, np.max(timem), TS_increment * 1440)  # Regular interval in minutes
                a = np.interp(t, t_, a_)
                i = np.append(np.diff(a), 0)
                if np.sum(i) > 0:
                    i = i * p / np.sum(i)

                start_time = 12 * int(1 / (TS_increment * 24)) - round(len(i) / 2)
                TSRainfall[int(1 / TS_increment) * dy + start_time - 1: int(1 / TS_increment) * dy + start_time + len(i) - 1, 0] = i

        AllTSMelt[:, loc] = TSMelt[:, 0]
        AllTSRainfall[:, loc] = TSRainfall[:, 0]

    return AllTSRainfall, AllTSMelt
