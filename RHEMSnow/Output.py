import os
import numpy as np
from datetime import datetime
from datetime import timedelta
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def write_ts_output(FName, years, months, days, TS_increment, ids, TSPrecip, DailyPrecip, sat, ice):
    # Function to write diaggregated model output from RHEM-Snow to Kineros input files
    # 
    # Inputs
    #   FName - the folder directory where the output files are written
    #   years - a vector of years corresponding to the daily model output
    #   months - a vector of months corresponding to the daily model output
    #   days - a vector of days corresponding to the daily model output
    #   TS_increment - the desired timestep (fraction of a day)
    #   ids: 
    #   TSPrecip: Disaggregated net water input timeseries
    #   DailyPrecip: Non-disaggregated net water input timeseries (to ensure mass balance)
    #   sat: soil saturation value for each day
    #   ice: ice fraction value for each day
    # 
    # This function writes a directory of input files for kineros simulations
    
    # Patrick Broxton (broxtopd@arizona.edu) - March 2022

    print('Writing hourly precip files to ' + FName)

    for d in range(len(years)):
        year = str(int(years[d]))
        if len(year) < 4:
            year = '0' + year
        if len(year) < 4:
            year = '0' + year
        if len(year) < 4:
            year = '0' + year
        month = str(int(months[d]))
        if len(month) < 2:
            month = '0' + month
        day = str(int(days[d]))
        if len(day) < 2:
            day = '0' + day
            
        Locs = range(int(1 / TS_increment) * d + 1, int(1 / TS_increment) * (d + 1))

        if TSPrecip.ndim == 2:
            truth = np.sum(np.amax(TSPrecip[Locs, :], axis=1)) > 1E-3
        else:
            truth = np.sum(np.amax(TSPrecip[Locs])) > 1E-3

        if truth:
            fdir = FName + '/' + year + '/' + month
            if not os.path.exists(fdir):
                os.makedirs(fdir)

            if DailyPrecip[d] > 0:
                output_file = open(fdir + '/' + day + '.txt', 'w')
                print('! Created by Python RHEM-Snow: ' + datetime.now().strftime("%m/%d/%Y %H:%M"), file=output_file)

                for loc in range(len(ids)):

                
                    if TSPrecip.ndim == 2:
                        locs_gt = np.amax(TSPrecip[Locs, :], axis=1) > 1E-3
                        Precip_day = TSPrecip[Locs, loc]
                    else:
                        locs_gt = TSPrecip[Locs] > 1E-3
                        Precip_day = TSPrecip[Locs]

                    indices_gt = [i for i, x in enumerate(locs_gt) if x]
                    first_ts = indices_gt[0]
                    last_ts = indices_gt[-1]
                    
                    hr_start = np.floor(first_ts*TS_increment*24)
                    mn_start = np.floor((first_ts*TS_increment*24 - hr_start)*60)
                    hr_start = str(int(hr_start))
                    mn_start = str(int(mn_start))
                    if len(hr_start) < 2:
                        hr_start = '0' + hr_start
                    if len(mn_start) < 2:
                        mn_start = '0' + mn_start
                
                    print('', file=output_file)
                    print('BEGIN GAUGE', file=output_file)
                    print('! Event started: ' + month + '/' + day + '/' + year + ' 00:00', file=output_file)
                    print('! Element ID: ' + ids[loc], file=output_file)
                    print('SAT = ' + '{:.2f}'.format(sat[d]), file=output_file)
                    print('ICE = ' + '{:.2f}'.format(ice[d]), file=output_file)
                    print('N = ' + '{:.0f}'.format(last_ts - first_ts + 2), file=output_file)
                    print('TIME    DEPTH ! (mm)', file=output_file)

                    amount = 0
                    i = 0
                    Precip_day = Precip_day * DailyPrecip[d] / np.sum(Precip_day)
                    mult = np.sum(Precip_day) / np.sum(Precip_day[first_ts:last_ts+1])

                    for ts in np.arange(first_ts, last_ts+2):
                        i = i + 1
                        minute_str = '{:.1f}'.format((i - 1) * 1440 * TS_increment)
                        amount_str = '{:.3f}'.format(amount)
                        print(minute_str.rjust(8) + amount_str.rjust(8), file=output_file)
                        amount = amount + Precip_day[min(len(Precip_day)-1, int(ts))] * mult
                        
                    print('END GAUGE\n', file=output_file)

                output_file.close()
                

def write_daily_mat(output_file, model_output, forcing_data, model_pars, TS_vec, TSPrecip):

    print('Writing ' + output_file)
    mdic = {'model_output': model_output,
            'forcing_data': forcing_data,
            'model_pars': model_pars,
            'TSPrecip': TSPrecip,
            'StartDate': TS_vec[0].strftime('%m/%d/%Y'),
            'EndDate': TS_vec[-1].strftime('%m/%d/%Y')}

    sio.savemat(output_file, mdic)


def date_range(StartDate, EndDate):
    start = datetime.strptime(StartDate, '%d-%b-%Y')
    end = datetime.strptime(EndDate, '%d-%b-%Y')
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days


def display_snowpack_balance(TS_vec, model_output, forcing_data, disploc, dispdaterange):
    # Function to display mass and energy balances from a Rhem-Snow simulation. 
    #
    # Inputs
    #   TS_vec is a vector of matlab timestamps corresponding to model output
    #   forcing_data: structure will all of the forcing data that will be used
    #   model_output: structure containing all model outputs
    #   validation_files: Namess of the files that contains the validation data
    #   disploc: Modeled location to display output for
    #   dispdaterange: range of dates to restrict model output display
    #
    # Patrick Broxton (broxtopd@arizona.edu) - March 2010

    # Model constants
    TS = 86400  # Model Timestep [s]

    # Locations to subset the data for (based on the given date range)
    a = TS_vec >= dispdaterange[0]
    b = TS_vec <= dispdaterange[1]
    tlocs = a & b

    ## Create Balances Figure
    my_dpi = 96
    fig = plt.figure(figsize=(1900 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    plt.rcParams['font.size'] = 12

    # Plot overall mass balance
    plt.subplot(2, 3, 1)
    plt.plot(TS_vec[tlocs],
             np.cumsum(forcing_data['snowfall'][tlocs, disploc] + model_output['rain_on_snow'][tlocs, disploc]), '-k',
             linewidth=2, label='Precipitation on Snow')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['snowpack_sublimation'][tlocs, disploc]), '-r', linewidth=2,
             label='Snowpack Sublimation')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['canopy_sublimation'][tlocs, disploc]), '-g', linewidth=2,
             label='Canopy Sublimation')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['melt'][tlocs, disploc]), '-b', linewidth=2, label='Snow Melt')
    plt.plot(TS_vec[tlocs], model_output['swe'][tlocs, disploc], '-c', linewidth=2, label='SWE')
    plt.plot(TS_vec[tlocs], np.cumsum(
        forcing_data['snowfall'][tlocs, disploc] + model_output['rain_on_snow'][tlocs, disploc] -
        model_output['snowpack_sublimation'][tlocs, disploc] - model_output['canopy_sublimation'][tlocs, disploc] -
        model_output['melt'][tlocs, disploc]) - model_output['swe'][tlocs, disploc] -
             model_output['canopy_snow_storage'][tlocs, disploc], '--k', linewidth=0.5, label='Balance')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.xlim(dispdaterange)
    plt.ylabel('Accumulated Mass (mm $h_2o$)')
    plt.title('Overall Mass Balance')

    # Plot Snowpack Mass Balance
    plt.subplot(2, 3, 2)
    plt.plot(TS_vec[tlocs], np.cumsum(
        model_output['tsfall'][tlocs, disploc] + model_output['rain_on_snow'][tlocs, disploc] +
        model_output['snow_unload'][tlocs, disploc] + model_output['melt_drip'][tlocs, disploc]), '--b', linewidth=2,
             label='Precipitation on Snow')
    plt.plot(TS_vec[tlocs], model_output['swe'][tlocs, disploc], '-c', linewidth=2, label='Snow Water Equivalent')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['snowpack_sublimation'][tlocs, disploc]), '-r', linewidth=2,
             label='Snowpack Sublimation')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['melt'][tlocs, disploc]), '-b', linewidth=2, label='Snow Melt')
    plt.plot(TS_vec[tlocs], np.cumsum(
        model_output['tsfall'][tlocs, disploc] + model_output['rain_on_snow'][tlocs, disploc] +
        model_output['snow_unload'][tlocs, disploc] + model_output['melt_drip'][tlocs, disploc] -
        model_output['snowpack_sublimation'][tlocs, disploc] - model_output['melt'][tlocs, disploc]) -
             model_output['swe'][tlocs, disploc], '--k', linewidth=1, label='Balance')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.xlim(dispdaterange)
    plt.ylabel('Accumulated Mass (mm $h_2o$)')
    plt.title('Snowpack Mass Balance')

    # Plot Canopy Mass Balance
    plt.subplot(2, 3, 3)
    plt.plot(TS_vec[tlocs],
             np.cumsum(forcing_data['snowfall'][tlocs, disploc] - model_output['tsfall'][tlocs, disploc]), linewidth=2,
             label='Snowfall Caught in Canopy')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['snow_unload'][tlocs, disploc]), linewidth=2,
             label='Snow Unload from Canopy')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['melt_drip'][tlocs, disploc]), linewidth=2,
             label='Melt Drip from Canopy')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['canopy_sublimation'][tlocs, disploc]), linewidth=2,
             label='Canopy Sublimation')
    plt.plot(TS_vec[tlocs], model_output['canopy_snow_storage'][tlocs, disploc], linewidth=2,
             label='Canopy Snow Storage')
    plt.plot(TS_vec[tlocs], np.cumsum(
        forcing_data['snowfall'][tlocs, disploc] - model_output['tsfall'][tlocs, disploc] - model_output['snow_unload'][
            tlocs, disploc] - model_output['melt_drip'][tlocs, disploc] - model_output['canopy_sublimation'][
            tlocs, disploc]) - model_output['canopy_snow_storage'][tlocs, disploc], '--k', linewidth=0.5,
             label='Balance')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.xlim(dispdaterange)
    plt.ylabel('Accumulated Mass (mm $h_2o$)')
    plt.title('Canopy Mass Balance')

    # Plot radiative balance
    plt.subplot(2, 3, 4)
    plt.plot(TS_vec[tlocs], np.cumsum(forcing_data['srad'][tlocs, disploc] * TS), '-r', linewidth=2,
             label='Shortwave In')
    plt.plot(TS_vec[tlocs],
             -np.cumsum((forcing_data['srad'][tlocs, disploc] - model_output['Qsn'][tlocs, disploc]) * TS), '--r',
             linewidth=2, label='Shortwave Out')
    plt.plot(TS_vec[tlocs], np.cumsum(forcing_data['lrad'][tlocs, disploc] * TS), '-b', linewidth=2,
             label='Longwave In')
    plt.plot(TS_vec[tlocs], -np.cumsum(model_output['Qle'][tlocs, disploc] * TS), '--b', linewidth=2,
             label='Longwave Out')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qn'][tlocs, disploc] * TS), '-g', linewidth=2,
             label='Net Radiation')
    plt.plot(TS_vec[tlocs], np.cumsum(
        model_output['Qsn'][tlocs, disploc] + forcing_data['lrad'][tlocs, disploc] - model_output['Qle'][
            tlocs, disploc] - model_output['Qn'][tlocs, disploc]), '--k', label='Balance')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.xlim(dispdaterange)
    plt.ylabel('Accumulated Energy (J)');
    plt.title('Radiative Balance')

    # Plot Snowpack Energy Balance
    plt.subplot(2, 3, 5)
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qn_snow'][tlocs, disploc]) * TS, linewidth=2, label='Net Radiation')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qh'][tlocs, disploc]) * TS, linewidth=2, label='Sensible Heat')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qe'][tlocs, disploc]) * TS, linewidth=2, label='Latent Heat')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qp'][tlocs, disploc]) * TS, linewidth=2, label='Heat from Precip')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qm'][tlocs, disploc] * TS), linewidth=2, label='Phase Change Heat')
    plt.plot(TS_vec[tlocs], np.cumsum(model_output['Qg'][tlocs, disploc] * TS), linewidth=2, label='Ground Heat')
    plt.plot(TS_vec[tlocs], model_output['Q'][tlocs, disploc], linewidth=2, label='Cold Content')
    plt.plot(TS_vec[tlocs], np.cumsum(
        model_output['Qn_snow'][tlocs, disploc] + model_output['Qg'][tlocs, disploc] + model_output['Qh'][
            tlocs, disploc] + model_output['Qe'][tlocs, disploc] + model_output['Qp'][tlocs, disploc] +
        model_output['Qm'][tlocs, disploc]) * TS - model_output['Q'][tlocs, disploc], '--k', linewidth=1,
             label='Balance')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.xlim(dispdaterange)
    plt.ylabel('Accumulated Energy (J)');
    plt.title('Snowpack Energy Balance')

    # Plot Temperatures
    model_output['Tm'][model_output['swe'] == 0] = np.nan;
    plt.subplot(2, 3, 6)
    plt.plot(TS_vec[tlocs], forcing_data['tmean'][tlocs, disploc], '--r', linewidth=2, label='Model Air Temperature')
    plt.plot(TS_vec[tlocs], model_output['Tm'][tlocs, disploc], '--b', linewidth=2, label='Est. Snowpack Temperature')
    plt.plot(TS_vec[tlocs], np.zeros(model_output['Tm'][tlocs, disploc].shape), '--k', label='Freezing Level')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.xlim(dispdaterange)
    plt.ylabel('Temperature ($^oC$)');
    plt.title('Temperatures')

    fig.autofmt_xdate()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def display_snowpack_validation(TS_vec, model_output, forcing_data, validation_file, disploc, dispdaterange):
    # Function to display model validation of a RHEM-Snow simulation
    #
    # Inputs
    #   TS_vec is a vector of matlab timestamps corresponding to model output
    #   forcing_data: structure will all of the forcing data that will be used
    #   model_output: structure containing all model outputs
    #   validation_files: Namess of the files that contains the validation data
    #   disploc: Modeled location to display output for
    #   dispdaterange: range of dates to restrict model output display
    #
    # Patrick Broxton (broxtopd@arizona.edu) - March 2022

    # Model constants
    TS = 86400  # Model Timestep [s]

    # Locations to subset the data for (based on the given date range)
    a = TS_vec >= dispdaterange[0]
    b = TS_vec <= dispdaterange[1]
    tlocs = a & b

    if not validation_file == '':

        md = sio.loadmat(validation_file)

        if 'swe' in md:
            obs_swe = md['swe'][:, disploc]
        else:
            obs_swe = np.ones(tlocs.shape) * np.nan

        model_swe = model_output['swe'][:, disploc]

        if 'depth' in md:
            obs_depth = md['depth'][:, disploc]
        else:
            obs_depth = np.ones(tlocs.shape) * np.nan

        model_depth = model_output['depth'][:, disploc]

        if 'snowfall' in md:
            obs_snowfall = md['snowfall'][:, 0]
            obs_snowfall_nanlocs = np.isnan(obs_snowfall)
            mod_snowfall = forcing_data['snowfall'][:, disploc]
            obs_snowfall[obs_snowfall_nanlocs] = mod_snowfall[obs_snowfall_nanlocs]
            obs_cum_snowfall = np.cumsum(obs_snowfall)
        else:
            obs_cum_snowfall = np.ones(tlocs.shape) * np.nan

        model_cum_snowfall = np.cumsum(forcing_data['snowfall'][:, disploc])
        model_cum_rainfall = np.cumsum(forcing_data['rainfall'][:, disploc])

        if 'sublimation' in md:
            obs_sublimation = md['sublimation'][:, 0]
            obs_sublimation_nanlocs = np.isnan(obs_sublimation) + ((obs_swe < 10) + (obs_depth < 50))
            mod_sublimation = model_output['snowpack_sublimation'][:, disploc] + model_output['canopy_sublimation'][:, disploc]
            obs_sublimation[obs_sublimation_nanlocs] = mod_sublimation[obs_sublimation_nanlocs]
            obs_cum_sublimation = np.cumsum(obs_sublimation)
        else:
            obs_cum_sublimation = np.ones(tlocs.shape) * np.nan

        model_cum_sublimation = np.cumsum(model_output['snowpack_sublimation'][:, disploc] + model_output['canopy_sublimation'][:, disploc])

        obs_snowfall_i = 0
        obs_sublimation_i = 0
        model_snowfall_i = 0
        model_rainfall_i = 0
        model_sublimation_i = 0
        for i in range(len(TS_vec)):
            if TS_vec[i].month == 10 and TS_vec[i].day == 1:
                obs_snowfall_i = obs_cum_snowfall[i]
                obs_sublimation_i = obs_cum_sublimation[i]
                model_snowfall_i = model_cum_snowfall[i]
                model_rainfall_i = model_cum_rainfall[i]
                model_sublimation_i = model_cum_sublimation[i]

            obs_cum_snowfall[i] = obs_cum_snowfall[i] - obs_snowfall_i
            obs_cum_sublimation[i] = obs_cum_sublimation[i] - obs_sublimation_i
            model_cum_snowfall[i] = model_cum_snowfall[i] - model_snowfall_i
            model_cum_rainfall[i] = model_cum_rainfall[i] - model_rainfall_i
            model_cum_sublimation[i] = model_cum_sublimation[i] - model_sublimation_i

        if 'snowfall' in md:
            obs_cum_snowfall[obs_snowfall_nanlocs] = np.nan

        if 'sublimation' in md:
            obs_cum_sublimation[obs_sublimation_nanlocs] = np.nan

        TS_avg = date_range('1-Oct-1998', '30-Sep-1999')

        doys_avg = []
        for TS in TS_avg:
            doys_avg.append(TS.timetuple().tm_yday)
        doys_avg = np.array(doys_avg)

        doys = []
        for TS in TS_vec:
            doys.append(TS.timetuple().tm_yday)
        doys = np.array(doys)

        median_model_cum_snowfall = np.ones(len(doys_avg)) * np.nan
        q25_model_cum_snowfall = np.ones(len(doys_avg)) * np.nan
        q75_model_cum_snowfall = np.ones(len(doys_avg)) * np.nan
        median_obs_cum_snowfall = np.ones(len(doys_avg)) * np.nan
        q25_obs_cum_snowfall = np.ones(len(doys_avg)) * np.nan
        q75_obs_cum_snowfall = np.ones(len(doys_avg)) * np.nan
        median_model_cum_rainfall = np.ones(len(doys_avg)) * np.nan
        q25_model_cum_rainfall = np.ones(len(doys_avg)) * np.nan
        q75_model_cum_rainfall = np.ones(len(doys_avg)) * np.nan
        median_model_swe = np.ones(len(doys_avg)) * np.nan
        q25_model_swe = np.ones(len(doys_avg)) * np.nan
        q75_model_swe = np.ones(len(doys_avg)) * np.nan
        median_obs_swe = np.ones(len(doys_avg)) * np.nan
        q25_obs_swe = np.ones(len(doys_avg)) * np.nan
        q75_obs_swe = np.ones(len(doys_avg)) * np.nan
        median_model_depth = np.ones(len(doys_avg)) * np.nan
        q25_model_depth = np.ones(len(doys_avg)) * np.nan
        q75_model_depth = np.ones(len(doys_avg)) * np.nan
        median_obs_depth = np.ones(len(doys_avg)) * np.nan
        q25_obs_depth = np.ones(len(doys_avg)) * np.nan
        q75_obs_depth = np.ones(len(doys_avg)) * np.nan
        median_model_cum_sublimation = np.ones(len(doys_avg)) * np.nan
        q25_model_cum_sublimation = np.ones(len(doys_avg)) * np.nan
        q75_model_cum_sublimation = np.ones(len(doys_avg)) * np.nan
        median_obs_cum_sublimation = np.ones(len(doys_avg)) * np.nan
        q25_obs_cum_sublimation = np.ones(len(doys_avg)) * np.nan
        q75_obs_cum_sublimation = np.ones(len(doys_avg)) * np.nan
        c = -1
        for doy_avg in doys_avg:
            c = c+1
            if 'snowfall' in md:
                locs = doys == doy_avg * ~np.isnan(obs_cum_snowfall)
                if sum(locs) > 0:
                    median_model_cum_snowfall[c] = np.median(model_cum_snowfall[locs])
                    q25_model_cum_snowfall[c] = np.percentile(model_cum_snowfall[locs], 25)
                    q75_model_cum_snowfall[c] = np.percentile(model_cum_snowfall[locs], 75)
                    median_obs_cum_snowfall[c] = np.median(obs_cum_snowfall[locs])
                    q25_obs_cum_snowfall[c] = np.percentile(obs_cum_snowfall[locs], 25)
                    q75_obs_cum_snowfall[c] = np.percentile(obs_cum_snowfall[locs], 75)
                    median_model_cum_rainfall[c] = np.median(model_cum_rainfall[locs])
                    q25_model_cum_rainfall[c] = np.percentile(model_cum_rainfall[locs], 25)
                    q75_model_cum_rainfall[c] = np.percentile(model_cum_rainfall[locs], 75)

            if 'swe' in md:
                locs = doys == doy_avg * ~np.isnan(obs_swe)
                if sum(locs) > 0:
                    median_model_swe[c] = np.median(model_swe[locs])
                    q25_model_swe[c] = np.percentile(model_swe[locs], 25)
                    q75_model_swe[c] = np.percentile(model_swe[locs], 75)
                    median_obs_swe[c] = np.median(obs_swe[locs])
                    q25_obs_swe[c] = np.percentile(obs_swe[locs], 25)
                    q75_obs_swe[c] = np.percentile(obs_swe[locs], 75)

            if 'depth' in md:
                locs = doys == doy_avg * ~np.isnan(obs_depth)
                if sum(locs) > 0:
                    median_model_depth[c] = np.median(model_depth[locs])
                    q25_model_depth[c] = np.percentile(model_depth[locs], 25)
                    q75_model_depth[c] = np.percentile(model_depth[locs], 75)
                    median_obs_depth[c] = np.median(obs_depth[locs])
                    q25_obs_depth[c] = np.percentile(obs_depth[locs], 25)
                    q75_obs_depth[c] = np.percentile(obs_depth[locs], 75)

            if 'sublimation' in md:
                locs = doys == doy_avg * ~np.isnan(obs_cum_sublimation)
                if sum(locs) > 0:
                    median_model_cum_sublimation[c] = np.median(model_cum_sublimation[locs])
                    q25_model_cum_sublimation[c] = np.percentile(model_cum_sublimation[locs], 25)
                    q75_model_cum_sublimation[c] = np.percentile(model_cum_sublimation[locs], 75)
                    median_obs_cum_sublimation[c] = np.median(obs_cum_sublimation[locs])
                    q25_obs_cum_sublimation[c] = np.percentile(obs_cum_sublimation[locs], 25)
                    q75_obs_cum_sublimation[c] = np.percentile(obs_cum_sublimation[locs], 75)

        #  Create temporal validation figure (if a validation file is given)

        ## Create Balances Figure
        my_dpi = 96
        fig = plt.figure(figsize=(1900 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        plt.rcParams['font.size'] = 12

        # Precipitation
        fig.add_subplot(4, 3, (1, 2))
        if 'snowfall' in md:
            plt.plot(TS_vec[tlocs], obs_cum_snowfall[tlocs], '-k', linewidth=2, label='Observed Snowfall')
        plt.plot(TS_vec[tlocs], model_cum_snowfall[tlocs], '-b', linewidth=2, label='Modeled Snowfall')
        plt.plot(TS_vec[tlocs], model_cum_rainfall[tlocs] + model_cum_snowfall[tlocs], '-r', linewidth=2, label='Total Precip.')
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.xlim(dispdaterange)
        plt.ylabel('Precipitation (mm)')
        ylimits = plt.gca().get_ylim()

        plt.subplot(4, 3, 3)
        if 'snowfall' in md:
            plt.plot(TS_avg, median_obs_cum_snowfall, '-k', linewidth=2, label='Observed SWE')
            plt.plot(TS_avg, q25_obs_cum_snowfall, '--k')
            plt.plot(TS_avg, q75_obs_cum_snowfall, '--k')

        plt.plot(TS_avg, median_model_cum_snowfall, '-b', linewidth=2, label='Modeled SWE')
        plt.plot(TS_avg, q25_model_cum_snowfall, '--b')
        plt.plot(TS_avg, q75_model_cum_snowfall, '--b')
        plt.plot(TS_avg, median_model_cum_rainfall+median_model_cum_snowfall, '-r', linewidth=2, label='Modeled SWE')
        plt.plot(TS_avg, q25_model_cum_rainfall+q25_model_cum_snowfall, '--r')
        plt.plot(TS_avg, q75_model_cum_rainfall+q75_model_cum_snowfall, '--r')
        plt.ylabel('Precipitation (mm)')
        plt.xlim([min(TS_avg), max(TS_avg)])
        plt.ylim(ylimits)
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%d'))
        plt.gca().set_xticks([TS_avg[0], TS_avg[61], TS_avg[122], TS_avg[182], TS_avg[243], TS_avg[304], TS_avg[364]])

        #  SWE
        fig.add_subplot(4, 3, (4, 5))
        if 'swe' in md:
            plt.plot(TS_vec[tlocs], obs_swe[tlocs], '-k', linewidth=2, label='Observed SWE')

        plt.plot(TS_vec[tlocs], model_swe[tlocs], '-r', linewidth=2, label='Modeled SWE')
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.ylabel('SWE (mm)')
        ylimits = plt.gca().get_ylim()

        plt.subplot(4, 3, 6)
        if 'swe' in md:
            plt.plot(TS_avg, median_obs_swe, '-k', linewidth=2, label='Observed SWE')
            plt.plot(TS_avg, q25_obs_swe, '--k')
            plt.plot(TS_avg, q75_obs_swe, '--k')

        plt.plot(TS_avg, median_model_swe, '-r', linewidth=2, label='Modeled SWE')
        plt.plot(TS_avg, q25_model_swe, '--r')
        plt.plot(TS_avg, q75_model_swe, '--r')
        plt.ylabel('SWE (mm)')
        plt.xlim([min(TS_avg), max(TS_avg)])
        plt.ylim(ylimits)
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%d'))
        plt.gca().set_xticks([TS_avg[0], TS_avg[61], TS_avg[122], TS_avg[182], TS_avg[243], TS_avg[304], TS_avg[364]])

        # Depth
        fig.add_subplot(4, 3, (7, 8))
        if 'depth' in md:
            plt.plot(TS_vec[tlocs], obs_depth[tlocs], '-k', linewidth=2, label='Observed Depth')

        plt.plot(TS_vec[tlocs], model_depth[tlocs], '-r', linewidth=2, label='Modeled Depth')
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.ylabel('Depth (mm)')
        ylimits = plt.gca().get_ylim()

        plt.subplot(4, 3, 9)
        if 'depth' in md:
            plt.plot(TS_avg, median_obs_depth, '-k', linewidth=2, label='Observed Depth')
            plt.plot(TS_avg, q25_obs_depth, '--k')
            plt.plot(TS_avg, q75_obs_depth, '--k')
        plt.plot(TS_avg, median_model_depth, '-r', linewidth=2, label='Modeled Depth')
        plt.plot(TS_avg, q25_model_depth, '--r')
        plt.plot(TS_avg, q75_model_depth, '--r')
        plt.ylabel('Depth (mm)')
        plt.xlim([min(TS_avg), max(TS_avg)])
        plt.ylim(ylimits)
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%d'))
        plt.gca().set_xticks([TS_avg[0], TS_avg[61], TS_avg[122], TS_avg[182], TS_avg[243], TS_avg[304], TS_avg[364]])

        # Sublimation
        fig.add_subplot(4, 3, (10, 11))
        if 'sublimation' in md:
            plt.plot(TS_vec[tlocs], obs_cum_sublimation[tlocs], '-k', linewidth=2, label='Observed Sublimation')

        plt.plot(TS_vec[tlocs], model_cum_sublimation[tlocs],  '--r', linewidth=2, label='Modeled Sublimation')
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.ylabel('Sublimation (mm)')
        ylimits = plt.gca().get_ylim()

        plt.subplot(4, 3, 12)
        if 'sublimation' in md:
            plt.plot(TS_avg, median_obs_cum_sublimation, '-k', linewidth=2, label='Observed Sublimation')
            plt.plot(TS_avg, q25_obs_cum_sublimation, '--k')
            plt.plot(TS_avg, q75_obs_cum_sublimation, '--k')

        plt.plot(TS_avg, median_model_cum_sublimation, '-r', linewidth=2, label='Modeled Sublimation')
        plt.plot(TS_avg, q25_model_cum_sublimation, '--r')
        plt.plot(TS_avg, q75_model_cum_sublimation, '--r')
        plt.ylabel('Sublimation (mm)')
        plt.xlim([min(TS_avg), max(TS_avg)])
        plt.ylim(ylimits)
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%d'))
        plt.gca().set_xticks([TS_avg[0], TS_avg[61], TS_avg[122], TS_avg[182], TS_avg[243], TS_avg[304], TS_avg[364]])
