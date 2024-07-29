"""
Script to collocate the P5 MiRAC (point) measurements during the ACLOUD, AFLUX, MOSAiC-ACA, HALO-AC3 campaigns 
with MODIS data of the Aqua and Terra satellite. The MODIS data is given as granule files. 
The collocation follows the following prinicple steps within a loop (could be improved to do collocation without using a loop):

    - All MODIS granule files within the P5 flight timerange ± 90 minutes (if not, RF is skipped) are listed
    - For every specified P5 timestamp, the temporally closest MODIS granule of the above mentioned list is selected
    - This granule is then searched for the spatially closest MODIS value to the P5 measurement
    - If both, the temporal and spatial threshold are met, the MODIS value is saved
    - If not, the list of MODIS files is searched for granules which are just ±5 minutes away from the previously selected one
    - If one is found same procedure for collocation is done

Last part of the script plots the MODIS LWP data along with P5 track and the coordinates of collocated MODIS values. 

Author: Maximilian Ringel (maximilian_ringel@icloud.com)

Hamburg July 2024
"""


import numpy as np
import xarray as xr
from cartopy import crs as ccrs, feature as cfeature
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import src
from glob import glob
import ac3airborne
from dotenv import load_dotenv
import os
import warnings
import datetime
import typhon as ty

# ignore all warnings
warnings.simplefilter('ignore')

# use typhon ploting style
plt.style.use(ty.plots.styles.get('typhon'))

cat, meta, credentials, kwds = src.get_ac3_meta()

# choose satellite (aqua or terra) and the timedelta in seconds between each collocated P5-MODIS value
satellite = 'aqua'
timedelta = 30

# loop over all campaigns and P5 research flights
platform='P5'
instrument = 'MiRAC-A'
for campaign in list(cat): 
    if campaign in ['PAMARCMiP','HAMAG']: continue
    for flight_id in list(cat[campaign][platform][instrument]):
        if flight_id in ['ACLOUD_P5_RF04','ACLOUD_P5_RF06','ACLOUD_P5_RF11','ACLOUD_P5_RF14',
                         'ACLOUD_P5_RF15','ACLOUD_P5_RF16','ACLOUD_P5_RF17','ACLOUD_P5_RF21',
                         'ACLOUD_P5_RF22','ACLOUD_P5_RF23','ACLOUD_P5_RF25','AFLUX_P5_RF09',
                         'AFLUX_P5_RF10','AFLUX_P5_RF11','AFLUX_P5_RF13','AFLUX_P5_RF13',
                         'MOSAiC-ACA_P5_RF09','HALO-AC3_P5_RF01']: continue
        print("")
        print(flight_id)

        # GET CLOSEST MODIS GRANULE FILES
        #################################

        flight = meta[campaign][platform][flight_id]

        # get some date specifics for later processing
        date_dt = meta[campaign][platform][flight_id]['date']
        date_doy = str(date_dt.timetuple().tm_yday)
        year = date_dt.strftime("%Y")
        date = year + date_dt.strftime("%m%d")

        # load gps file (either locally or from the AC3 cloud)
        try:
            gpsins = cat[campaign][platform]['GPS_INS'][flight_id](**credentials,storage_options=kwds).to_dask().sel(
                time=slice(flight['takeoff'],flight['landing']))
        except:
            gpsins = cat[campaign][platform]['GPS_INS'][flight_id](storage_options=kwds).to_dask().sel(
                time=slice(flight['takeoff'],flight['landing']))

        # save every timedelta-th P5 timestamp for collocation
        P5_timestamps = gpsins.time.values[::timedelta]

        # create timerange of P5 flight ± 90 Minutes
        # wihtin which satellite overpasses will be searched
        start = flight['takeoff'] - datetime.timedelta(minutes=90)
        end = flight['landing'] + datetime.timedelta(minutes=90)
        
        timerange = []
        delta = datetime.timedelta(seconds=1)
        while start <= end:
            timerange.append(start)
            start += delta

        ###
        ### create list of closest MODIS granule files (if any available) of aqua or terra
        ###

        if satellite == "aqua":
            if len(date_doy)==2:
                modis_data_files = sorted(glob(f"/work2/modis_lwp/MYD06_L2.A{year}0{date_doy}*"))
                modis_coord_files = sorted(glob(f"/work2/tmp2_modis/MYD03.A{year}0{date_doy}*"))
            if len(date_doy)==3:
                modis_data_files = sorted(glob(f"/work2/modis_lwp/MYD06_L2.A{year}{date_doy}*"))
                modis_coord_files = sorted(glob(f"/work2/tmp2_modis/MYD03.A{year}{date_doy}*"))

            # for some flights there are extra (not needed) 
            # MODIS coordinate files which are removed
            if flight_id == 'ACLOUD_P5_RF18':
                del modis_coord_files[-1]
            if flight_id == 'AFLUX_P5_RF03':
                del modis_coord_files[3]
            if flight_id == 'AFLUX_P5_RF04':
                del modis_coord_files[1]
            if flight_id == 'AFLUX_P5_RF15':
                del modis_coord_files[-1]

        if satellite == "terra":
            if len(date_doy)==2:
                modis_data_files = sorted(glob(f"/work2/modis_lwp/MOD06_L2.A{year}0{date_doy}*"))
                modis_coord_files = sorted(glob(f"/work2/modis_lwp/MOD03.A{year}0{date_doy}*"))
            if len(date_doy)==3:
                modis_data_files = sorted(glob(f"/work2/modis_lwp/MOD06_L2.A{year}{date_doy}*"))
                modis_coord_files = sorted(glob(f"/work2/modis_lwp/MOD03.A{year}{date_doy}*"))
            
            # for some flights there are extra (not needed) 
            # MODIS coordinate files which are removed
            if flight_id == 'ACLOUD_P5_RF05':
                del modis_coord_files[1]
            if flight_id == 'ACLOUD_P5_RF07':
                del modis_coord_files[1]
            if flight_id == 'AFLUX_P5_RF04':
                del modis_coord_files[-1]
            if flight_id == 'AFLUX_P5_RF08':
                del modis_coord_files[-1]
            if flight_id == 'AFLUX_P5_RF14':
                del modis_coord_files[5]
        
        # following code block is used to identify extra MODIS coordinate files which
        # are then manually removed from modis_coord_files (see code above)
        if len(modis_data_files) != len(modis_coord_files):
            print("data files: ",len(modis_data_files))
            print("coord files: ",len(modis_coord_files))
            raise Exception("Unequal number of data and coord files!")

        # create lists which only contain modis granule files within the P5 flight (±90min) timerange
        modis_timestamps = []
        modis_data_files_filtered = []
        modis_coord_files_filtered = []
        for i, file in enumerate(modis_data_files):
            
            # get modis granule timestamp in datetime format
            time_dt = datetime.time(
                int(file[35:37]), 
                int(file[37:39]))
            date_time_dt = datetime.datetime.combine(
                date_dt, time_dt)

            # if modis granule timestamp is in P5 flight (±90min) timerange
            # add it to filtered list (later to be collocated)
            if date_time_dt in timerange:
                modis_data_files_filtered.append(modis_data_files[i])
                modis_coord_files_filtered.append(modis_coord_files[i])
                modis_timestamps.append(date_time_dt)

        # if there are no MODIS granule files in P5 flight (±90min) timerange, skip this RF
        if len(modis_timestamps) == 0:
            print(f"! No {satellite} overflights during {flight_id}!")
            continue

        # COLLOCATION
        #############

        # go through every MODIS granule file of filtered list (see above) and search for collocations with P5

        # list index used to check whether the modis granule file has changed since the last iteration
        modis_file_ind_old = 0

        # create 1km resolution dataset from the modis data and coordination files
        modis_ds = src.create_modis_1km_dataset(modis_coord_files_filtered[modis_file_ind_old],
                                                modis_data_files_filtered[modis_file_ind_old],
                                                varnames=["Cloud_Water_Path",
                                                          "Cloud_Water_Path_Uncertainty"])

        # save MODIS pixel coordinates and corresponding lwp values to numpy arrays
        modis_lons = modis_ds.lon.values
        modis_lats = modis_ds.lat.values
        modis_lwp = modis_ds.Cloud_Water_Path.values
        modis_lwp_unc = modis_ds.Cloud_Water_Path_Uncertainty.values   

        # allocate empty arrays / lists for saving the collocated data
        colloc_lwp = np.empty(len(P5_timestamps))
        colloc_lwp_unc = np.empty(len(P5_timestamps))
        colloc_lons = np.empty(len(P5_timestamps))
        colloc_lats = np.empty(len(P5_timestamps))
        colloc_modis_time = []

        colloc_lwp[:] = np.nan
        colloc_lwp_unc[:] = np.nan
        colloc_lons[:] = np.nan
        colloc_lats[:] = np.nan

        # loop through every (timedelta-th) P5 coordinate and search for nearest MODIS pixel
        for j, P5_ts in enumerate(P5_timestamps):

            # get index of modis file with nearest timestamp to current P5 timestamp
            timediff = np.array([(src.to_datetime(P5_ts)-MODIS_ts).total_seconds() 
                                for MODIS_ts in modis_timestamps])
            modis_file_ind_new = np.abs(timediff).argmin()

            t_min = np.abs(timediff[modis_file_ind_new]/60)

            # if modis file index has changed, open corresponding new file
            if modis_file_ind_new != modis_file_ind_old:
                modis_ds.close()
                modis_ds = src.create_modis_1km_dataset(modis_coord_files_filtered[modis_file_ind_new],
                                                        modis_data_files_filtered[modis_file_ind_new],
                                                        varnames=["Cloud_Water_Path",
                                                                  "Cloud_Water_Path_Uncertainty"])

                modis_lons = modis_ds.lon.values
                modis_lats = modis_ds.lat.values
                modis_lwp = modis_ds.Cloud_Water_Path.values   
                modis_lwp_unc = modis_ds.Cloud_Water_Path_Uncertainty.values

                modis_file_ind_old = modis_file_ind_new

            # get coordinates of P5 at current timestamp
            P5_lon = gpsins.sel(time=P5_ts).lon.values
            P5_lat = gpsins.sel(time=P5_ts).lat.values 

            # find nearest MODIS pixel to P5 location
            nn_ind, d_min = src.find_nn_gridcell(modis_lons,modis_lats,
                                                 P5_lon,P5_lat)
            
            # if the nearest MODIS pixel is within 5km distance and 90min timerange to the P5 observation save it
            if (d_min <= 5.) & (t_min <= 90.):

                colloc_lwp[j] = float(modis_lwp[nn_ind])
                colloc_lons[j] = modis_lons[nn_ind]
                colloc_lats[j] = modis_lats[nn_ind]
                colloc_modis_time.append(modis_timestamps[modis_file_ind_new])
            
            # if not, check whether there is a MODIS granule file with a timestamp 5 minutes before (some files just have 5 minutes in between them)
            # if so, search for possible collocations within this MODIS file
            elif (len(modis_timestamps)>=2.) & (((modis_timestamps[modis_file_ind_new]-modis_timestamps[modis_file_ind_new-1]).total_seconds()/60)==+5.):
                modis_file_ind_new -= 1
                modis_ds = src.create_modis_1km_dataset(modis_coord_files_filtered[modis_file_ind_new],
                                                        modis_data_files_filtered[modis_file_ind_new],
                                                        varnames=["Cloud_Water_Path",
                                                                  "Cloud_Water_Path_Uncertainty"])

                modis_lons = modis_ds.lon.values
                modis_lats = modis_ds.lat.values
                modis_lwp = modis_ds.Cloud_Water_Path.values
                modis_lwp_unc = modis_ds.Cloud_Water_Path_Uncertainty.values

                # find nearest neighbour gridcell of modis to P5 location
                nn_ind, d_min = src.find_nn_gridcell(modis_lons,modis_lats,
                                                     P5_lon,P5_lat)

                if (d_min <= 5.) & (t_min <= 90.):
                    
                    colloc_lwp[j] = float(modis_lwp[nn_ind])
                    colloc_lons[j] = modis_lons[nn_ind]
                    colloc_lats[j] = modis_lats[nn_ind]
                    colloc_modis_time.append(modis_timestamps[modis_file_ind_new])

                modis_file_ind_old = modis_file_ind_new
            
            # if not, check whether there is a MODIS granule file with a timestamp 5 minutes after (some files just have 5 minutes in between them)
            # if so, search for possible collocations within this MODIS file
            elif (len(modis_timestamps)>=2.) & (modis_file_ind_new<=(len(modis_timestamps)-2)):
                if (((modis_timestamps[modis_file_ind_new]-modis_timestamps[modis_file_ind_new+1]).total_seconds()/60)==5.):
                    modis_file_ind_new += 1
                    modis_ds = src.create_modis_1km_dataset(modis_coord_files_filtered[modis_file_ind_new],
                                                            modis_data_files_filtered[modis_file_ind_new],
                                                            varnames=["Cloud_Water_Path",
                                                                      "Cloud_Water_Path_Uncertainty"])

                    modis_lons = modis_ds.lon.values
                    modis_lats = modis_ds.lat.values
                    modis_lwp = modis_ds.Cloud_Water_Path.values
                    modis_lwp_unc = modis_ds.Cloud_Water_Path_Uncertainty.values

                    # find nearest neighbour gridcell of modis to P5 location
                    nn_ind, d_min = src.find_nn_gridcell(modis_lons,modis_lats,
                                                         P5_lon,P5_lat)
                    
                    if (d_min <= 5.) & (t_min <= 90.):
                        
                        colloc_lwp[j] = float(modis_lwp[nn_ind])
                        colloc_lons[j] = modis_lons[nn_ind]
                        colloc_lats[j] = modis_lats[nn_ind]
                        colloc_modis_time.append(modis_timestamps[modis_file_ind_new])
                        
                    modis_file_ind_old = modis_file_ind_new

            # save MODIS granule timestamp 
            if len(colloc_modis_time) < (j+1):
                colloc_modis_time.append(modis_timestamps[modis_file_ind_new])

        print(np.count_nonzero(~np.isnan(colloc_lwp))," collocations")

        # writing collocated modis data to xarray dataset
        modis_lwp_ds = xr.Dataset(data_vars=dict(
                modis_time=(['time'],np.array(colloc_modis_time)),
                lon=(['time'],colloc_lons),
                lat=(['time'],colloc_lats),
                lwp=(['time'],colloc_lwp),
                lwp_unc=(['time'],colloc_lwp_unc)
                ),coords=dict(time=P5_timestamps))
        modis_lwp_ds.lon.attrs['units'] = 'Degrees east'
        modis_lwp_ds.lat.attrs['units'] = 'Degrees north'
        modis_lwp_ds.lwp.attrs['units'] = 'g/m²'
        modis_lwp_ds.lwp_unc.attrs['units'] = '%'
        modis_lwp_ds.time.encoding['units'] = "seconds since 1970-01-01"

        # saving xarray dataset to netcdf
        modis_lwp_ds.to_netcdf(f'/net/secaire/mringel/data/lwp_collocated/lwp_modis/{flight_id}_MODIS_{satellite}_lwp.nc')

        # PLOTTING RESULTS
        ##################

        # plot timeseries of collocated MODIS LWP, P5 LWP and ERA5 LWP as a sanity check

        fig, ax = plt.subplots(figsize=(15,4))

        colors = cm.get_cmap('cividis', len(modis_timestamps)+1).colors 

        # plot the collocated MODIS LWP values of each granule seperately with different colors
        for j,timestamp in enumerate(modis_timestamps):
            modis_granule_ds = modis_lwp_ds.where(modis_lwp_ds.modis_time==np.datetime64(timestamp),drop=True)
            if len(modis_granule_ds.lwp) != 0:
                lwp = modis_granule_ds.lwp.values
                lwp[lwp==-9999] = np.nan
                ax.plot(modis_granule_ds.time,lwp,
                        linestyle=None,
                        c=colors[j],
                        alpha=0.75,
                        label=f'MODIS ({timestamp.strftime("%H:%M")} UTC)')

        # load collocated ERA5 LWP
        ERA5_lwp = src.get_ERA5_along_P5(flight_id)
        ax.plot(ERA5_lwp.time,ERA5_lwp.lwp*1000,linestyle='-',label='ERA5',color='C0',alpha=0.75)

        # open retrieved LWP from P5 MiRAC-A
        lwp_P5 = xr.open_dataset(f'/net/secaire/mringel/data/lwp_retrieved/{flight_id}_lwp.nc')
        lwp_p5 = lwp_P5.lwp.values
        # add NaN values for missing timestamps 
        lwp_new = np.empty(len(timerange))
        for i, ts in enumerate(timerange):
            try:
                lwp_new[i] = lwp_P5.sel(time=ts).lwp.values
            except:
                lwp_new[i] = np.nan
        ax.plot(timerange,lwp_new*1000,linestyle='-',label='P5',linewidth=3,color='C2')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel('LWP [g m$^{-2}$]')
        ax.set_title(f'{flight_id} ({date_dt.strftime("%d.%m.%Y")})')
        ax.grid(alpha=0.4)
        ax.axhline(y=0,linestyle='dashed',color='gray',alpha=0.75)
        ax.legend(loc='upper right',fontsize=12)

        plt.savefig(f'/home/mringel/lwp_project/figures/Satellite_collocations/Retrieved_LWP_vs_MODIS_{satellite}_quicklook_{flight_id}.png',
                    bbox_inches='tight',
                    dpi=200)
        
        plt.close()

        # plotting maps of LWP of each MODIS granule with P5 track and collocated MODIS pixel coordinates as a sanity check

        for j,timestamp in enumerate(modis_timestamps):
            modis_granule_ds = modis_lwp_ds.where(modis_lwp_ds.modis_time==np.datetime64(timestamp),drop=True)
            if len(modis_granule_ds.lon) != 0:

                modis_ds = src.create_modis_1km_dataset(modis_coord_files_filtered[j],
                                                        modis_data_files_filtered[j],
                                                        varnames=["Cloud_Water_Path"])

                fig = plt.figure(figsize=(6, 10))
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-10.0, 20.0, 75.0, 83.0], crs=ccrs.PlateCarree())

                # plot MODIS LWP of granule
                pc = ax.pcolormesh(modis_ds['lon'], modis_ds['lat'], modis_ds["Cloud_Water_Path"],
                                    zorder=-1,
                                    cmap='Blues',
                                    vmin=0,
                                    vmax=500,
                                    transform=ccrs.PlateCarree())
                ax.coastlines(resolution='50m', color='black', linewidth=1,zorder=1)
                fig.colorbar(pc,fraction=0.053, pad=0.04, label='LWP [g m$^{-2}$]')
                ax.set_title(f'MODIS ({satellite}) granule at \n{modis_timestamps[j]}'
                            f'\n with {flight_id} ',fontsize=15)

                # plot P5 track
                ax.scatter(gpsins.lon.values,
                           gpsins.lat.values,
                           color='black',
                           s=5,
                           zorder=2,
                           transform=ccrs.PlateCarree(),
                           label='P5 track')

                # plot coordinates of collocated MODIS pixels
                ax.scatter(modis_granule_ds.lon.values,
                           modis_granule_ds.lat.values,
                           color='red',
                           s=0.75,
                           zorder=2,
                           transform=ccrs.PlateCarree(),
                           label='Collocations')

                lgnd = ax.legend(loc='lower left')
                lgnd.legendHandles[0]._sizes = [50]
                lgnd.legendHandles[1]._sizes = [50]
                
                plt.savefig(f'/home/mringel/lwp_project/figures/Satellite_collocations/MODIS/Colloc_MODIS_{satellite}_{flight_id}_granule_{modis_timestamps[j].strftime("%H%M")}.png',
                            bbox_inches='tight',
                            dpi=200)
                
                plt.close() 