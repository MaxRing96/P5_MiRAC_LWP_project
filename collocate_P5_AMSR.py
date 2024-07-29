"""
Script to collocate the P5 MiRAC (point) measurements during the ACLOUD, AFLUX, MOSAiC-ACA, HALO-AC3 campaigns with AMSR2 data (on a regular lat-lon grid) .
The collocation follows the following prinicple steps within a loop (could be improved to do collocation without using a loop):

    - Calcuation of spatial distance of every AMSR2 gridcell to P5 measurement
    - Only select AMSR2 gridcells closer than specified spatial threshold value 
    - Out of these select the AMSR2 gridcell, having the smallest timedelta to the P5 measurement
    - If this AMSR2 gridcell has a timestamp within the specified temporal threshold value save it to the collocated data 

Last part of the script plots the AMSR2 LWP data along with P5 track and the coordinates of collocated AMSR2 cells. 

Author: Maximilian Ringel (maximilian_ringel@icloud.com)

Hamburg July 2024
"""

import numpy as np
import xarray as xr
from cartopy import crs as ccrs
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


# set collocation criteria
"""
timedelta: Length of iteration step over MiRAC timestamps to be collocated
d_max: Maximum spatial distance for collocation of satellite data
t_max: Maximum temporal distance for collocation of satellite data
"""
timedelta = 10
d_max = 10
t_max = 90
###

cat, meta, credentials, kwds = src.get_ac3_meta()

# loop over all P5 flights of all missions, excluding flights for which no LWP was retrieved
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

        # get flight date
        flight = meta[campaign][platform][flight_id]
        flight_date = meta[campaign][platform][flight_id]['date'].strftime("%Y%m%d")

        # get flight gps data (either locally or from the AC3 cloud)
        try:
            gpsins = cat[campaign][platform]['GPS_INS'][flight_id](**credentials,storage_options=kwds).to_dask().sel(
                time=slice(flight['takeoff'],flight['landing']))
        except:
            gpsins = cat[campaign][platform]['GPS_INS'][flight_id](storage_options=kwds).to_dask().sel(
                time=slice(flight['takeoff'],flight['landing']))

        # load every timedelta-th P5 timestamp
        P5_timestamps = gpsins.time.values[::timedelta]

        # open amsr data (regular lat-lon grid with 0.25 deg resolution)
        amsr = xr.open_dataset(f'/net/secaire/mringel/amsr/amsr2/RSS_AMSR2_ocean_L3_daily_{flight_date[0:4]}-{flight_date[4:6]}-{flight_date[6:8]}_v08.2.nc')
        amsr = amsr.rename({'pass':'overpass'})

        amsr_lons, amsr_lats = np.meshgrid(amsr.lon.values,
                                           amsr.lat.values)

        # allocate lists for saving collocated amsr data
        lwp = []
        time_P5 = []
        time_amsr = []
        colloc_lons = []
        colloc_lats = []
        overpass = []
        colloc_counter = 0
        amsr_cell_ind = 0
        amsr_cell_inds = []
        # loop over all specified P5 timestamps
        for i,timestamp in enumerate(P5_timestamps):

            # get coordinates of P5 at current timestamp
            P5_lon = gpsins.sel(time=timestamp).lon.values
            P5_lat = gpsins.sel(time=timestamp).lat.values 

            # calculate distance of every AMSR gridcell to P5 location
            nn_ind, d = src.find_nn_gridcell(amsr_lons,amsr_lats,
                                             P5_lon,P5_lat)
            
            # only select AMSR gridcells closer than <d_min> km to P5
            amsr_aroundP5 = amsr.isel(lon=np.where(d<=d_max)[1],
                                      lat=np.where(d<=d_max)[0],drop=True)
            
            # select AMSR gridcell closest in time to P5 measurement
            if (len(amsr_aroundP5.lon.values) != 0) & (len(amsr_aroundP5.lat.values) != 0):
            
                delta_t = (timestamp-amsr_aroundP5.time.values).astype('timedelta64[m]')
                delta_t_min_ind = np.unravel_index(np.argmin(delta_t, axis=None), delta_t.shape)

                amsr_aroundP5_lons, amsr_aroundP5_lats = np.meshgrid(amsr_aroundP5.lon.values,
                                                                     amsr_aroundP5.lat.values)
                
                # calculate distance of every surrounding AMSR gridcell to P5 location
                nn_ind, d_close = src.find_nn_gridcell(amsr_aroundP5_lons,amsr_aroundP5_lats,
                                                       P5_lon,P5_lat)

                # check whether AMSR measurement is within ±90 minutes of P5 measurement
                if (np.abs(delta_t[delta_t_min_ind]) <= np.timedelta64(t_max,'m')):

                    # if temporal threshold fullfilled save amsr data to allocated lists
                    amsr_lwp = amsr_aroundP5.cloud_liquid_water.values[delta_t_min_ind]
                    amsr_lon = amsr_aroundP5_lons[nn_ind]
                    amsr_lat = amsr_aroundP5_lats[nn_ind]
                    amsr_time = amsr_aroundP5.time.values[delta_t_min_ind]

                    if len(amsr_cell_inds)==0:
                        amsr_lon_0 = amsr_lon
                        amsr_lat_0 = amsr_lat
                        amsr_time_0 = amsr_time

                    lwp.append(amsr_lwp)
                    colloc_lons.append(amsr_lon)
                    colloc_lats.append(amsr_lat)
                    time_P5.append(timestamp)
                    time_amsr.append(amsr_time)
                    overpass.append(delta_t_min_ind[0])                    

                    # check whether amsr grid cell has changed
                    if (amsr_time != amsr_time_0) or (amsr_lon != amsr_lon_0) or (amsr_lat != amsr_lat_0):
                        amsr_cell_ind += 1
                    
                    # append amsr gridcell index (for averaging corresponding P5 measurements later on)
                    amsr_cell_inds.append(flight_id+'_'+str(amsr_cell_ind))

                    amsr_lon_0 = amsr_lon
                    amsr_lat_0 = amsr_lat
                    amsr_time_0 = amsr_time

                    colloc_counter += 1

        print(f"{colloc_counter} collocations")
        print("")

        # save collocated amsr data to netcdf
        amsr_lwp_ds = xr.Dataset(
            data_vars=dict(
                amsr_time_var=(['time'],np.array(time_amsr)),
                amsr_overpass=(['time'],np.array(overpass)),
                lon=(['time'],colloc_lons),
                lat=(['time'],colloc_lats),
                lwp=(['time'],lwp),
                cell=(['time'],np.array(amsr_cell_inds))
                ),
            coords=dict(
                time=time_P5))
        
        amsr_lwp_ds.amsr_overpass.attrs['units'] = '0: Ascending overpass | 1: Descending Overpass'
        amsr_lwp_ds.lon.attrs['units'] = 'Degrees east'
        amsr_lwp_ds.lat.attrs['units'] = 'Degrees north'
        amsr_lwp_ds.lwp.attrs['units'] = 'g/m²'
        #amsr_lwp_ds.cell.attrs['long_name'] = 'Index of collocated AMSR cell'
        amsr_lwp_ds.time.encoding['units'] = "seconds since 1970-01-01"

        # saving xarray dataset to netcdf
        amsr_lwp_ds.to_netcdf('/net/secaire/mringel/data/lwp_collocated/lwp_amsr2/{0:s}_AMSR_lwp_v3.nc'.format(flight_id))

        # plot amsr lwp data on map around Svalbard along with P5 track and the coordinates of collocated amsr cells
        """
        for overpass in range(2):
      
            amsr_overpass_ds = amsr_lwp_ds.where(amsr_lwp_ds.amsr_overpass==overpass,drop=True)   

            if len(amsr_overpass_ds.time.values) != 0:
                
                fig = plt.figure(figsize=(6, 10))
                
                ax = plt.axes(projection=ccrs.NorthPolarStereo())
                ax.set_extent([-10.0, 20.0, 75.0, 83.0], crs=ccrs.PlateCarree())
                ax.coastlines(resolution='50m', color='black', linewidth=1)
                
                pc = ax.pcolormesh(amsr.lon,amsr.lat,amsr.isel(overpass=overpass).cloud_liquid_water.fillna(-9999.)*1000,
                                cmap='Blues',vmin=0,vmax=500,
                                zorder=1,
                                transform=ccrs.PlateCarree())
                fig.colorbar(pc,fraction=0.053, pad=0.04, label='LWP [g m$^{-2}$]')

                ax.scatter(gpsins.lon.values,
                        gpsins.lat.values,
                        color='black',
                        s=5,
                        zorder=2,
                        transform=ccrs.PlateCarree(),
                        label='P5 track')
                
                ax.scatter(amsr_overpass_ds.lon.values,amsr_overpass_ds.lat.values,
                        color='red',
                        s=1,
                        label='Collocations',
                        zorder=3,
                        transform=ccrs.PlateCarree()
                        )
                
                lgnd = ax.legend(loc='lower left')

                lgnd.legendHandles[0]._sizes = [50]
                lgnd.legendHandles[1]._sizes = [50]

                if overpass == 0:
                        ax.set_title(f'{flight_id}\nAMSR2 ascending overpass',fontsize=14)
                if overpass == 1:
                        ax.set_title(f'{flight_id}\nAMSR2 descending overpass')

                if overpass==0:
                    plt.savefig(f'/home/mringel/lwp_project/figures/Satellite_collocations/AMSR2/Colloc_AMSR2_P5_map_{flight_id}_ascending.png',
                                bbox_inches='tight',
                                dpi=200)
                if overpass==1:
                    plt.savefig(f'/home/mringel/lwp_project/figures/Satellite_collocations/AMSR2/Colloc_AMSR2_P5_map_{flight_id}_descending.png',
                                bbox_inches='tight',
                                dpi=200)
        """


