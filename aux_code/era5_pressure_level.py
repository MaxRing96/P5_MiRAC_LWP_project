"""
Script extracts ERA-5 surface variables along the flight track

Currently: ERA-5 only for HALO flight days available

So far, only for one flight ERA5 data is available
"""


import numpy as np
import xarray as xr
import os
import ac3airborne
from src.find_closest_gridcell import nearest_gridcell


if __name__ == '__main__':
    
    path_base = '/home/nrisse/igmk_work_gha/data/era5/'
    
    # read meta data
    meta = ac3airborne.get_flight_segments()
    cat = ac3airborne.get_intake_catalog()
    
    ac3cloud_username = os.environ['AC3_USER']
    ac3cloud_password = os.environ['AC3_PASSWORD']
    credentials = dict(user=ac3cloud_username, password=ac3cloud_password)
    
    kwds = {'simplecache': dict(
        cache_storage='/home/nrisse/igmk_work_sec/intake-cache/',
        same_names=True,
    )}
    
    # read ERA-5 for HALO-AC3
    file = path_base+'era5_pre_hourly_rf03_20220313_0700_20220313_1700.grib'
    ds_era5 = xr.open_dataset(file)
    
    #%% calculate sea ice on path for each flight
    for mission in meta.keys():
        
        if mission != 'HALO-AC3':
            continue
    
        for platform in meta[mission].keys():
            
            if platform != 'HALO':
                continue
            
            for flight_id, flight in meta[mission][platform].items():
                
                if flight_id in ['HALO-AC3_HALO_RF00', 'HALO-AC3_HALO_RF01']:  # test flight and transfer flight
                    continue
                
                if flight_id not in ['HALO-AC3_HALO_RF03']:
                    continue
                
                print(flight_id)
                                
                try:
                    ds_gps = cat[mission][platform]['GPS_INS'][flight_id](
                        storage_options=kwds, **credentials).read()
                    
                except TypeError:
                    ds_gps = cat[mission][platform]['GPS_INS'][flight_id](
                        storage_options=kwds).read()
                
                # remove duplicates in gps data
                ds_gps = ds_gps.sel(time=~ds_gps.indexes['time'].duplicated())
                
                # drop where lon or lat is nan
                ds_gps = ds_gps.sel(time=(~np.isnan(ds_gps.lat)) | (~np.isnan(ds_gps.lon)))
                
                # create meshgrid of lat/lon
                lon_2d, lat_2d = np.meshgrid(ds_era5['longitude'], ds_era5['latitude'])
                
                # get index of nearest pixels
                indices1d, indices2d, distances = nearest_gridcell(
                    lon_2d=lon_2d, 
                    lat_2d=lat_2d, 
                    lon_1d=ds_gps.lon.values, 
                    lat_1d=ds_gps.lat.values,
                    )

                # get era-5 data along track at the same hour
                ix_time = ds_gps.time.astype('datetime64[h]')
                ix_lat = ds_era5.latitude.isel(latitude=indices2d[0]).rename({'latitude': 'time'})
                ix_lon = ds_era5.longitude.isel(longitude=indices2d[1]).rename({'longitude': 'time'})
                
                ds_flight = ds_era5.sel(time=ix_time, longitude=ix_lon, latitude=ix_lat)
                                
                # rename time in era5 time
                ds_flight = ds_flight.rename({'time': 'time_era5'})
                
                # add polar 5 variables
                ds_flight['time'] = (('time_era5'), ds_gps.time.values)
                ds_flight['longitude_p5'] = (('time_era5'), ds_gps.lon.values)
                ds_flight['latitude_p5'] = (('time_era5'), ds_gps.lat.values)
                
                # make polar 5 time the only coordinate
                ds_flight = ds_flight.swap_dims({'time_era5': 'time'})
                
                # coordinates to variables
                ds_flight = ds_flight.reset_coords(names=['time_era5', 'longitude', 'latitude', 'valid_time', 'number', 'step'])
                
                # save to file
                path_out = f'{path_base}along_track/{flight["mission"].lower()}/{flight["platform"].lower()}/'
                outfile = path_out+flight['mission']+'_'+flight['platform']+'_era5_pressure_level_'+flight['date'].strftime('%Y%m%d')+'_'+flight['name']+'.nc'
                ds_flight.to_netcdf(outfile)
