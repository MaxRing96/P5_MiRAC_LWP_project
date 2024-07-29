"""
Extraction of ERA-5 surface variables along the flight track. The viewing
geometry is not considered, only the sub-aircraft point is taken.
"""
import numpy as np
import xarray as xr
import os
import ac3airborne
from src.find_closest_gridcell import nearest_gridcell
from dotenv import load_dotenv

load_dotenv()


def clean_gps(ds):
    """Clean GPS data by dropping duplicate times and nan."""

    ds = ds.sel(time=~ds.indexes['time'].duplicated())
    ds = ds.sel(time=(~np.isnan(ds.lat)) | (~np.isnan(ds.lon)))
    
    return ds


if __name__ == '__main__':
    
    # read meta data
    meta = ac3airborne.get_flight_segments()
    cat = ac3airborne.get_intake_catalog()
    
    ac3cloud_username = os.environ['AC3_USER']
    ac3cloud_password = os.environ['AC3_PASSWORD']
    credentials = dict(user=ac3cloud_username, password=ac3cloud_password)
    
    kwds = {'simplecache': dict(
        cache_storage=os.environ['INTAKE_CACHE'],
        same_names=True,
    )}
    
    # select flights
    skip = ['HALO-AC3_HALO_RF00', 'HALO-AC3_HALO_RF01']  # non-Arctic flights
    flights = [(mission, platform, flight_id)
               for mission in meta.keys()
               if mission != 'PAMARCMiP'
               for platform in meta[mission].keys()
               for flight_id in meta[mission][platform].keys()
               if flight_id not in skip
               ]
    
    for mission, platform, flight_id in flights:
        
        print(flight_id)
        
        # flight meta information
        flight = meta[mission][platform][flight_id]
           
        # read gps data of flight
        try:
            ds_gps = cat[mission][platform]['GPS_INS'][flight_id](
                storage_options=kwds, **credentials).read()
            
        except TypeError:
            ds_gps = cat[mission][platform]['GPS_INS'][flight_id](
                storage_options=kwds).read()
        
        # read era5 data of the flight day
        file = os.path.join(
            os.environ['PATH_ERA5'],
            flight['date'].strftime('%Y/%m'),
            f'era5-single-levels_60n_{flight["date"].strftime("%Y%m%d")}.nc',
            )
        ds_era5 = xr.open_dataset(file)
        
        # clean gps data
        ds_gps = clean_gps(ds_gps)
        
        # create meshgrid of era5 grid
        lon_2d, lat_2d = np.meshgrid(ds_era5['longitude'], ds_era5['latitude'])
        
        # get index of nearest pixels
        indices1d, indices2d, distances = nearest_gridcell(
            lon_2d=lon_2d, 
            lat_2d=lat_2d, 
            lon_1d=ds_gps.lon.values, 
            lat_1d=ds_gps.lat.values,
            )
        
        assert (distances < 20000).all()
        
        # get era-5 data along track at the same hour
        ix_time = ds_gps.time.astype('datetime64[h]')
        ix_lat = ds_era5.latitude.isel(latitude=indices2d[0]).rename(
            {'latitude': 'time'})
        ix_lon = ds_era5.longitude.isel(longitude=indices2d[1]).rename(
            {'longitude': 'time'})
        
        ds_flight = ds_era5.sel(
            time=ix_time, longitude=ix_lon, latitude=ix_lat)
        
        # rename time in era5 time
        ds_flight = ds_flight.rename({'time': 'time_era5'})
        
        # add polar 5 variables
        ds_flight['time'] = (('time_era5'), ds_gps.time.values)
        ds_flight['longitude_p5'] = (('time_era5'), ds_gps.lon.values)
        ds_flight['latitude_p5'] = (('time_era5'), ds_gps.lat.values)
        
        # make polar 5 time the only coordinate
        ds_flight = ds_flight.swap_dims({'time_era5': 'time'})
        
        # coordinates to variables
        ds_flight = ds_flight.reset_coords()
        
        # save to file
        file_out = os.path.join(
            os.environ['PATH_CAMPAIGNS'],
            flight['mission'].lower(),
            flight['platform'].lower(),
            'era5',
            flight['mission']+'_'+flight['platform']+'_era5_single_levels_'+
            flight['date'].strftime('%Y%m%d')+'_'+flight['name']+'.nc'
            )
        ds_flight.to_netcdf(file_out)
