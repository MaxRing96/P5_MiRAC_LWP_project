#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:54:50 2023

@author: Mario Mech (mario.mech@uni-koeln.de)
"""

import os
from cdo import Cdo
import ac3airborne

import xarray as xr
import numpy as np
import pandas as pd

from scipy import spatial
import cartopy.crs as ccrs

from dotenv import load_dotenv
load_dotenv()

# path to save pamtra output
#data_path = os.environ['LOCAL_DATA']+'lwp_project/'

# needed for data stored in AC3 cloud
ac3cloud_username = os.environ['AC3_USER']
ac3cloud_password = os.environ['AC3_PASSWORD']

credentials = dict(user=ac3cloud_username, password=ac3cloud_password)

# local caching
kwds = {'simplecache': dict(
    cache_storage=os.environ['INTAKE_CACHE'], 
    same_names=True
)}

cat = ac3airborne.get_intake_catalog() # intake catalogues
meta = ac3airborne.get_flight_segments() # flight segmentation

def nearest_gridcell(lon_2d, lat_2d, lon_1d, lat_1d):
    """
    Find nearest gridcell and return the distance in meters. Note, that the
    distance is not the great-circle distance but the euclidian distance in
    the geocentric coordinate system.

    Parameters
    ----------
    lon_2d : np.ndarray
        Longitude of the grid.
    lat_2d : np.ndarray
        Latitude of the grid.
    lon_1d : np.ndarray
        Longitude of the trajectory.
    lat_1d : np.ndarray
        Latitude of the trajectory.

    Returns
    -------
    indices1d : np.ndarray 
        Index of the nearest grid cell for each trajectory point in the 
        flattened grid.
    indices2d : tupel
        Index of the nearest grid cell for each trajectory point for every
        dimension of the model grid. Can be also other than 2D.
    distance : TYPE
        Euclidian distance to the nearest grid cell.
    """
    
    # transform from geodetic to geocentric coordinates
    coords_2d = ccrs.Geocentric().transform_points(
        src_crs=ccrs.Geodetic(),
        x=lon_2d.ravel(),
        y=lat_2d.ravel(),
        z=np.zeros_like(lon_2d.ravel()),
        )
    
    coords_1d = ccrs.Geocentric().transform_points(
        src_crs=ccrs.Geodetic(),
        x=lon_1d,
        y=lat_1d,
        z=np.zeros_like(lon_1d),
        )
    
    # get indices of nearest grid cell
    tree = spatial.KDTree(coords_2d)
    distance, indices1d = tree.query(coords_1d)
    indices2d = np.unravel_index(indices1d, lon_2d.shape)
    
    return indices1d, indices2d, distance

def runCDO(flight_id=None,area=[-30,50,65,89],yyyy=2019,mm=4,dd=1,threads="32",outPath='/tmp/'):
    """
    cut area from ERA5 model output, selects timestep, transforms grid, and stores as netcdf files.
    
    area: array of integer giving the area of choice [minlon,maxlon,minlat,maxlat]
    yyyy: integer year
    mm:   integer month
    dd:   integer day of month
    timestep: integer starting with 1 = 0 UTC, 13 = 12 UTC
    threads: integer giving threads to be used for the cdo process
    outPath: string giving the path where to store the output
    """
    
    cdo = Cdo()
    yyyy = "%4d" % (yyyy,)
    mm = "%02d" % (mm,)
    dd = "%02d" % (dd,)
    area = ','.join([str(coord) for coord in area])
    era5_datetime = yyyy+mm+dd

    infiles = list()
    for var in ['130', '152']:
        cdo_string = "-sp2gpl -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"tmp_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append(outPath+"/tmp_ml_"+era5_datetime+"_"+var+".nc")
        
    for var in ['075', '076', '133', '246', '247']:
        cdo_string = "-setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"tmp_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append(outPath+"/tmp_ml_"+era5_datetime+"_"+var+".nc")
    
    for var in ['031','034','134','137','151','165','166','235']: #'078','079' total coloumn liqud and ice
        cdo_string = " -setgridtype,regular "+"/pool/data/ERA5/E5/sf/an/1H/"+var+"/E5sf00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"tmp_sf_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append(outPath+"/tmp_sf_"+era5_datetime+"_"+var+".nc")
    
    for var in ['129','172']: #'078','079' total coloumn liqud and ice
        cdo_string = " -setgridtype,regular "+"/pool/data/ERA5/E5/sf/an/IV/"+var+"/E5sf00_IV_2000-01-01_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"tmp_sf_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append(outPath+"/tmp_sf_"+era5_datetime+"_"+var+".nc")

    cdo.merge(input=infiles, output=outPath+'field_'+flight_id+'.nc')
    for file in infiles:
        os.remove(file)
    
    return

from os.path import exists

#for mission in ['ACLOUD', 'AFLUX', 'MOSAiC-ACA', 'HALO-AC3']:
#    for flight in list(cat[mission]['P5']['GPS_INS']):
for mission in ['HALO-AC3']:
#    for flight in list(cat[mission]['HALO']['GPS_INS']):
    for flight in ['HALO-AC3_HALO_RF16']:
        if mission not in ['HALO-AC3']:
            ds_gps = cat[mission]['HALO']['GPS_INS'][flight].to_dask()
        else:
            ds_gps = cat[mission]['HALO']['GPS_INS'][flight](storage_options=kwds,**credentials).to_dask()
        
        # remove duplicates in gps data
        ds_gps = ds_gps.sel(time=~ds_gps.indexes['time'].duplicated())

        # drop where lon or lat is nan
        ds_gps = ds_gps.sel(time=(~np.isnan(ds_gps.lat)) | (~np.isnan(ds_gps.lon)))
        if not exists('/scratch/b/b380702/tmp/field_'+flight+'.nc'):
            lon_min = ds_gps.lon.values.min()-0.25
            lon_max = ds_gps.lon.values.max()+0.25
            lat_min = ds_gps.lat.values.min()-0.25
            lat_max = ds_gps.lat.values.max()+0.25
            year = pd.Timestamp(ds_gps['time'].values[0]).year
            month = pd.Timestamp(ds_gps['time'].values[0]).month
            day = pd.Timestamp(ds_gps['time'].values[0]).day
            start_hour = pd.Timestamp(ds_gps['time'].values[0]).hour
            end_hour = pd.Timestamp(ds_gps['time'].values[-1]).hour+1
#            runCDO(flight_id=flight,area=[-12,25,75,84],yyyy=year,mm=month,dd=day,outPath='/scratch/b/b380702/tmp/')
            runCDO(flight_id=flight,area=[lon_min, lon_max, lat_min, lat_max],yyyy=year,mm=month,dd=day,outPath='/scratch/b/b380702/tmp/')

ds_era5 = xr.open_dataset('/scratch/b/b380702/tmp/field_'+flight+'.nc')

# create meshgrid of lat/lon
lon_2d, lat_2d = np.meshgrid(ds_era5['lon'], ds_era5['lat'])

# get index of nearest pixels
indices1d, indices2d, distances = nearest_gridcell(
    lon_2d=lon_2d, 
    lat_2d=lat_2d, 
    lon_1d=ds_gps.lon.values, 
    lat_1d=ds_gps.lat.values,
    )

# get era-5 data along track at the closest hour
ix_time = ds_gps.time.dt.round('H')
ix_lat = ds_era5.lat.isel(lat=indices2d[0]).rename({'lat': 'time'})
ix_lon = ds_era5.lon.isel(lon=indices2d[1]).rename({'lon': 'time'})

ds_flight = ds_era5.sel(time=ix_time, lon=ix_lon, lat=ix_lat)

# rename time in era5 time
ds_flight = ds_flight.rename({'time': 'time_era5'})

# add polar 5 variables
ds_flight['time'] = (('time_era5'), ds_gps.time.values)
ds_flight['longitude_p5'] = (('time_era5'), ds_gps.lon.values)
ds_flight['latitude_p5'] = (('time_era5'), ds_gps.lat.values)

# make polar 5 time the only coordinate
ds_flight = ds_flight.swap_dims({'time_era5': 'time'})

# coordinates to variables
ds_flight = ds_flight.reset_coords(names=['time_era5'])

ds_flight = ds_flight.rename({'var31':'ci','var34':'sst','var129':'z','var134':'sp','var137':'tcwv','var151':'msl','var165':'u10','var166':'v10','var172':'lsm','var235':'skt'})

# save to file
path_out = '/scratch/b/b380702/tmp/'
outfile = path_out+'ERA5_'+flight+'.nc'
ds_flight.to_netcdf(outfile)
