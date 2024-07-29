import numpy as np
from pyhdf.SD import SD, SDC
from cartopy import crs as ccrs, feature as cfeature
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os
import ac3airborne
import xarray as xr
import datetime
import pandas as pd
import typhon as ty 

#### load ac3airborne meta data
load_dotenv(dotenv_path='/home/mringel/lwp_project/code/.env')

# needed for data stored in AC3 cloud
ac3cloud_username = os.environ['AC3_USER']
ac3cloud_password = os.environ['AC3_PASSWORD']

credentials = dict(user=ac3cloud_username, password=ac3cloud_password)

# local caching
kwds = {'simplecache': dict(
    cache_storage=os.environ['INTAKE_CACHE'],
    #cache_storage='/home/mringel/lwp_project/data/', 
    same_names=True
)}

cat = ac3airborne.get_intake_catalog() # intake catalogues
meta = ac3airborne.get_flight_segments() # flight segmentation
#####

def find_nearest_arrindex(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nn_gridcell(lon_grid, lat_grid, lon_point, lat_point):
    """
    Caclulate the 2D distances in km between a given point coordinate and a 2D grid
    (both given in decimal degrees) on the sphere of Earth, using the harvesine function.
    """
    # convert lons and lats from decimal degrees to radians for calculation
    lon_grid = np.deg2rad(lon_grid)
    lat_grid = np.deg2rad(lat_grid)
    lon_point = np.deg2rad(lon_point)
    lat_point = np.deg2rad(lat_point)

    # create array of point lons, lats of model grid shape
    lon_point_arr = np.zeros([lon_grid.shape[0],lon_grid.shape[1]])
    lat_point_arr = np.zeros([lat_grid.shape[0],lat_grid.shape[1]])
    lon_point_arr[:,:] = lon_point
    lat_point_arr[:,:] = lat_point

    # calculate lon, lat distances between model and point
    dlon = lon_point_arr - lon_grid
    dlat = lat_point_arr - lat_grid 

    # calculate 2D distances between model and 
    # point on the sphere of earth using the harvesine function
    a = np.sin(dlat/2)**2 + np.cos(lat_grid) * np.cos(lat_point) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    d = c * r

    # returning model grid index of closest grid box 
    # to given point coordinate and the corresponding distance
    nn_ind = np.unravel_index(np.argmin(d, axis=None), d.shape)

    return nn_ind, d

def plot_map(ds,varname,area,title=None):
    data_crs = ccrs.PlateCarree()
    if area == 'Global':
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
        ax.set_global()
    if area == 'Svalbard':
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        ax.set_extent([-30.0, 30.0, 67.0, 90.0], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.pcolormesh(ds['lon'], ds['lat'], ds[varname], transform=data_crs)
    if title is not None:
        ax.set_title(title)

    return ax

def normed_log10_pdf_v0(var,bin_nr=10,bin_start=-1,bin_end=4):
    """
    Function to calculate relative frequencies of occurence (pdf)
    for a specified variable in logarthmic space, for specified 
    range and bins.
    """
    
    bins = np.linspace(start=bin_start, stop=bin_end, num=bin_nr)
    
    # count zeros and exclude them from variable
    n_zeros = len(var[var==0.])
    var = var[var!=0.]
    
    n_absolute, bins = np.histogram(np.log10(var),bins=bins)
    
    n_total = np.sum(n_absolute)+n_zeros
    
    prop_zero = n_zeros/n_total
    
    n_normed = np.array([n_absolute[i]/n_total for i in range(len(n_absolute))])
    n_normed[n_normed==0]=np.nan
    
    return n_normed, prop_zero, bins

def create_modis_1km_dataset(coord_file,sdc_file,varnames = ['Cloud_Water_Path']):
    
    geo_file = SD(coord_file, SDC.READ)

    lat = geo_file.select('Latitude').get()
    lon = geo_file.select('Longitude').get()

    ds = xr.Dataset(
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat)
        ),
        attrs=dict(description="MODIS on 1 km grid."),
    )
    data_file = SD(sdc_file, SDC.READ)

    for name in varnames:
        try: 
            data = data_file.select(name).get()
        except:
            data = coord_file.select(name).get()
        ds[name] = (('x','y'),data)
    
    return ds

def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.datetime.utcfromtimestamp(timestamp)

def get_ERA5_along_P5(flight_id,timestamps=None):
    
    campaign = flight_id.split('_')[0]
    platform = flight_id.split('_')[1]

    flight = meta[campaign][platform][flight_id]

    # load gps and amsr file corresponding to given P5 flight id
    try:
        gpsins = cat[campaign][platform]['GPS_INS'][flight_id](**credentials,storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
    except:
        gpsins = cat[campaign][platform]['GPS_INS'][flight_id](storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
    try:
        amsr2_sic = cat[campaign][platform]['AMSR2_SIC'][flight_id](**credentials,storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
    except:
        amsr2_sic = cat[campaign][platform]['AMSR2_SIC'][flight_id](storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))

    # load ERA5-based PAMTRA simulations along P5 flight
    pamtra_era5 = xr.open_dataset(f'/net/secaire/mech/data/era5_pamtra/flights/ERA5_{flight_id}_passive.nc')

    # convert timestamps of era5 (hours) to corresponding collocated P5 timestamps (seconds)
    try:
        pamtra_era5 = pamtra_era5.assign_coords(time=gpsins.time)
    except:
        pamtra_era5 = pamtra_era5.assign_coords(time=amsr2_sic.time)

    # select era5 at specified timestamps (e.g. of retrieved LWP)
    if timestamps is not None:
        pamtra_era5 = pamtra_era5.sel(time = timestamps)
    
    cwp_era5_all = pamtra_era5.wp.values[:,0,:,0]
    crp_era5_all = pamtra_era5.wp.values[:,0,:,2]
    
    lons = pamtra_era5.lon.values[:]
    lats = pamtra_era5.lat.values[:]

    levels = pamtra_era5.outlevels.values

    # loop through all PAMTRA-ERA5 timestamps and get ERA5 path values
    # at altitude closest to P5 altitude (given by gps file)
    cwp = np.zeros(len(pamtra_era5.time))
    crp = np.zeros(len(pamtra_era5.time))
    lwp = np.zeros(len(pamtra_era5.time))
    for i, ts in enumerate(pamtra_era5.time.values):

        alt = gpsins.sel(time=ts).alt.values
        alt_ind = find_nearest_arrindex(levels,alt)
        
        cwp[i] = cwp_era5_all[i,alt_ind]
        crp[i] = crp_era5_all[i,alt_ind]
        lwp[i] = cwp[i] + crp[i]

    ERA5_wp = xr.Dataset(data_vars=dict(
        cwp=(['time'],cwp),
        crp=(['time'],crp),
        lwp=(['time'],lwp),
        lon=(['time'],lons),
        lat=(['time'],lats)
        ),coords=dict(time=pamtra_era5.time))
    ERA5_wp.cwp.attrs['units'] = 'kg/m²'
    ERA5_wp.crp.attrs['units'] = 'kg/m²'
    ERA5_wp.lwp.attrs['units'] = 'kg/m²'
    ERA5_wp.time.encoding['units'] = "seconds since 1970-01-01"

    return ERA5_wp

def get_MODIS_along_P5(flight_id,satellite,mask_seaice=False):

    campaign = flight_id.split('_')[0]
    platform = flight_id.split('_')[1]

    flight = meta[campaign][platform][flight_id]
    
    lwp_modis = xr.open_dataset(f'/net/secaire/mringel/data/lwp_collocated/lwp_modis/{flight_id}_MODIS_{satellite}_lwp.nc')

    if mask_seaice is True:

        try:
            amsr2_sic = cat[campaign][platform]['AMSR2_SIC'][flight_id](**credentials,storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
        except:
            amsr2_sic = cat[campaign][platform]['AMSR2_SIC'][flight_id](storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
            
        sic = amsr2_sic.sel(time=lwp_modis.time,method='nearest',tolerance=np.timedelta64(1, 'm')).sic
        lwp_modis = lwp_modis.where(sic==0,drop=True)

    return lwp_modis

def get_identical_times(ds_1,ds_2,drop_others=True):

    mask = np.isin(ds_1.time.values,ds_2.time.values)
    mask = xr.DataArray(mask,coords={'time':ds_1.time.values})

    ds_1 = ds_1.where(mask==True,drop=drop_others)

    return ds_1

def make_timerange(timestamp,delta_t):
    
    time = pd.to_datetime(timestamp)-datetime.timedelta(seconds=delta_t)
    end = pd.to_datetime(timestamp)+datetime.timedelta(seconds=delta_t+1)
    step = datetime.timedelta(seconds=1)

    timerange = []

    while time < end:
        timerange.append(time)
        time += step

    return np.array(timerange)

def get_ds_at_times_of_CWT(ds,cwt):

    cwts = {
        'W':        1.,
        'NW':       2.,
        'N':        3.,
        'NE':       4,
        'E':        5.,
        'SE':       6.,
        'S':        7.,
        'SW':       8.,
        'cycl':     9.,
        'anticycl': 10.
    }

    CWT_3h_ds = xr.open_mfdataset('/home/mringel/lwp_project/data/cwt_mcao_ind/cwt_is_*_850.nc').compute()

    CWT_3h_time = CWT_3h_ds.where(CWT_3h_ds.CWT==cwts[cwt],drop=True).time.values

    CWT_1s_time = make_timerange(CWT_3h_time[0],delta_t=90*60)
    for j in range(1,len(CWT_3h_time)):

        timerange = make_timerange(CWT_3h_time[j],delta_t=90*60)

        CWT_1s_time = np.concatenate((CWT_1s_time,timerange))

    CWT = sorted(set(CWT_1s_time))

    CWT_ds = xr.DataArray(CWT,coords={'time':CWT})

    ds_new = ds.reindex(time=CWT_ds.time)

    ds_new = ds_new.isel(time=np.argwhere(~np.isnan(ds_new.lwp.values)).squeeze(),drop=True)

    return ds_new


def get_P5_around_AMSR_gridcells(flight_id):

    p5 = xr.open_dataset(f'/home/mringel/lwp_project/data/lwp_retrieved/{flight_id}_lwp.nc')
    amsr = xr.open_dataset(f'/home/mringel/lwp_project/data/lwp_collocated/lwp_amsr2/{flight_id}_AMSR_lwp_v2.nc')

    if len(amsr.time) == 0:
        print(f"No AMSR collocations for {flight_id}!")
    
    amsr_sel = get_identical_times(amsr,p5,drop_others=True)

    cells = [list(amsr_sel.groupby(amsr_sel.cell))[i][0] 
            for i in range(len(amsr_sel.groupby(amsr_sel.cell)))]

    amsr_lwp = []
    amsr_time = []

    p5_lwp_mean = []
    p5_lwp_std = []
    p5_lwp_var = []
    p5_lwp_min = []
    p5_lwp_max = []

    for cell in cells:

        amsr_time_j = amsr_sel.where(amsr_sel.cell==cell,drop=True).amsr_time_var.values[0]
        amsr_lwp_j = amsr_sel.where(amsr_sel.cell==cell,drop=True).lwp.values[0]

        p5_lwp_mean_j = np.nanmean(p5.where(amsr_sel.cell==cell,drop=True).lwp.values)
        p5_lwp_std_j = np.nanstd(p5.where(amsr_sel.cell==cell,drop=True).lwp.values)
        p5_lwp_var_j = np.nanvar(p5.where(amsr_sel.cell==cell,drop=True).lwp.values)
        p5_lwp_min_j = np.nanmin(p5.where(amsr_sel.cell==cell,drop=True).lwp.values)
        p5_lwp_max_j = np.nanmax(p5.where(amsr_sel.cell==cell,drop=True).lwp.values)
        
        amsr_time.append(amsr_time_j)
        amsr_lwp.append(amsr_lwp_j)

        p5_lwp_mean.append(p5_lwp_mean_j)
        p5_lwp_std.append(p5_lwp_std_j)
        p5_lwp_var.append(p5_lwp_var_j)
        p5_lwp_min.append(p5_lwp_min_j)
        p5_lwp_max.append(p5_lwp_max_j)

    dataset = xr.Dataset(
        data_vars=dict(
            time_amsr=(['amsr_cell'],np.array(amsr_time)),
            lwp_amsr=(['amsr_cell'],np.array(amsr_lwp)),
            lwp_p5_mean=(['amsr_cell'],np.array(p5_lwp_mean)),
            lwp_p5_std=(['amsr_cell'],np.array(p5_lwp_std)),
            lwp_p5_var=(['amsr_cell'],np.array(p5_lwp_var)),
            lwp_p5_min=(['amsr_cell'],np.array(p5_lwp_min)),
            lwp_p5_max=(['amsr_cell'],np.array(p5_lwp_max)),
            ),
            coords=dict(
                amsr_cell=np.array(cells))
            )
    
    return dataset

def integrate_lwc(lwc,p,T,q,z,z_max=None,eq_distant=True,rho_moist=False,axis=2):
    
    # calculates the wet density by the virtual temperature
    if rho_moist==True:
        Tv = T*(1+0.61*q)
        rho = ty.physics.density(p,Tv)
    # dry density
    else:
        rho = ty.physics.density(p,T)
    
    # height difference = integration length
    if eq_distant == True: # constant height diff
        dz = np.diff(z)[0]
    if eq_distant == False: # height diff vector
        dz = np.diff(z,prepend=0)
        
    # find height grid-index of maximum height, if specified
    if z_max is not None:
        z_diffs = np.absolute(z - z_max)
        z_max_index = np.argmin(z_diffs)+1
    if z_max is None:
        z_max_index = None
    
    
    # Vertically integrate frozen/liquid hydrometeors and water vapor
    lwp = np.nansum(lwc[:z_max_index] * rho[:z_max_index] * dz)
    
    return lwp

def get_ac3_meta():

    load_dotenv(dotenv_path='/home/mringel/lwp_project/code/.env')

    # needed for data stored in AC3 cloud
    ac3cloud_username = os.environ['AC3_USER']
    ac3cloud_password = os.environ['AC3_PASSWORD']

    credentials = dict(user=ac3cloud_username, password=ac3cloud_password)

    # local caching
    kwds = {'simplecache': dict(
        cache_storage=os.environ['INTAKE_CACHE'],
        #cache_storage='/home/mringel/lwp_project/data/', 
        same_names=True
    )}

    cat = ac3airborne.get_intake_catalog() # intake catalogues
    meta = ac3airborne.get_flight_segments() # flight segmentation

    return cat, meta, credentials, kwds

def get_GPS_along_track(flight_id):

    campaign = flight_id.split("_")[0]
    platform = flight_id.split("_")[1]

    flight = meta[campaign][platform][flight_id]

    try:
        gps = cat[campaign][platform]['GPS_INS'][flight_id](**credentials,storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
    except:
        gps = cat[campaign][platform]['GPS_INS'][flight_id](storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))

    # clean gps data
    gps = gps.sel(time=~gps.indexes['time'].duplicated())
    gps = gps.sel(time=(~np.isnan(gps.lat)) | (~np.isnan(gps.lon)))

    return gps

def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.datetime.utcfromtimestamp(timestamp)