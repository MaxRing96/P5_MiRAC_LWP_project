from pyhdf.SD import SD, SDC
#import numpy as np
from cartopy import crs as ccrs, feature as cfeature
from matplotlib import pyplot as plt
import xarray as xr
from osgeo import gdal
import rioxarray
import read_seaice
from find_closest_gridcell import nearest_gridcell

def plot_map(ds,name):
    data_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
    ax.set_global()
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.pcolormesh(ds['lon'], ds['lat'], ds[name], transform=data_crs)

    return

def create_dataset(coord_file,sdc_file,varnames = ['Cloud_Water_Path']):
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
        data = data_file.select(name).get()
        ds[name] = (('x','y'),data)
    return ds

#cfile = '/home/mech/Downloads/MOD03.A2022091.0035.061.2022091072227.hdf'
#dfile = '/home/mech/Downloads/MOD06_L2.A2022091.0035.061.2022095004714.hdf'
cfile = '/work2/tmp2_modis/MYD03.A2022090.1330.061.2022091150505.hdf'
dfile = '/work2/modis_lwp/MYD06_L2.A2022090.1330.061.2022091152418.hdf'

ds = create_dataset(cfile,dfile)

# sifile = SD('/home/mech/Downloads/MOD29.A2022091.0035.061.2022091133705.hdf', SDC.READ)
# datasets_dic = sifile.datasets()
# for idx,sds in enumerate(datasets_dic.keys()):
#     print(idx,sds)
seaice = read_seaice.read('20220401','/data/obs/campaigns/halo-ac3/auxiliary/sea_ice/daily_grid/')
# get index of nearest pixels
indices1d, indices2d, distances = nearest_gridcell(
    lon_2d=seaice['lon'].values,
    lat_2d=seaice['lat'].values,
    lon_1d=ds.lon.values.flatten(),
    lat_1d=ds.lat.values.flatten(),
    )

# get sea ice with the new index
ds['sic'] = (('x','y'), seaice.sic.values.flatten()[indices1d].reshape(2030,1354))

plot_map(ds,'sic')





