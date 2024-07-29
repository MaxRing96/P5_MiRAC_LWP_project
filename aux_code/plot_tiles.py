# 2017 for i in `seq 143 180`; do ii=`printf "%03d" $i`;python ~/projects/modis_products/plot_tiles.py 2017$ii >> ~/projects/modis_products/granules_myd.txt;done
# 2019 for i in `seq 78 101`; do ii=`printf "%03d" $i`;python ~/projects/modis_products/plot_tiles.py 2019$ii >> ~/projects/modis_products/granules_myd.txt;done
# 2020 for i in `seq 243 257`; do ii=`printf "%03d" $i`;python ~/projects/modis_products/plot_tiles.py 2020$ii >> ~/projects/modis_products/granules_myd.txt;done
# 2022 for i in `seq 79 100`; do ii=`printf "%03d" $i`;python ~/projects/modis_products/plot_tiles.py 2022$ii >> ~/projects/modis_products/granules_myd.txt;done

import sys
from pyhdf.SD import SD, SDC
import geopandas
from shapely.geometry import Polygon
import glob
from cartopy import crs as ccrs
from matplotlib import pyplot as plt

def plot_tiles(filenames='/work2/tmp2_modis/MOD03.A2022085.*'):
    map_proj = ccrs.Orthographic(central_latitude=90.0, central_longitude=0.0)
    #map_proj = ccrs.NorthPolarStereo()
    map_proj._threshold /= 100.

    fig, axs = plt.subplots(4,4,subplot_kw={'projection': map_proj},figsize=(11,11))
    ax=axs.flatten()

    # These 2 lines of code grab extents in projection coordinates
    lonlatproj = ccrs.PlateCarree()
    _, y_min = map_proj.transform_point(0, 60, lonlatproj)  #(0.0, -3189068.5)
    x_max, _ = map_proj.transform_point(90, 60, lonlatproj) #(3189068.5, 0)

    # prep extents of the axis to plot map
    pad = 25000
    xmin,xmax,ymin,ymax = y_min-pad, x_max+pad, y_min-pad, x_max+pad
    files = sorted(glob.glob(filenames))
    date = files[0].split('/')[-1].split('.')[1][1::]
    sat = files[0].split('/')[-1].split('.')[0]
    times = [date]
    for i,file in enumerate(files):
        #ax[i] = plt.axes(projection=map_proj)

#        ax[i].set_global()
        timestamp = file.split('/')[-1].split('.')[2]
        times.append(timestamp)
        ax[i].gridlines()

        ax[i].coastlines(linewidth=0.5, color='k', resolution='50m')
        ax[i].set_title(timestamp)

        ac3area = geopandas.GeoDataFrame({'name': ['Svalbard'],
                                          'geometry': [Polygon(
                                              [(-7.5, 76.5), (21.5, 76.5), (21.5, 84.5), (-7.5, 84.5), (-7.5, 76.5)])]})
        ax[i].add_geometries(ac3area.geometry, ec='r', lw=1, fc='none', crs=ccrs.Geodetic())
        try:
            geos = SD(file,SDC.READ)
        except:
            continue

        lat = geos.select('Latitude').get()
        lon = geos.select('Longitude').get()
        geojson = {"type": "Polygon",
            "coordinates": [
            [
                [lon[0, 0], lat[0, 0]], [lon[-1, 0], lat[-1, 0]],
                [lon[-1, -1], lat[-1, -1]], [lon[0, -1], lat[0, -1]], [lon[0, 0], lat[0, 0]]
            ]
        ]}
        granule = geopandas.GeoDataFrame({'name':['Granule'],'geometry' : [Polygon(geojson['coordinates'][0])]})
        ax[i].add_geometries(granule.geometry, ec='g',lw=1, fc='none', crs=ccrs.Geodetic())
        ax[i].set_extent([xmin, xmax, ymin, ymax], crs=map_proj)
        geos.end()

    #fig.delaxes(ax[-1])
    plt.suptitle(date)

    plt.savefig('/work2/modis_plots/' + sat + '_' + date + '_granules.png')

    return times

filepattern = sys.argv[1]

table_string = plot_tiles(filenames='/work2/tmp2_modis/MYD03.A' + filepattern + '.*')
print(table_string)