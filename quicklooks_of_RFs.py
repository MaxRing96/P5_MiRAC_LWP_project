import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
import ac3airborne
import os
import src
import typhon as ty
import datetime

# use typhon ploting style
plt.style.use(ty.plots.styles.get('typhon'))

load_dotenv(dotenv_path='/home/mringel/lwp_project/code/.env')

# needed for data stored in AC3 cloud
ac3cloud_username = os.environ['AC3_USER']
ac3cloud_password = os.environ['AC3_PASSWORD']

credentials = dict(user=ac3cloud_username, password=ac3cloud_password)

# local caching
kwds = {'simp lecache': dict(
    cache_storage=os.environ['INTAKE_CACHE'],
    #cache_storage='/home/mringel/lwp_project/data/', 
    same_names=True
)}

cat = ac3airborne.get_intake_catalog() # intake catalogues
meta = ac3airborne.get_flight_segments() # flight segmentation

platform='P5'
instrument = 'MiRAC-A'
for campaign in list(cat):   
    if campaign == 'PAMARCMiP': continue
    for flight_id in list(cat[campaign][platform][instrument]):
        if flight_id in ['ACLOUD_P5_RF04','ACLOUD_P5_RF06','ACLOUD_P5_RF11','ACLOUD_P5_RF14',
                         'ACLOUD_P5_RF15','ACLOUD_P5_RF16','ACLOUD_P5_RF17','ACLOUD_P5_RF21',
                         'ACLOUD_P5_RF22','ACLOUD_P5_RF23','ACLOUD_P5_RF25','AFLUX_P5_RF09',
                         'AFLUX_P5_RF10','AFLUX_P5_RF11','AFLUX_P5_RF13','AFLUX_P5_RF13',
                         'MOSAiC-ACA_P5_RF09','HALO-AC3_P5_RF01']: continue
        print("")
        print(flight_id)

        flight = meta[campaign][platform][flight_id]

        try:
            miraca = cat[campaign][platform]['MiRAC-A'][flight_id](**credentials,storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
        except:
            miraca = cat[campaign][platform]['MiRAC-A'][flight_id](storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
        try:
            amsr2_sic = cat[campaign][platform]['AMSR2_SIC'][flight_id](**credentials,storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))
        except:
            amsr2_sic = cat[campaign][platform]['AMSR2_SIC'][flight_id](storage_options=kwds).to_dask().sel(time=slice(flight['takeoff'],flight['landing']))

        lwp_p5_ds = xr.open_dataset(f'/home/mringel/lwp_project/data/lwp_retrieved/{flight_id}_lwp.nc')
        lwp_era5_ds = src.get_ERA5_along_P5(flight_id)
        lwp_amsr_ds = xr.open_dataset(f'/home/mringel/lwp_project/data/lwp_collocated/lwp_amsr2/{flight_id}_AMSR_lwp.nc')
        try:
            lwp_aqua_ds = src.get_MODIS_along_P5(flight_id,satellite='aqua',mask_seaice=False)
        except FileNotFoundError:
            lwp_aqua_ds = None
        try: 
            lwp_terra_ds = src.get_MODIS_along_P5(flight_id,satellite='terra',mask_seaice=False)
        except FileNotFoundError:
            lwp_terra_ds = None

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(17,9), sharex=True, gridspec_kw=dict(height_ratios=(1, 1, 1, 0.1)))

        # 1st: plot flight altitude and radar reflectivity
        ax1.plot(miraca.time, miraca.alt*1e-3, label='Flight altitude', color='k')

        im = ax1.pcolormesh(miraca.time, miraca.height*1e-3, 10*np.log10(miraca.Ze).T, vmin=-40, vmax=30, cmap='jet', shading='nearest')
        fig.colorbar(im, ax=ax1, pad=-0.17, label='Radar reflectivity [dBz]')
        ax1.set_ylabel('Height [km]')
        legend = ax1.legend(frameon=True, loc='upper left')
        frame = legend.get_frame()
        frame.set_facecolor('white')

        # 2nd: plot 89 GHz TBa
        ax2.grid(0.15)
        ax2.plot(miraca.time, miraca.tb, label='Tb(89 GHz)', color='k')
        ax2.set_ylabel('$T_b$ [K]')
        ax2.legend(frameon=False,
                bbox_to_anchor=(1.01, 0.5), 
                loc='center left',
                fontsize=13)

        if lwp_aqua_ds is not None:
            lwp_aqua = lwp_aqua_ds.lwp.values
            lwp_aqua[lwp_aqua==-9999] = np.nan
        if lwp_terra_ds is not None:
            lwp_terra = lwp_terra_ds.lwp.values
            lwp_terra[lwp_terra==-9999] = np.nan

        # add NaN values for missing timestamps 
        lwp_p5 = np.empty(len(miraca.time.values))
        for i, ts in enumerate(miraca.time.values):
            try:
                lwp_p5[i] = lwp_p5_ds.sel(time=ts).lwp.values
            except:
                lwp_p5[i] = np.nan

        # 3rd: plot LWP
        ax3.grid(0.15)
        ax3.axhline(y=0,linestyle='dashed',color='gray',alpha=0.75)
        ax3.plot(lwp_era5_ds.time,lwp_era5_ds.lwp*1000,linestyle='-',label='ERA5',color='C0')
        if lwp_aqua_ds is not None:
            ax3.plot(lwp_aqua_ds.time,lwp_aqua,color='darkred',label='MODIS - Aqua')
        if lwp_terra_ds is not None:
            ax3.plot(lwp_terra_ds.time,lwp_terra,color='tomato',label='MODIS - Terra')
        ax3.plot(lwp_amsr_ds.time,lwp_amsr_ds.lwp.values*1000,color='m',label='AMSR2 - GCOM-W1')
        ax3.plot(miraca.time,lwp_p5*1000,color='green',label='MiRAC - P5')
        ax3.set_ylabel('LWP [g m$^{-2}$]')
        ax3.legend(frameon=False,
                bbox_to_anchor=(1.01, 0.5), 
                loc='center left',
                ncol=1,
                fontsize=13)

        # plot AMSR2 sea ice concentration
        im = ax4.pcolormesh(amsr2_sic.time,
                    np.array([0, 1]),
                    np.array([amsr2_sic.sic,amsr2_sic.sic]), cmap='Blues_r', vmin=0, vmax=100,
                    shading='auto')
        cax = fig.add_axes([0.87, 0.075, 0.1, ax4.get_position().height])
        fig.colorbar(im, cax=cax, orientation='horizontal', label='Sea ice [%]')
        ax4.tick_params(axis='y', labelleft=False, left=False)
        ax4.spines[:].set_visible(True)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.set_xlabel('Time (hh:mm) [UTC]')

        plt.savefig(f'/home/mringel/lwp_project/figures/quicklooks/Quicklook_{flight_id}.png',
                    bbox_inches='tight',
                    dpi=200)


        