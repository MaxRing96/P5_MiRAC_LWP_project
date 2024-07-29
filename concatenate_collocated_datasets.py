import xarray as xr
from glob import glob
import ac3airborne
import os
import src

cat, meta, credentials, kwds = src.get_ac3_meta()

terra_collocations = sorted(glob('/net/secaire/mringel/data/lwp_collocated/lwp_modis/*_P5_RF*terra_lwp.nc'))
aqua_collocations = sorted(glob('/net/secaire/mringel/data/lwp_collocated/lwp_modis/*_P5_RF*aqua_lwp.nc'))

for i,campaign in enumerate(list(cat)):   
    if campaign in ['PAMARCMiP','HAMAG']: continue
    
    # create list of flight ids for each campaign and remove flights
    # for which no LWP could be retrieved
    flight_ids = list(cat[campaign]['P5']['MiRAC-A'])
    if campaign == 'ACLOUD':
        flights_to_remove = ['ACLOUD_P5_RF04','ACLOUD_P5_RF06','ACLOUD_P5_RF11',
                             'ACLOUD_P5_RF14','ACLOUD_P5_RF15','ACLOUD_P5_RF16',
                             'ACLOUD_P5_RF17','ACLOUD_P5_RF21','ACLOUD_P5_RF22',
                             'ACLOUD_P5_RF23','ACLOUD_P5_RF25']
        for x in range(len(flights_to_remove)): 
            flight_ids.remove(flights_to_remove[x]) 
    if campaign == 'AFLUX':
        flights_to_remove = ['AFLUX_P5_RF09','AFLUX_P5_RF10','AFLUX_P5_RF11','AFLUX_P5_RF13']
        for x in range(len(flights_to_remove)): 
            flight_ids.remove(flights_to_remove[x]) 
    if campaign == 'MOSAiC-ACA':
        flight_ids.remove('MOSAiC-ACA_P5_RF09')
    if campaign == 'HALO-AC3':
        flight_ids.remove('HALO-AC3_P5_RF01')

    for j,flight_id in enumerate(flight_ids):
        
        print(flight_id)

        aqua_path = f'/net/secaire/mringel/data/lwp_collocated/lwp_modis/{flight_id}_MODIS_aqua_lwp.nc'
        terra_path = f'/net/secaire/mringel/data/lwp_collocated/lwp_modis/{flight_id}_MODIS_terra_lwp.nc'

        if j == 0:
            p5_campaign = xr.open_dataset(f'/net/secaire/mringel/data/lwp_retrieved/{flight_id}_lwp_v4.nc')
            amsr_campaign = xr.open_dataset(f'/net/secaire/mringel/data/lwp_collocated/lwp_amsr2/{flight_id}_AMSR_lwp_v3.nc')
            era5_campaign = src.get_ERA5_along_P5(flight_id)
        
        else:
            p5_flight = xr.open_dataset(f'/net/secaire/mringel/data/lwp_retrieved/{flight_id}_lwp_v4.nc')
            amsr_flight = xr.open_dataset(f'/net/secaire/mringel/data/lwp_collocated/lwp_amsr2/{flight_id}_AMSR_lwp_v3.nc')
            era5_flight = src.get_ERA5_along_P5(flight_id)
            
            p5_campaign = xr.concat([p5_campaign,p5_flight],dim='time')
            era5_campaign = xr.concat([era5_campaign,era5_flight],dim='time')
            if len(amsr_flight.time)!=0:
                amsr_campaign = xr.concat([amsr_campaign,amsr_flight],dim='time')
        
        if aqua_path in aqua_collocations:
            if aqua_collocations.index(aqua_path) == 0:
                aqua_campaign = src.get_MODIS_along_P5(flight_id,satellite='aqua',mask_seaice=True)
            else:
                aqua_flight = src.get_MODIS_along_P5(flight_id,satellite='aqua',mask_seaice=True)
                aqua_campaign = xr.concat([aqua_campaign,aqua_flight],dim='time')

        if terra_path in terra_collocations:
            if terra_collocations.index(terra_path) == 0:
                terra_campaign = src.get_MODIS_along_P5(flight_id,satellite='terra',mask_seaice=True)
            else:
                terra_flight = src.get_MODIS_along_P5(flight_id,satellite='terra',mask_seaice=True)
                terra_campaign = xr.concat([terra_campaign,terra_flight],dim='time')

    p5_campaign.to_netcdf(f'/net/secaire/mringel/data/lwp_retrieved/{campaign}_allRFs_P5_lwp_v4.nc')
    amsr_campaign.to_netcdf(f'/net/secaire/mringel/data/lwp_collocated/lwp_amsr2/{campaign}_allRFs_P5_AMSR_lwp_v4.nc')
    era5_campaign.to_netcdf(f'/net/secaire/mringel/data/lwp_collocated/{campaign}_allRFs_P5_ERA5_lwp_v2.nc')
    aqua_campaign.to_netcdf(f'/net/secaire/mringel/data/lwp_collocated/lwp_modis/{campaign}_allRFs_P5_MODIS_aqua_lwp_v2.nc')
    terra_campaign.to_netcdf(f'/net/secaire/mringel/data/lwp_collocated/lwp_modis/{campaign}_allRFs_P5_MODIS_terra_lwp_v2.nc')

    if i == 0:
        p5_all_campaigns = p5_campaign
        amsr_all_campaigns = amsr_campaign
        era5_all_campaigns = era5_campaign
        aqua_all_campaigns = aqua_campaign
        terra_all_campaigns = terra_campaign
    if i > 0:
        p5_all_campaigns = xr.concat([p5_all_campaigns,p5_campaign],dim='time')
        amsr_all_campaigns = xr.concat([amsr_all_campaigns,amsr_campaign],dim='time')
        era5_all_campaigns = xr.concat([era5_all_campaigns,era5_campaign],dim='time')
        aqua_all_campaigns = xr.concat([aqua_all_campaigns,aqua_campaign],dim='time')
        terra_all_campaigns = xr.concat([terra_all_campaigns,terra_campaign],dim='time')
    
    p5_campaign.close()
    amsr_campaign.close()
    era5_campaign.close()
    aqua_campaign.close()
    terra_campaign.close()

p5_all_campaigns.to_netcdf('/net/secaire/mringel/data/lwp_retrieved/all_campaigns_P5_lwp_v4.nc')
amsr_all_campaigns.to_netcdf('/net/secaire/mringel/data/lwp_collocated/lwp_amsr2/all_campaigns_P5_AMSR_lwp_v4.nc')
era5_all_campaigns.to_netcdf('/net/secaire/mringel/data/lwp_collocated/all_campaigns_ERA5_lwp_v2.nc')
aqua_all_campaigns.to_netcdf('/net/secaire/mringel/data/lwp_collocated/lwp_modis/all_campaigns_P5_MODIS_aqua_lwp_v2.nc')
terra_all_campaigns.to_netcdf('/net/secaire/mringel/data/lwp_collocated/lwp_modis/all_campaigns_P5_MODIS_terra_lwp_v2.nc')


