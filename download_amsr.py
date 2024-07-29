import fnmatch
import sys
import os
import ac3airborne
from ftplib import FTP
from dotenv import load_dotenv

campaign = 'MOSAiC-ACA'
platform = 'P5'

meta = ac3airborne.get_flight_segments() # flight segmentation

# list of campaign flight dates
flight_dates = [meta[campaign][platform][flight_id]['date'].strftime(("%Y%m%d")) 
                for flight_id in meta[campaign][platform].keys()]
# list of amsr files of campaign flight dates
amsr_files = [f'/amsr2/ocean/L3/v08.2/daily/{date[0:4]}/RSS_AMSR2_ocean_L3_daily_{date[0:4]}-{date[4:6]}-{date[6:8]}_v08.2.nc' 
              for date in flight_dates]

ftp = FTP('ftp.remss.com')
ftp.login('maximilian_ringel@icloud.com','maximilian_ringel@icloud.com')    

# Download files of amsr_files
for i,file in enumerate(amsr_files):
    
    print(f"{i+1}/{len(amsr_files)} | Downloading {file}")

    try:
        ftp.retrbinary("RETR " + file ,open("/net/secaire/mringel/amsr/amsr2/" + file.split('/')[-1], 'wb').write)

    except EOFError: 
        pass # To avoid EOF errors.

ftp.close()