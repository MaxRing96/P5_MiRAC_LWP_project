#!/usr/bin/env python3

import sys
from pyhdf.HDF import HDF
from pyhdf.SD import SD, SDC
from pyhdf.VS import VS
import numpy as np
from pprint import pprint

filename = sys.argv[1]

### READ VDATA ###
f = HDF(filename, SDC.READ)

vs = f.vstart()

# Print info
print("VDATA variables:")
data_info_list = vs.vdatainfo()
pprint(data_info_list)
###

vs_lat = vs.attach("Latitude")
vs_lon = vs.attach("Longitude")

lat = np.array(vs_lat[:])
lon = np.array(vs_lon[:])

vs_lat.detach()
vs_lon.detach()

vs.end()
f.close()

### READ SDS DATA ###
f = SD(filename, SDC.READ)
# Print info
print("SDS DATA variables:")
pprint(f.datasets())
###

iwc = f.select("IWC")[:]

pprint(lat)
pprint(lon)
pprint(iwc)
