import numpy as np
import csv
import glob
import matplotlib as plt
import matplotlib.pyplot as plt
# %matplotlib inline
import xarray as xr
import seaborn as sns;
sns.set(color_codes=True)
plt.style.use('ggplot')
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

from pylab import *
import os
from matplotlib import ticker
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import datetime
import calendar as cal
from matplotlib.colors import ListedColormap, BoundaryNorm
import cmocean
#
# data = xr.open_dataset('/Volumes/WorkDrive/melt_dates/seaiceconc.nc') # opening sit for xarray
# ## Arange months up to Aug to December ##
# def is_months(month):
#     return (month >= 8) & (month <= 12)
# data = data.sel(time=is_months(data['time.month']))

# ## Select SIC variable ##
# sic_xr = data.SI_12km_SH_ICECON_DAY_SpPolarGrid12km
# latitude = data.GridLat_SpPolarGrid12km
# longitude = data.GridLon_SpPolarGrid12km
#
# ## Here we want all the values that are over 100 (missing or land mask) to be set to nan ##
# sic_xr = sic_xr.where(sic_xr < 101,np.nan)

## Because the desired output is one slice/ image/grid for each year, I will process year by year
def continuous_meet(cond, window_size, dim):
    """
    Continuously meet a given condition along a dimension.
    """
    _found = cond.rolling(dim={'time': window_size},
                          center=True).sum(skipna=True).fillna(False).astype(np.float)

    detected = np.array(
        _found.rolling(dim={'time': window_size})
        .reduce(lambda a, axis: (a == window_size).any(axis=axis))
        .fillna(False)
        .astype(bool)
    )

    indices = (detected * np.arange(detected.shape[0]).reshape(detected.shape[0], 1, 1))
    indices[indices == 0] = detected.shape[0]
    output = indices.argmin(axis=0)

    return xr.DataArray(output)

sub_data = xr.open_dataset('/Volumes/WorkDrive/melt_dates/files/5d_30p_dec_nc/netcdf/y12_break5d_30.nc')
print(sub_data)
ice =  sub_data.__xarray_dataarray_variable__
ice.plot()
