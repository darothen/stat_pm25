## Process Lu's original observational data into a single netCDF file that
## will be easier to read in Python

library(ncdf4)

## PRECIPITATION
# Load in the data
load("monthly-prate_1999-2013.Rdata")
# times = seq(ISOdate(1999, 1, 1), by="month", length.out=180)
times = 0:179
# times = seq(as.Date("1999/1/1"), as.Date("2013/12/1"), "months")

# define dimensions for the netCDF file
londim <- ncdim_def("lon", "degrees_east", as.double(lon.out))
latdim <- ncdim_def("lat", "degrees_north", as.double(lat.out))
# May just need to re-compute the months on the fly when we read back.
timedim <- ncdim_def("time", "months since 1999-01-01 00:00:00", times)

# define variable
precip_def <- ncvar_def("PRECIP", "m/s", list(londim, latdim, timedim), 
                        longname="Precipitation rate")

# Open file for writing
ncout <- nc_create("monthly-prate_1999-2013.nc",list(precip_def),force_v4=T)

# Put data in file
ncvar_put(ncout,precip_def,precip)
# put additional attributes into dimension and data variables
ncatt_put(ncout,"lon","axis","X") 
ncatt_put(ncout,"lat","axis","Y")
ncatt_put(ncout,"time","axis","T")

# add global attributes
ncatt_put(ncout,0,"timestamp_start","1999/1/1")
ncatt_put(ncout,0,"timestamp_freq","monthly")

# Finalize file
nc_close(ncout)


## PM25
# Load in the data
load("PM25_1999_2013_invdist_2.5x2.5_500km.Rdata")
pm_dims <- dim(total_PM)
times = 1:pm_dims[3]

# define dimensions for the netCDF file
londim <- ncdim_def("lon", "degrees_east", as.double(lon.frame))
latdim <- ncdim_def("lat", "degrees_north", as.double(lat.frame))
# This will be an integer index that we can worry about later
timedim <- ncdim_def("time", "", times)

# define variables
pm25_def <- ncvar_def("PM25", "ug/m3", list(londim, latdim, timedim))
year_def <- ncvar_def("year", "", list(timedim))
month_def <- ncvar_def("month", "", list(timedim))
day_def <- ncvar_def("day", "", list(timedim))

# Open file for writing
ncout <- nc_create("PM_region.nc",list(pm25_def,year_def,month_def,day_def),force_v4=T)

# Put data in file
ncvar_put(ncout,pm25_def,total_PM)
ncvar_put(ncout,year_def,total_time[,1])
ncvar_put(ncout,month_def,total_time[,2])
ncvar_put(ncout,day_def,total_time[,3])

# put additional attributes into dimension and data variables
ncatt_put(ncout,"lon","axis","X") 
ncatt_put(ncout,"lat","axis","Y")
ncatt_put(ncout,"time","axis","T")

# Finalize file
nc_close(ncout)
