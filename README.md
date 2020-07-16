# poes_metop_sem
Utilities for analyzing and plotting POES and Metop Space Environment Monitor (SEM-2) data.



#####
# Maintenance
#####

1) Updating Master CDL and NetCDF binary:
!!!
!!! WARNING, older versions (e.g. 4.3) of ncgen (and possibly ncdump) do not handle UINT's correctly (e.g. flag_values).
!!! Version 4.6 is known to work. You can check your instance and version using "which ncgen; ncgen -v".
!!!
cd data/
ncgen -k 'netCDF-4' -o poes_raw-aggregate_master.nc  -x poes_raw-aggregate_master.cdl
ncgen -k 'netCDF-4' -o poes_l2-aggregate_master.nc   -x poes_l2-aggregate_master.cdl
