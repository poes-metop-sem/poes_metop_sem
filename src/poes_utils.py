__author__ = 'jgreen@spacehaz.com'
#####
# Authors Rob Redmon and Janet Green
# Disclaimer: Users assume all risk related to their use of these routines and authors disclaim
#       any and all warranties, whether expressed or implied, including (without limitation) any implied warranties of
#       merchantability or fitness for a particular purpose.
# History:
# 
#####


#####
# Setup
#####
import datetime as dtm, logging, numpy as np, os, shutil, sys, traceback
import configparser
import netCDF4 as nc4
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib as mpl; mpl.use('agg')
from scipy import interpolate
import struct
from scipy.signal import find_peaks
import xarray as xr
try:
   import cPickle as pickle
except:
   import pickle
from scipy import stats
import timeit

# Matplotlib default fonts sizes
#mpl.rcParams['font.size'] = 6
#mpl.rcParams['axes.titlesize'] = 12
#mpl.rcParams['axes.labelsize'] = 6
#mpl.rcParams['xtick.labelsize'] = 6
#mpl.rcParams['ytick.labelsize'] = 6

# Logging:
#     more info: https://docs.python.org/release/2.6/library/logging.html
log_level_console = logging.CRITICAL
log_level_file = logging.INFO
logger = logging.getLogger( 'poes_utils' )
#logger.setLevel( log_level_console )
console = logging.StreamHandler()
console.setLevel( log_level_console )
console.setFormatter( logging.Formatter( "%(asctime)s - %(name)s - %(levelname)s - %(message)s" ) )
logger.addHandler( console )
fh = logging.FileHandler('./spam.log')
fh.setLevel( log_level_file )
fh.setFormatter( logging.Formatter( "%(asctime)s - %(name)s - %(levelname)s - %(message)s" ) )
logger.addHandler( fh )


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                Misc Utils                                     "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def satID_to_satname( satID ):

    '''
    Gets the short satellite name from the satellite ID that is in each file

    Maps SatID in the Raw and L2 files to n|mXX (e.g. n15 for NOAA-15, m03 for Metop-C).
    Table was taken from ted_cal_coefficients.txt

    :param satID :
    :return satellite name or names (str or array of str's):
    '''
    satID2name = ['unk','unk','n16','unk','n15','unk','n17','n18','n19','unk','unk','m01','m02','m03']

    # Scalar
    if isinstance( satID, np.uint8 ):
        return satID2name[ satID ]

    # Numpy Array
    else:
        n = len( satID )

        sat_names = np.ndarray( n, dtype='a3' )
        sat_names[:] = 'nXX'
        sat_names[ satID == 4 ] = 'n15'
        sat_names[ satID == 2 ] = 'n16'
        sat_names[ satID == 6 ] = 'n17'
        sat_names[ satID == 7 ] = 'n18'
        sat_names[ satID == 8 ] = 'n19'
        sat_names[ satID == 11 ] = 'm01'
        sat_names[ satID == 12 ] = 'm02'
        sat_names[ satID == 13 ] = 'm03'

        return sat_names


def unix_time_ms_to_datetime( time_msec ):
    '''
    Returns an array of datetimes from the times in the files (Unix msec)

    :param: time_msec (Numpy array or masked array, list, single value): milliseconds since 1970
    :returns: time_dt (Numpy array of datetimes or single datetime if a single value (not array len =1) is passed
                        If an array of length 1 is passed an array of length 1 is returned)

    NOTE: This always passes back a numpy array even if you pass it a list
           If you pass a masked array it will send back a masked array
    '''

    # Check the size of what is passed
    n_times   = np.size( time_msec )
    if not hasattr(time_msec, '__len__'):
        # It is a scalar
        time_dt = dtm.datetime(1970,1,1) + dtm.timedelta( microseconds=1000.*time_msec )
    else:
        # Sometimes you can have an array or list that is just one value
        if n_times ==1:
            # It is a list or array with just 1 value
            if isinstance(time_msec, list):
                # It is a list of 1. Return an array of 1 here
                time_dt = np.array(dtm.datetime(1970, 1, 1) + dtm.timedelta(microseconds=1000. * time_msec[0]))
            else:
                # It is an array of len 1. Return and array of 1
                time_dt = np.array(dtm.datetime(1970, 1, 1) + dtm.timedelta(microseconds=1000. * time_msec))
        else:
            # It is a list, array or masked array with more than 1 value
            # This checks to see if it is a masked array
            if isinstance(time_msec,np.ma.MaskedArray):
                time_dt1 = [dtm.datetime(1970, 1, 1) + dtm.timedelta(microseconds=1000. * time_msec.data[i]) for i in np.arange(n_times)]
                time_dt = np.ma.array(time_dt1, mask = time_msec.mask)
            else:
                time_dt = np.array( [dtm.datetime(1970,1,1) + dtm.timedelta( microseconds=1000.*time_msec[i] ) for i in np.arange( n_times ) ] )

    return( time_dt )


def lon_360_to_180(lon):
    ''' Changes longitudes from 0 to 360 to -180 to 180
    :param lon:
    :returns lon_180
    '''
    lon_180 = lon
    lon_180[lon > 180] -= 360.
    return lon_180

def unixtime(date1):
    '''Change a datetime to ctime or seconds since 1970

    :param date1:
    :returns ctime
    NOTE: This is not set up to handle all the various types of lists, arrays, masked arrays that could be passed'''
    # Todo make this handle all types of intputs
    n_times = np.size(date1)

    if 1!=n_times:
        ctime = [(x - dtm.datetime(1970, 1, 1)).total_seconds() for x in date1]
    else:
        ctime = (date1 - dtm.datetime(1970, 1, 1)).total_seconds()
    return ctime


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                 File I/O                                      "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



def assign_cols(data, temp, cols, conv):
    '''Used by the read_poes_binary code
    '''
    for vco in range(0, len(temp)):
        if conv == 1:
            data[cols[vco]].append(temp[vco])
        else:
            data[cols[vco]].append(float(temp[vco]) / conv)


def repeat_vals(data, rnum):
    ''' This is used by the read_poes_binary code '''
    newdat = list()
    for x in data:
        for tco in range(0, rnum):
            newdat.append(x)
    return newdat


def fill_vals(data, rnum, filld):
    ''' This is used by the read_poes_binary code
    :param data:
    :param rnum:
    :param filld:
    :return:
    '''
    newdat = list()
    for x in data:
        for tco in range(0, rnum):
            if tco == 0:
                newdat.append(x)
            else:
                newdat.append(filld)
    return newdat

def process_meped_omni(data):
    '''Used by the read_poes_binary code
       Creates meped omni processed data from counts

    NOTE: Needs to be updated with actual algorithm to change poes omni counts to flux
          Right now it just sets the processed omni data to flags so that there are values'''

    omni_cols_proc = ['mep_omni_flux_p1', 'mep_omni_flux_p2', 'mep_omni_flux_p3', 'mep_omni_flux_flag_fit',
                      'mep_omni_flux_flag_iter_lim', 'mep_omni_gamma_p1', 'mep_omni_gamma_p2', 'mep_omni_gamma_p3']

    for col in omni_cols_proc:
        data[col] = [-99 for x in data['time']]
    return data,omni_cols_proc

def process_meped_tel(data,mep0_cols,mep90_cols):
    ''' Used by the read_poes_binary code

    Processes the meped telescope counts in the binary files to flux
    Creates a processed data structure just like the new netcdf files

    :param data: from read binary

    NOTE: Uncertainties are currently set to flags
    Proton contaminated data in the e4 channel is not flagged'''

    # These are the processed meped 0 deg telescope column names
    mep0_cols_proc = ['mep_pro_tel0_flux_p1', 'mep_pro_tel0_flux_p2', 'mep_pro_tel0_flux_p3',
                      'mep_pro_tel0_flux_p4', 'mep_pro_tel0_flux_p5', 'mep_pro_tel0_flux_p6',
                      'mep_ele_tel0_flux_e1', 'mep_ele_tel0_flux_e2', 'mep_ele_tel0_flux_e3']

    # These are the processed meped 90 deg telescope column names
    mep90_cols_proc = ['mep_pro_tel90_flux_p1', 'mep_pro_tel90_flux_p2', 'mep_pro_tel90_flux_p3',
                   'mep_pro_tel90_flux_p4', 'mep_pro_tel90_flux_p5', 'mep_pro_tel90_flux_p6',
                   'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3']

    # These are the geometric factors from the bowtie analysis available at NCEI
    mep_pro_G = [14.96, 47.43, 167.5, 573.42, 2243.53, .18]
    mep_ele_G = [.62, .32, .19, .40]

    # Do the meped conversions from counts to flux
    mep_err_col0 = list()
    mep_err_col90 = list()

    # First change protons from counts to flux
    for mpco in range(0, 6):
        data[mep0_cols_proc[mpco]] = [x * 100 / mep_pro_G[mpco] for x in data[mep0_cols[mpco]]]
        data[mep90_cols_proc[mpco]] = [x * 100 / mep_pro_G[mpco] for x in data[mep90_cols[mpco]]]

        mep_err_col0.append(mep0_cols_proc[mpco] + '_err')
        mep_err_col90.append(mep90_cols_proc[mpco] + '_err')

        # TODO: Need to update the uncertainties
        data[mep_err_col0[mpco]] = [-99 for x in data[mep0_cols[mpco]]]
        data[mep_err_col90[mpco]] = [-99 for x in data[mep90_cols[mpco]]]

    # Now change electrons from counts to flux
    for epco in range(0, 3):
        data[mep0_cols_proc[epco + 6]] = [x * 100 / mep_ele_G[epco] for x in data[mep0_cols[epco + 6]]]
        data[mep90_cols_proc[epco + 6]] = [x * 100 / mep_ele_G[epco] for x in data[mep90_cols[epco + 6]]]

        mep_err_col0.append(mep0_cols_proc[epco + 6] + '_err')
        mep_err_col90.append(mep90_cols_proc[epco + 6] + '_err')

        # NOTE: Need to update the uncertainties
        data[mep_err_col0[epco + 6]] = [-99 for x in data[mep0_cols[epco + 6]]]
        data[mep_err_col90[epco + 6]] = [-99 for x in data[mep90_cols[epco + 6]]]

    # Add e4
    # TODO: Check if I proton contaminated data is flagged in the netcdf files
    data['mep_ele_tel0_flux_e4'] = [x * 100 / mep_ele_G[3] for x in data[mep0_cols[5]]]
    data['mep_ele_tel90_flux_e4'] = [x * 100 / mep_ele_G[3] for x in data[mep90_cols[5]]]
    data['mep_ele_tel0_flux_e4_err'] = [-99 for x in data[mep0_cols[5]]]
    data['mep_ele_tel90_flux_e4_err'] = [-99 for x in data[mep90_cols[5]]]

    return data,mep0_cols_proc,mep90_cols_proc,mep_err_col0,mep_err_col90
def ted_cals_numflux(satID):
    '''Used by the read_poes_binary code

    Returns calibration tables that change ted spectral counts to number flux(#/cm2-s-str-eV)
           with [[ E1 0 electron, 30 electron, 0 proton, 30 proton],
                 [E2 0 electron, 30 electron, 0 proton, 30 proton], ...
    :param satID: this is the satID from the raw file
    :returns alltcals ([16,4] list): the whole calibration table for all 15 TEDenergies

    From https://www.ngdc.noaa.gov/stp/satellite/poes/docs/NGDC/TED%20processing%20ATBD_V1.pdf Table 4'''

    #N15 0
    if satID ==4:
        alltcals = [[6.964E+02, 9.272E+02, 4.022E+02, 4.121E+02],
                    [4.773E+02, 6.355E+02, 2.608E+02 ,2.758E+02],
                    [3.268E+02, 4.351E+02, 1.690E+02, 1.843E+02],
                    [2.248E+02, 2.993E+02, 1.101E+02, 1.237E+02],
                    [1.550E+02, 2.064E+02, 7.193E+01, 8.333E+01],
                    [1.065E+02, 1.418E+02, 4.678E+01, 5.587E+01],
                    [7.324E+01, 9.752E+01, 3.047E+01, 3.753E+01],
                    [5.033E+01, 6.702E+01, 1.983E+01, 2.518E+01],
                    [5.567E+01, 5.984E+01, 1.058E+02, 7.990E+01],
                    [4.292E+01, 4.625E+01, 6.579E+01, 4.900E+01],
                    [3.311E+01, 3.568E+01, 4.074E+01, 3.006E+01],
                    [2.553E+01, 2.760E+01, 2.528E+01, 1.845E+01],
                    [1.968E+01, 2.134E+01, 1.568E+01, 1.132E+01],
                    [1.521E+01, 1.649E+01, 9.729E+00, 6.946E+00],
                    [1.171E+01, 1.275E+01, 6.034E+00, 4.262E+00],
                    [9.032E+00, 9.843E+00, 3.741E+00, 2.615E+00]]
    # N16
    if satID ==2:
        alltcals = [[1.146E+03, 4.476E+02, 5.389E+02, 7.404E+02],
                    [7.857E+02, 4.476E+02, 3.338E+02, 4.741E+02],
                    [5.379E+02, 4.476E+02, 2.066E+02, 3.031E+02],
                    [3.700E+02, 4.476E+02, 1.285E+02, 1.950E+02],
                    [2.552E+02, 4.476E+02, 8.025E+01, 1.257E+02],
                    [1.753E+02, 4.476E+02, 4.983E+01, 8.064E+01],
                    [1.206E+02, 4.476E+02, 3.101E+01, 5.187E+01],
                    [8.286E+01, 1.002E+02, 1.927E+01, 3.330E+01],
                    [6.397E+01, 4.476E+02, 1.493E+02, 1.132E+02],
                    [5.087E+01, 4.476E+02, 9.100E+01, 7.024E+01],
                    [4.048E+01, 3.595E+01, 5.537E+01, 4.349E+01],
                    [3.223E+01, 4.476E+02, 3.371E+01, 2.698E+01],
                    [2.566E+01, 4.476E+02, 2.057E+01, 1.672E+01],
                    [2.041E+01, 1.689E+01, 1.253E+01, 1.037E+01],
                    [1.626E+01, 4.476E+02, 7.639E+00, 1.037E+01],
                    [1.294E+01, 4.476E+02, 4.658E+00, 6.431E+00]]
    # N17
    if satID ==6:
        alltcals = [[7.445E+02, 9.689E+02, 4.959E+02, 4.564E+02],
                    [5.103E+02, 6.641E+02, 3.146E+02, 3.027E+02],
                    [3.493E+02, 4.546E+02, 1.994E+02, 2.004E+02],
                    [2.403E+02, 3.127E+02, 1.271E+02, 1.335E+02],
                    [1.657E+02, 2.157E+02, 8.121E+01, 8.915E+01],
                    [1.138E+02, 1.481E+02, 5.165E+01, 5.922E+01],
                    [7.830E+01, 1.019E+02, 3.293E+01, 3.944E+01],
                    [5.381E+01, 7.003E+01, 2.096E+01, 2.622E+01],
                    [7.162E+01, 9.076E+01, 1.663E+02, 1.235E+02],
                    [5.453E+01, 6.753E+01, 1.012E+02, 7.434E+01],
                    [4.153E+01, 5.031E+01, 6.156E+01, 4.470E+01],
                    [3.162E+01, 3.743E+01, 3.753E+01, 2.693E+01],
                    [2.404E+01, 2.786E+01, 2.283E+01, 1.618E+01],
                    [1.832E+01, 2.075E+01, 1.389E+01, 9.744E+00],
                    [1.394E+01, 1.544E+01, 8.448E+00, 5.870E+00],
                    [1.062E+01, 1.148E+01, 5.143E+00, 3.531E+00]]
    # N18
    if satID == 7:
        alltcals = [[1.052E+03, 1.066E+03, 6.163E+02, 7.066E+02],
                    [7.392E+02, 6.860E+02, 3.848E+02, 4.322E+02],
                    [5.190E+02, 4.416E+02, 2.404E+02, 2.653E+02],
                    [3.654E+02, 2.857E+02, 1.503E+02, 1.643E+02],
                    [2.581E+02, 1.854E+02, 9.455E+01, 1.019E+02],
                    [1.816E+02, 1.193E+02, 5.911E+01, 6.266E+01],
                    [1.281E+02, 7.766E+01, 3.700E+01, 3.866E+01],
                    [9.017E+01, 5.020E+01, 2.314E+01, 2.389E+01],
                    [8.490E+01, 5.636E+01, 1.922E+02, 1.372E+02],
                    [6.770E+01, 4.372E+01, 1.168E+02, 8.268E+01],
                    [5.412E+01, 3.392E+01, 7.084E+01, 5.005E+01],
                    [4.315E+01, 2.629E+01, 4.288E+01, 3.018E+01],
                    [3.451E+01, 2.038E+01, 2.603E+01, 1.826E+01],
                    [2.748E+01, 1.582E+01, 1.578E+01, 1.103E+01],
                    [2.198E+01, 1.227E+01, 9.574E+00, 6.659E+00],
                    [1.753E+01, 9.526E+00, 5.798E+00, 4.025E+00]]
    # N19
    if satID ==8:
        alltcals = [[4.529E+02, 6.404E+02, 1.112E+03, 1.039E+03],
                    [3.324E+02, 4.322E+02, 6.680E+02, 5.958E+02],
                    [2.469E+02, 2.914E+02, 4.042E+02, 3.430E+02],
                    [1.800E+02, 1.960E+02, 2.427E+02, 1.960E+02],
                    [1.332E+02, 1.332E+02, 1.460E+02, 1.133E+02],
                    [9.790E+01, 8.951E+01, 8.825E+01, 6.493E+01],
                    [7.184E+01, 6.071E+01, 5.321E+01, 3.732E+01],
                    [5.289E+01, 4.114E+01, 3.202E+01, 2.146E+01],
                    [5.739E+01, 3.956E+01, 2.156E+02, 1.220E+02],
                    [4.432E+01, 3.002E+01, 1.279E+02, 7.295E+01],
                    [3.428E+01, 2.288E+01, 7.616E+01, 4.349E+01],
                    [2.650E+01, 1.741E+01, 4.522E+01, 2.593E+01],
                    [2.047E+01, 1.326E+01, 2.688E+01, 1.547E+01],
                    [1.582E+01, 1.011E+01, 1.598E+01, 9.228E+00],
                    [1.224E+01, 7.694E+00, 9.511E+00, 5.510E+00],
                    [9.435E+00, 5.867E+00, 5.654E+00, 3.284E+00]]
    # MEtop A or MetOP2
    if satID == 12:
        alltcals = [[1.042E+03, 1.335E+03, 7.235E+02, 1.411E+03],
                    [7.148E+02, 8.461E+02, 4.358E+02, 8.118E+02],
                    [4.906E+02, 5.357E+02, 2.622E+02, 4.668E+02],
                    [3.379E+02, 3.414E+02, 1.588E+02, 2.702E+02],
                    [2.337E+02, 2.180E+02, 9.650E+01, 1.569E+02],
                    [1.607E+02, 1.386E+02, 5.829E+01, 9.061E+01],
                    [1.108E+02, 8.833E+01, 3.533E+01, 5.247E+01],
                    [7.624E+01, 5.615E+01, 2.139E+01, 3.032E+01],
                    [8.134E+01, 8.069E+01, 1.509E+02, 1.262E+02],
                    [6.355E+01, 6.187E+01, 9.159E+01, 7.575E+01],
                    [4.966E+01, 4.746E+01, 5.553E+01, 4.555E+01],
                    [3.874E+01, 3.650E+01, 3.371E+01, 2.737E+01],
                    [3.027E+01, 2.795E+01, 2.043E+01, 1.648E+01],
                    [2.364E+01, 2.146E+01, 1.241E+01, 9.898E+00],
                    [1.849E+01, 1.651E+01, 7.519E+00, 5.951E+00],
                    [1.445E+01, 1.266E+01, 4.565E+00, 3.578E+00]]
    # Metop B or MetOp1
    if satID == 11:
        alltcals = [[1.318E+03, 1.204E+03, 6.819E+02, 1.676E+03],
                    [8.819E+02, 7.738E+02, 4.348E+02, 1.007E+03],
                    [5.890E+02, 4.976E+02, 2.767E+02, 6.038E+02],
                    [3.954E+02, 3.211E+02, 1.772E+02, 3.644E+02],
                    [2.664E+02, 2.081E+02, 1.138E+02, 2.207E+02],
                    [1.785E+02, 1.342E+02, 7.269E+01, 1.327E+02],
                    [1.199E+02, 8.664E+01, 4.655E+01, 8.049E+01],
                    [8.038E+01, 5.589E+01, 2.977E+01, 4.828E+01],
                    [8.763E+01, 6.573E+01, 2.203E+02, 1.756E+02],
                    [6.989E+01, 5.171E+01, 1.262E+02, 1.019E+02],
                    [5.553E+01, 4.065E+01, 7.271E+01, 5.910E+01],
                    [4.416E+01, 3.192E+01, 4.179E+01, 3.432E+01],
                    [3.518E+01, 2.510E+01, 2.398E+01, 1.994E+01],
                    [2.797E+01, 1.977E+01, 1.380E+01, 1.156E+01],
                    [2.232E+01, 1.550E+01, 7.935E+00, 6.701E+00],
                    [1.774E+01, 1.219E+01, 4.558E+00, 3.888E+00]]
    # Metop C or MetOp3
    if satID == 13:
        alltcals = [[1.281E+03, 1.318E+03, 8.100E+02, 1.271E+03],
                    [8.630E+02, 8.564E+02, 5.080E+02, 7.675E+02],
                    [5.819E+02, 5.558E+02, 3.181E+02, 4.628E+02],
                    [3.937E+02, 3.624E+02, 2.004E+02, 2.808E+02],
                    [2.672E+02, 2.370E+02, 1.266E+02, 1.710E+02],
                    [1.803E+02, 1.543E+02, 7.961E+01, 1.036E+02],
                    [1.221E+02, 1.007E+02, 5.018E+01, 6.288E+01],
                    [8.251E+01, 6.561E+01, 3.158E+01, 3.810E+01],
                    [8.763E+01, 6.918E+01, 2.440E+02, 1.922E+02],
                    [6.819E+01, 5.369E+01, 1.401E+02, 1.108E+02],
                    [5.323E+01, 4.162E+01, 8.028E+01, 6.359E+01],
                    [4.140E+01, 3.231E+01, 4.600E+01, 3.660E+01],
                    [3.231E+01, 2.503E+01, 2.641E+01, 2.104E+01],
                    [2.516E+01, 1.946E+01, 1.513E+01, 1.210E+01],
                    [1.958E+01, 1.506E+01, 8.686E+00, 6.960E+00],
                    [1.527E+01, 1.171E+01, 4.988E+00, 3.998E+00]]

    return alltcals


def process_ted_spec(data,tspec0_cols,tspec30_cols):
    '''Used by the read_poes_binary code

    Changes TED spectra data that is measured at 4 energies from counts to flux #/cm2-s-str-eV

    :param data: object with data from binary file
    :param tspec0_cols: names of the unprocessed 0 degree ted spectra cols
    :param tspec30_cols: names of the unprocessed 0 degree ted spectra cols

    TO DO: check this with overlapping new netcdf and binary data'''

    # ted 0 degree detector spectra new processed col names
    tspec0_cols_proc = ['ted_ele_tel0_flux_4', 'ted_ele_tel0_flux_8', 'ted_ele_tel0_flux_11', 'ted_ele_tel0_flux_14',
                        'ted_pro_tel0_flux_4', 'ted_pro_tel0_flux_8', 'ted_pro_tel0_flux_11', 'ted_pro_tel0_flux_14']

    # ted 30 degree detector spectra new processed col names
    tspec30_cols_proc = ['ted_ele_tel30_flux_4', 'ted_ele_tel30_flux_8', 'ted_ele_tel30_flux_11', 'ted_ele_tel30_flux_14',
                     'ted_pro_tel30_flux_4', 'ted_pro_tel30_flux_8', 'ted_pro_tel30_flux_11', 'ted_pro_tel30_flux_14']

    # These are from https://www.ngdc.noaa.gov/stp/satellite/poes/docs/NGDC/TED%20processing%20ATBD_V1.pdf
    # Returns a (16, 4) list with ele 0 ele 30, pro 0 pro 30. The spectra channels returned 4, 8, 11, 14
    alltcals = ted_cals_numflux(data['satID'][0])

    tcals0 =[alltcals[3][0], alltcals[7][0], alltcals[10][0], alltcals[13][0],
             alltcals[3][2], alltcals[7][2], alltcals[10][2], alltcals[13][2]]

    tcals30 =[alltcals[3][1], alltcals[7][1], alltcals[10][1], alltcals[13][1],
             alltcals[3][3], alltcals[7][3], alltcals[10][3], alltcals[13][3]]

    for tco in range(0, 8):
        data[tspec0_cols_proc[tco]] = [x / tcals0[tco] for x in data[tspec0_cols[tco]]]
        data[tspec30_cols_proc[tco]] = [x / tcals30[tco] for x in data[tspec30_cols[tco]]]

    return data,tspec0_cols_proc,tspec30_cols_proc


def ted_cals_eflux(satID):
    ''' Used by the read_poes_binary code

    Returns the calibration tables that change ted energy flux counts to energy flux mW/m2

    :param satID: this is the satID from the raw file
    :return efluxcals(list) : calibration table with 0-lowe, 30-lowe, 0-hie, 30-hie, 0-lowp, 30-lowp, 0-hip, 30-hip
    '''
    if satID==2: #N15
        efluxcals = [1.564E-6, 2.083E-6, 5.04E-5, 5.43E-5, 9.29E-7, 1.002E-6, 3.47E-5, 2.57E-5]
    if satID ==4: #N16
        efluxcals = [2.58E-6, 3.12E-6, 6.07E-5, 5.44E-6, 1.158E-6, 1.675E-6, 4.75E-5, 3.70E-5]
    if satID ==6: #N17
        efluxcals = [1.672E-6, 2.176E-6, 6.36E-5, 7.79E-5, 1.106E-6, 1.093E-6, 5.28E-5, 3.85E-5]
    if satID ==7: #N18
        efluxcals = [2.45E-6, 2.18E-6, 8.09E-5, 5.14E-5, 1.34E-6, 1.50E-6, 6.08E-5, 4.30E-5]
    if satID ==8: #N19
        efluxcals = [1.133E-6, 1.402E-6, 5.20E-5, 3.51E-5, 2.281E-6, 1.988E-6, 6.60E-5, 3.77E-5]
    if satID ==12: #MetOp2 or MetOp A
        efluxcals = [2.345E-6, 2.672E-6, 7.50E-5, 7.25E-5, 1.516E-6, 2.714E-6, 4.86E-5, 3.93E-5]
    if satID == 11: #MetOp1 or MetOp B
        efluxcals = [2.875E-6, 2.460E-6, 8.39E-5, 6.13E-5, 1.532E-6, 3.433E-6, 6.41E-5, 5.19E-5]
    if satID == 13: # MetOp3 or MetOp C
        efluxcals = [2.805E-6, 2.734E-6, 8.05E-5, 6.32E-5, 1.774E-6, 2.624E-6, 7.08E-5, 5.61E-5]

    return efluxcals

def process_ted_eflux(data,ted_eflux_cols,tback0_cols,tback30_cols):
    '''Changes the ted eflux counts to ted eflux (mW/m2)

    NOTE: This is not complete. Just flags are returned.
    '''

    # TO DO: apply the TED calibrations here
    ted0eflux_cols_proc = ['ted_ele_tel0_low_eflux', 'ted_pro_tel0_low_eflux',
                           'ted_ele_tel0_hi_eflux', 'ted_pro_tel0_hi_eflux']
    ted30eflux_cols_proc = ['ted_ele_tel30_low_eflux', 'ted_pro_tel30_low_eflux',
                            'ted_ele_tel30_hi_eflux', 'ted_pro_tel30_hi_eflux']
    ted0eflux_err = list()
    ted30eflux_err = list()

    # There are 4 ted electron eflux cols 0-lowe, 30-lowe, 0-hie, 30-hie
    # What is the name of the unprocessed data?

    for tco in range(0, 4):
        data[ted0eflux_cols_proc[tco]] = [-99 for x in data['time']]

        ted0eflux_err.append(ted0eflux_cols_proc[tco] + '_error')
        data[ted0eflux_err[tco]] = [-99 for x in data['time']]

    for tco in range(0, 4):
        data[ted30eflux_cols_proc[tco]] = [-99 for x in data['time']]

        ted30eflux_err.append(ted30eflux_cols_proc[tco] + '_error')
        data[ted30eflux_err[tco]] = [-99 for x in data['time']]


    ted_eflux_cols_err = list()

    for tco in range(0, len(ted_eflux_cols)):
        ted_eflux_cols_err.append(ted_eflux_cols[tco] + '_error')
        data[ted_eflux_cols_err[tco]] = [-99 for x in data['time']]

    tback0_cols_proc = list()
    for tco in range(0, len(tback0_cols)):
        tback0_cols_proc.append(tback0_cols[tco][0:-4])
        data[tback0_cols_proc[tco]] = [-99 for x in data['time']]

    tback30_cols_proc = list()
    for tco in range(0, len(tback30_cols)):
        tback30_cols_proc.append(tback30_cols[tco][0:-4])
        data[tback30_cols_proc[tco]] = [-99 for x in data['time']]
    return data, ted0eflux_cols_proc, ted30eflux_cols_proc, ted0eflux_err, ted30eflux_err, ted_eflux_cols_err, tback0_cols_proc, \
           tback30_cols_proc

def read_poes_bin(filename, datatype='raw', procvars=None):
    ''' Reads the POES archive data that was processed by the Space Weather Prediction
            center into a binary format and archived at NGDC. This is the format of the
            POES data prior to 2012. This will read it and translate to either raw
            or processed data formats that matches the later netcdf data at NGDC.
            NOTE: The code is currently incomplete and some values are simply returned as flags. These flagged
            values include some of the processed TED parameters, processed MEPED Omni parameters, and
            uncertainties
    :param  filename : full structure to the file and name
    :param  datatype='raw' or 'processed'
             If type ='raw' then the output will be a dictionary (data) with all the variables in the ngdc raw data files
             NOTE: The voltages and temps do not match!!!
             If type ='processed' then the output will be a dictionary (data) with all the variables in the ngdc processed data files
    :returns a dictionary of data

    DATE   9/2019
    AUTHOR Janet Green'''
    # Todo add the parameters that are currently still flags
    # =========================================================================
    # cnvrt is used to convert the data numbers to counts. This is to undo the
    # oboard compression
    cnvrt = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, \
             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, \
             25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 34.5, 36.5, 38.5, 40.5, 42.5, \
             44.5, 46.5, 48.5, 50.5, 53.0, 56.0, 59.0, 62.0, 65.5, 69.5, 73.5, 77.5, 81.5, \
             85.5, 89.5, 93.5, 97.5, 101.5, 106.5, 112.5, 118.5, 124.5, 131.5, 139.5, \
             147.5, 155.5, 163.5, 171.5, 179.5, 187.5, 195.5, 203.5, 213.5, 225.5, \
             237.5, 249.5, 263.5, 279.5, 295.5, 311.5, 327.5, 343.5, 359.5, 375.5, \
             391.5, 407.5, 427.5, 451.5, 475.5, 499.5, 527.5, 559.5, 591.5, 623.5, \
             655.5, 687.5, 719.5, 751.5, 783.5, 815.5, 855.5, 903.5, 951.5, 999.5, \
             1055.5, 1119.5, 1183.5, 1247.5, 1311.5, 1375.5, 1439.5, 1503.5, 1567.5, \
             1631.5, 1711.5, 1807.5, 1903.5, 1999.5, 2111.5, 2239.5, 2367.5, 2495.5, \
             2623.5, 2751.5, 2879.5, 3007.5, 3135.5, 3263.5, 3423.5, 3615.5, 3807.5, \
             3999.5, 4223.5, 4479.5, 4735.5, 4991.5, 5247.5, 5503.5, 5759.5, 6015.5, \
             6271.5, 6527.5, 6847.5, 7231.5, 7615.5, 7999.5, 8447.5, 8959.5, 9471.5, \
             9983.5, 10495.5, 11007.5, 11519.5, 12031.5, 12543.5, 13055.5, 13695.5, \
             14463.5, 15231.5, 15999.5, 16895.5, 17919.5, 18943.5, 19967.5, 20991.5, \
             22015.5, 23039.5, 24063.5, 25087.5, 26111.5, 27391.5, 28927.5, 30463.5, \
             31999.5, 33791.5, 35839.5, 37887.5, 39935.5, 41983.5, 44031.5, 46079.5, \
             48127.5, 50175.5, 52223.5, 54783.5, 57855.5, 60927.5, 63999.5, 67583.5, \
             71679.5, 75775.5, 79871.5, 83967.5, 88063.5, 92159.5, 96255.5, 100351.5, \
             104447.5, 109567.5, 115711.5, 121855.5, 127999.5, 135167.5, 143359.5, \
             151551.5, 159743.5, 167935.5, 176127.5, 184319.5, 192511.5, 200703.5, \
             208895.5, 219135.5, 231423.5, 243711.5, 255999.5, 270335.5, 286719.5, \
             303103.5, 319487.5, 335871.5, 352255.5, 368639.5, 385023.5, 401407.5, \
             417791.5, 438271.5, 462847.5, 487423.5, 511999.5, 540671.5, 573439.5, \
             606207.5, 638975.5, 671743.5, 704511.5, 737279.5, 770047.5, 802815.5, \
             835583.5, 876543.5, 925695.5, 974847.5, 1023999.5, 1081343.5, \
             1146879.5, 1212415.5, 1277951.5, 1343487.5, 1409023.5, 1474559.5, \
             1540095.5, 1605631.5, 1671167.5, 1753087.5, 1851391.5, 1949695.5, \
             1998848.0]

    # Open the file
    with open(filename, "rb") as file:

        # Read in the binary archive data
        # Each data record is 2544 bytes long and contains 32 seconds of data
        # Go to the end of file to get the size
        file.seek(0, 2)
        filesize = file.tell()
        # The number of data records is the file size divided by the bytes per record
        nums = int(filesize / 2544)

        # The way the data is laid out is not very intuitive
        # Each data record has 32 seconds of data but some values are only given each 8 sec

        # Define all the cols and the format and then read them in
        # The first bit is called header material because it does not have actual flux data

        # This is the format of the header data which gets to byte 1031
        head_fmt = '>2IH10B17i'

        # This is written once per 32s rec
        head_cols = ['cSumFlag', 'cSum', 'major_frame', 'MEPon', 'TEDon', 'mep_IFC_on', 'ted_IFC_on',
                     'ted_ele_PHD_level', 'ted_pro_PHD_level',
                     'ted_ele_HV_step', 'ted_pro_HV_step', 'microp', 'microp_flag', 'ted_V', 'ted_electron_CEM_V',
                     'ted_proton_CEM_V', 'ted_sweepV', 'TED_temp', 'MEPED_V', 'mep_circuit_temp', 'mep_omni_biase_V',
                     'MEP_ptel_biasV', 'MEP_etel_biasV', 'mep_proton_tel_temp', 'MEP_e_T', 'MEP_omni_T',
                     'DPU_V', 'microA_V', 'microB_V', 'DPU_temp']

        # These are 4 times per rec
        # Each on is 4 bytes and we will read 8 at a time and repeat 4 times
        # Repeat 4 x
        # -----------------------
        latlon_fmt = '>32i'
        latlon_cols = ['lat', 'lat', 'lat', 'lat', 'lon', 'lon', 'lon', 'lon']

        # These get repeated 4 times ihdcols+160 missing data are 4 times per rec
        # Not sure what to do with missing data yet

        # Repeats 4 times
        # ------------------------------
        ihd_mdf_fmt = '>9i4H160B9i4H160B9i4H160B9i4H160B'

        ihd_cols = ['satID', 'year', 'day', 'msec', 'alt', 'inc', 'orbit', 'gap1', 'gap2', 'minor_frame', 'minor_frame',
                    'minor_frame', 'minor_frame']
        ihd_data_cols = ['satID', 'year', 'day', 'msec', 'alt', 'inc', 'orbit', 'gap1', 'gap2', 'minor_frame']

        mdf_cols = list()
        for dco in range(0, 160):
            mdf_cols.append('mdf' + str(dco))

        # This whole chunk repeats 4 times
        # ------------------------------
        data_fmt = '>36B36B16B16B16B16B16B36B36B16B16B16B16B16B36B36B16B16B16B16B16B36B36B16B16B16B16B16B'
        # 4x
        mep90_cols = ['mep_pro_tel90_cps_p1', 'mep_pro_tel90_cps_p2', 'mep_pro_tel90_cps_p3',
                      'mep_pro_tel90_cps_p4', 'mep_pro_tel90_cps_p5', 'mep_pro_tel90_cps_p6',
                      'mep_ele_tel90_cps_e1', 'mep_ele_tel90_cps_e2', 'mep_ele_tel90_cps_e3']
        # 4x
        mep0_cols = ['mep_pro_tel0_cps_p1', 'mep_pro_tel0_cps_p2', 'mep_pro_tel0_cps_p3',
                     'mep_pro_tel0_cps_p4', 'mep_pro_tel0_cps_p5', 'mep_pro_tel0_cps_p6',
                     'mep_ele_tel0_cps_e1', 'mep_ele_tel0_cps_e2', 'mep_ele_tel0_cps_e3']
        # 4x
        omni_cols = ['mep_omni_cps_p6', 'mep_omni_cps_p7', 'mep_omni_cps_p8', 'mep_omni_cps_p9']
        # 4x
        ted0eflux_cols = ['ted_ele_tel0_low_eflux_cps', 'ted_pro_tel0_low_eflux_cps',
                          'ted_ele_tel0_hi_eflux_cps', 'ted_pro_tel0_hi_eflux_cps']
        # 4x
        ted0maxE_cols = ['ted_ele_energy_tel0', 'ted_pro_energy_tel0', 'ted_ele_max_flux_tel0', 'ted_pro_max_flux_tel0']
        # 4x
        ted30eflux_cols = ['ted_ele_tel30_low_eflux_cps', 'ted_pro_tel30_low_eflux_cps',
                           'ted_ele_tel30_hi_eflux_cps', 'ted_pro_tel30_hi_eflux_cps']
        # 4x
        ted30maxE_cols = ['ted_ele_energy_tel30', 'ted_pro_energy_tel30', 'ted_ele_max_flux_tel30', 'ted_pro_max_flux_tel30']
        data_cols = mep0_cols + mep0_cols + mep0_cols + mep0_cols + \
                    mep90_cols + mep90_cols + mep90_cols + mep90_cols + \
                    omni_cols + omni_cols + omni_cols + omni_cols + \
                    ted0eflux_cols + ted0eflux_cols + ted0eflux_cols + ted0eflux_cols + \
                    ted0maxE_cols + ted0maxE_cols + ted0maxE_cols + ted0maxE_cols + \
                    ted30eflux_cols + ted30eflux_cols + ted30eflux_cols + ted30eflux_cols + \
                    ted30maxE_cols + ted30maxE_cols + ted30maxE_cols + ted30maxE_cols

        # This is the field info and repeats 4 times
        # ------------------------------------------

        B_cols = ['Br_sat', 'Bt_sat', 'Bp_sat', 'Btot_sat', 'ted_alpha_0_sat', 'ted_alpha_30_sat', 'meped_alpha_0_sat',
                  'meped_alpha_90_sat', 'Br_foot',
                  'Bt_foot', 'Bp_foot', 'Btot_foot', 'ted_alpha_0_foot', 'ted_alpha_30_foot', 'meped_alpha_0_foot',
                  'meped_alpha_90_foot', 'geod_lat_foot', 'geod_lon_foot', 'mag_lat_foot', 'mag_lon_foot',
                  'L_IGRF', 'cgm lat',
                  'MLT', 'LT fofl']

        B_fmt = '>96i'

        # TED spec and background 4 of each
        # ----------------------------------------
        tspec_fmt = '>36B36B'

        tspec0_cols = ['ted_ele_tel0_cps_4', 'ted_ele_tel0_cps_8', 'ted_ele_tel0_cps_11', 'ted_ele_tel0_cps_14',
                       'ted_pro_tel0_cps_4', 'ted_pro_tel0_cps_8', 'ted_pro_tel0_cps_11', 'ted_pro_tel0_cps_14']
        tback0_cols = ['ted_ele_eflux_bg_tel0_hi_cps', 'ted_ele_eflux_bg_tel0_low_cps', 'ted_pro_eflux_bg_tel0_hi_cps',
                       'ted_pro_eflux_bg_tel0_low_cps']

        tspec30_cols = ['ted_ele_tel30_cps_4', 'ted_ele_tel30_cps_8', 'ted_ele_tel30_cps_11', 'ted_ele_tel30_cps_14',
                        'ted_pro_tel30_cps_4', 'ted_pro_tel30_cps_8', 'ted_pro_tel30_cps_11', 'ted_pro_tel30_cps_14']
        tback30_cols = ['ted_ele_eflux_bg_tel30_hi_cps', 'ted_ele_eflux_bg_tel30_low_cps', 'ted_pro_eflux_bg_tel30_hi_cps',
                        'ted_pro_eflux_bg_tel30_low_cps']

        tspec_cols = tspec0_cols + tspec0_cols + tspec0_cols + tspec0_cols + tback0_cols + \
                     tspec30_cols + tspec30_cols + tspec30_cols + tspec30_cols + tback30_cols

        # The ted moni flux
        ted_omni_flux1_cols = ['ted_ele_eflux_atmo_low']
        ted_omni_flux2_cols = ['ted_ele_eflux_atmo_hi']
        ted_omni_flux3_cols = ['ted_ele_eflux_atmo_total']
        ted_omni_flux4_cols = ['ted_pro_eflux_atmo_low']
        ted_omni_flux5_cols = ['ted_pro_eflux_atmo_hi']
        ted_omni_flux6_cols = ['ted_pro_eflux_atmo_total']
        ted_omni_flux7_cols = ['ted_total_eflux_atmo']

        ted_omni_flux_cols = ted_omni_flux1_cols + ted_omni_flux1_cols + ted_omni_flux1_cols + ted_omni_flux1_cols + \
                             ted_omni_flux2_cols + ted_omni_flux2_cols + ted_omni_flux2_cols + ted_omni_flux2_cols + \
                             ted_omni_flux3_cols + ted_omni_flux3_cols + ted_omni_flux3_cols + ted_omni_flux3_cols + \
                             ted_omni_flux4_cols + ted_omni_flux4_cols + ted_omni_flux4_cols + ted_omni_flux4_cols + \
                             ted_omni_flux5_cols + ted_omni_flux5_cols + ted_omni_flux5_cols + ted_omni_flux5_cols + \
                             ted_omni_flux6_cols + ted_omni_flux6_cols + ted_omni_flux6_cols + ted_omni_flux6_cols + \
                             ted_omni_flux7_cols + ted_omni_flux7_cols + ted_omni_flux7_cols + ted_omni_flux7_cols
        ted_omni_flux_fmt = '>28i28i28i28i'

        # Create the data dictionary
        data = dict()
        all_cols = head_cols + ['lat', 'lon'] + ihd_data_cols + mdf_cols + mep90_cols + mep0_cols + \
                   omni_cols + ted0eflux_cols + ted0maxE_cols + ted30eflux_cols + ted30maxE_cols + \
                   B_cols + tspec0_cols + tback0_cols + tspec30_cols + tback30_cols + ted_omni_flux1_cols + \
                   ted_omni_flux2_cols + ted_omni_flux3_cols + ted_omni_flux4_cols + ted_omni_flux5_cols + \
                   ted_omni_flux6_cols + ted_omni_flux7_cols

        for vco in all_cols:
            data[vco] = list()

        recl = 2544
        file.seek(0, 0)

        # This is so that we can skip processing some columns if they are not requested
        # Its a little tricky because if you want processed data you still have to
        # read the raw data

        if procvars:
            docols = procvars
        else:
            docols = all_cols

        for co in range(0, nums):
            #print(co)

            # -----------Read header info once per rec
            file.seek(0 + co * recl)
            # header is 88 bytes
            temp = struct.unpack(head_fmt, file.read(88))
            assign_cols(data, temp, head_cols, 1)

            # ------------ Read 4 sets of 4 lat and lons at once----------------------
            file.seek(88 + co * recl)
            temp = struct.unpack(latlon_fmt, file.read(128))
            # The cols are repeated 4 times because we read in 4 sets at once
            cols = latlon_cols + latlon_cols + latlon_cols + latlon_cols
            assign_cols(data, temp, cols, 10000)

            # ----------- Read 4 sets of ihd data ------------
            file.seek(216 + co * recl)
            temp = struct.unpack(ihd_mdf_fmt, file.read(816))
            cols = ihd_cols + mdf_cols + ihd_cols + mdf_cols + ihd_cols + mdf_cols + ihd_cols + mdf_cols
            assign_cols(data, temp, cols, 1)
            # inc will need a conversion

            # ----------- Read the mep and ted data -----------------------
            file.seek(1032 + co * recl)
            temp = struct.unpack(data_fmt, file.read(608))
            cols = data_cols + data_cols + data_cols + data_cols
            assign_cols(data, temp, cols, 1)
            # These need to go through cnvrt

            # ---------- Read field and pitch data ----------------
            file.seek(1640 + co * recl)
            temp = struct.unpack(B_fmt, file.read(384))
            cols = B_cols + B_cols + B_cols + B_cols
            assign_cols(data, temp, cols, 10000)

            # ---------- Read ted spec and back data ------------------
            file.seek(2024 + co * recl)
            temp = struct.unpack(tspec_fmt, file.read(72))
            cols = tspec_cols
            assign_cols(data, temp, cols, 1)

            # ---------- Read ted omni Eflux data ----------------------
            file.seek(2096 + co * recl)
            temp = struct.unpack(ted_omni_flux_fmt, file.read(448))
            cols = ted_omni_flux_cols + ted_omni_flux_cols + ted_omni_flux_cols + ted_omni_flux_cols
            assign_cols(data, temp, cols, 1)

    # Done reading all the data now do some postprocessing
    #
    # --------------Time----------------------
    # Deal with time cols- in the ngdc raw netcdf files time is saved as a
    # uint64 and milliseconds since 1970
    # First make a list of datetimes
    time2 = list()
    #time1 = list()
    time1 = [dtm.datetime(data['year'][x], 1, 1) + dtm.timedelta(data['day'][x] - 1) +
             dtm.timedelta(seconds=data['msec'][x] / 1000) for x in range(0, len(data['year']))]

    for x in time1:
        for tco in range(0, 4):
            time2.append(x + tco * dtm.timedelta(seconds=2))
    # A lot of the data columns need to be repeated to have the right time cadence
    repeat_cols = ['year', 'day', 'msec', 'satID', 'alt']

    # I'm going to interpolate the B values here
    for col in B_cols:
        f = interpolate.interp1d(np.arange(0,4*len(data[col]),4),data[col],fill_value='extrapolate')
        temp = f( np.arange(0,len(time2 )))
        data[col] = temp
        if col =='MLT':
            data[col]=temp*24.0/360.0

    for col in repeat_cols:
        data[col] = repeat_vals(data[col], 4)

    repeat_cols = head_cols

    for col in repeat_cols:
        data[col] = repeat_vals(data[col], 16)


    # The challenge here is that the spwc time is every 8 seconds and we need to make it
    # every 2 sec to match the netcdf files
    # There is an issue here because the NOAA NGDC raw data writes the data by the actual day
    # The SWPC files don't split up 32 second records so the times are not always the same
    # We will have to open the day before also. Ugh.

    ctime1 = unixtime(time2)
    data['time'] = [1000 * x for x in ctime1]

    # Do final conversions to make binary file look like raw
    # ----------------- sat_direction --------------------------
    # This variable is not included in the SWPC data but is in the raw data
    # so create it here from lat :0 South 1 North
    lat_dif = np.diff(np.array(data['lat']))
    lat_dif[np.where(lat_dif <= 0)] = 0
    lat_dif[np.where(lat_dif > 0)] = 1

    satdir_temp = list()
    satdir_temp.append(lat_dif[0])
    for ldat in lat_dif:
        satdir_temp.append(ldat)
    data['sat_direction'] = satdir_temp

    # ----------------- convert meped cnts ----------------------------
    for col in mep0_cols + mep90_cols + omni_cols:
        dtemp = [cnvrt[x] for x in data[col]]
        data[col] = dtemp

    # ----------------- convert ted spectra and eflux cnts ------------------------------
    for col in tspec0_cols + tspec30_cols + ted0eflux_cols + ted30eflux_cols:
        dtemp = [cnvrt[x] for x in data[col]]
        data[col] = dtemp

    # --------------- omni p6 needs to be divided by two to get cps because it is accumulated for 2 sec
    for col in omni_cols[0:2]:
        dtemp = [x / 2 for x in data[col]]
        data[col] = dtemp
    for col in omni_cols[2:4]:
        dtemp = [x / 4 for x in data[col]]
        data[col] = dtemp

    # The raw data fills in zeros for the ted spectra counts when there is no data
    fill_cols = tspec0_cols + tspec30_cols
    for col in fill_cols:
        data[col] = fill_vals(data[col], 4, -99)

    # Create HK data
    data['HK_data'] = [-99 for x in range(0, len(data['time']))]
    data['HK_key'] = [-99 for x in range(0, len(data['time']))]


    # Additional cal of the analog data
    anacols = ['ted_V', 'ted_electron_CEM_V',
               'ted_proton_CEM_V', 'ted_sweepV', 'TED_temp', 'MEPED_V', 'mep_circuit_temp', 'mep_omni_biase_V',
               'MEP_ptel_biasV', 'MEP_etel_biasV', 'mep_proton_tel_temp', 'MEP_e_T', 'MEP_omni_T',
               'DPU_V', 'microA_V', 'microB_V', 'DPU_temp']
    if len(set(docols) - set(anacols)) < len(docols):
        for col in anacols:
            dtemp = data[col]
            data[col] = [x / 10000 for x in dtemp]


    if datatype == 'raw':
        # If raw data is requested then we are done
        # If ony specific cols (procvars) were requested then just return those.
        if procvars:
            raw_cols = procvars
        else:
            raw_cols = ['time', 'year', 'day', 'msec', 'satID', 'minor_frame', 'major_frame', 'sat_direction',
                    'alt', 'lat', 'lon'] + mep0_cols + mep90_cols + omni_cols + tspec0_cols + tspec30_cols + \
                   ted0eflux_cols + ted30eflux_cols + ['microA_V', 'microB_V', 'DPU_V', 'MEPED_V', \
                    'ted_V', 'ted_sweepV', 'ted_electron_CEM_V','ted_proton_CEM_V', 'mep_omni_biase_V', \
                     'mep_circuit_temp', 'mep_proton_tel_temp', 'TED_temp','DPU_temp', 'HK_data', \
                    'HK_key', 'ted_ele_PHD_level', 'ted_pro_PHD_level','ted_IFC_on', 'mep_IFC_on', 'ted_ele_HV_step','ted_pro_HV_step']
        return {k: np.array(data[k]) for k in raw_cols}
    else:
        # If processed data is requested then calibrations are needed to change counts to flux
        # If just some columns are needed then don't processs all

        if procvars:
            # Check to see if procvars has an strings with mep...flux
            test=[e for e in procvars if (e[0:3]=='mep') & ('flux' in e)]

            # If meped data is requested then process it otherwise don't bother
            if len(test)>0:
                # Processs the meped telescope counts into flux
                data,mep0_cols_proc,mep90_cols_proc,mep_err_col0,mep_err_col90 = process_meped_tel(data,mep0_cols,mep90_cols)

            # NOTE: This just returns flags for the processed omni data because NOAA code
            # to do this  is needed
            # Todo Add the code to process the meped counts to flux using NOAA method
            data,omni_cols_proc = process_meped_omni(data)

            # This processes the intermittent TED spectra counts to #/cm2-s-str-keV
            data, tspec0_cols_proc, tspec30_cols_proc = process_ted_spec(data, tspec0_cols, tspec30_cols)

            # These are the atmospheric energy input cols
            # ted_ele_eflux_atmo_low, ted_ele_eflux_atmo_hi, ted_ele_eflux_atmo_total, ted_pro_eflux_atmo_low,
            # ted_pro_eflux_atmo_hi, ted_pro_eflux_atmo_total, ted_total_eflux_atmo
            # These are already processed and stored in the binary data files
            ted_eflux_cols = ted_omni_flux1_cols + ted_omni_flux2_cols + ted_omni_flux3_cols + ted_omni_flux4_cols + \
                         ted_omni_flux5_cols + ted_omni_flux6_cols + ted_omni_flux7_cols

            # This processes the local energy flux counts to mW/m2
            data, ted0eflux_cols_proc, ted30eflux_cols_proc, ted0eflux_err, ted30eflux_err,ted_eflux_cols_err,tback0_cols_proc, \
                tback30_cols_proc = process_ted_eflux(data, ted_eflux_cols, tback0_cols, tback30_cols)
        else:
            # Process all the data

            # Processs meped telescope counts to flux
            data,mep0_cols_proc,mep90_cols_proc,mep_err_col0,mep_err_col90 = process_meped_tel(data,mep0_cols,mep90_cols)
            data,omni_cols_proc = process_meped_omni(data)

            data,tspec0_cols_proc,tspec30_cols_proc = process_ted_spec(data, tspec0_cols, tspec30_cols)

            ted_eflux_cols = ted_omni_flux1_cols + ted_omni_flux2_cols + ted_omni_flux3_cols + ted_omni_flux4_cols + \
                         ted_omni_flux5_cols + ted_omni_flux6_cols + ted_omni_flux7_cols

            data,ted0eflux_cols_proc,ted30eflux_cols_proc,ted0eflux_err,ted30eflux_err,ted_eflux_cols_err,tback0_cols_proc, \
                tback30_cols_proc= process_ted_eflux(data,ted_eflux_cols,tback0_cols,tback30_cols)

        data['aacgm_lat_foot'] = [-99 for x in data['time']]
        data['aacgm_lon_foot'] = [-99 for x in data['time']]
        data['mag_lat_sat'] = [-99 for x in data['time']]
        data['mag_lon_sat'] = [-99 for x in data['time']]
        data['Bx_sat'] = [-99 for x in data['time']]
        data['By_sat'] = [-99 for x in data['time']]
        data['Bz_sat'] = [-99 for x in data['time']]
        data['ted_ele_eflux_atmo_low_err']=[-99 for x in data['time']]
        data['ted_ele_eflux_atmo_hi_err'] = [-99 for x in data['time']]
        data['ted_ele_eflux_atmo_total_err'] = [-99 for x in data['time']]
        data['ted_pro_eflux_atmo_low_err']=[-99 for x in data['time']]
        data['ted_pro_eflux_atmo_hi_err'] = [-99 for x in data['time']]
        data['ted_pro_eflux_atmo_total_err'] = [-99 for x in data['time']]
        data['ted_total_eflux_atmo_err'] = [-99 for x in data['time']]

        if procvars:
            fin_cols = procvars
        else:
            proc_cols = ['time', 'year', 'day', 'msec', 'satID', 'sat_direction', 'alt', 'lat',
                         'lon'] + mep0_cols_proc + \
                        mep90_cols_proc + mep_err_col0 + mep_err_col90 + \
                        ['mep_ele_tel0_flux_e4', 'mep_ele_tel0_flux_e4_err', 'mep_ele_tel90_flux_e4',
                         'mep_ele_tel90_flux_e4_err'] + \
                        omni_cols_proc + tspec0_cols_proc + tspec30_cols_proc + \
                        ted0eflux_cols_proc + ted30eflux_cols_proc + ted0eflux_err + ted30eflux_err + ted_eflux_cols + \
                        ted_eflux_cols_err + ted0maxE_cols + ted30maxE_cols + tback0_cols + tback30_cols + \
                        tback0_cols_proc + tback30_cols_proc + ['ted_ele_eflux_atmo_low_err','ted_ele_eflux_atmo_hi_err',
                        'ted_ele_eflux_atmo_total_err','ted_pro_eflux_atmo_low_err','ted_pro_eflux_atmo_hi_err',
                        'ted_pro_eflux_atmo_total_err','ted_total_eflux_atmo_err'] + \
                        ['Br_sat', 'Bt_sat', 'Bp_sat', 'Btot_sat',
                         'Br_foot', 'Bt_foot', 'Bp_foot', 'Btot_foot', 'geod_lat_foot', 'geod_lon_foot',
                         'aacgm_lat_foot',
                         'aacgm_lon_foot', 'mag_lat_foot', 'mag_lon_foot', 'mag_lat_sat', 'mag_lon_sat', 'Bx_sat',
                         'By_sat', 'Bz_sat', 'ted_alpha_0_sat', 'ted_alpha_30_sat', 'ted_alpha_0_foot',
                         'ted_alpha_30_foot',
                         'meped_alpha_0_sat', 'meped_alpha_90_sat', 'meped_alpha_0_foot', 'meped_alpha_90_foot',
                         'L_IGRF', 'MLT', 'ted_IFC_on', 'mep_IFC_on']
            fin_cols = proc_cols

        return {k: np.array(data[k]) for k in fin_cols}

    return 0


def decode_filename( file_name ):
    """
    Decodes POES/MetOp data file names.

    :param  fn: Filename to decode.
    :return:   Dictionary.
    """
    my_name = 'decode_filename'

    # Remove directory prefix if exists:
    t_fn = os.path.basename( file_name )

    info = { 'file_name' : file_name,
             'sat_name'  : t_fn[5:8],
             'dt'        : dtm.datetime.strptime( t_fn[9:9+8], '%Y%m%d' ),
             'level'     : 'processed/ngdc/uncorrected/full',
             'type'      : 'NC4' }

    return( info )


def get_file_names( dir_root_list, fn_pattern_list ):
    """ Returns list of file names given a starting directory.
    Used by get_data_***

    :param  dir_root (str or list):        String or list of stirngs of absolute or relative user path (OPTIONAL).
    :param  fn_pattern_list (list):               List of filename patterns.
    :return fn_list (list):                List of files found or empty [].

    CHANGES 04/2020: JGREEN - made it so it can accept lists or strings as input
    """
    my_name = 'get_file_names'

    try:
        import fnmatch
        logger.info( my_name+': ['+', '.join( fn_pattern_list )+']' )
        logger.debug( fn_pattern_list )

        # TODO: Check the inputs.
        # This turns whatever you pass into a list so it can accept either a list of strings or a single string
        if not isinstance(dir_root_list, list):
            dir_root_list = [dir_root_list]
        if not isinstance(fn_pattern_list, list):
            fn_pattern_list = [fn_pattern_list]

        fn_list = []

        # Create a list of files
        for dir_root in dir_root_list:
            for root, dirnames, filenames in os.walk( dir_root ):
                for fn_pattern in fn_pattern_list:
                    for filename in fnmatch.filter( filenames, fn_pattern ):
                        fn_list.append( os.path.join( root, filename ) )

            if fn_list: break

        # sort file names
        fn_list = sorted( fn_list )

        return( fn_list )

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.fatal( my_name+': Exception Caught!' )
        logger.fatal( exc_type, fname, exc_tb.tb_lineno )
        logger.fatal( traceback.format_exc() )


def is_meped_var_supposed_to_be_empty_during_IFC( varname ):

    '''This function is used by get_data to set processed values to a flag if an IFC is in progress'''
    meped_vars_expected_empty_during_ifc = \
        ['mep_pro_tel0_flux_p1',
        'mep_pro_tel0_flux_p2',
        'mep_pro_tel0_flux_p3',
        'mep_pro_tel0_flux_p4',
        'mep_pro_tel0_flux_p5',
        'mep_pro_tel0_flux_p6',
        'mep_pro_tel0_flux_p1_err',
        'mep_pro_tel0_flux_p2_err',
        'mep_pro_tel0_flux_p3_err',
        'mep_pro_tel0_flux_p4_err',
        'mep_pro_tel0_flux_p5_err',
        'mep_pro_tel0_flux_p6_err',
        'mep_pro_tel90_flux_p1',
        'mep_pro_tel90_flux_p2',
        'mep_pro_tel90_flux_p3',
        'mep_pro_tel90_flux_p4',
        'mep_pro_tel90_flux_p5',
        'mep_pro_tel90_flux_p6',
        'mep_pro_tel90_flux_p1_err',
        'mep_pro_tel90_flux_p2_err',
        'mep_pro_tel90_flux_p3_err',
        'mep_pro_tel90_flux_p4_err',
        'mep_pro_tel90_flux_p5_err',
        'mep_pro_tel90_flux_p6_err',
        'mep_ele_tel0_flux_e1',
        'mep_ele_tel0_flux_e2',
        'mep_ele_tel0_flux_e3',
        'mep_ele_tel0_flux_e4',
        'mep_ele_tel0_flux_e1_err',
        'mep_ele_tel0_flux_e2_err',
        'mep_ele_tel0_flux_e3_err',
        'mep_ele_tel0_flux_e4_err',
        'mep_ele_tel90_flux_e1',
        'mep_ele_tel90_flux_e2',
        'mep_ele_tel90_flux_e3',
        'mep_ele_tel90_flux_e4',
        'mep_ele_tel90_flux_e1_err',
        'mep_ele_tel90_flux_e2_err',
        'mep_ele_tel90_flux_e3_err',
        'mep_ele_tel90_flux_e4_err',
        'mep_omni_flux_p1',
        'mep_omni_flux_p2',
        'mep_omni_flux_p3',
        # TODO: Add this one after correctly setting a _FillValue in master NetCDF CDL: 'mep_omni_flux_flag_fit',
        # TODO: Add this one after correctly setting a _FillValue in master NetCDF CDL: 'mep_omni_flux_flag_iter_lim'
        ]

    return varname in meped_vars_expected_empty_during_ifc

def get_file_list(sat_name, dt_start, dt_end, dir_root_list, dtype, swpc_root_list = None, all=False):
    ''' Used by get_data to create a list of file names to process

    :param sat_name (str):    i.e. 'n15'
    :param dt_start (datetime): start date
    :param dt_end (datetime): end date
    :param dtype (str): 'raw','proc','avg'
    :return fn_list (list): list of file names to look for
    '''

    my_name = 'get_file_list'
    # Create a list of ngdc data files to look for in the specified directories
    n_days = ((dtm.timedelta(days=1) \
               + dtm.datetime(year=dt_end.year, month=dt_end.month, day=dt_end.day) \
               - dtm.datetime(year=dt_start.year, month=dt_start.month, day=dt_start.day))).days

    file_pattern_list = []
    for i_day in np.arange(n_days):
        t_dt = dt_start + dtm.timedelta(days=int(i_day))
        # if it is not the average cdf files than add dtype to the end
        # That allows V2 files to be found as well
        if dtype != 'avg':
            file_pattern_list.append('poes_' + sat_name + '_%04d%02d%02d_%s.nc' % (t_dt.year, t_dt.month, t_dt.day,dtype))
        else:
            file_pattern_list.append('poes_' + sat_name + '_%04d%02d%02d.cdf' % (t_dt.year, t_dt.month, t_dt.day))

    logger.debug(my_name + ': Searching for files to aggregate: ' + '\n\t'.join(dir_root_list))

    # Run through all the directories in dir_root and look for files that match
    # Check if dir_root_list is a list of directories or a single directory
    for dir_root in dir_root_list:
        logger.debug(my_name + ': Searching %s' % dir_root)

        fn_list = get_file_names(dir_root, file_pattern_list)
        logger.debug(my_name + ': Found these %d files in \'%s\':' % (len(fn_list), dir_root))
        logger.debug(fn_list)
        # Once we have a list we'll move on (avoids finding same file names in multiple places).
        # Warning: If multiple file repositories exist and first one is incomplete then so will be the returned NetCDF4 object.
        if fn_list: break

    logger.debug('Found %d files to aggregate.' % len(fn_list))

    # Now check any remaining files and see if there is a swpc raw file with it
    # If a raw swpc binary file is found, process it and save an raw nc version
    # So when looking for swpc files first check for nc files and then
    # .bin files

    # IF the all key is set to True then return swpc binary file data as well
    if all:
        # Find what files were not found in the ngdc data search
        more_files = len(file_pattern_list) - len(fn_list)
        if more_files > 0:
            # First look for nc swpc files
            # Make a list of files that still need to be found
            more_files_get = list()
            for file in file_pattern_list:
                test = [x for x in fn_list if file in fn_list]
                if len(test) < 1:
                    more_files_get.append(file)
            for dir_root in swpc_root_list:
                fn_swpc_list_nc = get_file_names(dir_root, more_files_get)

            # Now check to see if there are still more files to find
            still_more = len(more_files_get) - len(fn_swpc_list_nc)

            # If there are still more files to find then check the bin data
            if still_more > 0:
                # change the file names to .bin
                still_more_files_get = list()
                for file in more_files_get:
                    test = [x for x in fn_swpc_list_nc if file in fn_swpc_list_nc]
                    if len(test) < 1:
                        still_more_files_get.append(file)
                swpc_files = [x[0:17] + '.bin' for x in still_more_files_get]
                for dir_root in swpc_root_list:
                    fn_swpc_list_bin = get_file_names(dir_root, swpc_files)

            # Now append the two types of swpc files to fn_list
            fn_list.extend(fn_swpc_list_nc)
            fn_list.extend(fn_swpc_list_bin)
    return (fn_list)

def get_data_processed( dataloc, sat_name, dt_start, dt_end, clobber=True, vars=None, all = True  ):
    """ Returns aggregated POES ngdc format processed data between dt_start and dt_end
    :param dataloc (str)                    A location and name of a either a configfile or a top directory
                                            The program checks to see if this is a configfile with multiple
                                            POES directories defined. If it is not, then it assumes it is the top
                                            level directory and looks for data in all below.
    :param sat_name (string):               One of ['m01', 'm02', 'm03', 'n15', ..., 'n19' ].
    :param dt_start (datetime):             Start datetime with year, month, day
    :param dt_end (datetime):               End datetime.
    :param clobber (Optional:True or False): If True will overwrite existing tmp aggregate files
    :param vars (Optional: list):            A list of variables to return vars = ['time','L_IGRF'] etc
    :param all (Optional: True or False):    If True will also include reformatted SWPC binary data
    :return (NetCDF object):
    """
    return get_data(dataloc, sat_name, dt_start, dt_end, clobber=clobber, vars=vars, all=all, dtype = 'proc')

def get_data_raw( dataloc, sat_name, dt_start, dt_end, clobber=True, vars=None, all=True  ):
    """ Returns aggregated POES ngdc format raw data
    :param dataloc (str)                    A location and name of a either a configfile or a top directory
                                            The program checks to see if this is a configfile with multiple
                                            POES directories defined. If it is not, then it assumes it is the top
                                            level directory and looks fordata in all below.
    :param sat_name (string):               One of ['m01', 'm02', 'm03', 'n15', ..., 'n19' ].
    :param dt_start (datetime):             Start datetime with year, month, day
    :param dt_end (datetime):               End datetime.
    :param clobber (Optional:True or False): If True will overwrite existing tmp aggregate files
    :param vars (Optional: list):           A list of variables to return
    :param all (Optional: True or False):    If True will also include reformatted SWPC binary data
    :return (NetCDF object):
    """
    return get_data(dataloc, sat_name, dt_start, dt_end, clobber=clobber, vars=vars, all=all, dtype = 'raw')

def get_data( dataloc, sat_name, dt_start, dt_end, clobber=True, vars=None, all=True, dtype = None  ):
    """ Returns aggregated POES ngdc format "raw" or processed data between dt_start and dt_end
        Unfortunately, there are some error in the NGDC netcdf file that prevent them from easily
        being concatentated. This code creates a new aggregated netcdf file and then reads that which makes
        it a bit slow.

    :param dataloc (str)                    A location and name of a configfile or a top directory
                                            The program checks to see if this is a configfile with multiple
                                            POES directories defined. If it is not, then it assumes it is the top
                                            level directory and looks for raw data in all below.
    :param sat_name (string):               One of ['m01', 'm02', 'm03', 'n15', ..., 'n19' ].
    :param dt_start (datetime):             Start datetime.
    :param dt_end (datetime):               End datetime.
    :param clobber (Optional:True or False): If True will overwrite existing tmp aggregate files
    :param vars (Optional: list):           A list of variables to return
    :param all (Optional: Treu or False):    If True will include SWPC binary data
    :param dtype (string)                    Type of data requested ('raw' or 'processed')
    :return (NetCDF object):

    UPDATES:
    09/2019 JGREEN: added the keyword all. If this is true then the code
    searches the raw swpc directory as well. If no ngdc data exists it will look for swpc data, create
    an equivalent netcdf file in the swpc directory and load that.
    04/2020: JGREEN: changed this so that you can either pass a configfile or a single top level directory
            And made it so that raw and processed used the same code
    08/2020: JGREEN: changed it so that data ist not first read to a temporary dictionary and only the
            variales requested are writtne to the new aggregated nc file
    """

    my_name = 'get_data'

    try:
        # Get the data location info
        # check if dataloc is a directory
        if os.path.isdir(dataloc):
            # dir_root_list is expecting a list so make sure a list is passed
            if isinstance(dataloc,str):
                dir_root_list = [dataloc]
            else:
                dir_root_list = dataloc

            # Also need the data netcdf template
            # Todo make it so you don't need these
            if dtype=='raw':
                fn_master_test = get_file_names(dir_root_list, ['poes_raw-aggregate_master.nc'])
            elif dtype=='proc':
                fn_master_test = get_file_names(dir_root_list, ['poes_l2-aggregate_master.nc'])
            # The average swpc data does not need a master file

            if ((dtype=='raw') | (dtype=='proc')) & len(fn_master_test)>0:
                fn_master=fn_master_test[0]
            else:
                print('Did not find raw master netcdf template poes_raw-aggregate_master.nc. No data will be returned')
                raise Exception('Did not find raw master netcdf template poes_raw-aggregate_master.nc')

            if all:
                # This assumes all data is under the same directory
                swpc_root_list = dir_root_list
            # aggregated data will be put under the same directory
            dir_agg = dir_root_list[0]
            is_dir = 1
        else:
            # If a top level directory is not passed then assume it is a config file with dirs
            config = load_config(dataloc)
            if dtype=='raw':
                fn_master = config['fn_master_raw']                 # raw data netcdf data template
                dir_root_list = config['dir_data_raw_search_paths']  # directory of the raw ngdc data
            elif dtype=='proc':
                fn_master = config['fn_master_l2']                  # processed data netcdf data template
                dir_root_list = config['dir_data_l2_search_paths']  # directory of processed data

            dir_agg = config['dir_aggregates']
            if all:
                swpc_root_list = config['dir_data_binary_search_paths'] # directory of swpc binary data
            is_dir = 0

        # Need to make sure it found fn_master for raw or processed data otherwise it will not return data
        if  (not os.path.isfile(fn_master)):
            print('Did not find master netcdf template poes_raw-aggregate_master.nc. or poes_l2-aggregate_master.nc. No data will be returned')
            raise ('Did not find raw master netcdf template poes_raw-aggregate_master.nc')

        # Todo Probably should check that the directories exist

        #####
        # Data: return Existing aggregate or create New aggregate
        #####

        # If file exists, skip aggregation:
        yyyymmdd1 = '%04d%02d%02d' % ( dt_start.year, dt_start.month, dt_start.day )
        yyyymmdd2 = '%04d%02d%02d' % ( dt_end.year,   dt_end.month,   dt_end.day   )

        if dtype =='raw':
            fn_agg = dir_agg + '/poes_%s_%s-%s_raw.nc' % ( sat_name, yyyymmdd1, yyyymmdd2 )
        elif dtype =='proc':
            fn_agg = dir_agg + '/poes_%s_%s-%s_proc.nc' % (sat_name, yyyymmdd1, yyyymmdd2)

        if not clobber and os.path.isfile( fn_agg ):
            logger.info( my_name+': Found existing aggregate, using %s.' % fn_agg )
            # This jumps to end and loads the old file
        else:
            # Clobber existing aggregate if desired
            if clobber and os.path.isfile( fn_agg ):
                logger.info( my_name+': Clobber is on, removing existing aggregate %s.' % fn_agg )
                os.unlink( fn_agg )

            '''-------- List of Files to Ingest ------------------------- '''
            # Returns a list of data files found (data are organized as day files):
            fn_list = get_file_list(sat_name, dt_start, dt_end, dir_root_list, dtype, swpc_root_list =swpc_root_list, all=all)

            ## If no files are found return None
            if len( fn_list ) == 0: return None

            ''' --------- Ingest Data -------------------------'''
            # Copy Master NetCDF to new file in '/tmp' or User choice:
            # The Master NetCDF has the correct variable names and attributes
            # The new aggregated data just has to be added
            # If only some vars are requested then copy those from the master

            if vars:
                with nc4.Dataset(fn_master,"r") as src, nc4.Dataset(fn_agg, "w") as dst:
                    # copy global attributes all at once via dictionary
                    dst.setncatts(src.__dict__)
                    # copy dimensions
                    for name, dimension in src.dimensions.items():
                        dst.createDimension(
                            name, (len(dimension) if not dimension.isunlimited() else None))
                    # copy all file data except for the excluded
                    for name, variable in src.variables.items():
                        if name in vars:
                            x = dst.createVariable(name, variable.datatype, variable.dimensions)
                            #dst[name][:] = src[name][:]
                            # copy variable attributes all at once via dictionary
                            dst[name].setncatts(src[name].__dict__)
            else:
                shutil.copy( fn_master, fn_agg )

            # Ingest each day file to Temporary Memory dictionary:
            nc_all = nc4.Dataset( fn_agg, 'r+')

            if vars:
                varlist = vars
            else:
                varlist = nc_all.variables

            #starttime = timeit.default_timer()
            varstart=0
            for fn in fn_list:
                # First check if it is a bin file or nc file
                # If it is nc then open with nc4 and add variables to t_data

                if fn[-3:] =='.nc':
                    # This opens the nc file and reads each variable into t_data
                    with nc4.Dataset( fn, 'r' ) as nc_day:
                        logger.debug( my_name+': Ingesting %s.' % fn )

                        idx_ifc = np.where(nc_day['mep_IFC_on'][:] > 0)[0]

                        for vname in varlist:
                            var_day = nc_day.variables[ vname ]

                            var_day.set_auto_mask(False)

                            # L2 processing doesn't correctly fill MEPED variables during IFC.
                            # Temporary variable to hold corrected values (if we need to correct them)
                            var_day_fixed = var_day[:]
                            if len(idx_ifc) > 0 and is_meped_var_supposed_to_be_empty_during_IFC(vname):
                                # Fill based on new master's _FillValue, noting that mep_IFC_on is only set every 16 time steps,
                                # so we Fill the full range +/- 16 time steps. Multiple IFCs in a day will result in over FillValue'ing
                                # the data but that's unlikely.
                                dt_ifc = nc4.num2date(nc_day['time'][idx_ifc], units=nc_all['time'].units)
                                logger.info( my_name + ': Filling %s during MEPED IFC for time range %s - %s.' % (vname, str(dt_ifc[0]), str(dt_ifc[-1]) ) )
                                var_day_fixed[ idx_ifc[0] - 16 : idx_ifc[-1] + 16 ] = nc_all[ vname ]._FillValue

                            # Add the data to the new nc files
                            nc_all[vname][varstart:varstart+len(var_day_fixed[:])] = var_day_fixed[:]

                else:
                    #If it is bin file then open with read_POES_bin
                    if dtype=='raw':
                        pdata = read_poes_bin(fn,datatype = 'raw',procvars = vars)
                    else:
                        pdata = read_poes_bin(fn, datatype='processed', procvars=vars)

                    for vname in varlist:
                        nc_all[vname][varstart:(varstart + len(pdata[vname][:]))] = pdata[vname][:]

                varstart=len(nc_all['time'])

        # Return NC4 object:
        logger.info( my_name+': Returning handle to %s.' % fn_agg )
        return( nc_all )

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error( my_name+': Exception Caught!' )
        # This was not working
        #logger.error( exc_type, fname, exc_tb.tb_lineno )
        logger.error( traceback.format_exc() )

        return( None )
def get_data_V2( dataloc, sat_name, dt_start, dt_end, vars=None, dtype = None  ):
    """ This doesn't really work well becasue xarray concatenates every variable which is slow
        Returns aggregated POES V2 "raw" or "processed" data between dt_start and dt_end
        V2 data are the new netcdf files created by fix_POES_netcdf.py. The original files at NGDC
        have an issue with the dimensions that makes them difficult to concatenate and use. The fix_POES_netcdf
        code rewrites the netcdf data and corrects this error so the dimension of the variables is time. It
        allows faster concatenation. If you are working with a lot of data it may make sense to fix the datafiles
        and then use this code. The code assumes the V2 files are in the same location as the processed and raw data

    :param dataloc (str)                    A location and name of a configfile or a top directory
                                            The program checks to see if this is a configfile with multiple
                                            POES directories defined. If it is not, then it assumes it is the top
                                            level directory and looks for raw data in all below.
    :param sat_name (string):               One of ['m01', 'm02', 'm03', 'n15', ..., 'n19' ].
    :param dt_start (datetime):             Start datetime.
    :param dt_end (datetime):               End datetime.
    :param vars (Optional: list):           A list of variables to return
    :param dtype (string)                    Type of data requested ('raw' or 'processed')
    :return (NetCDF object):

    UPDATES:

    """

    my_name = 'get_data_V2'

    try:
        # Get the data location info
        # check if dataloc is a directory
        if os.path.isdir(dataloc):
            # dir_root_list is expecting a list so make sure a list is passed
            if isinstance(dataloc,str):
                dir_root_list = [dataloc]
            else:
                dir_root_list = dataloc

            is_dir = 1
        else:
            # If a top level directory is not passed then assume it is a config file with dirs
            config = load_config(dataloc)
            if dtype=='raw':
                dir_root_list = config['dir_data_raw_search_paths']  # directory of the raw ngdc data
            elif dtype=='proc':
                dir_root_list = config['dir_data_l2_search_paths']  # directory of processed data

            is_dir = 0

        # NOTE: Probably should check that the directories exist

        #####
        # Data: return new concatenated data using xarray
        #####

        '''-------- List of Files to Ingest ------------------------- '''
        # Check if dtype has V2 on the end
        if dtype.find('V2')<0:
            dtype = dtype + '_V2'

        # Returns a list of data files found (data are organized as day files):
        fn_list = get_file_list(sat_name, dt_start, dt_end, dir_root_list, dtype, all=False)


        # If no files are found return None
        if len( fn_list ) == 0: return None

        ''' --------- Open V2 Data -------------------------'''
        # fn_list returns swpc bin and V2 files
        # First get the list of just V2 files
        V2_list = [s for s in fn_list if "V2" in s]
        # xarray automatically gets all the data so if you don't want some variables you need to drop them
        allprocvars = ['year','day','msec','satID','sat_direction','alt','lat','lon','mep_pro_tel0_flux_p1',
                       'mep_pro_tel0_flux_p2','mep_pro_tel0_flux_p3','mep_pro_tel0_flux_p4','mep_pro_tel0_flux_p5',
                       'mep_pro_tel0_flux_p6','mep_pro_tel0_flux_p1_err',
                       'mep_pro_tel0_flux_p2_err','mep_pro_tel0_flux_p3_err','mep_pro_tel0_flux_p4_err','mep_pro_tel0_flux_p5_err',
                       'mep_pro_tel0_flux_p6_err','mep_pro_tel90_flux_p1',
                       'mep_pro_tel90_flux_p2','mep_pro_tel90_flux_p3','mep_pro_tel90_flux_p4','mep_pro_tel90_flux_p5',
                       'mep_pro_tel90_flux_p6','mep_pro_tel90_flux_p1_err',
                       'mep_pro_tel90_flux_p2_err','mep_pro_tel90_flux_p3_err','mep_pro_tel90_flux_p4_err','mep_pro_tel90_flux_p5_err',
                       'mep_pro_tel90_flux_p6_err','mep_ele_tel0_flux_e1','mep_ele_tel0_flux_e2','mep_ele_tel0_flux_e3','mep_ele_tel0_flux_e4',
                       'mep_ele_tel0_flux_e1_err','mep_ele_tel0_flux_e2_err','mep_ele_tel0_flux_e3_err','mep_ele_tel0_flux_e4_err',
                       'mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2', 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4',
                       'mep_ele_tel90_flux_e1_err', 'mep_ele_tel90_flux_e2_err', 'mep_ele_tel90_flux_e3_err',
                       'mep_ele_tel90_flux_e4_err','mep_omni_flux_p1','mep_omni_flux_p2','mep_omni_flux_p3','mep_omni_flux_flag_fit',
                       'mep_omni_flux_flag_iter_lim','mep_omni_gamma_p1','mep_omni_gamma_p2','mep_omni_gamma_p3',
                       'ted_ele_tel0_flux_4','ted_ele_tel0_flux_8','ted_ele_tel0_flux_11','ted_ele_tel0_flux_14',
                       'ted_ele_tel30_flux_4', 'ted_ele_tel30_flux_8', 'ted_ele_tel30_flux_11', 'ted_ele_tel30_flux_14',
                       'ted_pro_tel0_flux_4', 'ted_pro_tel0_flux_8', 'ted_pro_tel0_flux_11', 'ted_pro_tel0_flux_14',
                       'ted_pro_tel30_flux_4', 'ted_pro_tel30_flux_8', 'ted_pro_tel30_flux_11', 'ted_pro_tel30_flux_14',
                       'ted_ele_tel0_low_eflux','ted_ele_tel30_low_eflux','ted_ele_tel0_hi_eflux','ted_ele_tel30_hi_eflux',
                       'ted_pro_tel0_low_eflux','ted_pro_tel30_low_eflux','ted_pro_tel0_hi_eflux','ted_pro_tel30_hi_eflux',
                       'ted_ele_tel0_low_eflux_error','ted_ele_tel30_low_eflux_error','ted_ele_tel0_hi_eflux_error',
                       'ted_ele_tel30_hi_eflux_error','ted_pro_tel0_low_eflux_error','ted_pro_tel30_low_eflux_error',
                       'ted_pro_tel0_hi_eflux_error','ted_pro_tel30_hi_eflux_error','ted_ele_eflux_atmo_low','ted_ele_eflux_atmo_hi',
                       'ted_ele_eflux_atmo_total','ted_ele_eflux_atmo_low_err','ted_ele_eflux_atmo_hi_err','ted_ele_eflux_atmo_total_err',
                       'ted_pro_eflux_atmo_low','ted_pro_eflux_atmo_hi','ted_pro_eflux_atmo_total','ted_pro_eflux_atmo_low_err',
                       'ted_pro_eflux_atmo_hi_err','ted_pro_eflux_atmo_total_err','ted_total_eflux_atmo','ted_total_eflux_atmo_err',
                       'ted_ele_energy_tel0','ted_ele_energy_tel30','ted_pro_energy_tel0','ted_pro_energy_tel30','ted_ele_max_flux_tel0',
                       'ted_ele_max_flux_tel30','ted_pro_max_flux_tel0','ted_pro_max_flux_tel30','ted_ele_eflux_bg_tel0_low',
                       'ted_ele_eflux_bg_tel30_low','ted_ele_eflux_bg_tel0_hi','ted_ele_eflux_bg_tel30_hi','ted_pro_eflux_bg_tel0_low',
                       'ted_pro_eflux_bg_tel30_low','ted_pro_eflux_bg_tel0_hi','ted_pro_eflux_bg_tel30_hi','ted_ele_eflux_bg_tel0_low_cps',
                       'ted_ele_eflux_bg_tel30_low_cps','ted_ele_eflux_bg_tel0_hi_cps','ted_ele_eflux_bg_tel30_hi_cps','ted_pro_eflux_bg_tel0_low_cps',
                       'ted_pro_eflux_bg_tel30_low_cps','ted_pro_eflux_bg_tel0_hi_cps','ted_pro_eflux_bg_tel30_hi_cps','Br_sat',
                       'Bt_sat','Bp_sat','Btot_sat','Br_foot','Bt_foot','Bp_foot','Btot_foot','geod_lat_foot','geod_lon_foot',
                       'aacgm_lat_foot','aacgm_lon_foot','mag_lat_foot','mag_lon_foot','mag_lat_sat','mag_lon_sat','Bx_sat',
                       'By_sat','Bz_sat','ted_alpha_0_sat','ted_alpha_30_sat','ted_alpha_0_foot','ted_alpha_30_foot','meped_alpha_0_sat',
                       'meped_alpha_90_sat','meped_alpha_0_foot','meped_alpha_90_foot','L_IGRF','MLT','ted_IFC_on','mep_IFC_on'
                       ]
        if vars:
            if dtype =='proc_V2':
                dropvars = [s for s in allprocvars if not any(xs in s for xs in vars)]
            else:
                dropvars = []
        else:
            dropvars = []
        if len(V2_list)>0:

            pdata = xr.open_mfdataset(V2_list,decode_times=False,drop_variables=dropvars)
            #pdata = xr.open_mfdataset(V2_list, decode_times=False)
            #print(timeit.default_timer() - start_time)
            print('Here')

        return( pdata )

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error( my_name +': Exception Caught!' )
        # This was not working
        #logger.error( exc_type, fname, exc_tb.tb_lineno )
        logger.error( traceback.format_exc() )

        return( None )


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                 File I/O SWPC                                 "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def decode_filename_swpc( file_name ):
    """
    Decodes POES/MetOp data file names.
    :param fn: Filename to decode.
    :return:   Dictionary.
    """
    my_name = 'decode_filename'

    # Remove directory prefix if exists:
    t_fn = os.path.basename( file_name )

    info = { 'file_name' : file_name,
             'sat_name'  : t_fn[5:8],
             'dt'        : dtm.datetime.strptime( t_fn[9:9+8], '%Y%m%d' ),
             'level'     : 'processed/swpc/uncorrected/avg',
             'type'      : 'CDF' }

    return( info )



def get_data_swpc_avg( dataloc, sat_name, dt_start, dt_end, clobber=False ):
    """ Returns aggregated POES 16 sec avg processed CDF .

    :param sat_name:      One of {'m01', 'm02', 'm03', 'n15', ..., 'n19' }.
    :param dt_start:      Start datetime.
    :param dt_end:        End datetime.
    :param dir_user_root: String of a single absolute or relative user path (OPTIONAL).
    :return:              CDF object.
    """
    my_name = 'get_data_swpc_avg'

    try:
        # Get the data location info
        # check if dataloc is a directory
        if os.path.isdir(dataloc):
            # dir_root_list is expecting a list
            if isinstance(dataloc,str):
                dir_root_list = [dataloc]
            else:
                dir_root_list = dataloc

            # aggregated data will be put under the same directory
            dir_agg = dir_root_list[0]
            is_dir = 1
        else:
            config = load_config(dataloc)
            dir_agg = config['dir_aggregates']

            is_dir = 0

        #####
        # Data: return Existing handle or create New aggregate
        #####

        # Imports
        import random, string, subprocess
        import pycdf

        # If file exists, skip aggregation:
        yyyymmdd1 = '%04d%02d%02d' % ( dt_start.year, dt_start.month, dt_start.day )
        yyyymmdd2 = '%04d%02d%02d' % ( dt_end.year,   dt_end.month,   dt_end.day   )
        # fn_agg should not have an addiitonal /tmp
        fn_agg = dir_agg + '/poes_%s_%s-%s.cdf' % ( sat_name, yyyymmdd1, yyyymmdd2 )
        if os.path.isfile( fn_agg ):
            logger.debug( my_name+': Found existing aggregate, using %s.' % fn_agg )

        else:
            ''' ---------- List of Files to Ingest ------------------ '''
            # File name pattern we need to look for (data are organized as day files):
            dtype='avg'
            fn_list = get_file_list(sat_name, dt_start, dt_end, dir_root_list, dtype, swpc_dir_root_list=[])

            logger.debug( 'Found %d files to aggregate.' % len( fn_list ) )
            if len( fn_list ) == 0: return None

            random_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
            fn_cdfmerge = '/Users/janet/PycharmProjects/SHELLS/tmp/flist_cdfmerge_%s.txt' % random_id
            with open( fn_cdfmerge, 'w' ) as fp:
                for fname in fn_list:
                    fp.write('%s\n' % fname )

                fp.write( fn_agg[:-3] )
            #JGREEN: Needed to change this directory because there is no cdf36
            cmd = ["/Applications/cdf34_1-dist/bin/cdfmerge", "-nolog", "-noprefix", "-dataonly", "-file" , fn_cdfmerge ]

            #cmd = ["/Applications/cdf36_1-dist/bin/cdfmerge", "-nolog", "-noprefix", "-dataonly", "-file", fn_cdfmerge ]
            logger.debug( 'Running command: %s' % ' '.join( cmd ) )
            subprocess.call( cmd )

        # Return CDF object:
        logger.info( my_name+': Returning handle to %s.' % fn_agg )
        cdf_all = pycdf.CDF( fn_agg )
        return( cdf_all )

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error( my_name+': Exception Caught!' )
        logger.error( exc_type, fname, exc_tb.tb_lineno )
        logger.error( traceback.format_exc() )

        return( None )


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                     Analysis                                  "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def bindata(bdata,xvar,yvar,xbin,ybin,procvars = None,):
    """
    The purpose is to bin the POES data
    The most common example would be to bin on passes and L

    :param bdata - a structured numpy array of or netcdf4 array
    :param xvar - the x variable to bin on ex 'L_IGRF' : this can be a string variable name or a data array
    :param yvar - the yvariable to bin on ex passes    : this can be a string variable name or a data array
    :param xbin - the bins for x ex np.arange(0,10,.25)
    :param ybin - the bins for y ex np.arange(min(passes),max(passes)+1)
    :param procvars - a list of columns to bin otherwise all in data will be binned"""

    # Check if x var is a varibale name
    if type(xvar)==str:
        xvar = bdata[xvar][:]
    if type(yvar)==str:
        yvar = bdata[yvar][:]

    # This is kind of tricky because bdata could be a numpy.ndarray
    # or it could be a netCDF4 object or possibly a dict
    dtemp = []
    if procvars:
        vars = procvars
        for var in vars:
            dtemp.append(list(bdata[var][:].data))
    else:
        # If no variables are passed then we have to get them and collect the data
        if type(bdata).__module__ =='netCDF4._netCDF4':
            vars = list(bdata.variables.keys())
            for var in vars:
                dtemp.append(list(bdata[var][:].data))
        elif type(bdata).__module__ == 'numpy':
            # Check to see if it is a structured array with cols
            if bdata.dtype.names:
                vars = list(bdata.dtype.names)
                for var in vars:
                    dtemp.append(list(bdata[var][:].data))
            else:
            # If not just use the straight data
                dtemp =bdata
        elif type(bdata).__module__ == 'numpy.ma.core':
            dtemp=bdata
        elif type(bdata)==list:
            dtemp = bdata
        else:
            return

    # This returns a numpy array witout the column names that is [cols,xvar,yvar]
    # This works for masked arrays
    bin_data = stats.binned_statistic_2d(xvar, yvar, dtemp, statistic=np.ma.mean, bins=[xbin, ybin])
    # Todo Should return vars so you know the order of the returned array
    return bin_data

def make_bin_data(data, passnums, binvars, bincol1, binres1):
    '''Creates poes data binned by pass (defined based on L) and another column such as L

    :param data (structured array or netcdf object such as returned by get_data_raw)
    :param passnums (list) Created by getLpass
    :param binvars (list of variables to bin)
    :param bincol1 (str):           name of first col to bin on
    :param binres1 (list):          list of bin values for bincol1 i.e. np.range(1,10,.5)
    :param dtype (Optional default='raw': type of POES data to get 'raw' or 'processed

    Note: this works for netcdf structures

    '''

    # Now bin it in chunks using whatever col is passed
    # This is kind of a pain because if it's a structure array I don't think you need the [:]
    # But if it is a netcdf object you do.
    bindat = data[bincol1][:]
    passbins = np.arange(np.nanmin(passnums),np.nanmax(passnums),1)
    allbinned= bindata(data, bindat, passnums, binres1, passbins, procvars = binvars )

    return allbinned



def calc_fluence( nc, dt_start, dt_end, channel, noise=1e3, gap_max=3. ):
    """ Calculate the particle fluence in one channel

    :param nc:         NetCDF object (can be multi-day if you used get_data(...).
    :param dt_start:   Starting Datetime.
    :param dt_end:     Ending Datetime.
    :param channel:    Electron or Proton variable from NC object, e.g. 'mep_ele_tel0_flux_e1' for MEPED e- (>40 keV) number flux from 0-degree (nadir) telescope.
    :param noise:      Ignore input flux < 'noise' [Default 1e3].
                       Good starting noise values appear to be:
                            MEPED E1-E4 (0,90) : 1000 #/cm2/s/str
                            MEPED P1-P4 (0,90) :   10 #/cm2/s/str
                            OMNI  P1-P2 (omni) :   50 #/cm2/s/str
                       *However, these are just rough starting points! We should really start from or near to the 1 count level.

    :param gap_max:    Will integrate over gaps of up to 'gap_max' (seconds). [Default 3 seconds == no gap integration]
                       Nominal data cadence is 2-seconds.
                       Value < 4 essentially ensures no integrating across gaps.

    :return:           Fluence in same units as 'channel' sans 'seconds'.
                       NaN if 1) big gap in data or 2) weird time sampling.
    """
    my_name = 'poes_utils:calc_fluence'

    ''' Setup '''
    DELTA_SAMPLE_NOMINAL = 2.0
    time_msec = nc.variables['time'][:]
    n_times   = len( time_msec )
    time_dt = np.array( [dtm.datetime(1970,1,1) + dtm.timedelta( microseconds=1000.*time_msec[i] ) for i in np.arange( n_times ) ] )

    units   = nc.variables[ channel ].units.replace( '-s-', '-' )

    idx_event = np.where( ( time_dt >= dt_start ) & ( time_dt < dt_end ) )[0]

    flux = nc.variables[ channel ][idx_event]

    # null out noise
    flux[ flux < noise ] = 0.

    # Integrate across gaps up to 'gap_max':
    # Case of 1 measurement
    if 1 == len( idx_event ):
        delta_sample = DELTA_SAMPLE_NOMINAL
    else:
        delta_sample = (time_msec[ idx_event[1:] ] - time_msec[ idx_event[0:-1] ])/1000.
        delta_sample = np.append( delta_sample, delta_sample[-1] )

    # Check for weird time sampling:
    if ( (np.max( delta_sample ) > gap_max) or (np.mean(delta_sample) > 1.1*DELTA_SAMPLE_NOMINAL) ):
        logger.info( my_name + ': Max gap (%.1f seconds) or Average exceeds Nominal+10%% (%.1f seconds). Sample time is Avg, Min, Max: %.1f, %.1f, %.1f' %
            (gap_max, DELTA_SAMPLE_NOMINAL, np.mean(delta_sample), np.min(delta_sample), np.max(delta_sample)) )
        return( { 'value': np.nan, 'units': units } )

    # Fluence = sample time * flux
    fluence = np.sum( delta_sample * flux )

    return( { 'value': fluence, 'units': units } )


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                 Graphics                                      "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def plot_stack_ted_raw( dataloc, sat_name, dt_start, dt_end, dpi=600, dir_output=None ):
    '''The purpose of this is to make a stack plot of the raw ted data

    :param dataloc (str)                    A location and name of a configfile or a top directory
                                            The program checks to see if this is a configfile with multiple
                                            POES directories defined. If it is not, then it assumes it is the top
                                            level directory and looks for raw data in all below.
    :param sat_name (string):               One of ['m01', 'm02', 'm03', 'n15', ..., 'n19' ].
    :param dt_start (datetime):             Start datetime.
    :param dt_end (datetime):               End datetime.
    :param dpi (int) (Optional)             dpi for plotting
    :param dir_output (Optional)            directory for output plot

    CHANGES:
    08/2020 JGREEN: Changed this to work with the new get_data'''

    my_name = 'plot_stack_ted_raw'

    #####
    # Setup Data
    #####
    ncraw = get_data_raw( dataloc, sat_name, dt_start, dt_end, clobber=True, vars=None, all=True  )
    #ncraw = get_data_raw( sat_name, dt_range[0], dt_range[1], dir_user_data=dir_user_data )
    data = {}
    data['sat_name'] = satID_to_satname(ncraw['satID'][:].filled()[0])
    assert( sat_name == data['sat_name'] )
    time_msec = ncraw['time'][:].filled()
    data['time_dt'] = nc4.num2date(time_msec, units=ncraw['time'].units)
    #if not dt_range:
    #    dt_range = [data['time_dt'][0], data['time_dt'][-1]]

    # Ephemeris
    data['lat'] = ncraw['lat'][:].filled( fill_value=np.nan )
    data['alt'] = ncraw['alt'][:].filled( fill_value=np.nan )
    data['lon_180'] = lon_360_to_180(ncraw['lon'][:])
    data['sat_direction'] = ncraw['sat_direction'][:].filled( fill_value=99 ).astype( float )
    data['sat_direction'][ data['sat_direction'] == 99 ] = np.nan

    # Electrons
    data['ted_ele_tel0_cps_4' ] = ncraw['ted_ele_tel0_cps_4' ][:].filled( fill_value=np.nan )
    data['ted_ele_tel0_cps_8' ] = ncraw['ted_ele_tel0_cps_8' ][:].filled( fill_value=np.nan )
    data['ted_ele_tel0_cps_11'] = ncraw['ted_ele_tel0_cps_11'][:].filled( fill_value=np.nan )
    data['ted_ele_tel0_cps_14'] = ncraw['ted_ele_tel0_cps_14'][:].filled( fill_value=np.nan )
    data['ted_ele_tel30_cps_4' ] = ncraw['ted_ele_tel30_cps_4' ][:].filled( fill_value=np.nan )
    data['ted_ele_tel30_cps_8' ] = ncraw['ted_ele_tel30_cps_8' ][:].filled( fill_value=np.nan )
    data['ted_ele_tel30_cps_11'] = ncraw['ted_ele_tel30_cps_11'][:].filled( fill_value=np.nan )
    data['ted_ele_tel30_cps_14'] = ncraw['ted_ele_tel30_cps_14'][:].filled( fill_value=np.nan )

    data['ted_ele_tel0_low_eflux_cps'  ] = ncraw['ted_ele_tel0_low_eflux_cps'  ][:].filled( fill_value=np.nan )
    data['ted_ele_tel30_low_eflux_cps' ] = ncraw['ted_ele_tel30_low_eflux_cps' ][:].filled( fill_value=np.nan )
    data['ted_ele_tel0_hi_eflux_cps'  ] = ncraw['ted_ele_tel0_hi_eflux_cps'  ][:].filled( fill_value=np.nan )
    data['ted_ele_tel30_hi_eflux_cps' ] = ncraw['ted_ele_tel30_hi_eflux_cps' ][:].filled( fill_value=np.nan )

    # Protons
    data['ted_pro_tel0_cps_4' ] = ncraw['ted_pro_tel0_cps_4' ][:].filled( fill_value=np.nan )
    data['ted_pro_tel0_cps_8' ] = ncraw['ted_pro_tel0_cps_8' ][:].filled( fill_value=np.nan )
    data['ted_pro_tel0_cps_11'] = ncraw['ted_pro_tel0_cps_11'][:].filled( fill_value=np.nan )
    data['ted_pro_tel0_cps_14'] = ncraw['ted_pro_tel0_cps_14'][:].filled( fill_value=np.nan )
    data['ted_pro_tel30_cps_4' ] = ncraw['ted_pro_tel30_cps_4' ][:].filled( fill_value=np.nan )
    data['ted_pro_tel30_cps_8' ] = ncraw['ted_pro_tel30_cps_8' ][:].filled( fill_value=np.nan )
    data['ted_pro_tel30_cps_11'] = ncraw['ted_pro_tel30_cps_11'][:].filled( fill_value=np.nan )
    data['ted_pro_tel30_cps_14'] = ncraw['ted_pro_tel30_cps_14'][:].filled( fill_value=np.nan )

    data['ted_pro_tel0_low_eflux_cps'  ] = ncraw['ted_pro_tel0_low_eflux_cps'  ][:].filled( fill_value=np.nan )
    data['ted_pro_tel30_low_eflux_cps' ] = ncraw['ted_pro_tel30_low_eflux_cps' ][:].filled( fill_value=np.nan )
    data['ted_pro_tel0_hi_eflux_cps'  ] = ncraw['ted_pro_tel0_hi_eflux_cps'  ][:].filled( fill_value=np.nan )
    data['ted_pro_tel30_hi_eflux_cps' ] = ncraw['ted_pro_tel30_hi_eflux_cps' ][:].filled( fill_value=np.nan )

    # Housekeeping
    data['ted_V'     ] = ncraw['ted_V'     ][:].filled( fill_value=np.nan )
    data['ted_sweepV'] = ncraw['ted_sweepV'][:].filled( fill_value=np.nan )

    data['ted_electron_CEM_V'] = ncraw['ted_electron_CEM_V'][:].filled( fill_value=np.nan )
    data['ted_proton_CEM_V'  ] = ncraw['ted_proton_CEM_V'  ][:].filled( fill_value=np.nan )

    data['TED_temp'] = ncraw['TED_temp'][:].filled( fill_value=np.nan )

    data['ted_ele_PHD_level'] = ncraw['ted_ele_PHD_level'][:].filled( fill_value=-1 ).astype( float )   # int8 => float
    data['ted_ele_PHD_level'][ data['ted_ele_PHD_level'] == -1 ] = np.nan
    data['ted_pro_PHD_level'] = ncraw['ted_pro_PHD_level'][:].filled( fill_value=-1 ).astype( float )   # int8 => float
    data['ted_pro_PHD_level'][ data['ted_pro_PHD_level'] == -1 ] = np.nan

    data['ted_ele_HV_step'] = ncraw['ted_ele_HV_step'][:].filled( fill_value=-1 ).astype( float )   # int8 => float
    data['ted_ele_HV_step'][ data['ted_ele_HV_step'] == -1 ] = np.nan
    data['ted_pro_HV_step'] = ncraw['ted_pro_HV_step'][:].filled( fill_value=-1 ).astype( float )   # int8 => float
    data['ted_pro_HV_step'][ data['ted_pro_HV_step'] == -1 ] = np.nan

    # Flags
    # IFC
    data['flag_ifc'] = ncraw['ted_IFC_on'][:].filled( fill_value=-1 ).astype( float )   # int8 => float
    data['flag_ifc'][data['flag_ifc'] == -1] = np.nan
    # Linearly combine flags
    data['flags_all'] = data['flag_ifc']

    # Close NetCDF
    ncraw.close()

    #####
    # Plot
    #####
    title = '%s TED Raw - %s to %s' % (sat_name, dt_start, dt_end)
    if dir_output:
        dt_range_str = '%d%02d%02d_%02d%02d-%d%02d%02d_%02d%02d' % \
                       (dt_start.year, dt_start.month, dt_start.day, dt_start.hour, dt_start.minute,
                        dt_end.year, dt_end.month, dt_end.day, dt_end.hour, dt_end.minute)
        file_plot = dir_output + '/' + 'poes_%s_%s_ted_raw.png' % (sat_name, dt_range_str)
    else:
        file_plot = None
    dt_range = [dt_start, dt_end]

    _plot_stack_ted_raw( sat_name, data, dt_range, title, dpi, file_plot )


def _plot_stack_ted_raw( sat_name, data, dt_range, title=None, dpi=600, file_plot=None ):
    my_name = '_plot_stack_ted_raw'

    #####
    # Configuration
    #####

    # Figures
    figure_size   = [np.sqrt(2)*210./25.4, 210./25.4]    # ISO A4
    font_legend = 5
    markersize = 1

    #####
    # Setup Figure
    #####
    if not title:
        title = '%s TED Raw - %s to %s' % (sat_name, dt_range[0], dt_range[1])

    fig = plt.figure( num=0, figsize=figure_size )
    gs = mpl.gridspec.GridSpec( 12, 1, height_ratios=[1,1,1,1,1,1,0.6,0.6,0.6,0.6,0.6,0.6] )
    axs_list = []

    # plt.tight_layout()
    fig.subplots_adjust( left=0.125, right=0.9, bottom=0.15, top=0.95, hspace=0.1 )


    """""""""""""""""""""""""""""""""""""""""""""""
    " Plots                                       "
    """""""""""""""""""""""""""""""""""""""""""""""
    logger.info( my_name+': Plotting %s %s to %s...' % (sat_name, str(dt_range[0])[0:19], str(dt_range[-1])[0:19]) )
    i_plot = -1

    # Electron 0-degree (Zenith): Counts
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_ele_tel0_cps_4'  ], 'r.', markersize=markersize, label='tel0_ch4'  )
    ax.plot( data['time_dt'], data['ted_ele_tel0_cps_8'  ], 'g.', markersize=markersize, label='tel0_ch8'  )
    ax.plot( data['time_dt'], data['ted_ele_tel0_cps_11' ], 'b.', markersize=markersize, label='tel0_ch11' )
    ax.plot( data['time_dt'], data['ted_ele_tel0_cps_14' ], 'k.', markersize=markersize, label='tel0_ch14' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e- 0deg' '\n' 'counts' ), ax.set_yscale('log'), ax.set_ylim( [1, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    plt.title( title )

    # Electron 30-degree (off-Zenith): Counts
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_ele_tel30_cps_4'  ], 'r.', markersize=markersize, label='tel30_ch4'  )
    ax.plot( data['time_dt'], data['ted_ele_tel30_cps_8'  ], 'g.', markersize=markersize, label='tel30_ch8'  )
    ax.plot( data['time_dt'], data['ted_ele_tel30_cps_11' ], 'b.', markersize=markersize, label='tel30_ch11' )
    ax.plot( data['time_dt'], data['ted_ele_tel30_cps_14' ], 'k.', markersize=markersize, label='tel30_ch14' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e- 30deg' '\n' 'counts' ), ax.set_yscale('log'), ax.set_ylim( [1, 1e6] )
    axs_list.append( ax )

    # Proton 0-degree (Zenith): Counts
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_pro_tel0_cps_4'  ], 'r.', markersize=markersize, label='tel0_ch4'  )
    ax.plot( data['time_dt'], data['ted_pro_tel0_cps_8'  ], 'g.', markersize=markersize, label='tel0_ch8'  )
    ax.plot( data['time_dt'], data['ted_pro_tel0_cps_11' ], 'b.', markersize=markersize, label='tel0_ch11' )
    ax.plot( data['time_dt'], data['ted_pro_tel0_cps_14' ], 'k.', markersize=markersize, label='tel0_ch14' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+ 0deg' '\n' 'counts' ), ax.set_yscale('log'), ax.set_ylim( [1, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton 30-degree (off-Zenith): Counts
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_pro_tel30_cps_4'  ], 'r.', markersize=markersize, label='tel30_ch4'  )
    ax.plot( data['time_dt'], data['ted_pro_tel30_cps_8'  ], 'g.', markersize=markersize, label='tel30_ch8'  )
    ax.plot( data['time_dt'], data['ted_pro_tel30_cps_11' ], 'b.', markersize=markersize, label='tel30_ch11' )
    ax.plot( data['time_dt'], data['ted_pro_tel30_cps_14' ], 'k.', markersize=markersize, label='tel30_ch14' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+ 30deg' '\n' 'counts' ), ax.set_yscale('log'), ax.set_ylim( [1, 1e6] )
    axs_list.append( ax )

    # Electron Energy Flux 0-degree and 30-degree: Counts/second (according to ProcessData.cpp line ~ 1511)
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_ele_tel0_low_eflux_cps'  ], 'r.', markersize=markersize, label='tel0_low' )
    ax.plot( data['time_dt'], data['ted_ele_tel0_hi_eflux_cps'   ], 'g.', markersize=markersize, label='tel0_hi'  )
    ax.plot( data['time_dt'], data['ted_ele_tel30_low_eflux_cps' ], 'b.', markersize=markersize, label='tel30_low' )
    ax.plot( data['time_dt'], data['ted_ele_tel30_hi_eflux_cps'  ], 'k.', markersize=markersize, label='tel30_hi'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e-' '\n' 'counts/second' ), ax.set_yscale('log'), ax.set_ylim( [1, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton Energy Flux 0-degree and 30-degree: Counts/second (according to ProcessData.cpp line ~ 1511)
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_pro_tel0_low_eflux_cps'  ], 'r.', markersize=markersize, label='tel0_low' )
    ax.plot( data['time_dt'], data['ted_pro_tel0_hi_eflux_cps'   ], 'g.', markersize=markersize, label='tel0_hi'  )
    ax.plot( data['time_dt'], data['ted_pro_tel30_low_eflux_cps' ], 'b.', markersize=markersize, label='tel30_low' )
    ax.plot( data['time_dt'], data['ted_pro_tel30_hi_eflux_cps'  ], 'k.', markersize=markersize, label='tel30_hi'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+' '\n' 'counts/second' ), ax.set_yscale('log'), ax.set_ylim( [1, 1e4] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # House Keeping: Voltages
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_V'],              'r.', markersize=markersize, label='ted_V' )
    ax.plot( data['time_dt'], data['ted_sweepV'],         'g.', markersize=markersize, label='ted_sweepV' )
    ax.plot( data['time_dt'], data['ted_electron_CEM_V'], 'b.', markersize=markersize, label='ted_electron_CEM_V' )
    ax.plot( data['time_dt'], data['ted_proton_CEM_V'],   'k.', markersize=markersize, label='ted_proton_CEM_V' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Voltages' )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # House Keeping: Temperature
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['TED_temp'], 'k.', markersize=markersize, label='TED_temp' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Temp' )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # House Keeping: PHD Level
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_ele_PHD_level'], 'r.', markersize=markersize, label='ted_ele_PHD_level' )
    ax.plot( data['time_dt'], data['ted_pro_PHD_level'], 'g.', markersize=markersize, label='ted_pro_PHD_level' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'PHD Level' )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # House Keeping: HV Step
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['ted_ele_HV_step'], 'r.', markersize=markersize, label='ted_ele_HV_step' )
    ax.plot( data['time_dt'], data['ted_pro_HV_step'], 'g.', markersize=markersize, label='ted_pro_HV_step' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'HV Step' ), ax.set_ylim( [-1,8] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Flags
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['flag_ifc'], 'r.', markersize=markersize, label='IFC'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Flags' ), ax.set_ylim( [-1,5] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Lat, Lon
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( data['time_dt'], data['lat'],     'r-', label='latitude'  )
    ax.plot( data['time_dt'], data['lon_180'], 'g-', label='longitude' )
    ax.set_xlim( dt_range )
    ax.grid()
    ax.set_ylabel( 'Degrees' ), ax.set_ylim([-180,180] )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Alt
    ax = ax.twinx()
    ax.plot( data['time_dt'], data['alt'][:], 'b-', label='altitude' )
    ax.set_ylabel( 'km' ), ax.set_ylim( [700,900] )
    plt.legend( prop={'size':font_legend}, loc='best' )

    ax_ephemeris = ax     # save for later.

    # Tick label spacing
    width_hours = ( dt_range[1] - dt_range[0] ).total_seconds() / 3600.
    if width_hours < 0.5:
        majloc = mpl.dates.MinuteLocator( interval=1  )
        minloc = mpl.dates.SecondLocator( interval=30 )
    elif width_hours <= 1:
        majloc = mpl.dates.MinuteLocator( interval=2 )
        minloc = mpl.dates.MinuteLocator( interval=1 )
    elif width_hours <= 3:
        majloc = mpl.dates.MinuteLocator( interval=5   )
        minloc = mpl.dates.SecondLocator( interval=150 )
    elif width_hours <= 5:
        majloc = mpl.dates.MinuteLocator( interval=30 )
        minloc = mpl.dates.MinuteLocator( interval=15 )
    elif width_hours <= 24:
        majloc = mpl.dates.HourLocator()
        minloc = mpl.dates.MinuteLocator( interval=30 )
    else:
        majloc = mpl.dates.HourLocator( interval=2)
        minloc = mpl.dates.HourLocator()

    ax.xaxis.set_major_locator( majloc )
    ax.xaxis.set_minor_locator( minloc )


    ''' Extra x-axis labels '''
    xlabels = []
    mpl_times = mpl.dates.date2num(data['time_dt'])
    xticks = ax_ephemeris.get_xticks()
    for tick in xticks:
        d_tick = np.abs(mpl_times - tick)
        i_tick = np.argmin(d_tick)
        if (d_tick[i_tick] / (xticks[1] - xticks[
            0])) < 0.01:  # Less than 1% of the distance is a good measure for matching ticks.
            # print( 'DEBUG: X-tick %s matched to Data time %s.' % ( mpl.dates.num2date( tick ), data['time_dt'][ i_tick ] ) )
            tickstr = '%02d:%02d' % (data['time_dt'][i_tick].hour, data['time_dt'][i_tick].minute) + \
                      '\n%.1f' % (data['lat'][:][i_tick]) + \
                      '\n%.1f' % (data['lon_180'][:][i_tick]) + \
                      '\n%.1f' % (data['alt'][:][i_tick]) + \
                      '\n%.1f' % (data['sat_direction'][:][i_tick])

        else:
            logger.warn('WARNING: No X-tick match found for %s.' % mpl.dates.num2date(tick))
            tickstr = '.\n.\n.\n.\n.\n.\n.\n.'  # This is when we have trouble lining up xticks.

        xlabels.append(tickstr)

    ax.set_xticklabels(xlabels)
    ax.annotate('UT\n' 'GLat\n' 'GLon\n' 'Alt\n' 'Sat Dir\n',
                xy=(0.1, 0.1425),
                xycoords='figure fraction', horizontalalignment='right',
                verticalalignment='top', fontsize=mpl.rcParams['xtick.labelsize'])

    # Ensure the x-axes are all lined up and default x-axis labels are hidden!
    for ax in axs_list:
        # All x-axis are lined up
        ax.set_xlim( dt_range )
        # Turn off default x-axis labels. Don't add the "ephemeris" axes to the axs_list[] or you'll erase the markings you want.
        ax.tick_params(labelbottom=False)
        # Grid on
        ax.grid()


    #####
    # Write Figure
    #####
    if file_plot:
        logger.info( my_name + ': Saving plot to %s.' % file_plot )
        fig.savefig( file_plot, dpi=dpi )
        plt.close(fig)
    else:
        plt.show()


def plot_stack_ted_l2( data, dt_range=None, dpi=600, dir_output=None ):
    """PURPOSE: To make a plot of the processed ted data
    This is slightly different than plot_stack_ted_raw because the data is already retrieved and passed

    :param data     a netcdf opbject with the l2 processed data from get_data_processed
    :param dt_range
    :param dpi
    :param dir_output """
    my_name = 'plot_stack_ted_l2'

    #####
    # Configuration
    #####

    # Figures
    figure_size   = [np.sqrt(2)*210./25.4, 210./25.4]    # ISO A4
    font_legend = 5
    markersize = 1

    #####
    # Setup Data
    #####
    sat_name = satID_to_satname( data['satID'][:].filled()[0] )
    time_msec = data['time'][:].filled()
    time_dt = nc4.num2date( time_msec, units=data['time'].units )
    if not dt_range:
        dt_range = [time_dt[0], time_dt[-1]]

    # Ephemeris
    lon_180 = lon_360_to_180(data['lon'][:])

    # Flags
    flag_ifc = data['ted_IFC_on'][:]
    flag_ifc[ flag_ifc == -1 ] = 0

    #####
    # Setup Figure
    #####

    title = '%s TED L2 - %s to %s' % (sat_name, dt_range[0], dt_range[1])

    fig = plt.figure( num=0, figsize=figure_size )
    gs = mpl.gridspec.GridSpec( 9, 1, height_ratios=[0.6,1,1,1,1,1,1,0.6,0.6] )
    axs_list = []

    # plt.tight_layout()
    fig.subplots_adjust( left=0.125, right=0.9, bottom=0.15, top=0.95, hspace=0.1 )


    """""""""""""""""""""""""""""""""""""""""""""""
    " Plots                                       "
    """""""""""""""""""""""""""""""""""""""""""""""
    logger.info( my_name+': Plotting %s %s to %s...' % (sat_name, str(dt_range[0])[0:19], str(dt_range[-1])[0:19]) )
    i_plot = -1

    # Pitch Angle
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_alpha_0_sat'  ][:], 'r-', label='alpha_0_sat'  )
    ax.plot( time_dt, data['ted_alpha_30_sat' ][:], 'g-', label='alpha_30_sat' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Tele P.A.' '\n' '(deg)' ), ax.set_ylim( [0,180] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    plt.title( title )

    # Electron 0-degree (Zenith) and 30-degree Telescopes: Flux
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_ele_tel0_low_eflux' ][:], 'r.', markersize=markersize, label='tel0_low'  )
    ax.plot( time_dt, data['ted_ele_tel0_hi_eflux'  ][:], 'g.', markersize=markersize, label='tel0_hi'   )
    ax.plot( time_dt, data['ted_ele_tel30_low_eflux'][:], 'b.', markersize=markersize, label='tel30_low' )
    ax.plot( time_dt, data['ted_ele_tel30_hi_eflux' ][:], 'k.', markersize=markersize, label='tel30_hi'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e-' '\n' r'mW/$m^2$sr' ), ax.set_yscale('log'), ax.set_ylim( [1e-6, 10] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Electron 0-degree (Zenith) and 30-degree Telescopes: Backgrounds
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_ele_eflux_bg_tel0_low' ][:], 'r.', markersize=markersize, label='tel0_low_bkg'  )
    ax.plot( time_dt, data['ted_ele_eflux_bg_tel0_hi'  ][:], 'g.', markersize=markersize, label='tel0_hi_bkg'   )
    ax.plot( time_dt, data['ted_ele_eflux_bg_tel30_low'][:], 'b.', markersize=markersize, label='tel30_low_bkg' )
    ax.plot( time_dt, data['ted_ele_eflux_bg_tel30_hi' ][:], 'k.', markersize=markersize, label='tel30_hi_bkg'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e- Background' '\n' r'mW/$m^2$sr' ), ax.set_yscale('log'), ax.set_ylim( [1e-6, 10] )
    axs_list.append( ax )

    # Proton 0-degree (Zenith) and 30-degree Telescopes: Flux
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_pro_tel0_low_eflux' ][:], 'r.', markersize=markersize, label='tel0_low'  )
    ax.plot( time_dt, data['ted_pro_tel0_hi_eflux'  ][:], 'g.', markersize=markersize, label='tel0_hi'   )
    ax.plot( time_dt, data['ted_pro_tel30_low_eflux'][:], 'b.', markersize=markersize, label='tel30_low' )
    ax.plot( time_dt, data['ted_pro_tel30_hi_eflux' ][:], 'k.', markersize=markersize, label='tel30_hi'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+' '\n' r'mW/$m^2$sr' ), ax.set_yscale('log'), ax.set_ylim( [1e-6, 10] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton 0-degree (Zenith) and 30-degree Telescopes: Backgrounds
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_pro_eflux_bg_tel0_low' ][:], 'r.', markersize=markersize, label='tel0_low_bkg'  )
    ax.plot( time_dt, data['ted_pro_eflux_bg_tel0_hi'  ][:], 'g.', markersize=markersize, label='tel0_hi_bkg'   )
    ax.plot( time_dt, data['ted_pro_eflux_bg_tel30_low'][:], 'b.', markersize=markersize, label='tel30_low_bkg' )
    ax.plot( time_dt, data['ted_pro_eflux_bg_tel30_hi' ][:], 'k.', markersize=markersize, label='tel30_hi_bkg'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+ Background' '\n' r'mW/$m^2$sr' ), ax.set_yscale('log'), ax.set_ylim( [1e-6, 10] )
    axs_list.append( ax )

    # Electron Energy Flux Atmosphere
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_ele_eflux_atmo_low'  ][:], 'r.', markersize=markersize, label='atmo_low' )
    ax.plot( time_dt, data['ted_ele_eflux_atmo_hi'   ][:], 'g.', markersize=markersize, label='atmo_hi'  )
    ax.plot( time_dt, data['ted_ele_eflux_atmo_total'][:], 'b.', markersize=markersize, label='total'        )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e- Atmos.' '\n' r'mW/$m^2$' ), ax.set_yscale('log'), ax.set_ylim( [1e-6, 10] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton Energy Flux Atmosphere
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['ted_pro_eflux_atmo_low'  ][:], 'r.', markersize=markersize, label='atmo_low' )
    ax.plot( time_dt, data['ted_pro_eflux_atmo_hi'   ][:], 'g.', markersize=markersize, label='atmo_hi'  )
    ax.plot( time_dt, data['ted_pro_eflux_atmo_total'][:], 'b.', markersize=markersize, label='total'        )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+ Atmos.' '\n' r'mW/$m^2$' ), ax.set_yscale('log'), ax.set_ylim( [1e-6, 10] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Flags
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, flag_ifc, 'r-', markersize=markersize, label='IFC'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Flags' ), ax.set_ylim( [-2,5] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Lat, Lon
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['lat'][:], 'r-', label='latitude'  )
    ax.plot( time_dt, lon_180,        'g-', label='longitude' )
    ax.set_xlim( dt_range )
    ax.grid()
    ax.set_ylabel( 'Degrees' ), ax.set_ylim([-180,180] )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Alt
    ax = ax.twinx()
    ax.plot( time_dt, data['alt'][:], 'b-', label='altitude' )
    ax.set_ylabel( 'km' ), ax.set_ylim( [700,900] )
    plt.legend( prop={'size':font_legend}, loc='best' )

    ax_ephemeris = ax     # save for later.

    # Tick label spacing
    width_hours = ( dt_range[1] - dt_range[0] ).total_seconds() / 3600.
    if width_hours < 0.5:
        majloc = mpl.dates.MinuteLocator( interval=1  )
        minloc = mpl.dates.SecondLocator( interval=30 )
    elif width_hours <= 1:
        majloc = mpl.dates.MinuteLocator( interval=2 )
        minloc = mpl.dates.MinuteLocator( interval=1 )
    elif width_hours <= 3:
        majloc = mpl.dates.MinuteLocator( interval=5   )
        minloc = mpl.dates.SecondLocator( interval=150 )
    elif width_hours <= 5:
        majloc = mpl.dates.MinuteLocator( interval=30 )
        minloc = mpl.dates.MinuteLocator( interval=15 )
    else:
        majloc = mpl.dates.HourLocator()
        minloc = mpl.dates.MinuteLocator( interval=30 )

    ax.xaxis.set_major_locator( majloc )
    ax.xaxis.set_minor_locator( minloc )


    ''' Extra x-axis labels '''
    xlabels = []
    mpl_times = mpl.dates.date2num( time_dt )
    xticks = ax_ephemeris.get_xticks()
    for tick in xticks:
        d_tick = np.abs( mpl_times - tick )
        i_tick = np.argmin( d_tick )
        if (d_tick[i_tick] / (xticks[1] - xticks[0])) < 0.01:     # Less than 1% of the distance is a good measure for matching ticks.
            # print( 'DEBUG: X-tick %s matched to Data time %s.' % ( mpl.dates.num2date( tick ), time_dt[ i_tick ] ) )
            tickstr = '%02d:%02d' % ( time_dt[ i_tick ].hour, time_dt[ i_tick ].minute ) + \
                '\n%.1f' % ( data['lat'][:][i_tick] ) + \
                '\n%.1f' % ( data['lon'][:][i_tick] ) + \
                '\n%.1f' % ( data['alt'][:][i_tick] ) + \
                '\n%.1f' % ( data['mag_lat_sat'][:][i_tick] ) + \
                '\n%.1f' % ( data['mag_lon_sat'][:][i_tick] ) + \
                '\n%.1f' % ( data['MLT'][:][i_tick] ) + \
                '\n%.1f' % ( data['L_IGRF'][:][i_tick] ) + \
                '\n%.1f' % ( data['meped_alpha_0_sat' ][:][i_tick] ) + \
                '\n%.1f' % ( data['meped_alpha_90_sat'][:][i_tick] )

        else:
            logger.warn( 'WARNING: No X-tick match found for %s.' % mpl.dates.num2date( tick ) )
            tickstr = '.\n.\n.\n.\n.\n.\n.\n.'     # This is when we have trouble lining up xticks.

        xlabels.append( tickstr )

    ax.set_xticklabels( xlabels )
    ax.annotate( 'UT\n' 'GLat\n' 'GLon\n' 'Alt\n' 'MLat\n' 'MLon\n' 'MLT\n' 'L_IGRF\n' 'alpha(0)\n' 'alpha(90)\n',  xy=(0.1,0.1425),
        xycoords='figure fraction', horizontalalignment='right',
        verticalalignment='top',  fontsize=mpl.rcParams['xtick.labelsize'] )

    # Ensure the x-axes are all lined up and default x-axis labels are hidden!
    for ax in axs_list:
        # All x-axis are lined up
        ax.set_xlim( dt_range )
        # Turn off default x-axis labels. Don't add the "ephemeris" axes to this list or you'll erase the markings you want.
        ax.tick_params(labelbottom=False)
        # Grid on
        ax.grid()

    #####
    # Write Figure
    #####
    if dir_output:
        dt_range_str = '%d%02d%02d_%02d%02d-%d%02d%02d_%02d%02d' % \
            (dt_range[0].year, dt_range[0].month, dt_range[0].day, dt_range[0].hour, dt_range[0].minute,
             dt_range[1].year, dt_range[1].month, dt_range[1].day, dt_range[1].hour, dt_range[1].minute)
        file_plot = dir_output + '/' + 'poes_%s_%s_ted_l2.png' % (sat_name, dt_range_str)
        if not os.path.exists( dir_output ): os.mkdir( dir_output )
        logger.info( my_name+': Saving plot to %s.' % file_plot )
        fig.savefig( file_plot, dpi=dpi )
        plt.close( fig )
    else:
        plt.show()



def plot_stack_meped_raw(dataloc, sat_name, dt_start, dt_end, dir_output=None, dpi=600 ):

    """ Plots aggregated POES ngdc format processed data between dt_start and dt_end
    :param dataloc (str)                    A location and name of a either a configfile or a top directory
                                            The program checks to see if this is a configfile with multiple
                                            POES directories defined. If it is not, then it assumes it is the top
                                            level directory and looks for data in all below.
    :param sat_name (string):               One of ['m01', 'm02', 'm03', 'n15', ..., 'n19' ].
    :param dt_start (datetime):             Start datetime with year, month, day
    :param dt_end (datetime):               End datetime.
    :param dir_output
    :param dpi"""
    my_name = 'plot_stack_meped_raw'

    #####
    # Setup Data
    #####
    ncraw = get_data_raw(dataloc, sat_name, dt_start, dt_end, clobber=True, vars=None, all=True)
    #ncraw = get_data_raw( sat_name, dt_range[0], dt_range[1], dir_user_data=dir_user_data )
    data = {}
    data['sat_name'] = satID_to_satname(ncraw['satID'][:].filled()[0])
    assert( sat_name == data['sat_name'] )
    time_msec = ncraw['time'][:].filled()
    data['time_dt'] = nc4.num2date(time_msec, units=ncraw['time'].units)
    dtrange= [dt_start,dt_end]
    #if not dt_range:
    #    dt_range = [data['time_dt'][0], data['time_dt'][-1]]

    # Ephemeris
    data['lat'] = ncraw['lat'][:].filled( fill_value=np.nan )
    data['alt'] = ncraw['alt'][:].filled( fill_value=np.nan )
    data['lon_180'] = lon_360_to_180(ncraw['lon'][:])
    data['sat_direction'] = ncraw['sat_direction'][:].filled( fill_value=99 ).astype( float )
    data['sat_direction'][ data['sat_direction'] == 99 ] = np.nan

    # Protons
    data['mep_pro_tel0_cps_p1'] = ncraw['mep_pro_tel0_cps_p1'][:].filled( fill_value=np.nan )
    data['mep_pro_tel0_cps_p2'] = ncraw['mep_pro_tel0_cps_p2'][:].filled( fill_value=np.nan )
    data['mep_pro_tel0_cps_p3'] = ncraw['mep_pro_tel0_cps_p3'][:].filled( fill_value=np.nan )
    data['mep_pro_tel0_cps_p4'] = ncraw['mep_pro_tel0_cps_p4'][:].filled( fill_value=np.nan )
    data['mep_pro_tel0_cps_p5'] = ncraw['mep_pro_tel0_cps_p5'][:].filled( fill_value=np.nan )
    data['mep_pro_tel0_cps_p6'] = ncraw['mep_pro_tel0_cps_p6'][:].filled( fill_value=np.nan )
    data['mep_pro_tel90_cps_p1'] = ncraw['mep_pro_tel90_cps_p1'][:].filled( fill_value=np.nan )
    data['mep_pro_tel90_cps_p2'] = ncraw['mep_pro_tel90_cps_p2'][:].filled( fill_value=np.nan )
    data['mep_pro_tel90_cps_p3'] = ncraw['mep_pro_tel90_cps_p3'][:].filled( fill_value=np.nan )
    data['mep_pro_tel90_cps_p4'] = ncraw['mep_pro_tel90_cps_p4'][:].filled( fill_value=np.nan )
    data['mep_pro_tel90_cps_p5'] = ncraw['mep_pro_tel90_cps_p5'][:].filled( fill_value=np.nan )
    data['mep_pro_tel90_cps_p6'] = ncraw['mep_pro_tel90_cps_p6'][:].filled( fill_value=np.nan )

    # Electrons
    data['mep_ele_tel0_cps_e1'] = ncraw['mep_ele_tel0_cps_e1'][:].filled( fill_value=np.nan )
    data['mep_ele_tel0_cps_e2'] = ncraw['mep_ele_tel0_cps_e2'][:].filled( fill_value=np.nan )
    data['mep_ele_tel0_cps_e3'] = ncraw['mep_ele_tel0_cps_e3'][:].filled( fill_value=np.nan )
    data['mep_ele_tel90_cps_e1'] = ncraw['mep_ele_tel90_cps_e1'][:].filled( fill_value=np.nan )
    data['mep_ele_tel90_cps_e2'] = ncraw['mep_ele_tel90_cps_e2'][:].filled( fill_value=np.nan )
    data['mep_ele_tel90_cps_e3'] = ncraw['mep_ele_tel90_cps_e3'][:].filled( fill_value=np.nan )

    #  omni's
    data['mep_omni_cps_p6'] = ncraw['mep_omni_cps_p6'][:].filled( fill_value=np.nan )
    data['mep_omni_cps_p7'] = ncraw['mep_omni_cps_p7'][:].filled( fill_value=np.nan )
    data['mep_omni_cps_p8'] = ncraw['mep_omni_cps_p8'][:].filled( fill_value=np.nan )
    data['mep_omni_cps_p9'] = ncraw['mep_omni_cps_p9'][:].filled( fill_value=np.nan )

    # Housekeeping
    data['MEPED_V'] = ncraw['MEPED_V'][:].filled( fill_value=np.nan )
    data['mep_omni_biase_V'] = ncraw['mep_omni_biase_V'][:].filled( fill_value=np.nan )
    data['mep_circuit_temp'] = ncraw['mep_circuit_temp'][:].filled( fill_value=np.nan )
    data['mep_proton_tel_temp'] = ncraw['mep_proton_tel_temp'][:].filled( fill_value=np.nan )

    # Flags:
    # IFC
    data['flag_ifc'] = ncraw['mep_IFC_on'][:].filled( fill_value=-1 ).astype( float )   # int8 => float
    data['flag_ifc'][data['flag_ifc'] == -1] = np.nan
    # Linearly combine flags
    data['flags_all'] = data['flag_ifc']

    # Close NetCDF
    ncraw.close()

    #####
    # Plot
    #####
    title = '%s MEPED Raw - %s to %s' % (sat_name, dt_range[0], dt_range[1])
    if dir_output:
        dt_range_str = '%d%02d%02d_%02d%02d-%d%02d%02d_%02d%02d' % \
                       (dt_range[0].year, dt_range[0].month, dt_range[0].day, dt_range[0].hour, dt_range[0].minute,
                        dt_range[1].year, dt_range[1].month, dt_range[1].day, dt_range[1].hour, dt_range[1].minute)
        file_plot = dir_output + '/' + 'poes_%s_%s_meped_raw.png' % (sat_name, dt_range_str)
    else:
        file_plot = None

    _plot_stack_meped_raw( sat_name, data, dt_range, title, dpi, file_plot )


def _plot_stack_meped_raw( sat_name, data, dt_range, title=None, dpi=600, file_plot=None ):
    my_name = '_plot_stack_meped_raw'

    #####
    # Configuration
    #####

    # Figures
    figure_size = [np.sqrt(2) * 210. / 25.4, 210. / 25.4]  # ISO A4
    font_legend = 5
    markersize = 1

    #####
    # Setup Figure
    #####
    if not title:
        title = '%s MEPED Raw - %s to %s' % (sat_name, dt_range[0], dt_range[1])

    fig = plt.figure(num=0, figsize=figure_size)
    gs = mpl.gridspec.GridSpec(10, 1, height_ratios=[1, 1, 1, 1, 1, 1, 0.6, 0.6, 0.6, 0.6])
    axs_list = []

    # plt.tight_layout()
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.15, top=0.95, hspace=0.1)

    """""""""""""""""""""""""""""""""""""""""""""""
    " Plots                                       "
    """""""""""""""""""""""""""""""""""""""""""""""
    logger.info(my_name + ': Plotting %s %s to %s...' % (sat_name, str(dt_range[0])[0:19], str(dt_range[-1])[0:19]))
    i_plot = -1

    # Electron 0-degree (Zenith) Telescope: E1 - E3
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_ele_tel0_cps_e1'][:], 'r.', markersize=markersize, label='tel0_e1')
    ax.plot(data['time_dt'], data['mep_ele_tel0_cps_e2'][:], 'g.', markersize=markersize, label='tel0_e2')
    ax.plot(data['time_dt'], data['mep_ele_tel0_cps_e3'][:], 'b.', markersize=markersize, label='tel0_e3')
    ax.set_xlim(dt_range)
    ax.set_ylabel('e- 0deg' '\n' 'counts/sec'), ax.set_yscale('log'), ax.set_ylim([0.1, 1e4])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    plt.title(title)

    # Electron 90-degree (Wake) Telescope: E1 - E3
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_ele_tel90_cps_e1'][:], 'r.', markersize=markersize, label='tel90_e1')
    ax.plot(data['time_dt'], data['mep_ele_tel90_cps_e2'][:], 'g.', markersize=markersize, label='tel90_e2')
    ax.plot(data['time_dt'], data['mep_ele_tel90_cps_e3'][:], 'b.', markersize=markersize, label='tel90_e3')
    ax.set_xlim(dt_range)
    ax.set_ylabel('e- 90deg' '\n' 'counts/second'), ax.set_yscale('log'), ax.set_ylim([0.1, 1e4])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Proton 0-degree (Zenith) Telescope: P1 - P5
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p1'], 'r.', markersize=markersize, label='tel0_p1')
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p2'], 'g.', markersize=markersize, label='tel0_p2')
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p3'], 'b.', markersize=markersize, label='tel0_p3')
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p4'], 'k.', markersize=markersize, label='tel0_p4')
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p5'], '.', markersize=markersize, color='cyan',   label='tel0_p5')
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p6'], '.', markersize=markersize, color='orange', label='tel0_p6')
    ax.set_xlim(dt_range)
    ax.set_ylabel('p+ 0deg' '\n' 'counts/second'), ax.set_yscale('log'), ax.set_ylim([0.1, 1e4])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Proton 90-degree (Wake) Telescope: P1 - P5
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p1'], 'r.', markersize=markersize, label='tel90_p1')
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p2'], 'g.', markersize=markersize, label='tel90_p2')
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p3'], 'b.', markersize=markersize, label='tel90_p3')
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p4'], 'k.', markersize=markersize, label='tel90_p4')
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p5'], '.', markersize=markersize, color='cyan',   label='tel90_p5')
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p6'], '.', markersize=markersize, color='orange', label='tel90_p6')
    ax.set_xlim(dt_range)
    ax.set_ylabel('p+ 90deg' '\n' 'counts/second'), ax.set_yscale('log'), ax.set_ylim([0.1, 1e4])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Proton 0,90-degree (Zenith and Wake) Telescopes: P6
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_pro_tel0_cps_p6'],  'r.', markersize=markersize, label='tel0_p6')
    ax.plot(data['time_dt'], data['mep_pro_tel90_cps_p6'], 'g.', markersize=markersize, label='tel90_p6')
    ax.set_xlim(dt_range)
    ax.set_ylabel('p+' '\n' 'counts/second'), ax.set_yscale('log'), ax.set_ylim([0.1, 1e4])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Omni all 4 detectors: P6, P7, P8, P9
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_omni_cps_p6'], 'r.', markersize=markersize, label='omni_p6')
    ax.plot(data['time_dt'], data['mep_omni_cps_p7'], 'g.', markersize=markersize, label='omni_p7')
    ax.plot(data['time_dt'], data['mep_omni_cps_p8'], 'b.', markersize=markersize, label='omni_p8')
    ax.plot(data['time_dt'], data['mep_omni_cps_p9'], 'k.', markersize=markersize, label='omni_p9')
    ax.set_xlim(dt_range)
    ax.set_ylabel('omni' '\n' 'counts/second'), ax.set_yscale('log'), ax.set_ylim([0.1, 1e4])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Housekeeping: Voltages
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['MEPED_V'],             'r.', markersize=markersize, label='MEPED_V')
    ax.plot(data['time_dt'], data['mep_omni_biase_V'],    'g.', markersize=markersize, label='mep_omni_biase_V')
    ax.set_xlim(dt_range)
    ax.set_ylabel('House' '\n' 'Keeping')
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Housekeeping: Temperatures
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['mep_circuit_temp'],    'b.', markersize=markersize, label='mep_circuit_temp')
    ax.plot(data['time_dt'], data['mep_proton_tel_temp'], 'k.', markersize=markersize, label='mep_proton_tel_temp')
    ax.set_xlim(dt_range)
    ax.set_ylabel('House' '\n' 'Keeping')
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Flags
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['flag_ifc'], 'r.', markersize=markersize, label='IFC')
    ax.set_xlim(dt_range)
    ax.set_ylabel('Flags'), ax.set_ylim([-1, 5])
    axs_list.append(ax)
    plt.legend(prop={'size': font_legend}, loc='best')

    # Lat, Lon
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot(data['time_dt'], data['lat'], 'r-', label='latitude')
    ax.plot(data['time_dt'], data['lon_180'], 'g-', label='longitude')
    ax.set_xlim(dt_range)
    ax.grid()
    ax.set_ylabel('Degrees'), ax.set_ylim([-180, 180])
    plt.legend(prop={'size': font_legend}, loc='best')

    # Alt
    ax = ax.twinx()
    ax.plot(data['time_dt'], data['alt'][:], 'b-', label='altitude')
    ax.set_ylabel('km'), ax.set_ylim([700, 900])
    plt.legend(prop={'size': font_legend}, loc='best')

    ax_ephemeris = ax  # save for later.

    # Tick label spacing
    width_hours = (dt_range[1] - dt_range[0]).total_seconds() / 3600.
    if width_hours < 0.5:
        majloc = mpl.dates.MinuteLocator(interval=1)
        minloc = mpl.dates.SecondLocator(interval=30)
    elif width_hours <= 1:
        majloc = mpl.dates.MinuteLocator(interval=2)
        minloc = mpl.dates.MinuteLocator(interval=1)
    elif width_hours <= 3:
        majloc = mpl.dates.MinuteLocator(interval=5)
        minloc = mpl.dates.SecondLocator(interval=150)
    elif width_hours <= 5:
        majloc = mpl.dates.MinuteLocator(interval=30)
        minloc = mpl.dates.MinuteLocator(interval=15)
    else:
        majloc = mpl.dates.HourLocator()
        minloc = mpl.dates.MinuteLocator(interval=30)

    ax.xaxis.set_major_locator(majloc)
    ax.xaxis.set_minor_locator(minloc)

    ''' Extra x-axis labels '''
    xlabels = []
    mpl_times = mpl.dates.date2num(data['time_dt'])
    xticks = ax_ephemeris.get_xticks()
    for tick in xticks:
        d_tick = np.abs(mpl_times - tick)
        i_tick = np.argmin(d_tick)
        if (d_tick[i_tick] / (xticks[1] - xticks[
            0])) < 0.01:  # Less than 1% of the distance is a good measure for matching ticks.
            # print( 'DEBUG: X-tick %s matched to Data time %s.' % ( mpl.dates.num2date( tick ), data['time_dt'][ i_tick ] ) )
            tickstr = '%02d:%02d' % (data['time_dt'][i_tick].hour, data['time_dt'][i_tick].minute) + \
                      '\n%.1f' % (data['lat'][:][i_tick]) + \
                      '\n%.1f' % (data['lon_180'][:][i_tick]) + \
                      '\n%.1f' % (data['alt'][:][i_tick]) + \
                      '\n%.1f' % (data['sat_direction'][:][i_tick])

        else:
            logger.warn('WARNING: No X-tick match found for %s.' % mpl.dates.num2date(tick))
            tickstr = '.\n.\n.\n.\n.\n.\n.\n.'  # This is when we have trouble lining up xticks.

        xlabels.append(tickstr)

    ax.set_xticklabels(xlabels)
    ax.annotate('UT\n' 'GLat\n' 'GLon\n' 'Alt\n' 'Sat Dir\n',
                xy=(0.1, 0.1425),
                xycoords='figure fraction', horizontalalignment='right',
                verticalalignment='top', fontsize=mpl.rcParams['xtick.labelsize'])

    # Ensure the x-axes are all lined up and default x-axis labels are hidden!
    for ax in axs_list:
        # All x-axis are lined up
        ax.set_xlim( dt_range )
        # Turn off default x-axis labels. Don't add the "ephemeris" axes to this list or you'll erase the markings you want.
        ax.tick_params(labelbottom=False)
        # Grid on
        ax.grid()

    #####
    # Write Figure
    #####
    if file_plot:
        logger.info( my_name + ': Saving plot to %s.' % file_plot )
        fig.savefig( file_plot, dpi=dpi )
        plt.close(fig)
    else:
        plt.show()


def plot_stack_meped_l2( data, dt_range=None, dpi=600, dir_output=None ):
    my_name = 'plot_stack_meped_l2'

    #####
    # Configuration
    #####

    # Figures
    figure_size = [np.sqrt(2)*210./25.4, 210./25.4]    # ISO A4
    font_legend = 5
    markersize = 1

    #####
    # Setup Data
    #####
    sat_name = satID_to_satname( data['satID'][:].filled()[0] )
    time_msec = data['time'][:].filled()
    time_dt = nc4.num2date( time_msec, units=data['time'].units )
    if not dt_range:
        dt_range = [time_dt[0], time_dt[-1]]

    # Ephemeris
    lon_180 = lon_360_to_180(data['lon'][:])

    # Flags
    flag_ifc = data['mep_IFC_on'][:]
    flag_ifc[ flag_ifc == -1 ] = 0
    flag_omni_fit  = data['mep_omni_flux_flag_fit'][:]
    flag_omni_iter = data['mep_omni_flux_flag_iter_lim'][:]

    flags_all = flag_ifc + flag_omni_fit + flag_omni_iter

    #####
    # Setup Figure
    #####

    title = '%s MEPED L2 - %s to %s' % (sat_name, dt_range[0], dt_range[1])

    fig = plt.figure( num=0, figsize=figure_size )
    gs = mpl.gridspec.GridSpec( 9, 1, height_ratios=[0.6,1,1,1,1,1,1,0.6,0.6] )
    axs_list = []

    # plt.tight_layout()
    fig.subplots_adjust( left=0.125, right=0.9, bottom=0.15, top=0.95, hspace=0.1 )


    """""""""""""""""""""""""""""""""""""""""""""""
    " Plots                                       "
    """""""""""""""""""""""""""""""""""""""""""""""
    logger.info( my_name+': Plotting %s %s to %s...' % (sat_name, str(dt_range[0])[0:19], str(dt_range[-1])[0:19]) )
    i_plot = -1

    # Pitch Angle
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['meped_alpha_0_sat'  ][:], 'r-', label='alpha_0_sat'  )
    ax.plot( time_dt, data['meped_alpha_90_sat' ][:], 'g-', label='alpha_90_sat' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Tele P.A.' '\n' '(deg)' ), ax.set_ylim( [0,180] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    plt.title( title )

    # Electron 0-degree (Zenith) Telescope: E1 - E4
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['mep_ele_tel0_flux_e1'][:], 'r.', markersize=markersize, label='tel0_e1' )
    ax.plot( time_dt, data['mep_ele_tel0_flux_e2'][:], 'g.', markersize=markersize, label='tel0_e2' )
    ax.plot( time_dt, data['mep_ele_tel0_flux_e3'][:], 'b.', markersize=markersize, label='tel0_e3' )
    ax.plot( time_dt, data['mep_ele_tel0_flux_e4'][:], 'k.', markersize=markersize, label='tel0_e4' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e- 0deg' '\n' r'#/$cm^2$sr-s-keV' ), ax.set_yscale('log'), ax.set_ylim( [1e2, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Electron 90-degree (Wake) Telescope: E1 - E4
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['mep_ele_tel90_flux_e1'][:], 'r.', markersize=markersize, label='tel90_e1' )
    ax.plot( time_dt, data['mep_ele_tel90_flux_e2'][:], 'g.', markersize=markersize, label='tel90_e2' )
    ax.plot( time_dt, data['mep_ele_tel90_flux_e3'][:], 'b.', markersize=markersize, label='tel90_e3' )
    ax.plot( time_dt, data['mep_ele_tel90_flux_e4'][:], 'k.', markersize=markersize, label='tel90_e4' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'e- 90deg' '\n' r'#/$cm^2$sr-s-keV' ), ax.set_yscale('log'), ax.set_ylim( [1e2, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton 0-degree (Zenith) Telescope: P1 - P5
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['mep_pro_tel0_flux_p1'][:], 'r.', markersize=markersize, label='tel0_p1' )
    ax.plot( time_dt, data['mep_pro_tel0_flux_p2'][:], 'g.', markersize=markersize, label='tel0_p2' )
    ax.plot( time_dt, data['mep_pro_tel0_flux_p3'][:], 'b.', markersize=markersize, label='tel0_p3' )
    ax.plot( time_dt, data['mep_pro_tel0_flux_p4'][:], 'k.', markersize=markersize, label='tel0_p4' )
    ax.plot( time_dt, data['mep_pro_tel0_flux_p5'][:], '.',  markersize=markersize, color='cyan', label='tel0_p5'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+ 0deg' '\n' r'#/$cm^2$sr-s-keV' ), ax.set_yscale('log'), ax.set_ylim( [1e-1, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton 90-degree (Wake) Telescope: P1 - P5
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['mep_pro_tel90_flux_p1'][:], 'r.', markersize=markersize, label='tel90_p1' )
    ax.plot( time_dt, data['mep_pro_tel90_flux_p2'][:], 'g.', markersize=markersize, label='tel90_p2' )
    ax.plot( time_dt, data['mep_pro_tel90_flux_p3'][:], 'b.', markersize=markersize, label='tel90_p3' )
    ax.plot( time_dt, data['mep_pro_tel90_flux_p4'][:], 'k.', markersize=markersize, label='tel90_p4' )
    ax.plot( time_dt, data['mep_pro_tel90_flux_p5'][:], '.',  markersize=markersize, color='cyan', label='tel90_p5'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+ 90deg' '\n' r'#/$cm^2$sr-s-keV' ), ax.set_yscale('log'), ax.set_ylim( [1e-1, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Proton 0,90-degree (Zenith and Wake) Telescopes: P6
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['mep_pro_tel0_flux_p6'][:],   'r.', markersize=markersize, label='tel0_p6'  )
    ax.plot( time_dt, data['mep_pro_tel90_flux_p6' ][:], 'g.', markersize=markersize, label='tel90_p6' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'p+' '\n' r'#/$cm^2$sr-s-keV' ), ax.set_yscale('log'), ax.set_ylim( [1e2, 1e6] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Omni spectral fits at 3 energies: P1, P2, P3
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['mep_omni_flux_p1'][:], 'r.', markersize=markersize, label='omni_p1 25MeV'  )
    ax.plot( time_dt, data['mep_omni_flux_p2'][:], 'g.', markersize=markersize, label='omni_p2 50MeV'  )
    ax.plot( time_dt, data['mep_omni_flux_p3'][:], 'b.', markersize=markersize, label='omni_p3 100MeV' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'omni' '\n' r'p+/$cm^2$sr-s-MeV' ), ax.set_yscale('log'), ax.set_ylim( [1e-3, 1e4] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Flags
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, flag_omni_fit,  'g.', markersize=markersize, label='Omni Fit'  )
    ax.plot( time_dt, flag_omni_iter, 'b.', markersize=markersize, label='Omni Iter'  )
    ax.plot( time_dt, flag_ifc,       'r.', markersize=markersize, label='IFC'  )
    ax.plot( time_dt, flags_all,      'k.', markersize=markersize, label='Flags Summed'  )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Flags' ), ax.set_ylim( [-2,5] )
    axs_list.append( ax )
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Lat, Lon
    i_plot += 1
    ax = fig.add_subplot(gs[i_plot])
    ax.plot( time_dt, data['lat'][:], 'r-', label='latitude'  )
    ax.plot( time_dt, lon_180,        'g-', label='longitude' )
    ax.set_xlim( dt_range )
    ax.set_ylabel( 'Degrees' ), ax.set_ylim([-180,180] )
    ax.grid()
    plt.legend( prop={'size':font_legend}, loc='best' )

    # Alt
    ax = ax.twinx()
    ax.plot( time_dt, data['alt'][:], 'b-', label='altitude' )
    ax.set_ylabel( 'km' ), ax.set_ylim( [700,900] )
    plt.legend( prop={'size':font_legend}, loc='best' )

    ax_ephemeris = ax     # save for later.

    # Tick label spacing
    width_hours = ( dt_range[1] - dt_range[0] ).total_seconds() / 3600.
    if width_hours < 0.5:
        majloc = mpl.dates.MinuteLocator( interval=1  )
        minloc = mpl.dates.SecondLocator( interval=30 )
    elif width_hours <= 1:
        majloc = mpl.dates.MinuteLocator( interval=2 )
        minloc = mpl.dates.MinuteLocator( interval=1 )
    elif width_hours <= 3:
        majloc = mpl.dates.MinuteLocator( interval=5   )
        minloc = mpl.dates.SecondLocator( interval=150 )
    elif width_hours <= 5:
        majloc = mpl.dates.MinuteLocator( interval=30 )
        minloc = mpl.dates.MinuteLocator( interval=15 )
    else:
        majloc = mpl.dates.HourLocator()
        minloc = mpl.dates.MinuteLocator( interval=30 )

    ax.xaxis.set_major_locator( majloc )
    ax.xaxis.set_minor_locator( minloc )


    ''' Extra x-axis labels '''
    xlabels = []
    mpl_times = mpl.dates.date2num( time_dt )
    xticks = ax_ephemeris.get_xticks()
    for tick in xticks:
        d_tick = np.abs( mpl_times - tick )
        i_tick = np.argmin( d_tick )
        if (d_tick[i_tick] / (xticks[1] - xticks[0])) < 0.01:     # Less than 1% of the distance is a good measure for matching ticks.
            # print( 'DEBUG: X-tick %s matched to Data time %s.' % ( mpl.dates.num2date( tick ), time_dt[ i_tick ] ) )
            tickstr = '%02d:%02d' % ( time_dt[ i_tick ].hour, time_dt[ i_tick ].minute ) + \
                '\n%.1f' % ( data['lat'][:][i_tick] ) + \
                '\n%.1f' % ( data['lon'][:][i_tick] ) + \
                '\n%.1f' % ( data['alt'][:][i_tick] ) + \
                '\n%.1f' % ( data['mag_lat_sat'][:][i_tick] ) + \
                '\n%.1f' % ( data['mag_lon_sat'][:][i_tick] ) + \
                '\n%.1f' % ( data['MLT'][:][i_tick] ) + \
                '\n%.1f' % ( data['L_IGRF'][:][i_tick] ) + \
                '\n%.1f' % ( data['meped_alpha_0_sat' ][:][i_tick] ) + \
                '\n%.1f' % ( data['meped_alpha_90_sat'][:][i_tick] )

        else:
            logger.warn( 'WARNING: No X-tick match found for %s.' % mpl.dates.num2date( tick ) )
            tickstr = '.\n.\n.\n.\n.\n.\n.\n.'     # This is when we have trouble lining up xticks.

        xlabels.append( tickstr )

    ax.set_xticklabels( xlabels )
    ax.annotate( 'UT\n' 'GLat\n' 'GLon\n' 'Alt\n' 'MLat\n' 'MLon\n' 'MLT\n' 'L_IGRF\n' 'alpha(0)\n' 'alpha(90)\n',  xy=(0.1,0.1425),
        xycoords='figure fraction', horizontalalignment='right',
        verticalalignment='top',  fontsize=mpl.rcParams['xtick.labelsize'] )

    # Ensure the x-axes are all lined up and default x-axis labels are hidden!
    for ax in axs_list:
        # All x-axis are lined up
        ax.set_xlim( dt_range )
        # Turn off default x-axis labels. Don't add the "ephemeris" axes to this list or you'll erase the markings you want.
        ax.tick_params(labelbottom=False)
        # Grid on
        ax.grid()

    #####
    # Write Figure
    #####
    if dir_output:
        dt_range_str = '%d%02d%02d_%02d%02d-%d%02d%02d_%02d%02d' % \
            (dt_range[0].year, dt_range[0].month, dt_range[0].day, dt_range[0].hour, dt_range[0].minute,
             dt_range[1].year, dt_range[1].month, dt_range[1].day, dt_range[1].hour, dt_range[1].minute)
        file_plot = dir_output + '/' + 'poes_%s_%s_meped_l2.png' % (sat_name, dt_range_str)
        if not os.path.exists( dir_output ): os.mkdir( dir_output )
        logger.info( my_name+': Saving plot to %s.' % file_plot )
        fig.savefig( file_plot, dpi=dpi )
        plt.close( fig )
    else:
        plt.show()


def plot_map( dataloc, sat_list, level, var_list, dt_start, dt_end, delta_lat=1., delta_lon=1.,
    log10=True, empty_cell_value=None, cb_range_list=None, dpi=600, dir_output=None ):
    '''
    Averages and plots SEM2 observations from multiple satellites and variables on a "Mill" projection map.
    :param dataloc
    :param sat_list: One or more satellite short names, e.g. ['n15','n18','n19','m01','m02','m03']
    :param level:    Processing level of the data "l2" or "raw".
    :param var_list: One or more variable names as keys to the "l2" or "raw" NetCDF repository (see also "dir_user" keyword).
    :param dt_start: Datetime Start
    :param dt_end:   Datetime End
    :param delta_lat: Grid resolution in latitude.
    :param delta_lon: Grid resolution in longitude.
    :param log10:    Apply log10() to data.
    :param empty_cell_value: Fill empty grid cells with this value. "None" (default) results in setting empty cells to
        the minimum value. Value (e.g. NaN) fills all empty cells with value.
    :param cb_range_list: Color bar range list. Length = number of variables to plot. [[min,max],[min,max],...]
    :param dir_output:    Optional, directory to save plot to. None => plot will appear on screen.
    :param dir_user_data: Optional, used for call to get_data_l2() or get_data_raw().
    :return:
    '''

    #####
    # Top-level entry point
    #####
    my_name = 'plot_map'
    logger.info( my_name+': Starting with sats:[%s], level:%s, vars:[%s], dt_range:%s to %s, log10:%s, dir_output:%s' % (','.join(sat_list), level, ','.join(var_list), str(dt_start), str(dt_end), log10, dir_output) )

    # For each variable:
    for i_var in range(len(var_list)):

        varname = var_list[i_var]
        logger.info( my_name+': Variable [%s]' % varname )

        # Colorbar Range (if set)
        if cb_range_list is not None:
            cb_range = cb_range_list[i_var]
        else:
            cb_range = [None, None]

        # Overall gridded data
        grid_all = None
        n_all = None

        for satname in sat_list:
            # Get Satellite Data (returned data is time range is rounded to nearest days, need to subset later)
            if level is 'raw':
                #nc_ts = get_data_raw( satname, dt_start, dt_end, dir_user_data=dir_user_data )
                nc_ts= get_data_raw(dataloc, satname, dt_start, dt_end, clobber=True, vars=None, all=True)

            elif level is 'l2':
                nc_ts = get_data_processed(dataloc, satname, dt_start, dt_end, clobber=True, vars=None, all=True)
                #nc_ts = get_data_l2( satname, dt_start, dt_end, dir_user_data=dir_user_data )
            else:
                assert( x == 'oops' )

            # Condition of no data for this satellite
            if nc_ts is None:
                continue
            else:
                time_msec = nc_ts['time'][:].filled()
                time_dt = nc4.num2date(time_msec, units=nc_ts['time'].units)
                idx = np.where((time_dt >= dt_start) & (time_dt < dt_end))[0]
                time_msec = time_msec[idx]
                time_dt = time_dt[idx]

                lats_ts = nc_ts['lat'][idx].filled(fill_value=np.nan)
                lons_ts = lon_360_to_180(nc_ts['lon'][idx].filled(fill_value=np.nan))

                # SAA Corrected E1, E2, or E3 0-degree detector
                if varname[-13:] == 'saa_corrected':
                    tmp_e123 = get_ele_tel0_e123_flux_saa_corrected( satname, dt_start, dt_end, dir_user_data=dir_user_data )
                    if tmp_e123 is None:
                        continue
                    if varname == 'mep_ele_tel0_flux_e1_saa_corrected':
                        data_ts = tmp_e123[0]
                    if varname == 'mep_ele_tel0_flux_e2_saa_corrected':
                        data_ts = tmp_e123[1]
                    if varname == 'mep_ele_tel0_flux_e3_saa_corrected':
                        data_ts = tmp_e123[2]
                    # Subset to time range
                    data_ts = data_ts[idx]

                else:
                    # Variable to plot
                    # Subset to time range
                    if nc_ts[varname].dtype == 'int8':
                        data_ts = nc_ts[varname][idx].filled(fill_value=-99).astype(float)  # int8 => float
                        data_ts[data_ts == -99] = np.nan
                    else:
                        data_ts = nc_ts[varname][idx].filled(fill_value=np.nan)

                # Close NetCDF
                nc_ts.close()

                # Safeguard
                assert( len(data_ts) == len(lats_ts) == len(lons_ts) )

                # Grid Data to Map
                grid_new, n_new, lats_grid, lons_grid = _average_data_to_map( data_ts, lats_ts, lons_ts, delta_lat=delta_lat, delta_lon=delta_lon )

                # Combine to overall Map: Weighted Average
                if grid_all is None:
                    grid_all = np.array(grid_new)
                    n_all    = np.array(n_new)
                else:
                    grid_all = (n_all*grid_all + n_new*grid_new)
                    n_all += n_new
                    grid_all[n_all > 0] = grid_all[n_all > 0] / n_all[n_all > 0]

        if grid_all is not None:
            # Final processing: log10() and empty grid cells:
            if log10:
                # Log10() the positive values and set the negative values to the resulting minimum.

                # Remember any negative data (<= 0), and set to minimum value later
                is_set_min = (grid_all <= 0)

                # Log scale any data > 0
                grid_all[grid_all > 0] = np.log10(grid_all[grid_all > 0])

                # Set negative data to minimum value
                grid_all[ is_set_min ] = np.min( grid_all )

            # Grid cells without observations will not be plotted
            if empty_cell_value is None:
                grid_all[n_all == 0] = np.min( grid_all )
            else:
                grid_all[n_all == 0] = empty_cell_value

            # Plot Map
            title = '%s: %s\n%s to %s' % (', '.join(sat_list), varname, str(dt_start)[:16], str(dt_end)[:16])
            dt_range_str = '%d%02d%02d_%02d%02d-%d%02d%02d_%02d%02d' % \
                (dt_start.year, dt_start.month, dt_start.day, dt_start.hour, dt_start.minute,
                 dt_end.year, dt_end.month, dt_end.day, dt_end.hour, dt_end.minute)
            file_plot = dir_output + '/' + 'poes_%s_%s_%s_map.png' % ('-'.join(sat_list), varname, dt_range_str)
            _plot_map( grid_all, lats_grid, lons_grid, cb_range=cb_range, variable_label=varname, title=title, dpi=dpi, file_plot=file_plot )


def _average_data_to_map( data, lats, lons, delta_lat, delta_lon ):
    '''
    Grids timeseries data onto 2D map.
    '''
    my_name = 'average_data_to_map'

    #####
    # Configuration
    #####


    #####
    # Grid the Data
    #####

    # Lats and Lons (centers of grid boxes)
    lats_grid = np.linspace(-90,90, endpoint=True, num=long(180./delta_lat + 1))
    # lats_grid = np.arange(-90,91,2)
    lons_grid = np.linspace(-180,180, endpoint=True, num=long(360./delta_lon + 1))
    # lons_grid = np.arange(-179,181,2)

    data_grid = np.zeros((len(lats_grid), len(lons_grid)))
    n_grid    = 0 * data_grid

    # Map each satellite observation to Grid
    for i in np.arange( len( lats ) ):
        # Find nearest Lat, Lon grid cell
        i_lat = np.argmin( np.abs(lats[i] - lats_grid) )
        i_lon = np.argmin( np.abs(lons[i] - lons_grid) )

        # Add data to grid point and increment counter (skipping NaNs)
        if np.isfinite( data[i] ):
            data_grid[ i_lat, i_lon ] += data[i]
            n_grid[ i_lat, i_lon ]    += 1

    # Average data
    data_grid[ n_grid > 0 ] = data_grid[ n_grid > 0 ] / n_grid[ n_grid > 0 ]

    return [data_grid, n_grid, lats_grid, lons_grid]

def _stats_data_to_map( data, grid1, grid2, latvals, grid1vals, grid2vals,  rvals, bwidth, do_log):
    '''
    Grids timeseries data onto 2D map and returns stats (cumulative distribution at that grid

    :param data         an input time series of data
    :param grid1        the time series of first grid value (likely L or lat)
    :param grid2        the time series of longitude values to be binned
    :param latvals      is the time series of latitudes because we want this seperate for north and south
    :param grid1vals    a list of grid values to use i.e. gridvals =[1,1.2,1.4...7]
    :param grid2vals    a list of grid values to use


    for multiple years I should be able to add the values
    The L data being passed here is L=1,1.25,1.5,1.75,2.0 ...

    And the L grid is np.arange(1,10.25,.25) 37 values
    '''
    my_name = 'stats_data_to_map'

    #####
    # Grid the Data
    #####


    data_mean = np.zeros((2, 2, len(grid1vals)-1, len(grid2vals)-1)) # This is the mean value lat,N/S,L,lon
    n_grid    = 0 * data_mean # number of values in each grid point

    # These are the logflux values to use for the cdfs
    # I'm using 0 to 8 at .1 increments
    binvals = np.arange(rvals[0],rvals[1],bwidth)

    # data_cdf holds the cdf for every grid point
    # But I also need to consider lat (that is the first index)
    # data_cdf(lat, north or south, Lgrids, longrids, cdfvals)

    # The last endpoint does not get included in the binned data. i.e. 360 is not a bin
    data_cdf = np.zeros((2, 2, len(grid1vals)-1, len(grid2vals)-1, len(binvals)-1))

    # Create the N'S indicator
    ways = 0*latvals
    ways[0:-1] = np.diff(latvals)
    #repeat the last val
    ways[-1] = ways[-2]

    ways[np.where(ways>=0)]=0
    ways[np.where(ways < 0)] = 1



    # Map each satellite observation to Grid
    # Start by stepping through north and south lats
    for lats in range(0,2):

        for dway in range(0,2):

            # Then step through the Lgrid
            # The grid that gets passed is 1,1.25...10
            for g1ind in np.arange(len(grid1vals)-1):

                # Then the lon grid
                # The longrid that gets passed is
                # longrid is 0,10,20, ...360
                for g2ind in np.arange(len(grid2vals)-1):

                    # Get all the data for that grid
                    # What to do with 0's?
                    if lats ==0:
                        vals = np.where((grid1>=grid1vals[g1ind]) & (grid1<grid1vals[g1ind+1] ) & (grid2>=grid2vals[g2ind]) & (grid2<grid2vals[g2ind+1] ) & (latvals>0) & (ways ==dway))[0]
                    else:
                        vals = np.where((grid1>=grid1vals[g1ind]) & (grid1<grid1vals[g1ind+1] ) & (grid2>=grid2vals[g2ind]) & (grid2<grid2vals[g2ind+1] ) & (latvals<0) & (ways ==dway))[0]

                    # This is the mean for convenience at each grid and northern or southern
                    # The trouble here is that I ignoe 0's
                    # I think I should change this so that 0's get set to the 1 count value?

                    # For the mean flux I include 0's but not negatives that show up
                    # because of the calibrations
                    vals2 = np.where(data[vals]>=0)
                    data_mean[lats,dway, g1ind,g2ind]= np.nanmean(data[vals[vals2]])

                    # How to make a cdf
                    # x = np.sort(data)
                    # y = np.arange(1, len(x) +1)/len(x)
                    # But this would make all my cdfs different lengths
                    # I want to bin first with set bins

                    # We have a problem with fluxes that are 0 or less
                    # than 1 because my binvals are from 10^0=1 to 10^8
                    # I put all the data between 0 and 1 in the lowest bin

                    zvals = np.where((data[vals] >=0) & (data[vals]<1))
                    # I want this to work on values that are not log
                    if do_log ==1:
                        if len(zvals[0])>0:
                            data[vals[zvals[0]]] = 10**binvals[0]

                        values, base = np.histogram(np.log10(data[vals[vals2]]), bins=binvals)
                        if len(data[vals[vals2]]) > 0:
                            if (np.max(np.log10(data[vals[vals2]])) > np.max(binvals)):
                                print("Oh no, the data is higher than the highest cdf bin")
                    else:
                        if len(zvals[0]) > 0:
                            data[vals[zvals[0]]] = binvals[0]

                        values, base = np.histogram((data[vals[vals2]]), bins=binvals)
                        if len(data[vals[vals2]]) > 0:
                            if (np.max((data[vals[vals2]])) > np.max(binvals)):
                                print("Oh no, the data is higher than the highest cdf bin")

                    cumulative = np.cumsum(values)
                    data_cdf[lats,dway, g1ind,g2ind,:]=cumulative



                    n_grid[lats, dway, g1ind, g2ind]=len(vals[vals2])
                    if np.max(cumulative)!=len(vals[vals2]):
                        print('uh oh, the cumulative number and the total vals do not match')



    return [data_mean, data_cdf, n_grid, grid1vals, grid2vals]



def _L_average_data_( data, grid1, grid1vals ):
    '''
    Todo This needs to be fixed
    PURPOSE: To average the time series as a function of L

    :param data    an input time series of data
    :param grid    the time series of first grid value (likely L or lat)
    :param gridvals a list of grid values to use i.e. gridvals =[1,1.2,1.4...7]

    '''
    my_name = 'L_average_data'
    # To do this, you need to step through each time step
    # Or figure out the Lpasses and average those chunks
    # Or average it with a filter and decimate it?
    for g1ind in np.arange(len(grid1vals)-1):
        for g2ind in np.arange(len(grid2vals)-1):

            vals = np.where((grid1>=grid1vals[g1ind]) & (grid1<grid1vals[g1ind+1] ) & (grid2>=grid2vals[g2ind]) & (grid2<grid2vals[g2ind+1] ))[0]

            data_grid[g1ind,g2ind]= np.nanmean(data[vals])

            n_grid[g1ind,g2ind,:]=len(vals)
            print('Here')


    return [data_grid, n_grid, grid1vals, grid2vals]

def _plot_map( data, lats, lons, cb_range=[None, None], variable_label=None, dpi=600, file_plot=None ):
    '''
    Plots a data grid onto a map.
    '''
    my_name = '_plot_map'

    #####
    # Configuration
    #####
    bkg_color  = (0.7,0.7,0.7,0.8)
    cont_color = (0.6,0.6,0.6,0.8)

    #####
    # Plot
    #####
    #from matplotlib.toolkits import Basemap, cm
    import matplotlib as mp

    # New figure
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # Mercator map with continents and grid lines
    #m = Basemap(llcrnrlon=-180.,llcrnrlat=-85,urcrnrlon=180.,urcrnrlat=85,\
    #            rsphere=(6378137.00,6356752.3142),\
    #            resolution='l',projection='mill',\
    #            lat_0=0.,lon_0=0.,lat_ts=30.)
    # Coastlines
    #m.drawcoastlines(color=cont_color)      # (red,green,blue,alpha)
    # Continents
    # m.fillcontinents( color=cont_color, lake_color=cont_color )
    # Background color
    #m.drawmapboundary( fill_color=bkg_color )
    # Label latitudes (but don't actually draw the lines)

    #m.drawparallels( np.arange(-80,90,20),labels=[True, True, False, True], linewidth=0 )
    # Label longitudes (but don't actually draw the lines)
    #m.drawmeridians( np.arange(-180,180,30),labels=[True, True, False, True], linewidth=0 )

    # Add Data Image and make sure it's on top (zorder)
    # Prepare Lats and Lons for mesh grid
    lons_grid, lats_grid = np.meshgrid(lons, lats)
    if cb_range is None:
        cb_range = [None, None]
    im1 = plt.pcolormesh(lons_grid, lats_grid, np.log10(data), shading='flat', cmap=plt.cm.jet, vmin=cb_range[0], vmax=cb_range[1] )#, zorder=10)

    # Colorbar
    #cb = plt.colorbar(im1, "bottom", size="5%", pad="5%")
    #cb.set_label( variable_label )

    if title:
        ax.set_title( title )

    #####
    # Write Figure
    #####
    if file_plot:
        fig.savefig( file_plot, dpi=dpi )
        plt.close( fig )
        logger.info( my_name+': Saved plot to %s.' % file_plot )
    else:
        plt.show()
def _plot_cdf( data, lats, lons, cb_range, data_cdf, rvals, bwidth, data_label=None, lat_label=None, lon_label=None, lat_ladpi=600, file_plot=None ):
    '''
    Plots a color grid map of the POES data with the average values as
    well as the cumulative distribution functions s at specific grid points

    :param data:    dictionary of POES data returned from any of the get_data functions
    :param lats:    list of lats from binning example [0,5,10,15,20] or np.range(0,90,1)
    :param lons:    list of lons for binning
    '''
    my_name = '_plot_cdf'

    #####
    # Configuration
    #####
    bkg_color  = (0.7,0.7,0.7,0.8)
    cont_color = (0.6,0.6,0.6,0.8)

    #####
    # Plot
    #####

    import matplotlib as mp

    # New figure
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])

    # Add Data Image and make sure it's on top (zorder)
    # Prepare Lats and Lons for mesh grid
    # these do not have to be lat and lon (It's whatever grid is passed

    # The first plot is the average values of the data passed in data
    plt.subplot(3,1,1)
    lons_grid, lats_grid = np.meshgrid(lons, lats)
    if cb_range is None:
        cb_range = [None, None]
    im1 = plt.pcolormesh(lons_grid, lats_grid, np.log10(data[0,:,:]), shading='flat', cmap=plt.cm.jet, vmin=cb_range[0], vmax=cb_range[1] )#, zorder=10)
    plt.yticks(lats[0:-1:4])
    plt.ylabel(lat_label)
    plt.xlabel(lon_label)
    plt.title(data_label)
    plt.colorbar()
    # Colorbar
    #cb = plt.colorbar(im1, "bottom", size="5%", pad="5%")
    #cb.set_label( variable_label )

    # The next six plots are the cdfs at a 6 lat values and all lon values
    xvals = np.arange(rvals[0],rvals[1],bwidth)

    for pco in range(0,6):
        plt.subplot(3,3,4+pco)

        latco = pco*np.floor(len(lats)/6.0)
        plt.title(lats[latco])
        for lonco in range(0,len(lons)):
            plt.plot(xvals[0:-1],data_cdf[0,int(latco),int(lonco),:])
        plt.ylabel('cdf')
        plt.xlabel('log10(flux)')
        plt.xticks(xvals[0:-1:10])
        plt.yticks([0,.25,.5,.75,1])
    plt.tight_layout()

    if file_plot:
        fig.savefig( file_plot, dpi=300 )
        plt.close( fig )
        logger.info( my_name+': Saved plot to %s.' % file_plot )

def comp_cdf(file1, file2):
    # PURPOSE: To compare two cdf pickle files created by _stats_data_to_map
    # That function creates a pickle file with
    # data_grid (the mean in each bin), data_cdf (the full binned cdf), n_grid (number of data points)
    # grid1vals ( the lat bins), grid2vals (The lon bins)

    # Unpickle the first file
    with open(file1, 'rb') as f:  # Python 3: open(..., 'wb')
        [dataf1_gride1, dataf1_cdfe1, dataf1_gride2, dataf1_cdfe2, dataf1_gride3, dataf1_cdfe3, dataf1_gride4, dataf1_cdfe4, n_grid, grid1valsf1, grid2valsf1, rvalsf1, bwidthf1] = pickle.load(f)

    with open(file2, 'rb') as f:  # Python 3: open(..., 'wb')
        [dataf2_gride1, dataf2_cdfe1, dataf2_gride2, dataf2_cdfe2, dataf2_gride3, dataf2_cdfe3, dataf2_gride4, dataf2_cdfe4, n_grid,
         grid1valsf2, grid2valsf2, rvalsf2, bwidthf1] = pickle.load(f)

    # How to compare the two?
    # KS test to see if there are statistically significant differences
    # How do we display it? Each grid is 28X36 = That's 1008 difference to look at.
    # Maybe make quick look plots showing
    # L = 1,2,3,4,5,6,7 and 13 lons
    xvals = np.arange(rvalsf1[0], rvalsf1[1], bwidthf1)
    for fpo in range(1,7):
        # I will make 7 figures (1 for each L)
        plt.figure(fpo)
        for lco in np.arange(0,36,3):
            plt.subplot(4,3,int(lco/3)+1)
            plt.semilogy(xvals[0:-1],1-dataf1_cdfe3[fpo*4,lco,:])

            plt.semilogy(xvals[0:-1], 1-dataf2_cdfe3[fpo * 4, lco,:])


    print('Here')

def getlatpass(lat):
    peaks = find_peaks(lat)
    valleys = find_peaks(-lat)
    equator = np.where(np.diff(np.sign(lat)))
    allbreaks = np.sort(np.append(peaks[0], np.append(valleys[0], equator[0])))
    pbreaks = 0 * lat
    pbreaks[allbreaks] = 1
    passes = np.cumsum(pbreaks)

    return passes,allbreaks

def getLpass(L,dist=200,prom=.5):
    '''Creates an arary with pass number for each data point that can be used to more easily average data or plot
     it pass by pass. Limit to between 0 and 100 because weird things happen at large L.
     :param L(data column)      L value that we are using to define a pass
     :param dist(int)           Required distance in datapoints between peaks
     :param prom (float)        The prominance defines how high it has to be to be considered a peak
     :return passes (list)      A list with passnumbers from 0 to ... for each dfatapoint
     :return allbreaks (list)   List of the indices that define the breaks between passes

    Usage: if the data is netcdf4 returned from poes_utils
     getLpass(poes['L_IGRF][:],dist=200,prom=.5):
     if the data is a numpy array
     getLpass(data['L_IGRF'][:],dist=200,prom=.5'''

    if isinstance(L, np.ma.MaskedArray):
        Ldata = L.data
    else:
        Ldata = L
    goodinds= np.where((Ldata[:]>0) & (Ldata[:]<100))[0]
    peaks = find_peaks(Ldata[goodinds],distance=dist,prominence=prom) # Find the maxima
    valleys = find_peaks(-1*(Ldata[goodinds]),distance=dist,prominence=prom) # Find the minima
    #plt.plot(L.data[goodinds])
    #plt.plot(peaks[0],L.data[goodinds[peaks[0]]],'*')
    #plt.plot(valleys[0], L.data[goodinds[valleys[0]]], '+')
    allbreaks = np.sort(np.append(goodinds[peaks[0]], goodinds[valleys[0]]))
    pbreaks = 0 * Ldata
    pbreaks[allbreaks] = 1
    passes = np.cumsum(pbreaks)

    return passes,allbreaks

def make_sar(datacdf, Lvals,lonvals,rvals,bwidth,lon,sat ):
    #PURPOSE To do a statistical asynchronou regression of the data from all lons to one lon
    # in the southern hemisphere
    # INPUTS:
    # data_cdf: This has the cdf values for each variable in a grid that is [2,L,lon,cdfbins]
    # lon: the reference longitude for mapping the sar to
    # The trouble here is that I need to consider north and southern hemisphereos

    # Basically all I am doing is regridding the cdfs on fixed percentile bins
    # So that we have percentile -> log flux
    # So knowing your flux at a given Lindex and lon index you woule
    # get all the flux values and find the index of the closest
    # and then find the flux for the same index at whatever ref lon you want.

    # Find the index of the baseline lon
    # I chose 260 as the base lon because it is likely to have data
    # baselon=np.where(lonvals==int(lon))[0]

    cols = list(datacdf.keys())

    xvals = np.arange(rvals[0], rvals[1], bwidth) # This is the log flux ranges np.arange(0,7.1,.1)

    # At each L, compare all lons to the chosen lon
    pnum=3

    # fluxin has energy, Lgrid ,longrid, fluxgrid)
    # fluxin and fluxref will be dictionaries with np arrays

    fluxref = {}

    # First regrid the cdfs
    # Normally, you have cdf values at a fixed log(flux) grid (i.e. .1 log(flux)
    # interpoalte this to get the log flux at fixed cdf values
    cdfgrid = np.arange(0, 1.01, .01)

    # Define the output data
    for es in cols:
        fluxref[ es ] = np.zeros((2, 2, len(Lvals)-1,len(lonvals)-1,len(cdfgrid)), dtype = float)

    for alt in range(0,2):
        for dway in range(0,2):
            for g1 in np.arange(0,len(Lvals)-1):

                # Step throught the L grid

                for g2 in np.arange(0,len(lonvals)-1):
                    # Step through the longrid

                    # Step through each col at the L an longrid
                    for es in cols:

                        # Normally you would plot the cdf values (y-axis) versus the log flux values (x-axis)
                        # I create the opposite interpolation so I can get the log(flux) at any cdf percentile level
                        # xvals is the log(flux) bins 0,.1,...7

                        # There is an issue here that every log(flux) above the max value has a cdf of 1
                        # So when you plot cdf, log10(flux) you get a straight line at 0 and 1 that makes the
                        # interpolation wonky (infinite).
                        # Need to interpolate from the last 0 to the first 1

                        ilow = np.where(datacdf[es][alt, dway, g1, g2, :]==0)[0]
                        # Sometimes there is no 0 value so there is a check here
                        if len(ilow)<1:
                            ilow = [0]
                        ihi = np.where(datacdf[es][alt, dway, g1, g2, :]==1)[0]
                        if len(ihi)<1:
                            ihi = [len(datacdf[es][alt, dway, g1, g2, :])-1]

                        # print(ilow,ihi)
                        # print(datacdf[es][alt,dway,  g1, g2, :])
                        # print('L and lon',g1,g2)
                        # print(ilow[-1],ihi[0])
                        if min(datacdf[es][alt,dway,  g1, g2, :])==1:
                            ilow=[0]
                            ihi = [len(datacdf[es][alt, dway, g1, g2, :])-1]
                            print('Here')
                        funclon = interpolate.interp1d( datacdf[es][alt, dway, g1, g2, ilow[-1]:(ihi[0]+1)], xvals[ilow[-1]:(ihi[0]+1)],fill_value='extrapolate')

                        # extrapolating makes nans where the cdf is at 0
                        # I fill that with the min value?

                        # This gives the flux values at the % levels
                        fluxlon = funclon(cdfgrid)
                        #if ((g1>20) & (es==cols[2])):
                        #    plt.plot(datacdf[es][alt, dway, g1, g2, ilow[-1]:(ihi[0]+1)], xvals[ilow[-1]:(ihi[0]+1)],'.')
                        #    plt.plot(cdfgrid,fluxlon)
                        #    plt.title(str(g1)+ ' '+str(g2))
                        #    print(g1,g2)

                        # Where it is a nan it is set to the min
                        fluxlon[np.isnan(fluxlon)]=np.nanmin(fluxlon)

                        # To do the mapping, you would find the index of the flux in fluxin
                        # at the L and lon bin. That tells you the percentile level.
                        # Then you use that index to get flux ref at the referenc lon
                        if (alt==0) & (g1==4) & (g2==10):
                            print(fluxlon)
                        fluxref[ es ][alt, dway, g1, g2, :] = fluxlon
                        if alt ==0:
                            print('Here"')



    return fluxref, cdfgrid

    print('Here')

def datetimeFromFields( year, month, day, hhmm ) :

   strDate = month + ' ' + day + ' ' + str(year) + ' ' + hhmm

   try :
      d = dtm.datetime.strptime( strDate, "%b %d %Y %H%M" )
   except ValueError :
      print( "Error parsing " + strDate )

   return d


# aggregateProcessedData
# Collects data from a passed set of satellites and variables over a passed time window and
# writes it to a netCDF file
#
# Returns an aggregated collection of L binned data for all satellites and variables

def aggregateProcessedData( sdate_all, edate_all, satlist, varlist, Lbins ) :

    flen = 4*25*((edate_all-sdate_all).days+1)*len(satlist)
    fwid = len(Lbins)-1

    # We sill create an Lbinnded data set for plotting
    # Each key will have an np array that is all orbits long and Lbins wide
    final_dat ={}
    for var in varlist:
        final_dat[var]=np.zeros((flen,fwid),dtype = float)
    # Add one more variable called passtime
    final_dat['passtime'] = list()
    final_dat['passdtime'] = list()
    indco = 0

    ############
    clobber = 1 # Delete any pre-existing aggregated file

    #============================= Loop through sats and time ============================
    # Loop through each satelite
    for sco in range(0,len(satlist)):

        sdate = sdate_all  # set start date to process the data
        edate = edate_all  # set the end date to process the data

        # This will loop through to the end in chunks set to tstep
        while sdate < edate:
            print(str(sdate))

            # get_data_processed reads the daily ngdc netcdf files, creates an aggregated netcdf file and then reads that.
            # It will call two days if start is 2012-01-01 and end is 2012-01-02
            # Here we get a month of data
            # This gets the first day of the next month but we don't want to
            # include that day so subtract one day
            mdate = sdate + dt.timedelta(days=31)
            mdate = mdate.replace(day=1)
            mdate = mdate - dt.timedelta(days=1)

            # check to see if mdate is greater than adate
            # Maybe we don't need the whole month
            if mdate > edate:
                mdate = edate

            poes = get_data_processed(satlist[sco], sdate, mdate, vars=varlist)

            if poes is None:
                print("No Data for " + satlist[sco] + " in month starting on " + str(
                    sdate.date()))  # If there is no data for the month then skip it

            else:

                ptime = unix_time_ms_to_datetime(poes['time'][:])  # Change the time to datetime
                # Find the indices of the min and max Ls
                # i.e. the indices of each Lpass
                Linds = findLmins(poes['L_IGRF'], 200)

                # Now loop through each Lpass using the indices from Linds
                # and average the data in each Lstep sized bin just for plotting

                for lco in range(0, len(Linds) - 1):
                    # ------------------------------------
                    # Now average the data in the Lbins for that pass
                    # Get the L data for the pass that we will binon
                    passLdat = poes['L_IGRF'][Linds[lco]:Linds[lco + 1]]

                    # Find the range of L values in the pass
                    # Lmin = np.floor(np.nanmin(passLdat[np.where(passLdat>0)]))
                    # Lmax = np.ceil(np.nanmax(passLdat[np.where(passLdat>0)]))
                    # Lmax = 25
                    # Lbins = np.arange(Lmin, Lmax, Lstep)  # Lbins = 1,1.5,2 etc

                    # --------- Bin and average each variable -------------
                    for ivar in range(0, len(varlist)):
                        var = varlist[ivar]
                        # JGREEN: Need to fix nans or infs
                        if var == 'lon':
                            # For longitude you have to do something different to average over 0 to 360
                            cosave = binned_statistic(passLdat,
                                                      np.cos(np.deg2rad(poes[var][Linds[lco]:Linds[lco + 1]])), 'mean',
                                                      bins=Lbins).statistic
                            sinave = binned_statistic(passLdat,
                                                      np.sin(np.deg2rad(poes[var][Linds[lco]:Linds[lco + 1]])), 'mean',
                                                      bins=Lbins).statistic
                            lonave = np.array(
                                [np.rad2deg(math.atan2(sinave[x], cosave[x])) for x in range(0, len(sinave))])
                            lonave[np.where(lonave < 0)] = lonave[np.where(lonave < 0)] + 360.0
                            # final_dat[var].append(lonave)
                            final_dat[var][indco, :] = lonave
                        elif var == 'L_IGRF':
                            # For L we want the bin values
                            final_dat[var][indco, :] = Lbins[0:-1]
                        else:
                            bin_means = binned_statistic(passLdat, poes[var][Linds[lco]:Linds[lco + 1]], 'mean',
                                                         bins=Lbins)
                            final_dat[var][indco, :] = bin_means.statistic

                            # ----------- Add a pass time -----------------------------------
                            # To plot the data we need a pass time. Set it to the time at L=4
                    L4ind = np.where(np.floor(passLdat) == 4)[0]
                    # Sometimes it is a partial pass and there is no value at L=4
                    # In that case set it to the max L value
                    if len(L4ind) < 1:
                        L4ind = np.where(passLdat == max(passLdat))[0]

                    # Get the datetimes for the pass
                    tdat = ptime[Linds[lco]:Linds[lco + 1]]

                    # This is the unix time at L=4 that I'm using as a way to
                    # track the individual passes. Each pass will have this same time
                    tL4 = np.floor((tdat[L4ind[0]] - dt.datetime(1970, 1, 1)).total_seconds())
                    final_dat['passtime'].append(tL4)
                    # Save the passtime as a datetime list because it is easy to deal with.
                    final_dat['passdtime'].append(tdat[L4ind[0]])

                    indco = indco + 1

                poes.close()  # Close the netcdf file
            sdate = sdate + dt.timedelta(days=31)
            sdate = sdate.replace(day=1)

    # Now you have a mess of all the satellite data appended on each other not in time order
    # sort on final_dat['passtime']
    tinds = np.argsort(final_dat['passtime'][1:indco - 1])

    return final_dat, tinds

# aggregateRawData
# Collects raw data from a passed set of satellites and variables over a passed time window and
# writes it to a netCDF file
#
# Returns an aggregated collection of L binned data for all satellites and variables

def aggregateRawData( sdate_all, edate_all, satlist, varlist, Lbins ) :

    flen = 4*25*((edate_all-sdate_all).days+1)*len(satlist)
    fwid = len(Lbins)-1

    # We sill create an Lbinnded data set for plotting
    # Each key will have an np array that is all orbits long and Lbins wide
    final_dat ={}
    for var in varlist:
        final_dat[var]=np.zeros((flen,fwid),dtype = float)
    # Add one more variable called passtime
    final_dat['passtime'] = list()
    final_dat['passdtime'] = list()
    indco = 0

    ############
    clobber = 1 # Delete any pre-existing aggregated file

    #============================= Loop through sats and time ============================
    # Loop through each satelite
    for sco in range(0,len(satlist)):

        sdate = sdate_all  # set start date to process the data
        edate = edate_all  # set the end date to process the data

        # This will loop through to the end in chunks set to tstep
        while sdate < edate:
            print(str(sdate))

            # get_data_processed reads the daily ngdc netcdf files, creates an aggregated netcdf file and then reads that.
            # It will call two days if start is 2012-01-01 and end is 2012-01-02
            # We don't want to get more than a month of data at a time because that is
            # too much
            if (edate-sdate).days>30:
                mdate = sdate+dt.timedelta(days = 30)
            else:

                mdate = edate

            # Get raw poes data from netcdf
            poes = get_data_raw(satlist[sco], sdate, mdate, dir_user_data=None, clobber=True)

            # Also need processed to get L
            procvarlist = ['L_IGRF', 'MLT']
            poes_processed = get_data_processed(satlist[sco], sdate, mdate, vars=procvarlist)

            if poes is None:
                print("No Data for " + satlist[sco] + " in month starting on " + str(
                    sdate.date()))  # If there is no data for the month then skip it

            else:

                ptime = unix_time_ms_to_datetime(poes['time'][:])  # Change the time to datetime
                # Find the indices of the min and max Ls
                # i.e. the indices of each Lpass
                Linds = findLmins(poes_processed['L_IGRF'], 200)

                # Now loop through each Lpass using the indices from Linds
                # and average the data in each Lstep sized bin just for plotting

                for lco in range(0, len(Linds) - 1):
                    # ------------------------------------
                    # Now average the data in the Lbins for that pass
                    # Get the L data for the pass that we will binon
                    passLdat = poes_processed['L_IGRF'][Linds[lco]:Linds[lco + 1]]

                    # Find the range of L values in the pass
                    # Lmin = np.floor(np.nanmin(passLdat[np.where(passLdat>0)]))
                    # Lmax = np.ceil(np.nanmax(passLdat[np.where(passLdat>0)]))
                    # Lmax = 25
                    # Lbins = np.arange(Lmin, Lmax, Lstep)  # Lbins = 1,1.5,2 etc

                    # --------- Bin and average each variable -------------
                    for ivar in range(0, len(varlist)):
                        var = varlist[ivar]
                        # JGREEN: Need to fix nans or infs
                        if var == 'lon':
                            # For longitude you have to do something different to average over 0 to 360
                            cosave = binned_statistic(passLdat,
                                                      np.cos(np.deg2rad(poes[var][Linds[lco]:Linds[lco + 1]])), 'mean',
                                                      bins=Lbins).statistic
                            sinave = binned_statistic(passLdat,
                                                      np.sin(np.deg2rad(poes[var][Linds[lco]:Linds[lco + 1]])), 'mean',
                                                      bins=Lbins).statistic
                            lonave = np.array(
                                [np.rad2deg(math.atan2(sinave[x], cosave[x])) for x in range(0, len(sinave))])
                            lonave[np.where(lonave < 0)] = lonave[np.where(lonave < 0)] + 360.0
                            # final_dat[var].append(lonave)
                            final_dat[var][indco, :] = lonave
                        elif var == 'L_IGRF':
                            # For L we want the bin values
                            final_dat[var][indco, :] = Lbins[0:-1]
                        elif var == 'MLT':
                            bin_means = binned_statistic(passLdat, poes_processed[var][Linds[lco]:Linds[lco + 1]],
                                                         'mean', bins=Lbins)
                            final_dat[var][indco, :] = bin_means.statistic
                        else:
                            bin_means = binned_statistic(passLdat, poes[var][Linds[lco]:Linds[lco + 1]], 'mean',
                                                         bins=Lbins)
                            final_dat[var][indco, :] = bin_means.statistic

                    # ----------- Add a pass time -----------------------------------
                    # To plot the data we need a pass time. Set it to the time at L=4
                    L4ind = np.where(np.floor(passLdat) == 4)[0]
                    # Sometimes it is a partial pass and there is no value at L=4
                    # In that case set it to the max L value
                    if len(L4ind) < 1:
                        L4ind = np.where(passLdat == max(passLdat))[0]

                    # Get the datetimes for the pass
                    tdat = ptime[Linds[lco]:Linds[lco + 1]]

                    # This is the unix time at L=4 that I'm using as a way to
                    # track the individual passes. Each pass will have this same time
                    tL4 = np.floor((tdat[L4ind[0]] - dt.datetime(1970, 1, 1)).total_seconds())
                    final_dat['passtime'].append(tL4)
                    # Save the passtime as a datetime list because it is easy to deal with.
                    final_dat['passdtime'].append(tdat[L4ind[0]])

                    indco = indco + 1

                # Close the netcdf files
                poes_processed.close()
                poes.close()

            #sdate = sdate + dt.timedelta(days=31)
            #sdate = sdate.replace(day=1)
            sdate = mdate


    # Now you have a mess of all the satellite data appended on each other not in time order
    # sort on final_dat['passtime']
    tinds = np.argsort(final_dat['passtime'][1:indco - 1])

    return final_dat, tinds

# plotLbinnedData
#

def plotLbinnedData( final_dat, sdate_all, edate_all, satlist, Lbins, timeOrderedIndices ) :

    # Plot the data
    # JGREEN: Should label axis and add datetimes etc

    cols = ['mep_omni_cps_p6','mep_omni_cps_p7','mep_omni_cps_p8','mep_omni_cps_p9']
    fig = plt.figure(1)
    sats = ' '

    duration = edate_all - sdate_all
    ndays = duration.days
    if ndays == 0 :
       ndays = 1
    nDataCols = len( final_dat['passtime'] )
    colsPerWeek = 7 * nDataCols / ndays
    empty_labels = []
    for n in range(0, ndays,7) :
       empty_labels.append(' ')

    for sat in range(0,len(satlist)):
       sats = sats + satlist[sat] + ' '
    fig.suptitle( 'Satellites: ' + sats )

    for icol in range(0,len(cols)):
        axs = plt.subplot(4,1,icol+1)
        axs.set_title( cols[icol] )
        axs.set_ylabel('L')
        axs.set_xlabel( str(sdate_all) + '   -   ' + str(edate_all) )
        axs.set_xticks( np.arange(0, nDataCols, colsPerWeek) )
        axs.set_xticklabels( empty_labels )

        vmin = np.nanmin(final_dat[cols[icol]][np.where(final_dat[cols[icol]]>0)])
        vmax = np.nanmax(final_dat[cols[icol]][np.where(final_dat[cols[icol]]>0)])
        plt.pcolormesh( np.arange(0,nDataCols-2), Lbins[0:-1],np.transpose(np.log10(final_dat[cols[icol]][timeOrderedIndices[0:nDataCols-1],:])),
                                      vmin=np.log10(vmin), vmax=np.log10(vmax), cmap='jet')

        cbar = plt.colorbar()
        cbar.ax.set_ylabel( 'log10( count )' )

    #plt.show()
    plt.savefig( sc.config['dir_aggregates'] + 'flux_by_Lbin_' + str(sdate_all) + '_' + str(edate_all) + '.png')

    plt.close(fig)

# readSEPEvents()
# Reads SEP events from a NOAA file ftp://ftp.swpc.noaa.gov/pub/indices/SPE.txt
#
# Returns results as a list of dictionaries. Each dictionary defines the fields
# related to an SEP. Fields are 'startdate', 'enddate', 'protonflux', 
# 'flaredate', 'importance', 'location', 'region'.

def readSEPEvents( eventFilePath ) :

   eventFile = open( eventFilePath, 'r' )

   lines = eventFile.readlines()

   year = None

   eventlist = []

   for line in lines :

      # ingnore everything until we encounter a line having only a year
   
      if year is None :
         testline = line.strip()
         try :
            year = int( testline )
         except ValueError :
            pass

      else :

         # if at the end of list of events for that year
         # reset year to None until we encounter another year line

         testline = line.strip()

         if not testline or testline.lower() == 'none' :
            year = None

         # Have an event for this year, process it

         else :

            dict = {}

            elements = line.split()

            # translate start into datetime
            day, hhmm = elements[1].split('/')
            #print( year, elements[0], day, hhmm )
            startAt = datetimeFromFields( year, elements[0], day, hhmm )
            dict[ 'startdate' ] = startAt

            # translate max date into datetime
            day, hhmm = elements[3].split('/')
            endAt = datetimeFromFields( year, elements[2], day, hhmm )
            dict[ 'enddate' ] = endAt

            try :
               pFlux = int( elements[ 4 ] )
            except ValueError :
               pFlux = 0
               pass
            dict[ 'protonflux' ] = pFlux

            # Note: file inconsistent in presence and format of remaining
            # fields, so currently we ignore them.

            # translate flare max date
            #day, hhmm = elements[ 6 ].split('/')
            #flareAt = datetimeFromFields( year, elements[ 5 ], day, hhmm )
            #dict[ 'flaredate' ] = flareAt

            #importanceCode = elements[ 7 ]
            #dict[ 'importance' ] = importanceCode

            #locationCode = elements[ 8 ]
            #dict[ 'location' ] = locationCode

            #try :
            #   regionNum = int( elements[ 9 ] )
            #except ValueError :
            #   regionNum = 0
            #dict[ 'region' ] = regionNum

            eventlist.append( dict )

   eventFile.close()

   return eventlist

def get_energy_mev( channel ):
    # PURPOSE: Return threshold energy (MeV) for a heo satellite, channel and particle species

    poes_channels = { "mep_omni_cps_p6" : 16.0, "mep_omni_cps_p7" : 36.0, "mep_omni_cps_p8" : 70, "mep_omni_cps_p9" : 140.0 }

    energyMeV = poes_channels[ channel ]

    return energyMeV

def str2date(string):
    "Parse a string into a datetime object."
    dateformats = ['%Y-%m-%d','%m-%d-%Y','%m/%d/%Y', '%Y/%m/%d']
    for fmt in dateformats:
        try:
            return dt.datetime.strptime(string, fmt)
        except ValueError:
            pass

    raise ValueError("'%s' is not a recognized date/time" % string)

def load_config( cfgFilePath ) :
    ''' Loads a config file with data directories'''

    config = {}
    configParser = configparser.ConfigParser()
    configParser.read( cfgFilePath )
    try:
        path_items = configParser.items("paths")
        for key, path in path_items:
            if (key =='dir_aggregates') | (key=='fn_master_l2') | (key =='fn_master_raw'):
                config[key] = path
            else:
                config[key] = list()
                config[key].append(path)
        return config
    except:
        raise Exception('Config file is not valid. Please check format')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                     MAIN                                      "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
''' Examples for how to use each of the functions. 

    Running the examples require some inputs. One is information about where data is located.
    That can be set in a configfile or passed as a top level directory. Both usages are shown. 
    
    USAGE 
    python -c '/config_poes_utils_example.ini' -ex '''
if __name__ == '__main__':
    import datetime as dt
    import math
    from scipy.stats import binned_statistic
    import argparse

    parser = argparse.ArgumentParser('These tools perform aggregation, manipulation and plotting of POES data')
    #
    parser.add_argument('-d',"--dataloc",
                        help="The configfile with data locations or a top level directory of data",
                        required=False, default =os.getcwd())
    parser.add_argument('-s', "--start_time",
                        help="The start date to get data (i.e. -s 2001-01-01)",
                        required=True, default ='2001-01-01')
    parser.add_argument('-e', "--end_time",
                        help="The end date to get data (i.e. -e 2001-01-02)",
                        required=True, default ='2001-01-02')
    parser.add_argument('-sat', "--satname",
                        help="A name of satellite data to get (i.e. -sat n15 or -sat n16 ",
                        type=str, required=True, default='n15')
    parser.add_argument('-ex', "--example",
                        help="Name of example to run", required=False, default='get_raw')
    args = parser.parse_args()

    #------------ Do some argument checking -----------------
    # Check to see if a directory is passed or a config file
    # If nothing is passed it looks in the current directory
    if os.path.isdir(args.dataloc):
        print('Looking for POES data below '+args.dataloc)
        dataloc = args.dataloc
    else:
        # If not a directory then see if it is a config file
        try:
            # Try as a config file
            load_config(args.dataloc)
            print('Found config file with data directories ' + args.dataloc)
            dataloc = args.dataloc
        except:
            print('Need to provide a configfile with data locations or top level directory')
            raise

    # Check the date formats are OK and set start and end time
    sdate = str2date(args.start_time)
    edate = str2date(args.end_time)

    if args.example == 'get_raw':
        # This example does nothing but retrieve a dictionary of raw poes data (count data)
        # Optional arguments and defaults are
        # clobber = False (clobber=True will overwrite tmp datafiles)
        # vars = None gets all available variables (vars=['time','mep_omni_cps_p6'] will get only variables in list)
        # all = True will get binary swpc data and turn it into raw dictionary (all=False will only get ngdc format data)

        data = get_data_raw( dataloc, args.satname, sdate, edate, clobber = True)
        # print the list of vairables
        data.variables.keys()

    if args.example == 'get_processed':
        # This example does nothing but retrieve a dictionary of raw poes data (count data)
        # Optional arguments and defaults are
        # clobber = False (clobber=True will overwrite tmp datafiles)
        # vars = None gets all available variables (vars=['time','mep_omni_cps_p6'] will get only variables in list)
        # all = True will get binary swpc data and turn it into raw dictionary (all=False will only get ngdc format data)

        data = get_data_processed( dataloc, args.satname, sdate, edate, clobber=True)
        # Print a list of variables
        print(data.variables.keys())
    print('Here')













