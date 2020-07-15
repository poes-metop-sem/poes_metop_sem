#####
# Disclaimer: Users assume all risk related to their use of these routines and NOAA disclaims any and all warranties,
#      whether expressed or implied, including (without limitation) any implied warranties of merchantability or
#      fitness for a particular purpose.
# History:
# 
#####


#####
# Config: Adjust these as needed.
#####

config = {}

# Directory search paths for L2 2-second archive. You can list as many as you wish, in order of preference -- first file found wins.
config['dir_data_l2_search_paths'] = ['data/processed/ngdc/uncorrected/full/']

# Directory search paths for "raw" 2-second archive. You can list as many as you wish, in order of preference -- first file found wins.
config['dir_data_raw_search_paths'] = ['data/poes/public_local/data/raw/ngdc/']

# Directory search paths for L2 16-second average archive. You can list as many as you wish, in order of preference -- first file found wins.
config['dir_data_l2_swpc_avg_search_paths'] = 'data/poes/private/data/processed/swpc/uncorrected/avg/cdf/'

# Directory to save Aggregated files
config['dir_aggregates'] = 'data/poes/private/data/tmp/'

# Master NetCDF for L2 (aka "processed")
config['fn_master_l2'] = 'data/poes_l2-aggregate_master.nc'

# Master NetCDF for L1b (aka "raw")
config['fn_master_raw'] = 'data/poes_raw-aggregate_master.nc'


################################# No User Configurable options below this line ##############################


#####
# Setup
#####
import datetime as dtm, glob, logging, numpy as np, os, shutil, sys, traceback

import netCDF4 as nc4
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt

# Matplotlib default fonts sizes
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6

# Logging:
#     more info: https://docs.python.org/release/2.6/library/logging.html
log_level = logging.DEBUG
logger = logging.getLogger( 'poes_utils' )
logger.setLevel( log_level )
console = logging.StreamHandler()
console.setLevel( log_level )
console.setFormatter( logging.Formatter( "%(asctime)s - %(name)s - %(levelname)s - %(message)s" ) )
logger.addHandler( console )


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                Misc Utils                                     "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def satID_to_satname( satID ):
    '''
    Maps SatID in the Raw and L2 files to n|mXX (e.g. n15 for NOAA-15, m03 for Metop-C).
    Table was taken from ted_cal_coefficients.txt
    :param satID:
    :return:
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
    """ Unix time in milliseconds to Datetime
    :param time_ms: Numpy array of milliseconds since 1970 (default).

    :return: Numpy array of Datetime.
    """
    n_times   = np.size( time_msec )
    if 1 == n_times:
        time_dt = dtm.datetime(1970,1,1) + dtm.timedelta( microseconds=1000.*time_msec )
    else:
        time_dt = np.array( [dtm.datetime(1970,1,1) + dtm.timedelta( microseconds=1000.*time_msec[i] ) for i in np.arange( n_times ) ] )

    return( time_dt )


def lon_360_to_180(lon):
    lon_180 = lon
    lon_180[lon > 180] -= 360.
    return lon_180


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                 File I/O                                      "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def decode_filename( file_name ):
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
             'level'     : 'processed/ngdc/uncorrected/full',
             'type'      : 'NC4' }

    return( info )


def get_file_names( dir_root, fn_pattern_list ):
    """ Returns list of file names given a starting directory.

    :param dir_root:        String of absolute or relative user path (OPTIONAL).
    :param fn_pattern_list: List of filename patterns.
    :return:                List of files found or empty [].
    """
    my_name = 'get_file_names'

    try:
        import fnmatch
        logger.info( my_name+': ['+', '.join( fn_pattern_list )+']' )
        logger.debug( fn_pattern_list )

        # TODO: Check the inputs.

        fn_list = []
        for root, dirnames, filenames in os.walk( dir_root ):
            for fn_pattern in fn_pattern_list:
                for filename in fnmatch.filter( filenames, fn_pattern ):
                    fn_list.append( os.path.join( root, filename ) )

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


def get_data_processed( sat_name, dt_start, dt_end, dir_user_data=None ):
    return get_data_l2( sat_name, dt_start, dt_end, dir_user_data )

def get_data_l2( sat_name, dt_start, dt_end, dir_user_data=None, clobber=False ):
    """ Returns aggregated POES processed NetCDF.
    Look Order:
        - Current Directory
        - User chosen root
        - Default
        - NGDC SatDat URL (Future)

    :param sat_name:      One of {'m01', 'm02', 'm03', 'n15', ..., 'n19' }.
    :param dt_start:      Start datetime.
    :param dt_end:        End datetime.
    :param dir_user_data: String of a single absolute or relative user path (OPTIONAL).
    :return:              NetCDF object.
    """
    my_name = 'get_data_l2'

    # FIXME: make sure TED IFC periods filled the data in the official L2, and/or fill in the aggregates produced here.

    try:
        #####
        # Config:
        #####
        fn_master = config['fn_master_l2']

        # Search path order: Optional caller choice, OR preconfigured paths.
        if dir_user_data:
            dir_root_list = [dir_user_data]
        else:
            dir_root_list = config['dir_data_l2_search_paths']

        #####
        # Data: return Existing handle or create New aggregate
        #####

        # If file exists, skip aggregation:
        yyyymmdd1 = '%04d%02d%02d' % ( dt_start.year, dt_start.month, dt_start.day )
        yyyymmdd2 = '%04d%02d%02d' % ( dt_end.year,   dt_end.month,   dt_end.day   )
        fn_agg = config['dir_aggregates'] + '/poes_%s_%s-%s_l2.nc' % ( sat_name, yyyymmdd1, yyyymmdd2 )
        if not clobber and os.path.isfile( fn_agg ):
            logger.info( my_name+': Found existing aggregate, using %s.' % fn_agg )

        else:
            # Clobber existing if desired
            if clobber and os.path.isfile( fn_agg ):
                logger.info( my_name+': Clobber is on, removing existing aggregate %s.' % fn_agg )
                os.unlink( fn_agg )

            ''' List of Files to Ingest '''
            # File name pattern we need to look for (data are organized as day files):
            n_days = ( ( dtm.timedelta( days=1 ) \
                + dtm.datetime( year=dt_end.year,   month=dt_end.month,   day=dt_end.day   ) \
                - dtm.datetime( year=dt_start.year, month=dt_start.month, day=dt_start.day ) ) ).days
            file_pattern_list = []
            for i_day in np.arange( n_days ):
                t_dt = dt_start + dtm.timedelta( days=i_day )
                file_pattern_list.append( 'poes_'+sat_name+'_%04d%02d%02d_proc.nc' % ( t_dt.year, t_dt.month, t_dt.day ) )

            logger.debug( my_name+': Searching for files to aggregate: ' + '\n\t'.join( dir_root_list ) )
            for dir_root in dir_root_list:
                logger.debug( my_name+': Searching %s' % dir_root )
                fn_list = get_file_names( dir_root, file_pattern_list )
                logger.debug( my_name+': Found these %d files in \'%s\':' % ( len( fn_list ), dir_root ) )
                logger.debug( fn_list )
                # Once we have a list we'll move on (avoids finding same file names in multiple places).
                # Warning: If multiple file repositories exist and first one is incomplete then so will be the returned NetCDF4 object.
                if fn_list: break

            logger.debug( 'Found %d files to aggregate.' % len( fn_list ) )
            if len( fn_list ) == 0: return None

            ''' Ingest Data '''
            # Copy Master NetCDF to new file in '/tmp' or User choice:
            shutil.copy( fn_master, fn_agg )

            # Ingest each day file to Temporary Memory dictionary:
            nc_all = nc4.Dataset( fn_agg, 'r+' )
            t_data = {}    # Temp mem dictionary.
            for fn in fn_list:
                with nc4.Dataset( fn, 'r' ) as nc_day:
                    logger.debug( my_name+': Ingesting %s.' % fn )

                    # MEPED IFC doesn't correctly set the data variables to fill values
                    idx_ifc = np.where(nc_day['mep_IFC_on'][:].filled(fill_value=-1) > 0)[0]

                    for vname in nc_all.variables:
                        var_day = nc_day[ vname ]

                        # TODO: Overall issue with existing L2 "processed" repository: Fix POES/Metop L2 Master netCDF template and existing L2 dataset to ensure 'valid_range' attributes are indeed 2-element arrays (hint use 'valid_min' if needed).

                        # TODO: 'time' variable's valid_range is wrongly set to [0., 0.] resulting in a completely masked array (all filled).
                        if vname == 'time':
                            logger.error( my_name+': Variable "time" has broken attribute valid_range set to [0.,0.] (results in 100% masked array), disabling Auto Masking to enable reading/copying.' )
                            var_day.set_auto_mask(False)

                        # TODO: Most L2 file variables have 'valid_range' set to a scalar (should be a vector).
                        if hasattr(var_day, 'valid_range') and not isinstance(var_day.valid_range, np.ndarray):
                            logger.error( my_name+': Variable %s has broken attribute valid_range (scalar, should be 2-element), disabling Auto Masking to enable reading/copying.' % (vname) )
                            var_day.set_auto_mask(False)

                        # TODO: L2 processing doesn't correctly fill MEPED variables during IFC.
                        # Temporary variable to hold corrected values (if we need to correct them)
                        var_day_fixed = var_day[:]
                        if len(idx_ifc) > 0 and is_meped_var_supposed_to_be_empty_during_IFC(vname):
                            # Fill based on new master's _FillValue, noting that mep_IFC_on is only set every 16 time steps, so we Fill the full range +/- 16 time steps. Multiple IFCs in a day will result in over FillValue'ing the data but that's unlikely.
                            dt_ifc = nc4.num2date(nc_day['time'][idx_ifc], units=nc_all['time'].units)
                            logger.info( my_name + ': Filling %s during MEPED IFC for time range %s - %s.' % (vname, str(dt_ifc[0]), str(dt_ifc[-1]) ) )
                            var_day_fixed[ idx_ifc[0] - 16 : idx_ifc[-1] + 16 ] = nc_all[ vname ]._FillValue

                        if not vname in t_data:
                            # TODO: Is using [:] the best way to ensure all Data and Attribute values are captured? What about things that contribute to masking?
                            t_data[ vname ] = var_day_fixed[:]
                        else:
                            t_data[ vname ] = np.append( t_data[ vname ], var_day_fixed )


            # Copy memory structure to NC4 aggregation object:
            logger.debug( my_name+': Filling NC4 object with data...' )
            for vname in nc_all.variables:
                nc_all.variables[ vname ][:] = t_data[ vname ]

            nc_all.close()


        # Move NC4 file repo and return NC4 object

        logger.info( my_name+': Returning handle to %s.' % fn_agg )
        nc_all = nc4.Dataset( fn_agg )
        return( nc_all )

    except Exception as e:
        logger.error( my_name+': Exception Caught!' )
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # logger.error( exc_type, fname, exc_tb.tb_lineno )
        logger.error( traceback.format_exc() )

        return( None )


def get_data_raw( sat_name, dt_start, dt_end, dir_user_data=None, clobber=False ):
    """ Returns aggregated POES "raw"
    Look Order:
        - Current Directory
        - User chosen root
        - User's default
        - NGDC SatDat URL (Future)

    :param sat_name:      One of {'m01', 'm02', 'm03', 'n15', ..., 'n19' }.
    :param dt_start:      Start datetime.
    :param dt_end:        End datetime.
    :param dir_user_root: String of a single absolute or relative user path (OPTIONAL).
    :return:              NetCDF object.
    """
    my_name = 'get_data_raw'

    try:
        #####
        # Config:
        #####
        fn_master = config['fn_master_raw']

        # Search path order: Optional caller choice, OR preconfigured paths.
        if dir_user_data:
            dir_root_list = [dir_user_data]
        else:
            dir_root_list = config['dir_data_raw_search_paths']

        #####
        # Data: return Existing handle or create New aggregate
        #####

        # If file exists, skip aggregation:
        yyyymmdd1 = '%04d%02d%02d' % ( dt_start.year, dt_start.month, dt_start.day )
        yyyymmdd2 = '%04d%02d%02d' % ( dt_end.year,   dt_end.month,   dt_end.day   )
        fn_agg = config['dir_aggregates'] + '/poes_%s_%s-%s_raw.nc' % ( sat_name, yyyymmdd1, yyyymmdd2 )
        if not clobber and os.path.isfile( fn_agg ):
            logger.info( my_name+': Found existing aggregate, using %s.' % fn_agg )

        else:
            # Clobber existing aggregate if desired
            if clobber and os.path.isfile( fn_agg ):
                logger.info( my_name+': Clobber is on, removing existing aggregate %s.' % fn_agg )
                os.unlink( fn_agg )

            ''' List of Files to Ingest '''
            # File name pattern we need to look for (data are organized as day files):
            n_days = ( ( dtm.timedelta( days=1 ) \
                + dtm.datetime( year=dt_end.year,   month=dt_end.month,   day=dt_end.day   ) \
                - dtm.datetime( year=dt_start.year, month=dt_start.month, day=dt_start.day ) ) ).days
            file_pattern_list = []
            for i_day in np.arange( n_days ):
                t_dt = dt_start + dtm.timedelta( days=i_day )
                file_pattern_list.append( 'poes_'+sat_name+'_%04d%02d%02d_raw.nc' % ( t_dt.year, t_dt.month, t_dt.day ) )

            logger.debug( my_name+': Searching for files to aggregate: ' + '\n\t'.join( dir_root_list ) )
            for dir_root in dir_root_list:
                logger.debug( my_name+': Searching %s' % dir_root )
                fn_list = get_file_names( dir_root, file_pattern_list )
                logger.debug( my_name+': Found these %d files in \'%s\':' % ( len( fn_list ), dir_root ) )
                logger.debug( fn_list )
                # Once we have a list we'll move on (avoids finding same file names in multiple places).
                # Warning: If multiple file repositories exist and first one is incomplete then so will be the returned NetCDF4 object.
                if fn_list: break

            logger.debug( 'Found %d files to aggregate.' % len( fn_list ) )
            if len( fn_list ) == 0: return None

            ''' Ingest Data '''
            # Copy Master NetCDF to new file in '/tmp' or User choice:
            shutil.copy( fn_master, fn_agg )

            # Ingest each day file to Temporary Memory dictionary:
            nc_all = nc4.Dataset( fn_agg, 'r+' )
            t_data = {}    # Temp mem dictionary.
            for fn in fn_list:
                with nc4.Dataset( fn, 'r' ) as nc_day:
                    logger.debug( my_name+': Ingesting %s.' % fn )

                    for vname in nc_all.variables:
                        var_day = nc_day.variables[ vname ]

                        # TODO: Fix POES/Metop Master netCDF template and existing dataset to ensure 'valid_range' attributes are indeed 2-element arrays (hint use 'valid_min' if needed).
                        # TODO: 'time' variable's valid_range is wrongly set to [0., 0.] resultingin a completely masked array (all filled).
                        if vname == 'time':
                            logger.error( my_name+': Variable "time" has broken attribute valid_range set to [0.,0.] (results in 100% masked array), disabling Auto Masking to enable reading/copying.' )
                            var_day.set_auto_mask(False)

                        # TODO: Most variables have 'valid_range' set to a scalar (should be a vector).
                        if hasattr(var_day, 'valid_range') and not isinstance(var_day.valid_range, np.ndarray):
                            logger.error( my_name+': Variable %s has broken attribute valid_range (scalar, should be 2-element), disabling Auto Masking to enable reading/copying.' % (vname) )
                            var_day.set_auto_mask(False)

                        if not vname in t_data:
                            # TODO: Is using [:] the best way to ensure all Data and Attribute values are captured? What about things that contribute to masking?
                            t_data[ vname ] = var_day[:]
                        else:
                            t_data[ vname ] = np.append( t_data[ vname ], var_day )


            # Copy memory structure to NC4 aggregation object:
            logger.debug( my_name+': Filling NC4 object with data...' )
            for vname in nc_all.variables:
                nc_all.variables[ vname ][:] = t_data[ vname ]

            nc_all.close()


        # Return NC4 object:
        logger.info( my_name+': Returning handle to %s.' % fn_agg )
        nc_all = nc4.Dataset( fn_agg )
        return( nc_all )

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error( my_name+': Exception Caught!' )
        logger.error( exc_type, fname, exc_tb.tb_lineno )
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


def get_data_processed_swpc( sat_name, dt_start, dt_end, dir_user_data=None ):
    return get_data_l2_swpc_avg( sat_name, dt_start, dt_end, dir_user_data )

def get_data_l2_swpc_avg( sat_name, dt_start, dt_end, dir_user_data=None ):
    """ Returns aggregated POES processed CDF.
    Look Order:
        - Current Directory
        - User chosen root
        - Default
        - NGDC SatDat URL (Future)

    :param sat_name:      One of {'m01', 'm02', 'm03', 'n15', ..., 'n19' }.
    :param dt_start:      Start datetime.
    :param dt_end:        End datetime.
    :param dir_user_root: String of a single absolute or relative user path (OPTIONAL).
    :return:              CDF object.
    """
    my_name = 'get_data_processed_swpc'

    try:
        #####
        # Config:
        #####

        # Search path order: Optional caller choice, then preconfigured paths.
        dir_root_list = []
        if dir_user_data: dir_root_list.append(dir_user_data)
        dir_root_list.append(config['dir_data_l2_swpc_avg_search_paths'])

        # TODO: Check the inputs.
        logger.info( my_name+': Starting: %s, %s to %s, %s.' % (sat_name, dt_start, dt_end, dir_user_root) )

        #####
        # Data: return Existing handle or create New aggregate
        #####

        # Imports
        import random, string, subprocess
        from spacepy import pycdf

        # If file exists, skip aggregation:
        yyyymmdd1 = '%04d%02d%02d' % ( dt_start.year, dt_start.month, dt_start.day )
        yyyymmdd2 = '%04d%02d%02d' % ( dt_end.year,   dt_end.month,   dt_end.day   )
        fn_agg = config['dir_aggregates'] + '/tmp/poes_%s_%s-%s.cdf' % ( sat_name, yyyymmdd1, yyyymmdd2 )
        if os.path.isfile( fn_agg ):
            logger.debug( my_name+': Found existing aggregate, using %s.' % fn_agg )

        else:
            ''' List of Files to Ingest '''
            # File name pattern we need to look for (data are organized as day files):
            n_days = ( ( dtm.timedelta( days=1 ) \
                + dtm.datetime( year=dt_end.year,   month=dt_end.month,   day=dt_end.day   ) \
                - dtm.datetime( year=dt_start.year, month=dt_start.month, day=dt_start.day ) ) ).days
            file_pattern_list = []
            for i_day in np.arange( n_days ):
                t_dt = dt_start + dtm.timedelta( days=i_day )
                file_pattern_list.append( 'poes_'+sat_name+'_%04d%02d%02d.cdf' % ( t_dt.year, t_dt.month, t_dt.day ) )

            logger.debug( my_name+': Searching in this order: ' + '\n\t'.join( dir_root_list ) )
            for dir_root in dir_root_list:
                logger.debug( my_name+': Searching %s' % dir_root )
                fn_list = get_file_names( dir_root, file_pattern_list )
                logger.debug( my_name+': Found these %d files in \'%s\':' % ( len( fn_list ), dir_root ) )
                logger.debug( fn_list )
                # Once we have a list we'll move on (avoids finding same file names in multiple places).
                # Warning: If multiple file repositories exist and first one is incomplete then so will be the returned NetCDF4 object.
                if fn_list: break

            logger.debug( 'Found %d files to aggregate.' % len( fn_list ) )
            if len( fn_list ) == 0: return None

            random_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
            fn_cdfmerge = '/tmp/flist_cdfmerge_%s.txt' % random_id
            with open( fn_cdfmerge, 'w' ) as fp:
                for fname in fn_list:
                    fp.write('%s\n' % fname )
                fp.write( fn_agg[:-3] )

            cmd = ["/Applications/cdf36_1-dist/bin/cdfmerge", "-nolog", "-noprefix", "-dataonly", "-file", fn_cdfmerge ]
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

def plot_stack_ted_raw( sat_name, dt_range, dir_user_data=None, dpi=600, dir_output=None ):
    my_name = 'plot_stack_ted_raw'

    #####
    # Setup Data
    #####
    ncraw = get_data_raw( sat_name, dt_range[0], dt_range[1], dir_user_data=dir_user_data )
    data = {}
    data['sat_name'] = satID_to_satname(ncraw['satID'][:].filled()[0])
    assert( sat_name == data['sat_name'] )
    time_msec = ncraw['time'][:].filled()
    data['time_dt'] = nc4.num2date(time_msec, units=ncraw['time'].units)
    if not dt_range:
        dt_range = [data['time_dt'][0], data['time_dt'][-1]]

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
    title = '%s TED Raw - %s to %s' % (sat_name, dt_range[0], dt_range[1])
    if dir_output:
        dt_range_str = '%d%02d%02d_%02d%02d-%d%02d%02d_%02d%02d' % \
                       (dt_range[0].year, dt_range[0].month, dt_range[0].day, dt_range[0].hour, dt_range[0].minute,
                        dt_range[1].year, dt_range[1].month, dt_range[1].day, dt_range[1].hour, dt_range[1].minute)
        file_plot = dir_output + '/' + 'poes_%s_%s_ted_raw.png' % (sat_name, dt_range_str)
    else:
        file_plot = None

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



def plot_stack_meped_raw( sat_name, dt_range, dir_output=None, dpi=600, dir_user_data=None ):
    my_name = 'plot_stack_meped_raw'

    #####
    # Setup Data
    #####
    ncraw = get_data_raw( sat_name, dt_range[0], dt_range[1], dir_user_data=dir_user_data )
    data = {}
    data['sat_name'] = satID_to_satname(ncraw['satID'][:].filled()[0])
    assert( sat_name == data['sat_name'] )
    time_msec = ncraw['time'][:].filled()
    data['time_dt'] = nc4.num2date(time_msec, units=ncraw['time'].units)
    if not dt_range:
        dt_range = [data['time_dt'][0], data['time_dt'][-1]]

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


def plot_map( sat_list, level, var_list, dt_start, dt_end, delta_lat=1., delta_lon=1.,
    log10=True, empty_cell_value=None, cb_range_list=None, dir_user_data=None, dpi=600, dir_output=None ):
    '''
    Averages and plots SEM2 observations from multiple satellites and variables on a "Mill" projection map.
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
                nc_ts = get_data_raw( satname, dt_start, dt_end, dir_user_data=dir_user_data )
            elif level is 'l2':
                nc_ts = get_data_l2( satname, dt_start, dt_end, dir_user_data=dir_user_data )
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


def _plot_map( data, lats, lons, cb_range=[None, None], variable_label=None, title=None, dpi=600, file_plot=None ):
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
    from mpl_toolkits.basemap import Basemap, cm

    # New figure
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # Mercator map with continents and grid lines
    m = Basemap(llcrnrlon=-180.,llcrnrlat=-85,urcrnrlon=180.,urcrnrlat=85,\
                rsphere=(6378137.00,6356752.3142),\
                resolution='l',projection='mill',\
                lat_0=0.,lon_0=0.,lat_ts=30.)
    # Coastlines
    m.drawcoastlines(color=cont_color)      # (red,green,blue,alpha)
    # Continents
    # m.fillcontinents( color=cont_color, lake_color=cont_color )
    # Background color
    m.drawmapboundary( fill_color=bkg_color )
    # Label latitudes (but don't actually draw the lines)
    m.drawparallels( np.arange(-80,90,20),labels=[True, True, False, True], linewidth=0 )
    # Label longitudes (but don't actually draw the lines)
    m.drawmeridians( np.arange(-180,180,30),labels=[True, True, False, True], linewidth=0 )

    # Add Data Image and make sure it's on top (zorder)
    # Prepare Lats and Lons for mesh grid
    lons_grid, lats_grid = np.meshgrid(lons, lats)
    if cb_range is None:
        cb_range = [None, None]
    im1 = m.pcolormesh(lons_grid, lats_grid, data, shading='flat', cmap=plt.cm.jet, latlon=True, vmin=cb_range[0], vmax=cb_range[1] )#, zorder=10)

    # Colorbar
    cb = m.colorbar(im1, "bottom", size="5%", pad="5%")
    cb.set_label( variable_label )

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


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                     MAIN                                      "
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
' This allows us to test and provide examples. '
if __name__ == '__main__':
    import datetime as dtm
    import poes_utils as pu

    t1 = dtm.datetime.utcnow()

    print( '\n++++++++++ Test: SWPC Aggregation START ++++++++++' )
    poes_swpc = pu.get_data_processed_swpc( 'm02', dtm.datetime(2014,1,1), dtm.datetime(2014,12,31) )
    print( 'From %s to %s.' % ( poes_swpc['EPOCH'][0], poes_swpc['EPOCH'][-1] ) )
    poes_swpc.close()

    print( '\n++++++++++ Test: NGDC Aggregation START ++++++++++' )
    poes = pu.get_data_processed( 'm02', dtm.datetime(2014,1,1), dtm.datetime(2014,12,31) )
    print( 'From %s to %s.' % ( unix_time_ms_to_datetime( poes.variables['time'][0] ), unix_time_ms_to_datetime( poes.variables['time'][-1] ) ) )
    poes.close()    # At the moment, this causes the first write to disk of the object which is about 5 seconds / day of data.

    t2 = dtm.datetime.utcnow()
    print( 'Seconds elapsed = %d' % (t2 - t1).total_seconds() )
    print( '\n---------- Test: Aggregation DONE  ----------' )
