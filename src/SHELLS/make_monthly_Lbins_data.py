import poes_utils as pu
import datetime as dt
import netCDF4 as nc4
import numpy as np
import math
from scipy import stats
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy.ma as ma
import timeit
import data_utils as du
import os

def valid_date(s):
    '''------------------------------------------------------------------
    PURPOSE: To check that a valid date is entered as an input
    :params s (str) a date in the format Y-m-d or Y-m-d H:M:S '''

    try:
        test = dt.datetime.strptime(s, "%Y-%m-%d")
        return test
    except:
        pass
    try:
        test = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return test
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def make_monthly_Lbin_data(sdate,edate,sat,config,odirec,vars,omvars,mplots):
    '''
    PURPOSE: To create monthly Lbin files of the POES/MetOp electron data that are faster to work with
       then the full NGDC netcdf files

    INPUTS:
    :param: sdate - (datetime) start date to create files
    :param: edate - (datetime) end date to create files
    :param: sat (str) - satellites name i.e. 'm02'
    :param: config (str) - the config file name (wiht path) or the top directory of POES data
    :param: odirec (str) - directory to put the Lbin data defaul (./Lbindata)
    :param: vars list(str) - variables to get, i.e. ['mep_ele_tel90_flux_e1','mep_ele_tel90_flux_e2']
    :param: omvars list(str) - omni variables to get, i.e. ['Kp*10', 'Dst']
    :param: mplots (0 or 1) – flag to make quality check plots or not

    OUTPUTS:
    Creates files called /odirec/yyyy/poes_Lbin_sat_yyyymm.nc

    '''

    # Check if the output directory exists and create if it doesn't
    if not os.path.isdir(odirec):
        os.mkdir(odirec)

    # Todo make this an input
    # This goes from 1 to 8.25 because to make the last bin average from 8-8.25
    # you need the extra edge
    Lbins = np.arange(1,8.5,.25)
    # configfile = '/Users/janet/PycharmProjects/SHELLS/config_poes_utils_example.ini' # config for poes_utils

    # These values are required for the output files
    loc_vars = ['time','L_IGRF','lat','lon','MLT']

    # This is the location variables plus the user requested variables and NS direction
    allvars = loc_vars +list(vars) +['NS'] # add on the north / south direction

    # Loop through the requested times and create monthly files
    while sdate<=edate:

        # mdate is the first date of the next month
        mdate = (sdate +dt.timedelta(days = 31)).replace(day=1,hour=0,minute=0,second=0)

        #------------ make the monthly nc output file ---------------------------
        yyyymm = '%04d%02d' % (sdate.year, sdate.month)

        # Already made sure odirec exists
        # Make sure odirec/year/ exists
        if not os.path.isdir(odirec+'/'+str(sdate.year).zfill(4)):
            os.mkdir(odirec+'/'+str(sdate.year).zfill(4))

        # output file name
        fn = odirec +'/'+str(sdate.year).zfill(4)+'/poes_Lbin_'+sat + '_'+ yyyymm +'.nc'

        sat_data = nc4.Dataset(fn, 'w')

        # The time column, time_med, will have the median time of each pass
        sat_data.createDimension('time_med', None)
        sat_data.createDimension('Lbin', len(Lbins)-1)

        # Create the dimension variables time and L
        sat_data.createVariable('time_med', np.uint64, ('time_med'))
        sat_data.createVariable('Lbin', np.float64, ('Lbin'))

        # Lbins goes from 1-8.25 to actually have an end bound the averaging
        # but the output data length is one less than that
        sat_data.variables['Lbin'] = Lbins[0:-1] # 1 up to and including L=8

        # Create all the variables
        for var in allvars:
            sat_data.createVariable(var, np.float64, ('time_med','Lbin'))

        # Create omni variables
        for var in omvars:
            sat_data.createVariable(var, np.float64, ('time_med'))

        time_idx = 0

        # Step through each day in the month and bin the data
        while sdate<mdate:
            # Get an extra day of data to deal with L passes that go over a day boundary. Always
            # allow the pass to end and not include incomplete passes at the beginning so that they are not
            # repeated
            print('Working on ', sdate)

            # Don't get 'NS' because that isn't in the datafile its created later
            data = pu.get_data_processed(config,sat,sdate,sdate+dt.timedelta(days=1),clobber=True,all=True,
                        vars=allvars[0:-1])
            # NOTE: get_data_processed will make a temp file in the tmp directory passed in the configfile or in the
            # top directory passed with the form poes_sat_YYYYMMDD-YYYYMMDD_proc
            # If you don't want a bunch of tmp files to build up then remove them after closing the file

            if data !=None:
                # There are some missing data file so you may just gt the second day
                # And on 2015-06-16 the data file is so short there's not a complete pass. So require
                # at least 30 minutes of data
                if len(np.where(data['time'][:]<(pu.unixtime(sdate + dt.timedelta(days=1))*1000.0))[0])>900:
                    # passes is a list of the L pass numbers and breaks is the indices between passes
                    passes, breaks = pu.getLpass(data['L_IGRF'][:])
                    # Check this with a plot
                    #plt.plot(np.arange(0,len(passes)),data['L_IGRF'])
                    #plt.plot(breaks,data['L_IGRF'][breaks],'r*')

                    # Add directional info (North/South) because it is an easy way to separate the data by MLT
                    # And the pitch angles are slightly different depending on the direction
                    NS = np.zeros((len(data['lat'])), dtype=float)
                    NS[0:-1] = np.diff(data['lat'])
                    # Repeat the last val
                    NS[-1] = NS[-2]

                    NS[np.where(NS >= 0)] = 0 # Northbound
                    NS[np.where(NS < 0)] = 1 # Southbound

                    # Find the last index needed to have complete passes.
                    # This is the last passnum after the end of the first day because
                    # we don't want partial passes at the end and beginning of the month

                    goodind = np.where(data['time'][:]<(pu.unixtime(sdate + dt.timedelta(days=1))*1000.0))[0]
                    # If the next day is missing then goodind will be []
                    if len(goodind)>0:
                        last_pass = passes[goodind[-1]] # pass number of the last point we want
                        last_ones = np.where(passes<=last_pass)[0]
                        last_ind = last_ones[-1]+1 # index of the last pass to include
                    else:
                        last_ind = len(data['time'][:])-1

                    # The binning is 2-3 times as fast if you don't include the extra day
                    # so create a shorter bit to bin
                    bdata = []

                    # This is the data to be L binned
                    for var in allvars[0:-1]:
                        bdata.append(list(data[var][0:last_ind].data))

                    pbins = np.arange(0,(passes[last_ind-1]+1)) # add one because bindata needs the edge bins
                    bindat = pu.bindata(bdata,passes[0:last_ind],data['L_IGRF'][0:last_ind],pbins,Lbins)
                    # This has bindat.statistic[cols, pbins, Lbins]

                    # This will be the time associated with each one
                    time_med = np.nanmedian(bindat.statistic[0,:,:],axis=1)
                    test_time = np.where(np.isnan(time_med))[0]
                    # if len(test_time)>0:
                    #   print('Here')

                    # Some time_med has nans because the start only has L values greater than Lbins
                    # i.e. the day starts with L=50 so the binning returns a nan for that pass
                    tinds = np.where(~np.isnan(time_med))[0]

                    # Check if the first pass is complete by checking the first and last Lbin?
                    if np.isnan(bindat.statistic[allvars.index('L_IGRF'),0,0]) | np.isnan(bindat.statistic[allvars.index('L_IGRF'),0,-1]):
                        start_ind = 1
                    else:
                        start_ind = 0
                    end_ind = len(time_med[tinds])

                    # Need to redo the lon and mlt binning because the average needs to be circular
                    # mstart = timeit.default_timer()

                    for col in ['lon','MLT']:
                        if col=='lon':
                            fac = np.pi/180.0
                        else:
                            fac = np.pi/12.0
                        xval = [math.sin(co*fac) for co in data[col][0:last_ind]]
                        yval = [math.cos(co *fac) for co in data[col][0:last_ind]]
                        x = stats.binned_statistic_2d(passes[0:last_ind],data['L_IGRF'][0:last_ind], xval, statistic=np.ma.mean, bins=[pbins,Lbins])
                        y = stats.binned_statistic_2d(passes[0:last_ind],data['L_IGRF'][0:last_ind],  yval, statistic=np.ma.mean, bins=[pbins,Lbins])
                        temp = (1.0/fac)*np.arctan2(x.statistic,y.statistic)
                        temp[temp<0] =temp[temp<0]+2*np.pi*(1/fac) # This is [Lbins,times]
                        sat_data.variables[col][time_idx:(time_idx + end_ind - start_ind), 0:len(Lbins) - 1] = temp[ tinds[start_ind:end_ind], :]
                        # Check that this works
                        #ntime = np.ravel(bindat.statistic[0,:,:])
                        #dats = np.ravel(np.transpose(temp))
                        #inds = np.argsort(np.array(bindat.statistic[0, :, :]), axis=None)
                        #plt.plot(ntime[inds],dats[inds],'.')
                        #plt.plot(data['time'], data['lon'])
                        #print('Here')
                    #print('MLT',timeit.default_timer()-mstart)

                    # Make a North South direction for dividing the data
                    NSbin = stats.binned_statistic_2d(passes[0:last_ind],data['L_IGRF'][0:last_ind], NS[0:last_ind], statistic=np.ma.mean, bins=[pbins, Lbins])
                    # Some end up in the middle
                    NSbin.statistic[NSbin.statistic<.5]=0
                    NSbin.statistic[NSbin.statistic >= .5] = 1
                    # Check this
                    # ntime = np.ravel(bindat.statistic[0,:,:])
                    # dats = np.ravel(NSbin.statistic)
                    # inds = np.argsort(np.array(bindat.statistic[0, :, :]), axis=None)
                    # plt.plot(ntime[inds],dats[inds])


                    # Write it to the netcdf file
                    # first time_med
                    sat_data.variables['time_med'][time_idx:(time_idx+end_ind-start_ind)] = time_med[tinds[start_ind:end_ind]]

                    # Now run through all the vars except NS
                    for co in range(0,len(allvars[0:-1])):
                        sat_data.variables[allvars[co]][time_idx:(time_idx+end_ind-start_ind),0:len(Lbins)-1] = bindat.statistic[co,tinds[start_ind:end_ind],:]
                        # statistic[var,pbin,Lbin]

                    # And the NS one
                    sat_data.variables['NS'][time_idx:(time_idx + end_ind - start_ind), 0:len(Lbins) - 1] = NSbin.statistic[tinds[start_ind:end_ind],:]

                    # Now get the Kp and other hourly omni data
                    Kp = du.get_omni(sdate, sdate+dt.timedelta(days=1), omvars)
                    Kpdate = [dt.datetime(np.int(Kp[co][0]), 1, 1, np.int(Kp[co][2]))+dt.timedelta(days = np.int(Kp[co][1])-1) for co in np.arange(0,len(Kp))]
                    Kptime1 = pu.unixtime(Kpdate)
                    Kptime = [1000*co for co in Kptime1]

                    for omco in range(0,len(omvars)):
                        # First three cols are time ones
                        Kpdata = [Kp[co][3+omco] for co in np.arange(0,len(Kp))]
                        Kpnew = np.interp(time_med[tinds[start_ind:end_ind]],Kptime, Kpdata)
                        Kpfin = np.floor(Kpnew)
                        sat_data.variables[omvars[omco]][time_idx:(time_idx + end_ind - start_ind)] = Kpfin


                    # Now update time_idx
                    time_idx = time_idx + end_ind - start_ind

                    # Get the filename of the tmp file created
                    pname = data.filepath()
                    # Close the files that netcdf file that was just read
                    data.close()
                    # remove the tmp file
                    os.remove(pname)

                    #print(timeit.default_timer()-starttime)
            sdate=sdate+dt.timedelta(days=1)


        # Make plots to check that it looks ok
        if mplots>0:
            xdates = [pu.unix_time_ms_to_datetime(x ) for x in sat_data['time_med'][:]]
            x = mdates.date2num(xdates)
            # Split into norht and south.
            n_inds_N = np.where( (np.nanmean(sat_data['lat'][:],axis=1)>0) & ((np.nanmean(sat_data['NS'][:],axis=1)>.5)))[0]
            n_inds_S = np.where( (np.nanmean(sat_data['lat'][:],axis=1)>0) & ((np.nanmean(sat_data['NS'][:],axis=1)<.5)))[0]
            s_inds_N = np.where( (np.nanmean(sat_data['lat'][:],axis=1)<0) & ((np.nanmean(sat_data['NS'][:],axis=1)>.5)))[0]
            s_inds_S = np.where( (np.nanmean(sat_data['lat'][:],axis=1)<0) & ((np.nanmean(sat_data['NS'][:],axis=1)<.5)))[0]
            for var in allvars[5:-1]:
                if (var.find('ele') != -1) | (var.find('pro') != -1):
                    vmi = np.nanmin(ma.masked_less(np.log10(np.transpose(sat_data[var])),-8))
                    vma = np.nanmax(ma.masked_less(np.log10(np.transpose(sat_data[var])),-8))
                else:
                    vmi = np.nanmin(ma.masked_less((np.transpose(sat_data[var])),-50000))
                    vma = np.nanmax(ma.masked_less((np.transpose(sat_data[var])),-50000))

                plt.set_cmap('jet')
                for aco in range(1,5):
                    if aco==1:
                        inds = n_inds_N
                        ti = ' N lat N-bound'
                    elif aco==2:
                        inds = n_inds_S
                        ti = ' N lat S-bound'
                    elif aco==3:
                        inds = s_inds_N
                        ti = ' S lat N-bound'
                    else:
                        inds = s_inds_S
                        ti = ' S lat S-bound'
                    ax = plt.subplot(4, 1, aco)
                    pcol = var
                    plt.title(pcol +' N lat N-bound')
                    if (var.find('ele') !=-1) | (var.find('pro') !=-1):
                        plt.pcolormesh([x[nco] for nco in inds], Lbins[:-1], ma.masked_less(np.log10(np.transpose(sat_data[var][inds,:])), -8),
                        vmin=vmi, vmax=vma)
                    else:
                        plt.pcolormesh([x[nco] for nco in inds], Lbins[:-1], ma.masked_less((np.transpose(sat_data[var][inds,:])), -8),
                        vmin=vmi, vmax=vma)
                    date_format = mdates.DateFormatter('%y-%m-%d')
                    ax.xaxis.set_major_formatter(date_format)
                    plt.colorbar()

                plt.tight_layout()
                fname = odirec+str((sdate-dt.timedelta(days = 1)).year).zfill(4)+'/poes_Lbin_'+sat+'_'+yyyymm +'_'+var+'.png'
                plt.savefig(fname)
                plt.close()
        sat_data.close()

        ('Done with '+yyyymm)

if __name__ == '__main__':
    import argparse
    '''
    PURPOSE: To create monthly data files of POES/MetOp data binned by L for each pass
    USAGE:
    -s 2013-01-01 -e 2013-01-01 -sat m02 -d /Users/PycharmProjects/configfile.ini 
        -od /Users/PycharmProjects/ -v mep_ele_tel90_flux_e1 mep_ele_tel90_flux_e2 -ov Kp*10 -pt
    '''

    parser = argparse.ArgumentParser('This creates new datafiles binned by L')
    #
    parser.add_argument('-s', "--startdate",
                        help="The Start Date - format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS ",
                        required=True,
                        default = None,
                        type=valid_date)
    parser.add_argument('-e', "--enddate",
                        help="The Start Date - format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS ",
                        required=True,
                        default = None,
                        type=valid_date)
    parser.add_argument('-sat', "--satname",
                        help="A name of satellite data to get (i.e. -sat n15 or -sat n16 ",
                        type=str, required=True)
    parser.add_argument('-d', "--dataloc",
                        help="The configfile with data locations or a top level directory of data",
                        required=False, default=os.getcwd())
    parser.add_argument('-od', "--odirec",
                        help="The output directory of data",
                        required=False, default=os.getcwd()+'/Lbindata/')
    parser.add_argument('-v', "--vars",
                        help="data variables to get",
                        required=False, default=['mep_ele_tel90_flux_e1','mep_ele_tel90_flux_e2',
                              'mep_ele_tel90_flux_e3','mep_ele_tel90_flux_e4','Btot_sat','meped_alpha_90_sat'],nargs='+')
    parser.add_argument('-ov', "--omvars",
                        help="omni data variables to get",
                        required=False, nargs='+', default = ['Kp*10'])
    parser.add_argument('-pt', "--plots", action='store_true', default=0)

    args = parser.parse_args()

    x = make_monthly_Lbin_data(args.startdate,args.enddate,args.satname,args.dataloc,args.odirec,args.vars,args.omvars,args.plots)