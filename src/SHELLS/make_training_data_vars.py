import datetime as dt
import poes_utils as pu
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import netCDF4 as nc4
import timeit
import numpy.ma as ma
try:
   import cPickle as pickle
except:
   import pickle

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

def make_training_data_vars(sdate,edate,satlist, varlist, cdf_dir,Lbin_dir, neur_dir, reflon,syear_all,eyear_all):
    '''
    PURPOSE: To create a datafile of electron flux mapped to one longitude with SAR to be used
    for developing the SHELLS neural network

    INPUTS:
    :param: sdate(datetime)-    time to start processing data
    :param: edate(datetime)-    time to end processing data
    :param: satlist(list(str))- i.e. ['n15','n18','n19','m01','m02']
    :param: varlist(list(str))- variables to process i.e. ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4']
    :param: cdf_dir(str)        directory where the cdf files are
    :param: Lbin_dir(str)       directory where the Lbin data files are
    :param: neur_dir(str)       directory for the output files
    :param: reflon(int)         E longitude to map to (degrees)
    :param: syear_all (int)     The start year of the accumulated cdf file
    :param: eyear_all           The end year of the accumulated cdf file

    OUTPUTS: monthly pickle files with the SAR modified data to be used by the SHELLS neural network

    USAGE(command line)
    python make_training_data.py -s 2013-01-01 -e 2013-05-01 -sats n15 n18 n19 m01 m02 -cd ./cdfdata/
       -ld ./Lbindat/ -nd ./neural_data/ -l 20 -sy 2015 -ey 2018:
    '''

    # These are the electron flux variables that have the percentile data
    evars = varlist

    svars = list()
    for var in evars:
    # These have the flux for each percentile
        svars.append(var+'_sar')

    # This is expected #orbits per day * Lpasses per orbit* # sats *days
    # that is used to estimate the len of array needed for make month long files
    flen = 20*4*len(satlist)*(31)

    Lbins = np.arange(1, 8.25, .25) # These are the Lbins
    cols = list()

    # make a list of columns for the output pickle file
    # The file will have columns with the SAR flux for each variable and Lbin
    for ecols in np.arange(0,len(varlist)):
        for lcols in np.arange(len(Lbins)):
            cols.append(varlist[ecols]+ ' '+str(Lbins[lcols]/4))

    # fin_dat will be  fluxE1 all Lbins, flux E2 all Lbins, flux e3 Lbins, flux e4 at all Lbins
    # For each L pass
    # And then a time vector with the time at the midpoint of each L pass

    # This sets everything to ref longitude
    ref_ind = int(np.floor(reflon/10))

    # All data will be referenced back to m02
    satref = 'm02'

    while sdate <edate:
        indco = 0  # This initializes the index of the final data
        # These will hold a month of data from all sats
        sar_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float) + -1  # The final data
        nosar_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float) + -1  # Data with no sar for comparison
        lat_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float)  # lat for plotting
        lon_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float)  # lon for plotting
        per_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float)  # percentage level for plotting
        way_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float) + -1  # direction of the sat
        full_lat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float)  # The actual lat not binned
        sat_dat = np.zeros((flen, len(varlist) * len(Lbins)), dtype=float) + -1  # index of the satellite

        sar_time = list()  # The time of the pass

        # Loop through all the satellites in satlist
        for sco in range(0,len(satlist)):
            # Read the SAR file for each sat
            sat = satlist[sco]
            print('Working on '+ satlist[sco]+sdate.strftime("%m/%d/%Y"))

            # Read in the monthly Lbin data for that sat
            datafile  = Lbin_dir+ str(sdate.year) + '/poes_Lbin_' + satlist[sco]+'_'+str(sdate.year) + str(sdate.month).zfill(2)+'.nc'
            if os.path.exists(datafile):
                with nc4.Dataset(datafile, 'r') as data:
                    # The data is stored as passes X Lbin for each var
                    # step through each variable
                    for eco in range(0, len(evars)):
                        # Cdfs for current sat
                        satsarfile = cdf_dir+'/'+sat+'/poes_cdf_' + sat + '_' + str(syear_all).zfill(4)+'_' \
                                        +str(eyear_all).zfill(4)+evars[eco]+'.nc'
                        sar_sat= nc4.Dataset(satsarfile,'r')
                        # cdfs for reference sat
                        srefile = cdf_dir+satref +'/poes_cdf_' + satref + '_' + str(syear_all).zfill(4)+'_' \
                                                +str(eyear_all).zfill(4)+evars[eco]+'.nc'
                        sar_ref = nc4.Dataset(srefile, 'r')

                        # This is the flux per percentile for the sat being processed
                        sar = sar_sat[evars[eco]][:]

                        # This is the flux per percentile for the ref sat
                        sarout = sar_ref[svars[eco]][:]

                        # Get the flux bins for the data being processed
                        fluxbin = np.round(np.log10(data[evars[eco]][:])*10)
                        fluxbin1=fluxbin.astype(int)

                        # Get the hemisphere
                        hemi = data['lat'][:]
                        hemi[hemi >= 0] = 0  # Northern hemi
                        hemi[hemi < 0] = 1  # Southern hemi
                        hemi1=hemi.astype(int)
                        # Get the NS direction
                        NSco = data['NS'][:]
                        NSco1 =NSco.astype(int)
                        lon = np.floor(data['lon'][:]/10)
                        lon1=lon.astype(int)
                        lon1[lon1>35]=0
                        Kp = np.floor(data['Kp*10'][:]/10)
                        Kp1 = Kp.astype(int)
                        # Need to make Kp into a vector
                        Kpvec = np.tile(Kp1,(len(Lbins),1)).T
                        # Need to make an array of Ls
                        Ls = np.zeros((len(data['time_med'][:]),len(Lbins)),dtype = int)

                        for lco in range(0,len(Lbins)):
                            Ls[:,lco] = Ls[:,lco]+lco


                        nan_inds = np.where((fluxbin1 < -10) | (hemi1 < -10) | (lon1 < -10) | (Kpvec < -10) | (NSco1 < -10))

                        # Set these to zero for now so that it is a valid index
                        # but flag it later
                        fluxbin1[nan_inds] = 0
                        hemi1[nan_inds] = 0
                        lon1[nan_inds] = 0
                        NSco1[nan_inds] = 0
                        Kpvec[nan_inds] = 0

                        # Get the percentile that corresponds to each flux for the current sat
                        per1 = sar[hemi1, NSco1, Ls, lon1, Kpvec, fluxbin1]
                        perbin1 = np.round(per1 * 100).astype(int)

                        # In northern some sar dat is nan
                        per_nan = np.where(perbin1<-10)[0]
                        perbin1[per_nan] = 0

                        # Get the flux at the ref satellite for the measured percentile
                        fluxval = sarout[1,1,Ls,ref_ind,Kpvec,perbin1]
                        # Flag the bad values again
                        fluxval[nan_inds] = -1
                        fluxval[per_nan] = -1

                        dlen = len(data['time_med'][:])
                        # Set the output data to the reference value
                        sar_dat[indco:indco + dlen, (eco * len(Lbins)):(eco * len(Lbins))+len(Lbins)] = fluxval
                        # Save the no sar data for comparison
                        nosar_dat[indco:indco + dlen, (eco * len(Lbins)):(eco * len(Lbins))+len(Lbins)] = np.log10(data[evars[eco]][:])

                        lat_dat[indco:indco+dlen,(eco * len(Lbins)):(eco * len(Lbins))+len(Lbins) ] = hemi[:]
                        full_lat[indco:indco+dlen,(eco * len(Lbins)):(eco * len(Lbins))+len(Lbins) ] = data['lat'][:]
                        lon_dat[indco:indco+dlen, (eco * len(Lbins)):(eco * len(Lbins))+len(Lbins)] = data['lon'][:]
                        per_dat[indco:indco+dlen, (eco * len(Lbins)):(eco * len(Lbins))+len(Lbins)] = perbin1
                        way_dat[indco:indco+dlen, (eco * len(Lbins)):(eco * len(Lbins))+len(Lbins)] = data['NS'][:]
                        sat_dat[indco:indco+dlen, (eco * len(Lbins)):(eco * len(Lbins))+len(Lbins)] = sco+1

                    dtimes = pu.unix_time_ms_to_datetime(data['time_med'][:])
                    sar_time.extend(dtimes.tolist())

                    indco=indco+dlen
            else:
                print('No datafile')

        # Now sort the data by time after going through all the sats
        tinds= np.argsort(sar_time)
        new_time = [sar_time[x] for x in tinds]

        # This orders data according to the new sorted time
        # and saves the month file

        new_dat = sar_dat[tinds[0:len(new_time)], :] # The re-ordered and mapped data
        new_nosar_dat = nosar_dat[tinds[0:len(new_time)], :] # The re-order but not mapped data
        temp_dat = 1.0*new_dat

        # Now fill in holes with the last value.
        # This could be a problem if the first row has missing data
        vec_last = 1.0*new_dat[0,:]

        # find the columns where the first record has holes
        # and fill them with the closest value
        mvals = np.where((np.isnan(vec_last)) | (np.isinf(vec_last)) | (vec_last==-1))[0]
        ico = 1
        if len(mvals)>0:
            for mco in mvals:
                fillval = vec_last[mco]
                ico=1
                while ( (fillval<0) & (ico<50) ):
                    fillval = new_dat[ico,mco]
                    ico = ico+1
                vec_last[mco] = fillval

        new_dat[0,:] = vec_last
        for ico in np.arange(1,len(new_time)):
            vec = 1.0*new_dat[ico,:]
            vec[np.where((np.isnan(vec)) | (np.isinf(vec)) | (vec ==-1))] = vec_last[np.where((np.isnan(vec)) | (np.isinf(vec)) | (vec==-1) )]
            new_dat[ico,:] = vec
            vec_last = vec

        # --------------- Plot the monthly data -------------------------
        # Make a list of dates every 5 days for plotting
        day_inds = list()
        date_list = list()
        for days in np.arange(1,30,5):
            dmin = [np.abs( (x-dt.datetime(sdate.year,sdate.month,days)).total_seconds() ) for x in new_time]
            close_ind = dmin.index(min(dmin))
            day_inds.append(close_ind)
            date_list.append(dt.datetime(sdate.year, sdate.month, days).strftime("%m/%d/%Y"))
        #sval = 300
        #lval = 400
        sval = 0
        lval = len(new_time)

        # Make a plot for each variable
        for eco in range(0,len(varlist)):
            fignum = plt.figure(eco+1)
            # plot the mapped data
            plt.subplot(5,1,1)
            im1 = plt.pcolormesh(np.arange(sval,lval), Lbins, np.transpose((new_dat[sval:lval,eco*len(Lbins):(eco+1)*len(Lbins)])), shading='flat',
                              cmap=plt.cm.jet, vmin=0, vmax = 7 )
            plt.title(varlist[eco])
            plt.colorbar()
            #plot the unmapped data
            plt.subplot(5, 1, 2)
            im1 = plt.pcolormesh(np.arange(sval,lval), Lbins, np.transpose((new_nosar_dat[sval:lval,eco*len(Lbins):(eco+1)*len(Lbins)])), shading='flat',
                                   cmap=plt.cm.jet, vmin=0, vmax = 7 )
            #plt.xticks(day_inds, date_list)
            plt.colorbar()

            #plot the NS direction
            plt.subplot(5, 1, 3)
            im1 = plt.pcolormesh(np.arange(sval,lval), Lbins, np.transpose(way_dat[sval:lval,eco*len(Lbins):(eco+1)*len(Lbins)]), shading='flat',
                                   cmap=plt.cm.jet, vmin=0, vmax = 3 )
            #plt.xticks(day_inds, date_list)
            plt.colorbar()

            # Plot the longitude
            plt.subplot(5, 1, 4)
            im1 = plt.pcolormesh(np.arange(sval,lval), Lbins, np.transpose(lon_dat[tinds[sval:lval],eco*len(Lbins):(eco+1)*len(Lbins)]), shading='flat',
                                   cmap=plt.cm.jet, vmin=0, vmax = 360 )
            plt.colorbar()

            # plot the percentile
            plt.subplot(5, 1, 5)
            im1 = plt.pcolormesh(np.arange(sval,lval), Lbins/4, ma.masked_less(np.transpose(per_dat[tinds[sval:lval],eco*len(Lbins):(eco+1)*len(Lbins)]),1), shading='flat',
                                   cmap=plt.cm.jet, vmin=0, vmax = 75 )
            plt.colorbar()
            plt.xticks(day_inds, date_list)

            #plt.savefig( 'neural_data/'+ 'dat' + satlist[0]+varlist[eco]+str(sdate.year) +str(sdate.month).zfill(2) + '.png')
            plt.savefig(neur_dir+ 'allsats_wsarsV5'+varlist[eco]+str(sdate.year) +str(sdate.month).zfill(2) + '.png')
            plt.close(fignum)

            fignum = plt.figure(eco + 10)
            # This figure will compare fluxes at some Lvalues for SAR and no SAR

            Lco = 1
            for Lp in [2,4,6]:
                plt.title(varlist[eco])
                fig = plt.subplot(3,1,Lco)
                stemp = new_dat[sval:lval, eco * len(Lbins)+Lp ]
                ginds = np.where(stemp>0)[0]
                nstemp = temp_dat[sval:lval, eco * len(Lbins) + Lp]
                ttime = [new_time[x] for x in ginds]
                plt.plot(ttime, nstemp[ginds], 'b')  # L=2 SAR
                plt.plot(ttime,stemp[ginds],'r') # L=2 SAR
                #plt.plot(ttime, nstemp[ginds], 'b')  # L=2 SAR
                Lco = Lco+1
                plt.ylim(2,6)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
                plt.gcf().autofmt_xdate()
            plt.savefig(
                    neur_dir + '/allsats_wsarsV5' + varlist[eco] + str(sdate.year) + str(sdate.month).zfill(
                        2) + 'lines.png')
            plt.close(fignum)


        datafile = neur_dir+'/allsats_wsarsV5' + str(sdate.year) + str(sdate.month).zfill(2) + '.p'
        with open(datafile, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([cols, new_time, new_dat], f)


if __name__ == "__main__":
    import argparse

    '''
    PURPOSE: To create a datafile of electron flux mapped to one longitude with SAR to be used
    for developing the SHELLS neural network

    INPUTS:
    :param: startdate - time to start processing data (ex 2013-01-01)
    :param: enddate -   time to end processing data (ex 2014-01-01)
    :param: sats -      i.e. n15 n18 n19 m01 m02'
    :param: vars-       variables to process i.e. mep_ele_tel90_flux_e1 mep_ele_tel90_flux_e2
                                mep_ele_tel90_flux_e3 mep_ele_tel90_flux_e4
    :param: cdfloc     directory where the cdf files are
    :param: Lbinloc    directory where the Lbin data files are
    :param: neurloc    directory for the output files
    :param: l     longitude bin to map to

    OUTPUTS: monthly pickle files with the SAR modified data to be used by the SHELLS neural network
    
    USAGE(command line)
    python make_training_data.py -s 2013-01-01 -e 2013-05-01 -sats n15 n18 n19 m01 m02 -cd ./cdfdata
       -ld ./Lbindat -nd ./neural_data -r 20:
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
    parser.add_argument('-sats', "--satlist",
                        help="A list of satellite data to get (i.e. -sat n15 n18) ",
                        required=False,
                        default = ['n15','n18','n19','m01','m02'],nargs='+')
    parser.add_argument('-v', "--vars",
                        help="data variables to use",
                        required=False, default=['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                                 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4'], nargs='+')
    parser.add_argument('-cd', "--cdfloc",
                        help="The location of the cdf data",
                        required=False, default=os.getcwd() + '/cdfdata/')
    parser.add_argument('-ld', "--Lbinloc",
                        help="The location of the Lbin data",
                        required=False, default=os.getcwd() + '/Lbindata/')
    parser.add_argument('-nd', "--neurloc",
                        help="The output directory of data",
                        required=False, default=os.getcwd() + '/neural_data/')
    parser.add_argument('-l', "--reflon",
                        help="longitude to map to",
                        required=False,
                        default = 20,
                        type=int)
    parser.add_argument('-sy', "--startyear",
                        help="start year for the cdf file",
                        required=False,
                        default = 2014,
                        type=int)
    parser.add_argument('-ey', "--endyear",
                        help="start year for the cdf file",
                        required=False,
                        default = 2018,
                        type=int)

    args = parser.parse_args()

    x = make_training_data_vars(args.startdate,args.enddate,args.satlist, args.vars, args.cdfloc, args.Lbinloc,
                           args.neurloc, args.reflon, args.startyear, args.endyear)