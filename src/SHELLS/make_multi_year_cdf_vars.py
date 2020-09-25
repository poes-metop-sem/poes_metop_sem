import os
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import copy
import scipy.ndimage
import scipy as sp

def make_multi_year_cdf_vars(syear_all, eyear_all, sat, dataloc,  evars,ofile,plots):
    '''
    PURPOSE: This program takes the yearly cdf files and turns them into total cdfs over many years

    :param: syear_all (int) - YYYY the start year to combine cdf data
    :param: eyear_all (int) - YYYY the end year to combine cdf data (will include this year)
    :param: sat (str) - the satellite name i.e. 'm02'
    :param: dataloc (str) - the directory of the yearly cdf files
    :param: evars (list(str))- the variables to create multi-year cdfs for (must be in the yearly files)
    :param: ofile(str) - a string to add onto the output file to identify it, i.e. could be 'V2'
    :param: plots(0 or 1) - 0 do not make plots (makes median plots and line plots of L vs lon for each variable/Kp)

    OUTPUTS: Creates a multi-year cdf file in dataloc/sat/ called poes_cdf_sat_YYYY_YYYY_variable_ofile.nc
            NOTE: Files are created for each variable so the file is not so huge

    USAGE (from command line):
    python make_multi_year_cdf -s 2014 -e 2019 -sat m02 -vars mep_ele_tel90_flux_e1 mep_ele_tel90_flux_e2
                mep_ele_tel90_flux_e3 mep_ele_tel90_flux_e4 meped_alpha_90_sat Btot_sat -d ./cdfdata/ -o V2 -pt

    USAGE (as a function):
    import make_multi_year_cdf_vars as mcdf
    mcdf.make_multi_year_cdf_vars(2014, 2019, 'm02','./cdfdata/',['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                                 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4',
                                                 'meped_alpha_90_sat','Btot_sat'],'V2',0)

    The two examples above will create cumulative distribution functions of data from 2014-2019 for the m02 satellite
    and the 6 variables listed using the yearly cdf files in the directory ./cdfdata. The commands will create netcdf
    files containing the cdf data for each variable with 'V2' appended on the end of the file name and no quality check
    plots.
    '''

    # These need to be included with in the yearly cdf files and will also be in the output files
    loc_vars = ['time', 'L_IGRF', 'lat', 'lon', 'MLT','NS','Kp*10']

    # These variables are to store the total number of points in each bin
    nvars = list()
    for var in evars:
       nvars.append(var+'_n')

    # These variables will hold the flux or value as a function of percentile from 0-1
    svars = list()
    for var in evars:
        svars.append(var+'_sar')

    # directory of the cdf files
    cdf_direc = dataloc

    syear = syear_all   # start year to use
    eyear = eyear_all   # end year to use

    # The base name of the output files to be created
    # The variable name and ofil will be added to this
    # ex file cdf_direc/m02/poes_cdf_m02_2013_2015_variable_V2.nc
    fnbase = cdf_direc + sat + '/poes_cdf_' + sat + '_' + str(syear_all).zfill(4) + '_' \
            + str(eyear_all).zfill(4)

    # These are temporary dictionaries to sum up the yearly data
    tempcdf = {}  # yearly cdf data (percentiles)
    n = {}  # yearly total #
    tempflux = {}  # yearly flux for each percentile

    allcdf = {} # combined cdf data
    alln = {} # combined n
    allflux = {} # combined data for each percentile

    dims = {} # dimension data i.e L etc
    var_types = {} # types for each variable
    var_dims = {}  # types for each variable


    #-------------------- Loop each year and combine the data ------------------
    while syear<=eyear:
        # The yearly file with cdf data fo open
        fn = cdf_direc+ sat+'/poes_cdf_'+sat + '_'+ str(syear).zfill(4) +'.nc'

        # Open the yearly dataset
        dsin = nc4.Dataset(fn,'r')

        if syear == syear_all:
            # If it is the first year then get some info we need once the files are closed
            dim_names = list(dsin.dimensions.keys())
            for dname in dim_names:
                dims[dname] = dsin[dname][:]
            for var in evars+ svars + nvars+ dim_names:
                var_types[var] = dsin.variables[var].dtype
                var_dims[var] = dsin.variables[var].dimensions

            print('Here')


        # Get the dimnsion variables
        #for dname in dsin.dimensions.keys():
        #    dname
        #if 'Bbins' in dsin.variables.keys():
        #    Bbins = dsin['Bbins'][:]

        # For
        #if 'pbins' in dsin.variables.keys():
        #    pbins = dsin['pbins'][:]
        #Lbins = dsin['Lbin'][:]

        for vco  in range(0,len(evars)):
            tempcdf[evars[vco]] = dsin[evars[vco]][:,:,:,:,:,:] # percentile at each flux, the size of this is different
            # for flux or B or pitch data
            n[nvars[vco]] = dsin[nvars[vco]][:,:,:,:,:] # total number of points
            tempflux[svars[vco]] = dsin[svars[vco]][:, :, :, :, :, :] # flux at each percentile
            # With hemi, NS, L,lon, Kp
            # percentile at each flux,pitch,B X n
            for pco in range(0, len(tempcdf[evars[vco]][0,0,0,0,0,:])):
                tempcdf[evars[vco]][:, :, :, :, :, pco] = tempcdf[evars[vco]][:, :, :, :, :, pco] * n[nvars[vco]]

            # flux,pitch,B X n at each percentile
            for fco in range(0,len(tempflux[svars[vco]][0,0,0,0,0,:])):
                tempflux[svars[vco]][:,:,:,:,:,fco] = tempflux[svars[vco]][:,:,:,:,:,fco]*n[nvars[vco]]

        # Now add the years together
        if syear==syear_all:
            # If its the first year then just copy
            allcdf = copy.deepcopy(tempcdf)
            alln = copy.deepcopy(n)
            allflux = copy.deepcopy(tempflux)
            # This is for writing out the final file
            #dimnames =
        else:
            for vco in range(0, len(evars)):
                allcdf[evars[vco]] = allcdf[evars[vco]]+tempcdf[evars[vco]]
                alln[nvars[vco]] = alln[nvars[vco]] +n[nvars[vco]]
                allflux[svars[vco]] = allflux[svars[vco]]+ tempflux[svars[vco]]
        #sco = syear-syear_all+1

        dsin.close() # close the year netcdf file
        print("Done with ", syear)
        syear = syear +1

    #----------------------------Done summing years, now summ high Kps and smooth --------------
    # Divide by total n in the end and sum Kp 5 and greater for each var
    for vco in range(0, len(evars)):
        # First sum up the big Kps and make that the last one

        for pco in range(0, len(allcdf[evars[vco]][0, 0, 0, 0, 0, :])):
            for bigKps in range(0,5):
                if bigKps ==0:
                    dum = allcdf[evars[vco]][:, :, :, :, 4, pco]
                    dumnum = alln[nvars[vco]][:, :, :, :, 4]
                else:
                    dum = dum +allcdf[evars[vco]][:, :, :, :, 4+bigKps, pco]
                    dumnum = dumnum+ alln[nvars[vco]][:, :, :, :, 4+bigKps]
            allcdf[evars[vco]][:, :, :, :, 9, pco] = dum/dumnum

        for fco in range(0, len(allflux[svars[vco]][0, 0, 0, 0, 0, :])):
            for bigKps in range(0, 5):
                if bigKps == 0:
                    dum = allflux[svars[vco]][:, :, :, :, 4, fco]
                    dumnum = alln[nvars[vco]][:, :, :, :, 4]
                else:
                    dum = dum + allflux[svars[vco]][:, :, :, :, 4 + bigKps, fco]
                    dumnum = dumnum + alln[nvars[vco]][:, :, :, :, 4+bigKps]
            allflux[svars[vco]][:, :, :, :, 9, fco] = dum / dumnum
            alln[nvars[vco]][:, :, :, :, 9] = dumnum

        # Set the last value to the summed Kps
        for Kp in range(0,9):

            for pco in range(0, len(allcdf[evars[vco]][0, 0, 0, 0, 0,:])):
                allcdf[evars[vco]][:, :, :, :, Kp, pco] = allcdf[evars[vco]][:, :, :, :, Kp, pco] / alln[nvars[vco]][:, :, :, :, Kp]


            for fco in range(0, len(allflux[svars[vco]][0, 0, 0, 0, 0, :])):
                allflux[svars[vco]][:, :, :, :, Kp, fco] = allflux[svars[vco]][:, :, :, :, Kp, fco] / alln[nvars[vco]][:, :, :, :, Kp]

        sigma = .6
        # Have to run this over each? allflux is flux for each percentile and allcdf is percentile for each flux
        # Do a gaussin smoothing
        plt.set_cmap('jet')
        for hemi  in range(0,2):
            for NSco in range(0,2):
                for Kp in range(0,10):
                    for pco in range(0, len(allflux[svars[vco]][0, 0, 0, 0, 0, :])):
                        tempdat = allflux[svars[vco]][hemi, NSco, :, :, Kp, pco]
                        dd2 = sp.ndimage.filters.gaussian_filter(tempdat, sigma, mode='constant')

                        dat3 = 0 * copy.copy(tempdat) + 1
                        dat3[np.isnan(tempdat) | np.isinf(tempdat) | (tempdat==0) | (tempdat.mask==True)] = 0
                        dd3 = sp.ndimage.filters.gaussian_filter(dat3, sigma, mode='constant')
                        allflux[svars[vco]][hemi, NSco, :, :, Kp, pco] = dd2/dd3


                    for fco in range(0, len(allcdf[evars[vco]][0, 0, 0, 0, 0, :])):
                        tempdat = allcdf[evars[vco]][hemi, NSco, :, :, Kp, fco]
                        dd2 = sp.ndimage.filters.gaussian_filter(tempdat, sigma, mode='constant')
                        dat3 = 0 * copy.copy(tempdat) + 1
                        dat3[np.isnan(tempdat) | np.isinf(tempdat) | (tempdat.mask==True)] = 0
                        dd3 = sp.ndimage.filters.gaussian_filter(dat3, sigma, mode='constant')
                        allcdf[evars[vco]][hemi, NSco, :, :, Kp, fco] = dd2/dd3

    #--------------------- Write the cdfs for each variable -----------------------
    # Had to do it this way because github only allows files <100mb
    for eco in range(0, len(evars)):
        if ofile =='':
            fnall = fnbase + evars[eco] + '.nc'
        else:
            fnall = fnbase + evars[eco] + '_'+ofile+'.nc'

        dsout = nc4.Dataset(fnall, 'w')

        # Create dimensions. There is no unlimited dimnesion here
        for dname in dim_names:
            dsout.createDimension(dname, len(dims[dname]))

        for v_name in [evars[eco],svars[eco], nvars[eco]] +dim_names:
            outVar = dsout.createVariable(v_name, var_types[v_name], var_dims[v_name])


        # Create the actual dimension variable data
        for dname in dim_names:
            dsout.variables[dname][:] = dims[dname]

        dsout[evars[eco]][:] = allcdf[evars[eco]][:]
        dsout[svars[eco]][:] = allflux[svars[eco]][:]
        dsout[nvars[eco]][:] = alln[nvars[eco]][:]

        print("Writing ",fnall)

        dsout.close()
    # ----------------- make plots to see if all is reasonable ---------------------------------
    if plots ==1:
        vmi = 1
        vma = 6
        hemisphere = ['N','S']
        NS = ['N','S']
        for eco in range(0,len(evars)):
            # If the variable is flux then plotting with min 1 and max 6 is good
            # If not, then use the value from the dimensions
            if evars[eco].find('B') > -1:
                vmi = dims['Bbins'][0]
                vma = dims['Bbins'][-1]
            if evars[eco].find('alpha') > -1:
                vmi = dims['pbins'][0]
                vma = dims['pbins'][-1]
            plt.set_cmap('jet')
            hemisphere = ['N', 'S']
            NS = ['N', 'S']
            # Make median plots for all vars and Kp
            for Kpval in range(0,10):
                pco = 1
                plt.figure(int(Kpval))
                plt.set_cmap('jet')
                plt.suptitle(evars[eco] + ' '+ str(syear).zfill(4)+' Kp='+str(Kpval))
                for hemi in range(0,2):
                    for NSco in range(0,2):
                        if (evars[eco].find('alpha') >0) & (NSco==1):
                            vmi = 90
                            vma = 135
                        if (evars[eco].find('alpha') >0) & (NSco == 0):
                            vmi = 45
                            vma = 90
                        plt.subplot(2, 2, pco)
                        plt.title( hemisphere[hemi] +'lat '+ NS[NSco])

                        plt.pcolormesh(dims['lonbin'], dims['Lbin'],
                                           ma.masked_less(allflux[svars[eco]][hemi,NSco,:,:,Kpval,50], -200),
                                           vmin=vmi, vmax=vma)
                        pco = pco+1
                        plt.colorbar()
                print('Working on Kp= ',Kpval)
                #plt.tight_layout()
                figname = cdf_direc +sat+' '+evars[eco]+'_'+str(syear_all).zfill(4)+'_'+str(eyear_all).zfill(4)+'_Kp'+str(Kpval)+'.png'
                plt.savefig(figname)
                plt.close()

            for L in range(0, len(dims['Lbin'[:]]), 4):
                for Kpval in range(0, 9):
                    plt.figure(20+L)
                    pco = 1
                    plt.suptitle(evars[eco] + ' ' + str(syear).zfill(4) + 'L=' + str(L))
                    for hemi in range(0, 2):
                        for NSco in range(0, 2):
                            plt.subplot(2, 2, pco)
                            plt.title( hemisphere[hemi] +'lat '+ NS[NSco])
                            plt.plot(dims['lonbin'], ma.masked_less(allflux[svars[eco]][hemi, NSco, L, :, Kpval, 50], -8))
                            plt.ylim(vmi,vma)
                            pco = pco + 1
                figname = cdf_direc + sat+' '+evars[eco] + '_' + str(syear_all).zfill(4) + '_' + str(eyear_all).zfill(
                    4) + '_L' + str(dims['Lbin'][L]) + '.png'
                plt.savefig(figname)
                plt.close()





if __name__ == '__main__':
    import argparse

    '''
    PURPOSE: To create multi-year files of the cumulative distribution of data
    as a function of hemisphere(N/S), sat direction(N,S), L, lon, Kp

    :param: syear - The start year (format YYYY)
    :param: eyear - The end year (format YYYY)
    :param: sataname - satellite name (format m02)
    :param: dataloc - The location of the L binned data files (default ./cdfdata/)
    :param: vars - The variables to make cdfs for (default ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                                 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4',
                                                 'meped_alpha_90_sat','Btot_sat'])
    :param: ofile - This is a string that will be added to the end of the output cdf file to identify it 
         default ('')
        i.e. the file is called poes_cdf_sat_YYYY_YYYY.nc and if ofile is passed it will be poes_cdf_sat_YYYY_YYYYofile.nc
    :param plots - (0 or 1) to make plots or not (default 0)
    
    OUTPUTS: Creates a multi-year cdf file in dataloc/sat/ called poes_cdf_sat_YYYY_YYYY.nc

    USAGE (from command line):
    python make_multi_year_cdf -s 2014 -e 2018 -sat m02 -d ./cdfdata/ -pt
    
    '''
    parser = argparse.ArgumentParser('This creates cdf files')
    #
    parser.add_argument('-s', "--syear",
                        help="The Start Year - format YYYY  ",
                        required=True,
                        default=None,
                        type=int)
    parser.add_argument('-e', "--eyear",
                        help="The end year - format YYYY ",
                        required=True,
                        default=None,
                        type=int)
    parser.add_argument('-sat', "--satname",
                        help="A name of satellite data to get (i.e. -sat n15 or -sat n16 ",
                        type=str, required=True)
    parser.add_argument('-d', "--dataloc",
                        help="The location of the cdf data",
                        required=False, default=os.getcwd() + '/cdfdata/')

    parser.add_argument('-v', "--vars",
                        help="data variables to use",
                        required=False, default=['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                                 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4'
                                                ], nargs='+')
    parser.add_argument('-o', "--ofile",
                        help="Output file name end",
                        required=False, default='')

    parser.add_argument('-pt', "--plots", action='store_true', default=0)

    args = parser.parse_args()

    x = make_multi_year_cdf_vars(args.syear, args.eyear, args.satname, args.dataloc,  args.vars,args.ofile,args.plots)