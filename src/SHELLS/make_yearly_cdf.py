import os
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import glob
import xarray as xr
import collections

def make_yearly_cdf(syear,eyear, sat, dataloc, odir, evars):
    '''
    PURPOSE: This program takes monthly Lbin files and turns them into yearly cumulative distributions
    as a function of hemisphere, satellite direction, L, lon, and Kp

    INPUTS:
    :param: syear (int)- The start year (YYYY) to create yearly cdfs for
    :param: eyear (int)- The end year (YYYY) to create yearly cdfs for
    :param: sat (str)- The satellite to create yearly cdfs for
    :param: odir (str)- The directory for putting the output files
    :param: evars list(str)- A list of the variables to create cdfs for

    NOTE!!! This is currently only set up to work well with the default variables.
    It will need to be updated in the future to work with any variable by automatically defining the cdfs bins
    from the data. Right now the bins are hard coded with what works best for particle fluxes, mag field,
    and pitch angles.

    OUTPUTS:
    netcdf files called odir/sat/poes_cdf_sat_YYYY.nc and two types of plots

    The output files contain the cdf percentiles for each binned variable, the value for each percentile, and the
    total counts in each bin. The cdfs are accumulated by hemisphere (N==0, S==1), satellite direction (Northward = 0
    Soutward =1), L (0 to 8 in .25 increments), longitude(0 to 360 in 10 degree bins) and Kp (0-10).

    For example data['mep_ele_tel90_e1][0,0,8,0,1,:] would give the percentile values for fluxbins (0 to 8 in .1 increments)
    in the Northern hemisphere when the satellite is moving northward at L=3-3.25, lon =0-10 and Kp=1-2.

    data['mep_ele_tel90_e1_sar][0,0,8,0,1,:] would give the flux values for percentile bin (0 to 1 in .1 increments)
    in the Northern hemisphere when the satellite is moving northward at L=3-3.25, lon =0-10 and Kp=1-2.

    data['mep_ele_tel90_e1_n][0,0,8,0,1] would give the number of values (0 to 1 in .1 increments)
    in the Northern hemisphere when the satellite is moving northward between L=3-3.25, lon =0-10 and Kp=1-2.

    '''

    #syear = # year to start with
    #eyear = 2014 # year to end with
    #sat = 'm02'  # satellite to user
    #m02,m01,n15,n18,n19

    # NOTE on zeros
    # When creating the cdfs we are working with the log flux so 0's would
    # normally be thrown out. Instead we set it to the half count value which is energy dependent

    Ge = [100/1.24,100/1.44,100/.75,100/.55]
    Gp = [100 / 42.95, 100 / 135.28, 100 / 401.09, 100 / 1128.67, 100/2202.93,100/.41]

    # Each monthly Lbin netcdf file has data[var][time_med,Lbins]
    # These variables are standard in the monthly Lbin files. They are required for this code
    loc_vars = ['time', 'L_IGRF', 'lat', 'lon', 'MLT','NS','Kp*10']

    # Add the variables to be processed
    # It will be up to the user to make sure the vars they want are actually in the Lbin data files
    allvars = loc_vars + list(evars)

    # These variables are to store the total number of points in each bin
    nvars = list()
    for var in evars:
       nvars.append(var+'_n')

    # These variables will hold the flux or value as a function of percentile from 0-1
    svars = list()
    for var in evars:
        svars.append(var+'_sar')

    # This is going to be for the output flux as a function of percentile
    # svars = ['mep_ele_tel90_flux_e1_sar','mep_ele_tel90_flux_e2_sar',
    #                          'mep_ele_tel90_flux_e3_sar','mep_ele_tel90_flux_e4_sar','Btot_sat_sar','meped_alpha_90_sat_sar']

    lonbins = np.arange(0, 360, 10) # These are the longitude bins
    Lbins = np.arange(1, 8.25, .25) # These are the Lbins
    binvals = np.arange(0, 8.1,.1) # These are the log electron flux bins
    Bbins = np.arange(16700,44500,100) # These are the bins for the magnetic field (if needed)
    pbins = np.arange(0,90,1) # These are the bins for the pitch angles (if needed)
    percentiles = np.arange(0,1.01,.01) # These are the percentile bins
    Kp = np.arange(0,10) # These are the Kp bins

    # Directory for storing the cdf info files
    cdf_direc = odir + '/'+sat +'/'
    #cdf_direc = '/Users/janet/PycharmProjects/SHELLS/cdfdataV5/'+sat+'/'

    # Check if the output directory exists and create it if not
    if not os.path.isdir(odir):
        os.mkdir(odir)

    # Now make the sat directory if it doesn't exist
    if not os.path.isdir(cdf_direc):
        os.mkdir(cdf_direc)

    while syear<=eyear:
        # This is the netcdf file for storing the cdf info
        fn = cdf_direc+ 'poes_cdf_'+sat + '_'+ str(syear).zfill(4) +'.nc'

        cdf_data = nc4.Dataset(fn, 'w', clobber =True)

        cdf_data.createDimension('hemi', 2) # North or south hemisphere
        cdf_data.createDimension('NS', 2) # North or south direction of sat(also corresponds to MLT)
        cdf_data.createDimension('Lbin', len(Lbins))
        cdf_data.createDimension('lonbin', len(lonbins))
        cdf_data.createDimension('fluxbins', len(binvals))
        cdf_data.createDimension('percentiles', len(percentiles))

        # Create the magnetic field bins if needed
        # This should work for any B field values
        if any([x.find('B')>-1 for x in evars]):
            cdf_data.createDimension('Bbins', len(Bbins))
            cdf_data.createVariable('Bbins', np.float64, ('Bbins'))
            cdf_data.variables['Bbins'][:] = Bbins

        # Create the pitch angle bins
        if any([x.find('meped_alpha')>-1 for x in evars]):
            cdf_data.createDimension('pbins', len(pbins))
            cdf_data.createVariable('pbins', np.float64, ('pbins'))
            cdf_data.variables['pbins'][:] = pbins

        # Kp is required
        # To do : change this so that other solar wind params could be used
        cdf_data.createDimension('Kp', len(Kp))
        cdf_data.createVariable('Kp', np.float64, ('Kp'))
        cdf_data.variables['Kp'][:] = Kp

        cdf_data.createVariable('Lbin', np.float64, ('Lbin'))
        cdf_data.createVariable('hemi', np.int, ('hemi'))
        cdf_data.createVariable('NS', np.int, ('NS'))
        cdf_data.createVariable('lonbin', np.float64, ('lonbin'))
        cdf_data.createVariable('fluxbins', np.float64, ('fluxbins'))
        cdf_data.createVariable('percentiles', np.float64, ('percentiles'))

        # This is the percentile for each electron fluxbin or other data
        for var in evars:
            if var.find('ele')>-1:
                cdf_data.createVariable(var, np.float64, ('hemi','NS','Lbin','lonbin','Kp','fluxbins'))
                cdf_data.variables[var][:, :, :, :, :, :] = 0
            if var.find('B')>-1:
                cdf_data.createVariable(var, np.float64, ('hemi', 'NS', 'Lbin', 'lonbin', 'Kp', 'Bbins'))
            if var.find('alpha')>-1:
                cdf_data.createVariable(var, np.float64, ('hemi', 'NS', 'Lbin', 'lonbin', 'Kp', 'pbins'))

        # These are the cdfs for the total B
        # This is the percentile for each Bbin
        # for var in Bvars:
        #    cdf_data.createVariable(var, np.float64, ('hemi','NS','Lbin','lonbin','Kp','Bbins'))

        # This is the cf for the pitchangle
        #for var in pvars:
        #    cdf_data.createVariable(var, np.float64, ('hemi','NS','Lbin','lonbin','Kp','pbins'))

        # This is the flux for each percentile bin
        # It's the same for the electron flux and B
        for var in svars:
            cdf_data.createVariable(var, np.float64, ('hemi','NS','Lbin','lonbin','Kp','percentiles'))
            cdf_data.variables[var][:, :, :, :, :,:] = 0

        # this is number of points in each bin
        for var in nvars:
            cdf_data.createVariable(var, np.float64, ('hemi','NS','Lbin','lonbin','Kp'))
            cdf_data.variables[var][:, :, :, :, :] = 0  # number in each bin

        cdf_data.variables['Lbin'][:] = Lbins
        cdf_data.variables['hemi'][:] = [0,1]
        cdf_data.variables['NS'][:] = [0, 1]
        cdf_data.variables['lonbin'][:] = lonbins
        cdf_data.variables['fluxbins'][:] = binvals
        cdf_data.variables['percentiles'][:] = percentiles

        # This is for making a plot of the median values in the end
        allmeds = {}
        meds = np.zeros((2,2,len(Lbins),len(lonbins),len(Kp)),dtype = float)

        # Get all the sat data for a year
        Ldirec = dataloc+'/'+str(syear).zfill(4) +'/'
        files = glob.glob(Ldirec + '*'+sat+'*.nc')
        files.sort()
        alldat = xr.open_mfdataset(files)
        # make a cdf for each hemi, direc (equates to MLT), L, lon, Kp

        # Add B and pitch angle to evars
        #evars.append(Bvars[0])
        #evars.append(pvars[0])

        for var in np.arange(0,len(evars)):
            # Step through each L
            for L in np.arange(0, len(Lbins) ):
                edat = (alldat[evars[var]][:, L].values) #  data value for one L
                # If it is a flux value then take the log10
                if evars[var].find('flux')>-1:
                    test_inds = np.where(edat==0)[0]
                    #if len(test_inds)>0:
                    #    print(test_inds)
                    # Check if its protons or eclectrons
                    if evars[var].find('ele')>-1:
                        # Find the energy
                        enum = int(evars[var][-1])
                        edat[edat==0]= .5*Ge[enum-1]
                    elif evars[var].find('pro')>-1:
                        enum = int(evars[var][-1])
                        edat[edat==0]= .5*Gp[enum-1]

                    fluxdat = np.log10(edat)

                elif evars[var].find('cps')>-1:
                    edat[edat == 0] =.5
                    fluxdat = np.log10(edat)
                else:
                    fluxdat = edat

                NSdat = alldat['NS'][:, L].values

                # Why are there lat values that are NaN?
                latdat = alldat['lat'][:, L].values
                latdat[latdat >=0] = 0 # Northern hemi
                latdat[latdat < 0] = 1 # Southern hemi
                londat = np.floor(alldat['lon'][:, L]/10)

                # Round Kp to the nearest whole number
                Kpdat1 = alldat['Kp*10'][:].values #
                Kpdat = np.floor(Kpdat1/10)

                # Create indices out of lat, lon, NS, Kp
                # This is faster than looping through each index
                indtest2 = (londat * 10000 + Kpdat * 100+ 10 * latdat + NSdat).__array__()
                # Get all the uniqe indeces
                inds, counts = np.unique(indtest2[~np.isnan(indtest2)], return_counts=True)

                result = collections.defaultdict(list)
                for val, idx in zip(fluxdat, indtest2):
                    if ((~np.isnan(val)) & (~np.isinf(val))):
                        result[idx].append(val)

                for idx in inds:

                    temp = np.array(result[idx])
                    #temp2 = temp[(~np.isnan(temp)) & (~np.isinf(temp))] # Get rid of fluxes <.01?
                    data_sorted = np.sort(temp)
                    n = len(temp)

                    if len(temp)>1:
                        p = 1. * np.arange(len(temp)) / (len(temp) - 1)

                        # Interpolate the values to the actual bin values
                        # Everything that is not mag field, or pitch angle is assumed to be flux
                        if evars[var].find('B') >-1:
                            per = np.interp(Bbins, data_sorted, p)
                        elif evars[var].find('alpha') >-1 :
                            per = np.interp(pbins, data_sorted, p)
                        else:
                            per = np.interp(binvals,data_sorted,p) # percentiles interpoalted to binvals (0,.1, ...8)

                        sar = np.interp(percentiles,p,data_sorted) # fluxes interpolated to percentiles (0,.01,...1)
                    else:
                        # If there are no values in the L,lat,lon,Kp bin then set it to 0
                        if evars[var].find('B') >-1:
                            per = 0*Bbins
                        elif evars[var].find('alpha') >-1:
                            per = 0*pbins
                        else:
                            per = 0*binvals
                        sar =0*percentiles
                        n = 0

                    #plt.subplot(2,1,1)
                    #plt.plot(data_sorted,p)
                    #plt.plot(binvals, per)
                    #plt.subplot(2,1,2)
                    #plt.plot(p,data_sorted)
                    #plt.plot(percentiles, sar)
                    #print('Plotting')

                    # Get the indices back
                    lonv = np.floor(idx/10000)
                    Kpv = np.floor((idx - lonv*10000)/100)
                    latv = np.floor((idx - lonv*10000 -Kpv*100)/10)
                    NSv = np.floor(idx -lonv*10000 -Kpv*100 - latv*10)
                    # There are some longitudes that equal exactly 360 that should b 0
                    # It only happens at one point
                    if lonv==36:
                        lonv=0
                    #print(lonv,Kpv, latv, NSv,idx )

                    cdf_data.variables[evars[var]][latv, NSv, L, lonv, Kpv, :] = per  # percentiles for fluxbins
                    #cdf_data.variables[evars2[var]][latv, NSv, L, lonv, Kpv, :] = per*n
                    cdf_data.variables[nvars[var]][latv, NSv, L, lonv, Kpv] = n # number in each bin
                    cdf_data.variables[svars[var]][latv, NSv, L, lonv, Kpv, :] = sar  # flux for percentiles

                    # median value for plotting later
                    meds[int(latv), int(NSv), int(L), int(lonv),int(Kpv)] = sar[np.where(percentiles==.5)]
                print(L,syear)

            allmeds[var] = meds
            # Now make a plot of the median for that variable
            pco=1
            vmi = 1
            vma = 5.5
            if evars[var].find('B') >-1:
                vmi = Bbins[0]
                vma = Bbins[-1]
            if evars[var].find('alpha') >-1:
                vmi = pbins[0]
                vma = pbins[-1]
            plt.set_cmap('jet')
            hemisphere = ['N', 'S']
            NS = ['N', 'S']

            # Make a plot for each Kp val
            for Kpval in Kp:
                pco = 1
                plt.figure(int(Kpval))

                plt.suptitle(sat+' '+evars[var] + ' '+ str(syear).zfill(4)+' Kp='+str(Kpval))
                for hemi in range(0,2):
                    for NSco in range(0,2):
                        # Pitch angles in the southern hemi go from 90 to 135
                        # and in the northern hemi they go from 45 to 90
                        if (evars[var].find('alpha') >0) & (NSco==1):
                            vmi = 90
                            vma = 135
                        if (evars[var].find('alpha') >0) & (NSco == 0):
                            vmi = 45
                            vma = 90
                        plt.subplot(2, 2, pco)
                        plt.title( hemisphere[hemi] +'lat '+ NS[NSco]+'bound')
                        plt.pcolormesh(lonbins, Lbins,
                                   ma.masked_less(((meds[hemi,NSco,:,:,int(Kpval)])), -8),
                                   vmin=vmi, vmax=vma)
                        if (pco == 1) | (pco == 3):
                            plt.ylabel('L IGRF')
                        if (pco == 3) | (pco == 4):
                            plt.xlabel('E longitude')
                        pco = pco+1
                        cbar = plt.colorbar()
                        if (evars[var].find('alpha') >0):
                            cbar.ax.set_ylabel('degrees')
                        cbar.ax.tick_params(labelsize='small')
                #print(Kpval)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Save the figure to a file
                figname = cdf_direc +sat+evars[var]+'_'+str(syear).zfill(4)+'_'+str(Kpval)+'.png'
                plt.savefig(figname)
                plt.close()

                # Make line plots for each L to show the longitude variaion
                plt.figure(int(Kpval+20))
                plt.suptitle(sat+' '+evars[var] + ' '+ str(syear).zfill(4)+' Kp='+str(Kpval))
                pco=1
                for hemi in range(0,2):
                    for NSco in range(0,2):
                        plt.subplot(2, 2, pco)
                        plt.title( hemisphere[hemi] +'lat '+ NS[NSco] +'bound')
                        for Lbin in np.arange(0,28,4):
                            plt.plot(lonbins, ma.masked_less(meds[hemi,NSco,Lbin,:,int(Kpval)],1),label=Lbin/4+1)
                        plt.ylim(1,vma)
                        if pco==1:
                            plt.legend()
                        pco = pco+1
                plt.tight_layout()
                figname = cdf_direc +sat+evars[var]+'_'+str(syear).zfill(4)+'_'+str(Kpval)+'lines.png'
                plt.savefig(figname)
                plt.close()

            print('MAde plot')

        cdf_data.close()
        alldat.close()
        syear = syear +1



if __name__ == '__main__':
    import argparse

    '''
    PURPOSE: To create yearly files of the cumulative distribution of data
    as a function of hemisphere(N/S), sat direction(N,S), L, lon, Kp
    
    :param: syear - The start year (format YYYY)
    :param: eyear - The end year (format YYYY)
    :param: sataname - satellite name (format m02)
    :param: dataloc - The location of the L binned data files (default ./Lbindata/)
    :param: odirec - The directory to put the cdf data (default ./cdfdata/)
    :param: vars - The variables to make cdfs for (default ['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                                 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4',
                                                 'meped_alpha_90_sat','Btot_sat'])
    
    USAGE (from command line):
    python make_yearly_cdf -s 2013 -e 2015 -sat m02 -d /SHELLS/Lbindata -od /SHELLS/cdfdata
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
                        help="The location of the Lbin data",
                        required=False, default=os.getcwd()+'/Lbindata/')
    parser.add_argument('-od', "--odirec",
                        help="The output directory of data",
                        required=False, default=os.getcwd() + '/cdfdata/')
    parser.add_argument('-v', "--vars",
                        help="data variables to use",
                        required=False, default=['mep_ele_tel90_flux_e1', 'mep_ele_tel90_flux_e2',
                                                 'mep_ele_tel90_flux_e3', 'mep_ele_tel90_flux_e4',
                                                 'meped_alpha_90_sat','Btot_sat'], nargs='+')


    args = parser.parse_args()

    x = make_yearly_cdf(args.syear, args.eyear, args.satname, args.dataloc, args.odirec, args.vars)