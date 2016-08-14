# Import Generic libraries
import numpy as np
import pandas as pd

# Import Libraries to process the kepler Data
import kplr
import pyfits
from PyKE.kepstitch import kepstitch
from PyKE.kepflatten import kepflatten
from algorithms.dtw import LB_Keogh

# Import Library for parallel computing
import multiprocessing
from joblib import Parallel, delayed

# Setting up log filename
LOG_FILENAME = 'processing_kepler_data.log'

# Read the planets list from the csv file
planets = pd.read_csv('data/candidate_planets.csv')

# Remove Baseline light curve chosen
planets = planets.ix[2:]

# Remove the K and transorm to float type allowing KPLR Library to process the name
planets['kepoi_name'] = planets['kepoi_name'].str.replace('K', '').astype(float)

# Get Some Count of the data
confirmed = planets.loc[planets['koi_disposition'] == 'CONFIRMED']
false_positive = planets.loc[planets['koi_disposition'] == 'FALSE POSITIVE']
candidate = planets.loc[planets['koi_disposition'] == 'CANDIDATE']

print "Number of total Planets: {}, Confirmed: {}, False Positive: {}, Candidates: {}" \
    .format(planets.shape[0], confirmed.shape[0], false_positive.shape[0], candidate.shape[0])

# Setting up the resultant dataframe columns name
DF_COLUMNS = ('kepoi_name', 'koi_disposition',
              'dtw_sap_flux', 'sap_flux_mean', 'sap_flux_std', 'sap_flux_min', 'sap_flux_max',
              'dtw_sap_bkg', 'sap_bkg_mean', 'sap_bkg_std', 'sap_bkg_min', 'sap_bkg_max',
              'dtw_pd_sap_flux', 'pd_sap_flux_mean', 'pd_sap_flux_std', 'pd_sap_flux_min', 'pd_sap_flux_max',
              'dtw_det_sap_flux', 'det_sap_flux_mean', 'det_sap_flux_std', 'det_sap_flux_min', 'det_sap_flux_max',
              'mom_centr1_mean', 'mom_centr1_std', 'mom_centr1_min', 'mom_centr1_max',
              'mom_centr2_mean', 'mom_centr2_std', 'mom_centr2_min', 'mom_centr2_max',
              'pos_corr1_mean', 'pos_corr1_std', 'pos_corr1_min', 'pos_corr1_max',
              'pos_corr2_mean', 'pos_corr2_std', 'pos_corr2_min', 'pos_corr2_max')

# Creating the Dataframe
df = pd.DataFrame(columns=DF_COLUMNS)

print "Holder DataFrame Created"

# Opening the baseline curve FITs File
hdu = pyfits.open("data/detrended/752.01.fits")

# Grab the data from the file
hdu_data = hdu[1].data

# Create the baseline time series for SAP_FLUX, SAP_BKG, PDCSAP_FLUX, DETSAP_FLUX
baseline_sap_flux = pd.Series(np.array(hdu_data["SAP_FLUX"]))
baseline_sap_bkg = pd.Series(np.array(hdu_data["SAP_BKG"]))
baseline_pd_sap_flux = pd.Series(np.array(hdu_data["PDCSAP_FLUX"]))
baseline_det_sap_flux = pd.Series(np.array(hdu_data["DETSAP_FLUX"]))

print "Baselines Variables Created"


def processPlanetData(index, planets, df, baseline_sap_flux, baseline_sap_bkg, baseline_pd_sap_flux,
                      baseline_det_sap_flux):
    # Get Kepler Object of Interest name and disposition
    kepoi_name = planets.iloc[index]['kepoi_name']
    koi_disposition = planets.iloc[index]['koi_disposition']

    # Set Files Names for the resultant Stiched File and Resultant Detrended File
    filenameStitch = "data/stitched/{}.fits".format(kepoi_name)
    filenameDetrend = "data/detrended/{}.fits".format(kepoi_name)

    # Initialize kplr API
    client = kplr.API()

    print "Processing Kepler Object: {} at index: {}".format(kepoi_name, index)

    # Find a Kepler Object of Interest
    koi = client.koi(planets.iloc[index]['kepoi_name'])

    # Get a list of light curve data sets.
    lcs = koi.get_light_curves(short_cadence=False, fetch=True, clobber=False)

    print "Got Kepler Object of Interest and Light Curve Files for {}".format(kepoi_name)

    lc_list = ""

    # Looping trough lcs to get list of light curves path
    for lc in lcs:
        lc_list += str(lc.filename) + ","

    # Removing trailing comma
    lc_list_clean = lc_list[:-1]

    print "Got path list of light curves."

    # Stitching together light curves quarters
    kepstitch(lc_list_clean, filenameStitch, True, False, LOG_FILENAME, 0)

    print "Finished Stiching the data for {}".format(kepoi_name)

    # Detrending Light Curve
    kepflatten(filenameStitch, filenameDetrend, "PDCSAP_FLUX", "PDCSAP_FLUX_ERR",
               3.0, 1.0, 3.0, 3, 10, "0,0", False, True, False, LOG_FILENAME, 0, True)

    print "Finished Detrending the data for {}".format(kepoi_name)

    # Opening Detrended FITs File
    hdu = pyfits.open(filenameDetrend)

    # Getting Detrended Data
    hdu_data = hdu[1].data

    # Getting all features form the Fit Files
    sap_flux = pd.Series(np.array(hdu_data["SAP_FLUX"]))
    sap_bkg = pd.Series(np.array(hdu_data["SAP_BKG"]))
    pd_sap_flux = pd.Series(np.array(hdu_data["PDCSAP_FLUX"]))
    det_sap_flux = pd.Series(np.array(hdu_data["DETSAP_FLUX"]))
    mom_centr1 = pd.Series(np.array(hdu_data["MOM_CENTR1"]))
    mom_centr2 = pd.Series(np.array(hdu_data["MOM_CENTR2"]))
    pos_corr1 = pd.Series(np.array(hdu_data["POS_CORR1"]))
    pos_corr2 = pd.Series(np.array(hdu_data["POS_CORR2"]))

    # Try to apply DTW with the baseline. If fails return NaN
    try:
        dtw_sap_flux = LB_Keogh(sap_flux, baseline_sap_flux, 10)
        dtw_sap_bkg = LB_Keogh(sap_bkg, baseline_sap_bkg, 10)
        dtw_pd_sap_flux = LB_Keogh(pd_sap_flux, baseline_pd_sap_flux, 10)
        dtw_det_sap_flux = LB_Keogh(det_sap_flux, baseline_det_sap_flux, 10)
    except:
        dtw_sap_flux, dtw_sap_bkg, dtw_pd_sap_flux, dtw_det_sap_flux = 'nan', 'nan', 'nan', 'nan'
        pass

    print "dtw_sap_flux: {}, dtw_sap_bkg: {}, dtw_pd_sap_flux: {}, dtw_det_sap_flux: {}".format(
        dtw_sap_flux, dtw_sap_bkg, dtw_pd_sap_flux, dtw_det_sap_flux)

    # Describe the Features Extracted (STD, MEAN, MIN, MAX)
    desc_sap_flux, desc_sap_bkg = sap_flux.describe(), sap_bkg.describe()
    desc_pd_sap_flux, desc_det_sap_flux = pd_sap_flux.describe(), det_sap_flux.describe()
    desc_mom_centr1, desc_mom_centr2 = mom_centr1.describe(), mom_centr2.describe()
    desc_pos_corr1, desc_pos_corr2 = pos_corr1.describe(), pos_corr2.describe()

    print "Features Described Correctly."

    # Apply the Features to the dataframe
    df.loc[index] = [kepoi_name, koi_disposition,
                     dtw_sap_flux, desc_sap_flux['mean'], desc_sap_flux['std'], desc_sap_flux['min'],
                     desc_sap_flux['max'],
                     dtw_sap_bkg, desc_sap_bkg['mean'], desc_sap_bkg['std'], desc_sap_bkg['min'], desc_sap_bkg['max'],
                     dtw_pd_sap_flux, desc_pd_sap_flux['mean'], desc_pd_sap_flux['std'], desc_pd_sap_flux['min'],
                     desc_pd_sap_flux['max'],
                     dtw_det_sap_flux, desc_det_sap_flux['mean'], desc_det_sap_flux['std'], desc_det_sap_flux['min'],
                     desc_det_sap_flux['max'],
                     desc_mom_centr1['mean'], desc_mom_centr1['std'], desc_mom_centr1['min'], desc_mom_centr1['max'],
                     desc_mom_centr2['mean'], desc_mom_centr2['std'], desc_mom_centr2['min'], desc_mom_centr2['max'],
                     desc_pos_corr1['mean'], desc_pos_corr1['std'], desc_pos_corr1['min'], desc_pos_corr1['max'],
                     desc_pos_corr2['mean'], desc_pos_corr2['std'], desc_pos_corr2['min'], desc_pos_corr2['max']
                     ]

    print "Dataframe Row Added Correctly."



# Setting the Number of cores in the computer (set up to 1 due to the fact that not all computers allow multiprocessing)
num_cores = 1
# Uncomment if you want to try multiprocessing
# num_cores = multiprocessing.cpu_count()

# Setting Up the Limit for notebook purpose
LIMIT = 2

Parallel(n_jobs=num_cores) \
    (delayed(processPlanetData) \
    (index, planets, df, baseline_sap_flux, baseline_sap_bkg, baseline_pd_sap_flux, baseline_det_sap_flux) \
     for index in range(0, LIMIT))

df.to_csv("data/kepler_pre_ml_notebook.csv")

print df.head()