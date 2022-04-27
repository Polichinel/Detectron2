# if local use new_torch_env

from genericpath import exists
import os

from scipy import stats
import numpy as np
import pandas as pd

import pickle

from utills_output import *


def open_df(full = False, threshold = 0.3):

    """Open a pre-made df with a specified probability-threshold. 
    I.e. how high prob was need to registre a object in the dataset.
    The dataset can either be the fullset (full=True) or the annotated subset (full = False).
    Used in get_df_dict."""
    
    path = '/home/simon/Documents/Bodies/data/OD_dataframes'

    if full == True:
        df_name = f'OD_DF_FULL_t{int(threshold*100)}.pkl'

    elif full == False:
        df_name = f'OD_DF_annotated_t{int(threshold*100)}.pkl'

    print(f'loading: {df_name}')

    df_path = os.path.join(path, df_name)

    with open(df_path, 'rb') as file:
        df = pickle.load(file)

    return(df)

def get_df_dict(full = False):

    """Creates a dict with all pre-made dfs - i.e. all the probability thresholds.
    The dict can either contain full datasets at each generated threshold (full = True)
     or the annotated sunsets (full = False).
     Used in find_best_ts and on to get_JSD and also in get_new_df"""

    df_dict = {}
    path = '/home/simon/Documents/Bodies/data/OD_dataframes'

    for root, dirs, files in os.walk(path):

        if full == False:
            ts_list = [file for file in files if file.split('_')[2] == 'annotated']

        elif full == True:
            ts_list = [file for file in files if file.split('_')[2] == 'FULL']
        
        t_range = [float(file.split('_')[-1].split('.')[0].split('t')[1])/100 for file in ts_list]
        t_range = sorted(np.array(t_range))

    for t in t_range:
        df_name = f'df_t{int(t*100)}'
        df_dict[df_name] = open_df(full, t)
        print(f'Done w/ {df_name}', end='\r')

    return(df_dict)


def jensen_shannon_distance(p, q):

    """
    Method to compute the Jenson-Shannon Distance between two probability distributions.
    Similar to the kulback leibler divergence but symatric. Solves issues with inf and nans. 
    From https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
    used in get_JSD
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def get_JSD(feature, df_dict):

    """Finds the df/threshold with the lowest (best) JSD given a specific feature. 
    Saves the info in a dict (best_dict). Also save the spefic JSD achived and the 
    corrolation between the annoated and the interpolated counts.
    As such, this functions always works on the annotated subset. Never the full set.
    The annotated df are automatically generated via the script "range_of_annotated_df.py". 
    Could also be generated one at a time via "output_to_df.py"
    Used in find_best_ts"""
    
    best_corr = 0 # just initial value to beat
    best_JSD = np.inf # just initial value to beat
    best_dict = {'feature':feature}

    for k in df_dict.keys():
        feature_mean = f'{feature}_mean'
        feature_annotated = f'{feature}_annotated'
        if feature_mean in np.array(df_dict[k].columns):
            #print(k, end = ': ')
            pk = df_dict[k][feature_mean] # is there strickly any reason you are not using a sample from the full set?
            qk = df_dict[k][feature_annotated]
            JSD = jensen_shannon_distance(pk, qk)
            corr = np.corrcoef(pk, qk)[0,1]            
            #KLD  = stats.entropy(pk, qk)
            #print(f'{JSD:.3f}')
            if JSD < best_JSD:
                best_JSD = JSD
                best_dict['JSD'] = best_JSD
                best_dict['df_JSD'] = f'{k.split("_")[1]}'

            if np.abs(corr) > best_corr:
                best_corr = corr
                best_dict['corr'] = best_corr
                best_dict['df_corr'] = f'{k.split("_")[1]}'            

            # Then really you want to save the best df.

        else:
            pass

    return(best_dict)


def find_best_ts():

    """Iterrates over all features and pass them to get_JSD. Thus you get a list (best_ts) 
    with info on wich df/threshold have the lowest (best) JSD given each feature.
    Used in get_new_df"""

    df_dict = get_df_dict(full = False) # full is always false here, because you need the annotated set here.
    best_ts = []

    for f in df_dict['df_t10'].columns[df_dict['df_t10'].columns.str.endswith('annotated')]: # t10 because it have the most features.
        feature = f.split('_')[0]
        best_dict = get_JSD(feature, df_dict)
        best_ts.append(best_dict)

    return(best_ts)


def get_new_df():

    """Creates to dataset. One with the full set of images and one woth only the annotated set.
    Uses the individual thresholds for each feature find by minimixing the JSD between interpolated counts
    and annoated counts given the annotated set."""

    # you should also make a fat df version of each...

    best_ts = find_best_ts()
    df_dict_full = get_df_dict(full = True)
    df_dict_annotated = get_df_dict(full = False)
    
    new_df_dict_full = {}
    new_df_dict_annotated = {}

    sub_features = ['mean', 'median', 'fasterR50', 'fasterR101', 'fasterX101', 'retinaR50', 'retinaR101']

    for feature in best_ts: # since you always use the annotated best_ts, you can do this for FULL as well

        if feature['corr'] >= 0.1: #only keep somewhat correlated stuff

            df_name = f'df_{feature["df_JSD"]}'

            for sub_feature in sub_features:

                sub_feature_name = f'{feature["feature"]}_{sub_feature}'

                if sub_feature_name in df_dict_full[df_name].columns:
                    new_df_dict_full[sub_feature_name] = df_dict_full[df_name][sub_feature_name]
                
                if sub_feature_name in df_dict_annotated[df_name].columns:
                    new_df_dict_annotated[sub_feature_name] = df_dict_annotated[df_name][sub_feature_name]


    new_df_full = pd.DataFrame(new_df_dict_full)
    new_df_full['img_id'] = df_dict_full['df_t10']['img_id'] # could just merge on index

    new_df_annotated = pd.DataFrame(new_df_dict_annotated)
    new_df_annotated['img_id'] = df_dict_annotated['df_t10']['img_id'] # could just merge on index

    feature_list_full = ['custom2', 'custom3', 'custom4', 'date created', 'time created', 'img_id', 
                         'city', 'province/state', 'sub-location', 'headline', 'by-line title', 'caption/abstract',
                         'object name'] # check for more meta

    new_large_df_full = new_df_full.merge(df_dict_full['df_t10'][feature_list_full], on='img_id', how='inner') 
    new_large_df_full.rename(columns= {'custom2' : 'publication', 'custom3' : 'year', 'custom4' : 'org img name'}, inplace= True)
    
    feature_list_annoated = ['person_annotated', 'child_annotated', 'male_annotated', 'adult_annotated',
                             'youth_annotated', 'falgIRQ_annotated', 'female_annotated', 'religiousGarmentFemale_annotated',
                             'uniformed_annotated', 'firearm_annotated', 'flagUS_annotated', 'infant_annotated',
                             'bloodedArea_annotated', 'militaryVehicle_annotated', 'prayerInformal_annotated',
                             'hostage_annotated', 'casualty_annotated', 'elderly_annotated', 'prayerSalah_annotated', 
                             'custom2', 'custom3', 'custom4', 'date created', 'time created', 'img_id', 
                             'city', 'province/state', 'sub-location', 'headline', 'by-line title', 'caption/abstract',
                             'object name'] # check for more meta

    new_large_df_annotated = new_df_annotated.merge(df_dict_annotated['df_t10'][feature_list_annoated], on='img_id', how='inner') 
    new_large_df_annotated.rename(columns= {'custom2' : 'publication', 'custom3' : 'year', 'custom4' : 'org img name'}, inplace= True)

    return(new_large_df_annotated, new_large_df_full)


def compile_OD_dfs():

    """Run everything above and save two csv files and two pickles. 
    One of each for the annotated set and one of each for the full set"""

    #Should also make a fat df...

    new_large_df_annotated, new_large_df_full = get_new_df()

    data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/' # meybe just make another dir...

    os.makedirs(data_dir, exist_ok = True)

    # You should just have a feature list for the slim dfs down here and make two sub dfs.
    # also makes it easier to edit.

    new_large_df_full.to_csv(f'{data_dir}df_od_full_slim.csv') # that should no just be there....
    new_large_df_annotated.to_csv(f'{data_dir}df_od_annotated_slim.csv')

    with open(f'{data_dir}df_od_full_slim.pkl', 'wb') as file:
        pickle.dump(new_large_df_full, file)

    with open(f'{data_dir}df_od_annotated_slim.pkl', 'wb') as file:
            pickle.dump(new_large_df_annotated, file)


if __name__ == "__main__":
    compile_OD_dfs()
