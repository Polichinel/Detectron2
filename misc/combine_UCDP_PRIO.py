# use geo_env_2022

import os
import numpy as np
import pandas as pd # original 1.2.3
import geopandas as gpd
from shapely.geometry import Point
import pickle
import urllib.request

import matplotlib.pyplot as plt


def get_prio_shape():

    location = '/home/simon/Documents/Bodies/data/PRIO'
    path_prio = location + '/priogrid_shapefiles.zip'

    if os.path.isfile(path_prio) == True:
        
        print('File already downloaded')
        prio_grid = gpd.read_file('zip://' + path_prio)

    else:
        print('Beginning file download PRIO...')
        url_prio = 'http://file.prio.no/ReplicationData/PRIO-GRID/priogrid_shapefiles.zip'

        urllib.request.urlretrieve(url_prio, path_prio)
        prio_grid = gpd.read_file('zip://' + path_prio)

    return prio_grid


def get_gwno():

    location = '/home/simon/Documents/Bodies/data/PRIO'
    #path_gwno = location + '/PRIO-GRID Yearly Variables for 2003-2009 - 2022-06-16.csv' #https://grid.prio.org/#/download # need to figrue out the API
    path_gwno = location + '/PRIO-GRID Yearly Variables for 1989-2014 - 2022-06-16.csv' #https://grid.prio.org/#/download # need to figrue out the API

    # why not just go 1989 - 2019 like ucdp...

    gwno = pd.read_csv(path_gwno)

    return gwno


def get_ucdp():
    location = '/home/simon/Documents/Bodies/data/UCDP' 
    path_ucdp = location + "/ged201-csv.zip"
    
    if os.path.isfile(path_ucdp) == True:
        print('file already downloaded')
        ucdp = pd.read_csv(path_ucdp)


    else: 
        print('Beginning file download UCDP...')

        url_ucdp = 'https://ucdp.uu.se/downloads/ged/ged201-csv.zip'
    
        urllib.request.urlretrieve(url_ucdp, path_ucdp)
        ucdp = pd.read_csv(path_ucdp)

    return ucdp


def add_month_id(ucdp): # you could also do a week_id....

    ucdp_tmp1 = ucdp.copy()

    ucdp_tmp1['year_months_start'] = ucdp_tmp1['date_start'].str.slice(start = 0, stop = 7) # Date YYYY-MM-DD
    ucdp_tmp1['year_months_end'] = ucdp_tmp1['date_start'].str.slice(start = 0, stop = 7) # Date YYYY-MM-DD


    mask1 = (ucdp_tmp1['year'] != ucdp_tmp1['year_months_start'].str.slice(start = 0, stop = 4).astype(int))
    mask2 = (ucdp_tmp1['year'] != ucdp_tmp1['year_months_end'].str.slice(start = 0, stop = 4).astype(int))

    # correction. Note that end and start year for the four entries that is corrected is the same.
    ucdp_tmp1.loc[mask1 | mask2, 'year'] = ucdp_tmp1.loc[mask1 | mask2,'year_months_start'].str.slice(start = 0, stop = 4).astype(int)

    ds_uniques = ucdp_tmp1['date_start'].str.slice(start = 0, stop = 7).unique()
    de_uniques = ucdp_tmp1['date_end'].str.slice(start = 0, stop = 7).unique() # do you need both?

    months_unique = np.union1d(ds_uniques, de_uniques)
    months_unique.sort()

    month_id = np.arange(109, months_unique.shape[0] + 109, 1) # this makes sure the month_id matches that of ViWES replication data. Just in case.

    month_df = pd.DataFrame({'month_id' : month_id, 'year_months_start' : months_unique, 'year_months_end' : months_unique})

    # I checked. There is no instance where the month id will differ if we take start or end.
    ucdp_tmp2 = ucdp_tmp1.merge(month_df[['month_id', 'year_months_start']], how = 'outer', on= 'year_months_start')

    return(ucdp_tmp2)


def trim_ucdp(ucdp_monthly):

    ucdp_slim = ucdp_monthly[['country','year', 'month_id', 'year_months_start', 'priogrid_gid','best','low','high']].copy()
    ucdp_gid = ucdp_slim.groupby(by=['priogrid_gid','month_id', 'year_months_start', 'year','country']).sum().reset_index() # so right now it is monthly units. you can change this..
    ucdp_gid.rename(columns={'priogrid_gid':'gid'}, inplace=True)

    ucdp_gid['log_best'] = np.log(ucdp_gid['best'] +1)
    ucdp_gid['log_low'] = np.log(ucdp_gid['low'] +1)
    ucdp_gid['log_high'] = np.log(ucdp_gid['high'] +1)

    return(ucdp_gid)


def add_years(ucdp, world_grid):

    diff = ucdp['year'].max() - world_grid['year'].max()

    subset_list = []

    for i in np.arange(1, diff+1, 1):

        subset = world_grid[world_grid['year'] == world_grid['year'].max()].copy()
        subset['year'] = world_grid['year'].max() + i

        subset_list.append(subset)

    new_years = pd.concat(subset_list)
    world_grid_all_years = pd.concat([world_grid, new_years])

    return world_grid_all_years


def combine_UCDP_PRIO(ucdp_gid, world_grid_all_years):

    month = [str(i).zfill(2) for i in np.arange(1,13,1)]
    world_grid_all_years.loc[:,'month'] = world_grid_all_years.apply(lambda _: month, axis=1)
    world_grid_monthly = world_grid_all_years.explode('month')

    world_grid_monthly['year_months_start'] = world_grid_monthly['year'].astype(str) + '-' +  world_grid_monthly['month'].astype(str)

    #ucdp_cliped = ucdp_gid[ucdp_gid['year']<2015].copy()# Could also just expand world_grid_monhtly but I do not need these values anyway (for now)..

    combined_df = world_grid_monthly.merge(ucdp_gid, how = 'left', on = ['gid', 'year_months_start', 'year']) # month id needs to be made after..
    combined_df.loc[:, ['best', 'low', 'high', 'log_best', 'log_low', 'log_high']] = combined_df.loc[:, ['best', 'low', 'high', 'log_best', 'log_low', 'log_high']].fillna(0)

    month_id_dict = dict(zip(ucdp_gid['year_months_start'],ucdp_gid['month_id']))
    combined_df['month_id'].fillna(combined_df['year_months_start'].map(month_id_dict), inplace = True)
    combined_df['month_id'] = combined_df['month_id'].astype(int)

    #combined_df['gwno'] = combined_df['gwno'].astype(int)
    # Does not really work...
    #contry_dict = dict(zip(combined_df.loc[combined_df['country'].notna(), 'country'].unique(), combined_df.loc[combined_df['country'].notna(), 'gwno'].unique()))
    #combined_df['country'].fillna(combined_df['gwno'].map(contry_dict), inplace = True)

    combined_df.drop(columns= ['country'], inplace = True)

    return(combined_df)


def compile_combined_df():

    prio_grid = get_prio_shape()
    gwno = get_gwno()
    world_grid = prio_grid.merge(gwno, how = 'right', on = 'gid') # if you just merge this on outer I think you get the full grid needed for R-UNET

    ucdp = get_ucdp()
    ucdp_monthly = add_month_id(ucdp)
    ucdp_gid = trim_ucdp(ucdp_monthly)

    world_grid_all_years = add_years(ucdp, world_grid)

    combined_df = combine_UCDP_PRIO(ucdp_gid, world_grid_all_years)

    data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/'

    with open(f'{data_dir}df_ucdp_prio.pkl', 'wb') as file:
        pickle.dump(combined_df, file)


if __name__ == "__main__":
    compile_combined_df()
