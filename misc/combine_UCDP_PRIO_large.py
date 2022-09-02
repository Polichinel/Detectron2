# use geo_env_2022

import os
import numpy as np
import pandas as pd # original 1.2.3
import geopandas as gpd
from shapely.geometry import Point
import pickle
import urllib.request

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


def add_months(ucdp, world_grid):

    diff = ucdp['year'].max() - world_grid['year'].max()

    subset_list = []

    for i in np.arange(1, diff+1, 1):

        subset = world_grid[world_grid['year'] == world_grid['year'].max()].copy()
        subset['year'] = world_grid['year'].max() + i

        subset_list.append(subset)

    new_years = pd.concat(subset_list)
    world_grid_all_years = pd.concat([world_grid, new_years])

    month = [str(i).zfill(2) for i in np.arange(1,13,1)]
    world_grid_all_years.loc[:,'month'] = world_grid_all_years.apply(lambda _: month, axis=1)
    world_grid_all_months = world_grid_all_years.sort_values('year').explode('month').copy()
    world_grid_all_months['year_months_start'] =  world_grid_all_months['year'].astype(str) + '-' +  world_grid_all_months['month'].astype(str)

    year_months = sorted(world_grid_all_months['year_months_start'].unique())
    ts = len(year_months)
    month_ids = np.arange(109, ts + 109, 1)
    month_id_dict = dict(zip(year_months,month_ids))
    month_df = pd.DataFrame({'year_months_start' : year_months, 'month_id': month_ids})
    world_grid_all_months_id = world_grid_all_months.merge(month_df, how = 'left', on = 'year_months_start')

    return world_grid_all_months_id


def prio_ucdp_merge(ucdp, world_grid_all_months):
    ucdp_tmp1 = ucdp.copy()

    ucdp_tmp1['year_months_start'] = ucdp_tmp1['date_start'].str.slice(start = 0, stop = 7) # Date YYYY-MM-DD
    ucdp_tmp1['year_months_end'] = ucdp_tmp1['date_start'].str.slice(start = 0, stop = 7) # Date YYYY-MM-DD


    mask1 = (ucdp_tmp1['year'] != ucdp_tmp1['year_months_start'].str.slice(start = 0, stop = 4).astype(int))
    mask2 = (ucdp_tmp1['year'] != ucdp_tmp1['year_months_end'].str.slice(start = 0, stop = 4).astype(int))

    # correction. Note that end and start year for the four entries that is corrected is the same.
    ucdp_tmp1.loc[mask1 | mask2, 'year'] = ucdp_tmp1.loc[mask1 | mask2,'year_months_start'].str.slice(start = 0, stop = 4).astype(int)

    sum_list = ['deaths_a','deaths_b', 'deaths_civilians', 'deaths_unknown','best', 'high', 'low']

    ucdp_monthly_unit = ucdp_tmp1.groupby(['year_months_start','year', 'priogrid_gid']).sum()[sum_list].reset_index()
    ucdp_monthly_unit.rename(columns={'priogrid_gid':'gid'}, inplace=True)

    ucdp_monthly_unit['log_best'] = np.log(ucdp_monthly_unit['best'] +1)
    ucdp_monthly_unit['log_low'] = np.log(ucdp_monthly_unit['low'] +1)
    ucdp_monthly_unit['log_high'] = np.log(ucdp_monthly_unit['high'] +1)

    prio_ucdp_df = world_grid_all_months.merge(ucdp_monthly_unit, how = 'left', on = ['gid', 'year_months_start', 'year'])
    prio_ucdp_df.fillna(0, inplace=True)

    return prio_ucdp_df


def compile_combined_df():

    prio_grid = get_prio_shape()
    gwno = get_gwno()
    ucdp = get_ucdp()

    world_grid = prio_grid.merge(gwno, how = 'right', on = 'gid') # if you just merge this on outer I think you get the full grid needed for R-UNET
    world_grid_all_months = add_months(ucdp, world_grid)
    prio_ucdp = prio_ucdp_merge(ucdp, world_grid_all_months)

    data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/'

    with open(f'{data_dir}g_df_ucdp_prio_large.pkl', 'wb') as file:
        pickle.dump(prio_ucdp, file)

    prio_ucdp_pd = pd.DataFrame(prio_ucdp.drop(columns= 'geometry').copy())

    with open(f'{data_dir}df_ucdp_prio_large.pkl', 'wb') as file:
        pickle.dump(prio_ucdp_pd, file)

    prio_ucdp_pd.to_csv(f'{data_dir}df_ucdp_prio_large.csv')

if __name__ == "__main__":
    compile_combined_df()
