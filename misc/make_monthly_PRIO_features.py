import os
import numpy as np
import pandas as pd # original 1.2.3
import pickle
import urllib.request
from scipy import interpolate



def get_data():

    data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/'

    with open(f'{data_dir}df_ucdp_prio.pkl', 'rb') as file:
        prio_ucdp = pickle.load(file)

    coords_long = prio_ucdp[['gid', 'xcoord', 'ycoord']].copy() # conveniet for plotting
    coords = coords_long.groupby('gid').mean()
    coords.reset_index(inplace = True)

    # FEATURES:
    # bdist3 capdist gcp_mer gcp_ppp excluded nlights_mean nlights_calib_mean 
    # petroleum_y pop_gpw_sum pop_gpw_sd prec_gpcp urban_ih agri_ih pasture_ih
    # barren_ih irrig_sum imr_mean cmr_mean mountain_mean  ttime_mean

    location = '/home/simon/Documents/Bodies/data/PRIO'
    path_yearly = location + '/PRIO-GRID Yearly Variables for 1989-2014 - 2022-09-08.csv' #https://grid.prio.org/#/download 
    path_static = location + '/PRIO-GRID Static Variables - 2022-09-08.csv' #https://grid.prio.org/#/download 

    yearly_df = pd.read_csv(path_yearly)
    yearly_df = yearly_df.merge(coords, on='gid', how='left') # for plotting. faster to scatter then geopandas plot

    static_df = pd.read_csv(path_static)
    static_df = static_df.merge(coords, on='gid', how='left') # for plotting. faster to scatter then geopandas plot

    yearly_df.sort_values(['gid', 'year'], inplace=True)
    #static_df.sort_values(['gid', 'year'], inplace=True)

    return(yearly_df, static_df)


def expl_interpl_extrapl(data, feature_list):
    
    n_months = 12
    n_gid = data['gid'].unique().shape[0] # number of groups
    n_years = data['year'].unique().shape[0] # number of years

    # list of lists with temporal sub unites. eg. months
    months = [list(np.arange(1, n_months+1))] * data.shape[0] # list of lists with temporal sub unites. eg. months

    # append the colum
    data['month'] = months 

    # explode the df in reference to the new col and drop the ond index
    data = data.explode('month').reset_index(drop=True) 

    # we'll only keep the original values in the first tempoeal sub unit. Interpolate rest.
    data.loc[data['month'] != 1, feature_list] = np.nan 

    # number of year x number of months. And then number of groups. +1 becaus Views month_id starts at 1
    data['month_id'] = list(np.arange(1,n_years * n_months+1)) * n_gid 

    # to be used below
    obs_features_list = []

    for f in feature_list:
        observed_col_name = f'{f}_observed'

        # an identifier for the non-interpolated values
        data[observed_col_name] = data.loc[:,f].notna() 
        obs_features_list.append(observed_col_name)

    # interpolate and extrapolate
    data = data.groupby('gid').apply(lambda x: x.interpolate(method='linear', axis = 0, limit_direction = 'both', fill_value='extrapolate'))


    # clip at group observed min and max for each feature.
    for f, o in list(zip(feature_list, obs_features_list)):
        data[f] = data.groupby('gid').apply(lambda x: np.clip(x[f], x[x[o] == True][f].min(), x[x[o] == True][f].max())).reset_index(drop=True)

    return(data)


def explode_static(static_df, yearly_df):
    
    n_months = 12
    years = sorted(yearly_df['year'].unique())
    n_gid = yearly_df['gid'].unique().shape[0] # number of groups - could be from either static or yearly. Same
    n_years = yearly_df['year'].unique().shape[0] # number of years

    df = static_df.copy()

    df['year'] = [years] * df.shape[0]
    df['month'] = [list(np.arange(1, n_months+1))] * df.shape[0]

    df = df.explode('year')
    df = df.explode('month')

    df.reset_index(inplace=True, drop= True)
    
    df['month_id'] = list(np.arange(1,n_years * n_months+1)) * n_gid 

    return(df)


def compile():
    
    data_dir = '/home/simon/Documents/Bodies/data/PRIO'

    # do the yearly
    yearly_df, static_df = get_data()
    yearly_feature_list = list(yearly_df.columns[2:-2])
    yearly_df_interpl = expl_interpl_extrapl(yearly_df, yearly_feature_list)
    yearly_df_interpl.fillna(0, inplace=True)

    yearly_df_interpl.to_pickle(f'{data_dir}/yearly_prio_interpl.pkl')

    # do the static
    static_df_interpl = explode_static(static_df, yearly_df)
    static_df_interpl.fillna(0, inplace=True)

    static_df_interpl.to_pickle(f'{data_dir}/static_prio_interpl.pkl')

    # merge
    merge_on = list(set.intersection(set(yearly_df_interpl.columns), set(static_df_interpl.columns)))
    full_df_interpl = yearly_df_interpl.merge(static_df_interpl, on=merge_on, how='outer')

    full_df_interpl.to_pickle(f'{data_dir}/full_prio_interpl.pkl')

if __name__ == "__main__":

    compile()