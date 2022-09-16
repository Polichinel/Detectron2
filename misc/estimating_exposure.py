# geo_env_2022

import numpy as np
import os

#import geopandas as gpd
import pandas as pd
import pickle
import sklearn
import urllib.request

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel


def get_cshapes():
    location = '/home/simon/Documents/Bodies/data/PRIO' 
    path_cshapes = location + "/CShapes-2.0.csv"
    
    if os.path.isfile(path_cshapes) == True:
        print('file already downloaded')
        cshapes = pd.read_csv(path_cshapes)


    else: 
        print('Beginning file download CShapes...')

        url_cshapes = 'https://icr.ethz.ch/data/cshapes/CShapes-2.0.csv'
    
        urllib.request.urlretrieve(url_cshapes, path_cshapes)
        cshapes = pd.read_csv(path_cshapes)

    return cshapes


def get_sub_df():
    
    data_dir = '/home/simon/Documents/Bodies/data/PRIO'

    with open(f'{data_dir}/prio_ucdp_full.pkl', 'rb') as file:
        df = pickle.load(file)

    cshapes = get_cshapes()
    cshapes_mask = ['Gaza', 'West Bank', 'Jordan', 'Palestine', 'Israel', 'Iraq', 'Syria', 'Lebanon', 'Turkey (Ottoman Empire)']
    gwno_to_keep = cshapes.loc[cshapes['cntry_name'].isin(cshapes_mask), 'gwcode'].unique()
    df_sub = df[df['gwno'].isin(gwno_to_keep)].copy()

    return df_sub


def esstimate_kernel(df_sub):
    
    # Kriteria log_best > 6 
    test_gids = df_sub.loc[df_sub['log_best'] > 6, 'gid'].unique()
    n_timelines = test_gids.shape[0]

    print(f'number of timelines used: {n_timelines}')

    mask = df_sub['gid'].isin(test_gids)
    df_test2 = df_sub[mask].copy()
    df_test2.sort_values(['month_id', 'gid'], inplace = True) # this make reshape work

    Y = np.array(df_test2['log_best']).reshape(-1, n_timelines)
    X = np.array(df_test2[['month_id']]).reshape(-1, n_timelines)

    # GP
    noise_std = Y.std()

    kernel_short = ConstantKernel() * Matern(length_scale=4.0, length_scale_bounds=(1, 32.0), nu=1.5) # nu = 1.5 gives v = 3/2 so matern32
    kernel_long = ConstantKernel() * RBF(length_scale=20.0, length_scale_bounds=(32, 320))

    kernel =  kernel_short + kernel_long + WhiteKernel(noise_level=noise_std)

    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gaussian_process.fit(X, Y)
    gaussian_process.kernel_

    return gaussian_process.kernel_


def predict_gp(df, kernel):
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    n_timelines_new = df['gid'].unique().shape[0]
    df.sort_values(['month_id', 'gid'], inplace = True) # this make reshape work
    x_new = np.array(df[['month_id']]).reshape(-1, n_timelines_new)
    y_new = np.array(df[['log_best']]).reshape(-1, n_timelines_new)

    gaussian_process.fit(x_new,y_new) #this refit is only needed bacouse you have more timelines here and treat them aas features...
    print(gaussian_process.kernel_)
    mean_prediction, std_prediction = gaussian_process.predict(x_new, return_std=True)

    return(mean_prediction, std_prediction)


def get_spatial(df):

    # Critaria log_best > 4
    sub_months = df[df['log_best'] > 4].groupby('month_id').count().sort_values('gid', ascending = False).iloc[:60,:].sample(5).index.values

    noise_std = df['tce'].std()
    kernel = ConstantKernel() * Matern(length_scale=0.5, length_scale_bounds=(0.1, 4.0), nu=1.5) + WhiteKernel(noise_level=noise_std)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    for month_id in sub_months:
        df_month = df[df['month_id'] == month_id].copy()
        y = np.array(df_month['tce']).reshape(-1,1)
        X = np.array(df_month[['xcoord', 'ycoord']]).reshape(-1,2)
        
        gaussian_process.fit(X, y)
        print(gaussian_process.kernel_)

    gaussian_process.kernel_.get_params()

    amplitude = ConstantKernel(constant_value = 0.7**2, constant_value_bounds = 'fixed')
    matern =  Matern(length_scale=0.5, nu=1.5, length_scale_bounds = 'fixed')
    epsilon = WhiteKernel(noise_level=0.4, noise_level_bounds = 'fixed')

    kernel = amplitude * matern + epsilon

    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    mean_prediction_list = []
    std_prediction_list = []
    gid_list = [] # for easy merge
    month_id_list = [] # for easy merge
    xcoord_list = []
    ycoord_list = []
    tce_list = []

    for month_id in df['month_id'].unique():
        df_month = df[df['month_id'] == month_id].copy()
        y = np.array(df_month['tce']).reshape(-1,1)
        X = np.array(df_month[['xcoord', 'ycoord']]).reshape(-1,2)
        
        gaussian_process.fit(X, y)
        #print(gaussian_process.kernel_)
        mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

        mean_prediction_list.append(mean_prediction)
        std_prediction_list.append(std_prediction)
        gid_list.append(np.array(df_month['gid']))
        month_id_list.append(np.array(df_month['month_id']))
        xcoord_list.append(np.array(df_month['xcoord']))
        ycoord_list.append(np.array(df_month['ycoord']))
        tce_list.append(np.array(df_month['tce']))


    tsce_df = pd.DataFrame({'tsce':np.array(mean_prediction_list).ravel(), 
                            'tsce_std':np.array(std_prediction_list).ravel(), 
                            'gid':np.array(gid_list).ravel(), 
                            'month_id':np.array(month_id_list).ravel(),
                            'xcoord':np.array(xcoord_list).ravel(),
                            'ycoord': np.array(ycoord_list).ravel(),
                            'tce': np.array(tce_list).ravel()})

    return(tsce_df)

def compile():

    df = get_sub_df()
    kernel = esstimate_kernel(df)

    # lock kernel
    kernel.set_params(k1__k1__k1__constant_value_bounds = 'fixed', 
                    k1__k1__k2__length_scale_bounds = 'fixed', 
                    k1__k2__k1__constant_value_bounds = 'fixed', 
                    k1__k2__k2__length_scale_bounds = 'fixed', 
                    k2__noise_level_bounds = 'fixed' )

    mean_prediction, std_prediction = predict_gp(df, kernel)

    df['tce'] = mean_prediction.reshape(-1,1)
    df['tce_std'] = std_prediction.reshape(-1,1)

    tsce_df = get_spatial(df)
    merge_on = list(set.intersection(set(tsce_df.columns),set(df.columns)))
    df_exposure = df.merge(tsce_df, how = 'left', on = merge_on)

    data_dir = '/home/simon/Documents/Bodies/data/PRIO/'
    with open(f'{data_dir}full_interpl_df_exposure.pkl', 'wb') as file:
        pickle.dump(df_exposure, file)

    print('Done.')

if __name__ == '__main__':
    compile()