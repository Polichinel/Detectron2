# use geo_env_2022
import numpy as np
import pandas as pd # original 1.2.3
import pickle

def get_data():

    image_data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/'

    pkl_file = open(f'{image_data_dir}bodies_df_exposure.pkl', 'rb')

    bodies_df = pickle.load(pkl_file)
    pkl_file.close()

    prio_data_dir = '/home/simon/Documents/Bodies/data/PRIO/'

    pkl_file = open(f'{prio_data_dir}full_interpl_df_exposure.pkl', 'rb')

    prio_df = pickle.load(pkl_file)
    pkl_file.close()

    return(bodies_df, prio_df)


def merge_dfs(bodies_df, prio_df):

    # drop this as it is making trouble.
    bodies_df['year'] = bodies_df['new_year'] # correct year feature
    bodies_df.drop(columns=['headline', 'new_year'], inplace=True)
    bodies_df.dropna(subset=['gid', 'month_id', 'year'], inplace= True) # there  

    merge_on = ['gid', 'month_id', 'year']

    for i in merge_on:
        bodies_df.loc[:,i] = bodies_df.loc[:,i].astype('int')

    bodies_df.reset_index(drop=True, inplace= True)

    bodies_prio_df = bodies_df.merge(prio_df, on = merge_on, how='left', suffixes=('_left', '_right'))

    # just drop the lefts and rename the rights (not that impoortant which one)
    bodies_prio_df.drop(columns=['log_best_left', 'log_high_left', 'log_low_left', 'tce_left', 'tsce_left'], inplace=True)
    bodies_prio_df.rename(columns = {'log_best_right':'log_best', 'log_high_right':'log_high' , 'log_low_right': 'log_low', 'tce_right': 'tce', 'tsce_right' : 'tsce'}, inplace = True)

    return(bodies_prio_df)


def compile():
    bodies_df, prio_df = get_data()
    bodies_prio_df = merge_dfs(bodies_df, prio_df)

    data_dir = '/home/simon/Documents/Bodies/data/done_dfs/'

    bodies_prio_df.to_pickle(f'{data_dir}bodies_df_2022_v1_0.pkl')
    bodies_prio_df.to_csv(f'{data_dir}bodies_df_2022_v1_0.csv')


if __name__ == '__main__':
    compile()