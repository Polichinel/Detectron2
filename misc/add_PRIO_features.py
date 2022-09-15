import os
import numpy as np
import pandas as pd # original 1.2.3
import pickle


def get_data(data_dir):

    #data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/'
    with open(f'{data_dir}df_ucdp_prio.pkl', 'rb') as file:
        prio_ucdp = pickle.load(file)


    #data_dir = '/home/simon/Documents/Bodies/data/PRIO'
    with open(f'{data_dir}/full_prio_interpl.pkl', 'rb') as file:
        full_df = pickle.load(file)

    return(prio_ucdp, full_df)


def merge_data(prio_ucdp, full_df):

    # correction for month_id
    offset = prio_ucdp['month_id'].min() -1
    full_df['month_id'] = full_df['month_id'] + offset

    # year cut of for ucdp and type correction
    prio_ucdp = prio_ucdp[prio_ucdp['year'] <= 2014]
    prio_ucdp['month_id'] = prio_ucdp['month_id'].astype(int)
    prio_ucdp['month'] = prio_ucdp['month'].astype(int)

    merge_on = list(set.intersection(set(prio_ucdp.columns),set(full_df.columns)))
    prio_ucdp_full = prio_ucdp.merge(full_df, on = merge_on, how='outer')

    return(prio_ucdp_full)


def compile():

    data_dir = '/home/projects/ku_00017/data/raw/PRIO/'

    prio_ucdp, full_df = get_data(data_dir)
    prio_ucdp_full = merge_data(prio_ucdp, full_df)

    prio_ucdp_full.to_pickle(f'{data_dir}/prio_ucdp_full.pkl')


if __name__ == '__main__':
    compile()