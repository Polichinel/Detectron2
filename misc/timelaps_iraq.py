# geo_env_2022
import os
import numpy as np
import pandas as pd # original 1.2.3
import pickle
import matplotlib.pyplot as plt


def get_data():

    data_dir = '/home/simon/Documents/Bodies/data/PRIO'

    with open(f'{data_dir}/full_interpl_df_exposure.pkl', 'rb') as file:
        full_df = pickle.load(file)

    return(full_df)


def plot_maps(df):

    map_dir = '/home/simon/Documents/Bodies/figures/maps/iraq_timelaps/'

    for feature in df.columns:

        feature_dir = os.path.join(map_dir, feature)
        os.makedirs(feature_dir, exist_ok = True)

        for month_id in df['month_id'].unique():

            x = df.loc[df['month_id'] == month_id, 'xcoord'] 
            y = df.loc[df['month_id'] == month_id, 'ycoord'] 
            c = df.loc[df['month_id'] == month_id, feature] 

            plt.figure(figsize=[28,12])
            plt.scatter(x, y, c = c, cmap = 'rainbow', marker = 's', s = 2, vmin= c.min(), vmax=c.max())
        
            fig_title = f'{feature}_{str(month_id).zfill(3)}'

            plt.title(fig_title)
            plt.savefig(feature_dir + '/' + fig_title + '.JPG', bbox_inches="tight")
            plt.cla() # idk if this is also needed..
            plt.close('all') # so they do not display or take up mem

def make_maps():
    df = get_data()
    plot_maps(df)

if __name__ == '__main__':
    make_maps()