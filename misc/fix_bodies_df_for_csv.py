# use ra_env

import os
import numpy as np
import pandas as pd
import pickle

bodies_pkl = '/home/simon/Documents/Bodies/data/done_dfs/bodies_df_2022_v1_0.pkl'
bodies_df = pd.read_pickle(bodies_pkl)
bodies_df.drop(columns=['by-line title', 'caption/abstract', 'object name'], inplace=True)

# fixed distance days
bodies_df['distance_days'] = bodies_df['distance_days'].astype(str).str.split(' ', expand = True).iloc[:,0]
mask = bodies_df['distance_days'] == 'NaT'
bodies_df.loc[mask, 'distance_days'] = np.nan
bodies_df.loc[:,'distance_days'].fillna(499, inplace = True)
bodies_df.loc[:, 'distance_days'] = bodies_df['distance_days'].astype(int)

# Fix data created
bodies_df['date created'] = bodies_df['date created'].astype(str).str.split(' ', expand = True).iloc[:,0].astype(int)

# Fix time created
bodies_df['time created'] = bodies_df['time created'].astype(str)

# fix city
bodies_df['city'] = bodies_df['city'].astype(str)

# fix province/state
bodies_df['province/state'] = bodies_df['province/state'].astype(str)

# fix sub-location
bodies_df['sub-location'] = bodies_df['sub-location'].astype(str)

path_df = '/home/simon/Documents/Bodies/data/done_dfs/'
bodies_df.to_csv(f'{path_df}bodies_df_2022_v1_1.csv', index=False)
bodies_df.to_pickle(f'{path_df}bodies_df_2022_v1_1.pkl')