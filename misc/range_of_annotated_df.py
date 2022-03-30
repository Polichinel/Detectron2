# if local use new_torch_env

import os
import sys
import time
import numpy as np
#import pandas as pd
#import seaborn as sns
#from collections import Counter
#import cv2
#import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
#import seaborn as sns
#from xml.etree import ElementTree, ElementInclude

import pickle
#from functools import reduce
#import iptcinfo3
#from iptcinfo3 import IPTCInfo
#import detectron2

from utills_output import *
from instance_to_output import *

def create_range_of_annotated_dfs():

    img_set = 'annotated'

    models = ['faster_rcnn_R_50_FPN_3x', 
            'faster_rcnn_R_101_FPN_3x', 
            'faster_rcnn_X_101_32x8d_FPN_3x', 
            'retinanet_R_50_FPN_3x', 
            'retinanet_R_101_FPN_3x'] 

    thresholds = np.arange(0.1,0.76,0.01)

    for threshold in thresholds:

        threshold = round(threshold, 2) # to avoid wierd rounding...

        df_name = f'OD_DF_{img_set}_t{int(threshold*100)}'
        df_dir = '/home/simon/Documents/Bodies/data/OD_dataframes'
        df_path = f'{df_dir}/{df_name}.pkl'
        os.makedirs(df_dir, exist_ok = True)

        for model_name in models:

            # This block can be used in the output ot df. -----------------------------------------------------
            print(f'Generating output for model {model_name} w/ threshold {threshold}...')
            
            output_list, all_img_feature_list = get_output_tX(model_name, threshold)

            print(f'\nOutput from {len(output_list)} images handled...')

            # pickle configurations and save
            new_output_dir =f'/home/simon/Documents/Bodies/data/computerome_outputs/alt_threshold_outputs/{model_name}'
            os.makedirs(new_output_dir, exist_ok = True)

            with open(new_output_dir + f'/output_list_t{int(threshold*100)}.pkl', 'wb') as file:
                pickle.dump(output_list, file)

            with open(new_output_dir + f'/all_img_feature_list_t{int(threshold*100)}.pkl', 'wb') as file:
                pickle.dump(all_img_feature_list, file)


        print(f'Outputs with threshold {threshold} generated, pickled and saved...')

        df = annotate_df(alt_threshold = threshold) # shitty function name still
        df = meta_to_df(df)


        with open(df_path, 'wb') as file:
            pickle.dump(df, file)

        print(f'df w/ t = {threshold} generated, pickled and saved...')


if __name__ == "__main__":
    create_range_of_annotated_dfs()

