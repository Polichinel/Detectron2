# if local use new_torch_env

import os
import sys
import time

import numpy as np
import pandas as pd
# import seaborn as sns

from collections import Counter

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from xml.etree import ElementTree, ElementInclude

import pickle
from functools import reduce

import iptcinfo3
from iptcinfo3 import IPTCInfo

import detectron2

from utills_output import *
from instance_to_output import *


# Shell ----------------------------------------- #

def make_df_tX():
    """...."""

    full_option = input('Use:\n a) Annoated subset \n b) Full set of images\n ')

    if full_option == 'a':
        img_set = 'annotated'
        models = ['faster_rcnn_R_50_FPN_3x', 
              'faster_rcnn_R_101_FPN_3x', 
              'faster_rcnn_X_101_32x8d_FPN_3x', 
              'retinanet_R_50_FPN_3x', 
              'retinanet_R_101_FPN_3x'] 

        print('You choose a, the annotated subset of images')


    elif full_option == 'b':
        img_set = 'FULL'

        models = ['faster_rcnn_R_50_FPN_3x_FULL', 
              'faster_rcnn_R_101_FPN_3x_FULL', 
              'faster_rcnn_X_101_32x8d_FPN_3x_FULL', 
              'retinanet_R_50_FPN_3x_FULL', 
              'retinanet_R_101_FPN_3x_FULL'] 

        print('You choose b, the Full (non-annotated) subset of images')


    else:
        print('Wrong answer....')
        sys.exit()

    threshold = input('Enter threshold (0.1 - 0.9): ')
    threshold = float(threshold)

    df_name = f'OD_DF_{img_set}_t{int(threshold*100)}'
    df_dir = '/home/simon/Documents/Bodies/data/OD_dataframes'
    df_path = f'{df_dir}/{df_name}.pkl'
    os.makedirs(df_dir, exist_ok = True)

    if os.path.exists(df_path):
        print(f'df {df_name} already exist...')
        overwrite_option = input(f'Overwrite existing df: {df_name} (y/n)\n')

        if overwrite_option == 'n':
            print('Script aborted')
            sys.exit()

        elif overwrite_option == 'y':
            print(f'Overwriting existing df: {df_name} in :')
            for i in range(10):
                print(f'{10-i} seconds. Press Ctrl+c to cancel script')
                time.sleep(1)
            print('Overwriting initiated!!!')

    for model_name in models:

            # This block can be used in the output ot df. -----------------------------------------------------
        print(f'Generating output for model {model_name} w/ threshold {threshold}...')
        
        output_list, all_img_feature_list = get_output_tX(model_name, threshold)

        print(f'\nOutput from {len(output_list)} images handled...')

        # pickle configurations and save
        #location = f'/home/projects/ku_00017/data/generated/bodies/detectron_outputs/{model_name}'
        new_output_dir =f'/home/simon/Documents/Bodies/data/computerome_outputs/alt_threshold_outputs/{model_name}'
        os.makedirs(new_output_dir, exist_ok = True)

        with open(new_output_dir + f'/output_list_t{int(threshold*100)}.pkl', 'wb') as file:
            pickle.dump(output_list, file)

        with open(new_output_dir + f'/all_img_feature_list_t{int(threshold*100)}.pkl', 'wb') as file:
            pickle.dump(all_img_feature_list, file)


    print(f'Outputs with threshold {threshold} generated, pickled and saved...')

    if full_option == 'a':
        img_set = 'annotated'
        print('You choose a, the annotated subset of images')

        df = annotate_df(alt_threshold = threshold) # shitty name still
        df = meta_to_df(df)

    elif full_option == 'b':
        df = annotate_df(FULL = True)
        full_img_dir = '/media/simon/Seagate Expansion Drive/images_spanner'
        
        if os.path.exists(full_img_dir):
            df = meta_to_df(df, img_dir=full_img_dir)

        else:
            print(f'External Bodies SSD not found on {full_img_dir}... Plug in SDD and try again..')
            sys.exit()

    with open(df_path, 'wb') as file:
        pickle.dump(df, file)

if __name__ == "__main__":
    make_df_tX()



# ------------------------------------------------# 

# if full_option == 'y' and alt_threshold_option == 'n':
#     df = annotate_df(FULL = True)
#     df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

#     with open('spanner_full_df_t10.pkl', 'wb') as file:
#         pickle.dump(df, file)


# elif full_option == 'n' and alt_threshold_option == 'n':
#     annotated_df = annotate_df()
#     annotated_df = meta_to_df(annotated_df)

#     with open('spanner_annotated_df_t10.pkl', 'wb') as file:
#         pickle.dump(annotated_df, file)


# elif full_option == 'y' and alt_threshold_option == 'y':
#     df = annotate_df(FULL = True, alt_threshold = True)
#     df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

#     with open('spanner_full_df_t30.pkl', 'wb') as file:
#         pickle.dump(df, file)


# elif full_option == 'n' and alt_threshold_option == 'y':
#     df = annotate_df(FULL = False, alt_threshold = True)
#     df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

#     with open('spanner_annotated_df_t30.pkl', 'wb') as file:
#         pickle.dump(df, file)

# else:
#     print('Unknown Error...')





































# Shell ----------------------------------------- #
# model_option_dict = {'a': 'faster_rcnn_R_50_FPN_3x', 'b': 'faster_rcnn_R_101_FPN_3x', 'c': 'faster_rcnn_X_101_32x8d_FPN_3x', 'd': 'retinanet_R_50_FPN_3x', 'e': 'retinanet_R_101_FPN_3x'}# ----------------------

# model_option = input(f"Choose model: \na) {model_option_dict['a']} \nb) {model_option_dict['b']} \nc) {model_option_dict['c']} \nd) {model_option_dict['d']} \ne) {model_option_dict['e']}\n")
# full_option = input('Use full set of images? (y/n): ')
# alt_threshold_option = input('Use alternative threshold (0.3)? (y/n)')


# if full_option == 'y':
#     model_name = model_option_dict[model_option] + '_FULL'

# elif full_option == 'n':
#     model_name = model_option_dict[model_option]

# if alt_threshold_option == 'y':
#     dir = 'alt_threshold_outputs'
#     output_list_name = 'output_list_t30.pkl'
#     feature_list_name = 'all_img_feature_list_t30.pkl'

# elif alt_threshold_option == 'n':
#     dir = 'detectron_outputs'
#     output_list_name = 'output_list.pkl'
#     feature_list_name = 'all_img_feature_list.pkl'

# # you did the name thing to be more secure.. and then you did not do it for alt thresholds...
# if full_option == 'y' and alt_threshold_option == 'n':
#     output_list_name = f'output_list_FULL.pkl'
#     feature_list_name = 'all_img_feature_list_FULL.pkl'

# path = f'/home/simon/Documents/Bodies/data/computerome_outputs/{dir}/{model_name}'
# output_list_path = os.path.join(path,output_list_name)
# feature_list_path = os.path.join(path,feature_list_name)

# if os.path.exists(output_list_path) & os.path.exists(feature_list_path):
#     print(f'Output path: {output_list_path} \nfeature_list_path: {feature_list_path} ')

# else:
#     print('file does not exist....')
# -------------------------------------------------------#


# # you did the name thing to be more secure.. and then you did not do it for alt thresholds...
# if model_name.split('_')[-1] == 'FULL':
#     instances_list_dir = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model_name}/instances_list_FULL.pkl'
#     #img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' #full run!!!
#     img_dir='/media/simon/Seagate Expansion Drive/images_spanner'

# else:
#     instances_list_dir = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model_name}/instances_list.pkl'
#     img_dir = '/home/simon/Documents/Bodies/data/jeppe/images'



# -------------------------------------------------------#

# annotated:

# annotated_df = annotate_df()
# annotated_df = meta_to_df(annotated_df)

# with open('spanner_annotated_df.pkl', 'wb') as file:
#     pickle.dump(annotated_df, file)

# # Full

# df = annotate_df(FULL = True)
# df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

# with open('spanner_full_df.pkl', 'wb') as file:
#     pickle.dump(df, file)
