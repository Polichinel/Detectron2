# if local use new_torch_env

import os

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


# Shell ----------------------------------------- #
full_option = input('Use full set of images? (y/n): ')
alt_threshold_option = input('Use alternative threshold (0.3)? (y/n)')

# ------------------------------------------------# 

if full_option == 'y' and alt_threshold_option == 'n':
    df = annotate_df(FULL = True)
    df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

    with open('spanner_full_df_t10.pkl', 'wb') as file:
        pickle.dump(df, file)


elif full_option == 'n' and alt_threshold_option == 'n':
    annotated_df = annotate_df()
    annotated_df = meta_to_df(annotated_df)

    with open('spanner_annotated_df_t10.pkl', 'wb') as file:
        pickle.dump(annotated_df, file)


elif full_option == 'y' and alt_threshold_option == 'y':
    df = annotate_df(FULL = True, alt_threshold = True)
    df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

    with open('spanner_full_df_t30.pkl', 'wb') as file:
        pickle.dump(df, file)


elif full_option == 'n' and alt_threshold_option == 'y':
    df = annotate_df(FULL = False, alt_threshold = True)
    df = meta_to_df(df, img_dir='/media/simon/Seagate Expansion Drive/images_spanner')

    with open('spanner_annotated_df_t30.pkl', 'wb') as file:
        pickle.dump(df, file)

else:
    print('Unknown Error...')





































# Shell ----------------------------------------- #
model_option_dict = {'a': 'faster_rcnn_R_50_FPN_3x', 'b': 'faster_rcnn_R_101_FPN_3x', 'c': 'faster_rcnn_X_101_32x8d_FPN_3x', 'd': 'retinanet_R_50_FPN_3x', 'e': 'retinanet_R_101_FPN_3x'}# ----------------------

model_option = input(f"Choose model: \na) {model_option_dict['a']} \nb) {model_option_dict['b']} \nc) {model_option_dict['c']} \nd) {model_option_dict['d']} \ne) {model_option_dict['e']}\n")
full_option = input('Use full set of images? (y/n): ')
alt_threshold_option = input('Use alternative threshold (0.3)? (y/n)')


if full_option == 'y':
    model_name = model_option_dict[model_option] + '_FULL'

elif full_option == 'n':
    model_name = model_option_dict[model_option]

if alt_threshold_option == 'y':
    dir = 'alt_threshold_outputs'
    output_list_name = 'output_list_t30.pkl'
    feature_list_name = 'all_img_feature_list_t30.pkl'

elif alt_threshold_option == 'n':
    dir = 'detectron_outputs'
    output_list_name = 'output_list.pkl'
    feature_list_name = 'all_img_feature_list.pkl'

# you did the name thing to be more secure.. and then you did not do it for alt thresholds...
if full_option == 'y' and alt_threshold_option == 'n':
    output_list_name = f'output_list_FULL.pkl'
    feature_list_name = 'all_img_feature_list_FULL.pkl'

path = f'/home/simon/Documents/Bodies/data/computerome_outputs/{dir}/{model_name}'
output_list_path = os.path.join(path,output_list_name)
feature_list_path = os.path.join(path,feature_list_name)

if os.path.exists(output_list_path) & os.path.exists(feature_list_path):
    print(f'Output path: {output_list_path} \nfeature_list_path: {feature_list_path} ')

else:
    print('file does not exist....')
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
