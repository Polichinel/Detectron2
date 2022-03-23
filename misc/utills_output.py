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



def get_img_path(img_dir):

    """Creates a list of all image paths."""

    # right now this does not take into account whether the image was anotated or not.
    # It also does not handle test or train.

    img_path_list = []

    for root, dirs, files in os.walk(img_dir):
        for img_name in files:
            if img_name.split('.')[1] == 'jpg':
                img_path = os.path.join(img_dir, img_name)                
                img_path_list.append(img_path)

    return(img_path_list)


def get_ensamble_est(df):

    """Creates a mean and median estimate from the 5 individual models"""

    front_str_list = []
    feature_dict = {'img_id': df['img_id'].to_numpy()}

    for i in df.columns[1:]:
        if '_' in i: # no meta features have this.
            front_str_list.append(i.split('_')[0])

    features = list(set(front_str_list))

    for i in features:
        mean_dict = {f'{i}_mean':df.loc[:,df.columns.str.startswith(i)].mean(axis=1)}
        median_dict = {f'{i}_median': np.median(df.loc[:,df.columns.str.startswith(i)], axis=1)}    
        feature_dict = {**feature_dict, **mean_dict}
        feature_dict = {**feature_dict, **median_dict}

    df_ensamble_est = pd.DataFrame(feature_dict)

    df_ensamble_est =  pd.merge(df, df_ensamble_est,on=['img_id'], how='outer')

    return(df_ensamble_est)


def make_df(model, FULL = False, alt_threshold = False):

    """Takes a specific model name and returns the corrosponding output generated on combuterome"""

    if FULL == True and alt_threshold == False:
        dir = 'detectron_outputs'
        output_list_name = 'output_list_FULL.pkl'
        feature_list_name = 'all_img_feature_list_FULL.pkl'
        model_name = f'{model}_FULL'
        # output_list_path = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model}_FULL/output_list_FULL.pkl' 
        # all_img_feature_list_path = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model}_FULL/all_img_feature_list_FULL.pkl'
        # instances_path = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model}_FULL/instances_list_FULL.pkl'


    elif FULL == False and alt_threshold == False:
        dir = 'detectron_outputs_test'
        output_list_name = 'output_list.pkl'
        feature_list_name = 'all_img_feature_list.pkl'
        model_name = model
        #output_list_path = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs_test/{model}/output_list.pkl' 
        #all_img_feature_list_path = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs_test/{model}/all_img_feature_list.pkl' 
        # instances_path = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs_test/{model}/instances_list.pkl' 


# ----------------------------
    elif FULL == True and alt_threshold == True:

        dir = 'alt_threshold_outputs'
        output_list_name = 'output_list_t30.pkl'
        feature_list_name = 'all_img_feature_list_t30.pkl'
        model_name = f'{model}_FULL'


    # alt treshold shoul be a number...
    elif FULL == False and alt_threshold == True:

        dir = 'alt_threshold_outputs'
        output_list_name = 'output_list_t30.pkl'
        feature_list_name = 'all_img_feature_list_t30.pkl'
        model_name = model

# ----------------------------


    path = f'/home/simon/Documents/Bodies/data/computerome_outputs/{dir}/{model_name}'
    output_list_path = os.path.join(path,output_list_name)
    all_img_feature_list_path = os.path.join(path,feature_list_name)

    if os.path.exists(output_list_path) & os.path.exists(all_img_feature_list_path):
        print(f'Files exists:\n Outputs path: {output_list_path} \n features path: {all_img_feature_list_path} ')

    else:
        print('Files does not exist....')


    with open(output_list_path, 'rb') as file:
        outputs_list = pickle.load(file)

    with open(all_img_feature_list_path, 'rb') as file:
        all_img_feature_list = pickle.load(file)

    # with open(instances_path, 'rb') as file: # you anot useing this now...
    #     instances_list = pickle.load(file)

    all_img_feature_list = list(set(all_img_feature_list)) # get the unique set of features

    df_thick = pd.DataFrame(outputs_list).fillna(0)
    df_thick.iloc[:,3:] = df_thick.iloc[:,3:].astype('int') # counts as ints not floats 

    df_thin = df_thick[['img_id'] + all_img_feature_list].copy() # making a df with just feature counts and img_id

    monkier_dict = {'faster_rcnn_R_50_FPN_3x' : 'fasterR50', 
                'faster_rcnn_R_101_FPN_3x' : 'fasterR101', 
                'faster_rcnn_X_101_32x8d_FPN_3x' : 'fasterX101',
                'retinanet_R_50_FPN_3x' : 'retinaR50',
                'retinanet_R_101_FPN_3x' : 'retinaR101'} 

    # dumb correction to help further down..
    df_thin.rename(columns={'flag_us':'flagUS'}, inplace= True)
    df_thin.rename(columns={'flag_iraqi':'falgIRQ'}, inplace= True)
    df_thin.rename(columns={'military_vehicle':'militaryVehicle'}, inplace= True)
    df_thin.rename(columns={'religious_garment_female':'religiousGarmentFemale'}, inplace= True)
    df_thin.rename(columns={'prayer_salah':'prayerSalah'}, inplace= True)
    df_thin.rename(columns={'prayer_informal':'prayerInformal'}, inplace= True)
    df_thin.rename(columns={'blooded_area':'bloodedArea'}, inplace= True)


    for old_name in df_thin.columns[1:]: # we do not change name image ID

        monkier = monkier_dict[model]
        new_name = f'{old_name}_{monkier}'
        df_thin.rename(columns={old_name:new_name}, inplace= True)

    return(df_thin)
    

def add_train_test_info(df_merged):

    """Takes a df with output data from all models and train/test info. Is only used later if we pass the annotated dataset."""

    train_test_index_path = '/home/simon/Documents/Bodies/scripts/OD/Detectron2/misc/train_test_index.pkl' # remenber to get to the right one on computerome
    #train_test_index_path = '/home/simon/Documents/Bodies/scripts/OD/Detectron2/misc/train_test_index.pkl' 

    with open(train_test_index_path, 'rb') as file:
        train_test_index = pickle.load(file)

    train_idx = pd.Series(train_test_index['train']).str.replace('.xml', '', regex=False)
    test_idx = pd.Series(train_test_index['test']).str.replace('.xml', '', regex=False)

    df_merged['train'] = df_merged['img_id'].isin(train_idx).astype('int')
    df_merged['test'] = df_merged['img_id'].isin(test_idx).astype('int')

    # Only keep annotated images
    df_sub = df_merged[(df_merged['test'] == 1) | (df_merged['train'] == 1)]

    return(df_sub)


def make_df_merged(FULL = False, alt_threshold = False):

    """Uses the function make_df to create 5 dfs. Then merges these dfs. 
    If it is the annotation set (FULL = False) we also add the train/test info."""


    df_faster_rcnn_R_50_FPN_3x = make_df('faster_rcnn_R_50_FPN_3x', FULL, alt_threshold)
    df_faster_rcnn_R_101_FPN_3x = make_df('faster_rcnn_R_101_FPN_3x', FULL, alt_threshold)
    df_faster_rcnn_X_101_32x8d_FPN_3x = make_df('faster_rcnn_X_101_32x8d_FPN_3x', FULL, alt_threshold)
    df_retinanet_R_50_FPN_3x = make_df('retinanet_R_50_FPN_3x', FULL, alt_threshold)
    df_retinanet_R_101_FPN_3x = make_df('retinanet_R_101_FPN_3x', FULL, alt_threshold)

    data_frames = [df_faster_rcnn_R_50_FPN_3x, df_faster_rcnn_R_101_FPN_3x, 
                df_faster_rcnn_X_101_32x8d_FPN_3x, df_retinanet_R_50_FPN_3x, 
                df_retinanet_R_101_FPN_3x]

    df_merged = reduce(lambda left,right: pd.merge(left,right,on=['img_id'], how='outer'), data_frames)    

    if FULL == False:
        df_merged = add_train_test_info(df_merged)

    df_merged = get_ensamble_est(df_merged)

    return(df_merged)


def get_annotations(annotated_img_dir, df_merged):

    """Get the annotated features so we can asses reliability. Is only used if we are on the anootated set."""


    ##### CHECK THAT IT IS THE RIGHT IMAGES§!!!!
    annotation_count_list = []

    img_path_list = get_img_path(annotated_img_dir)
    for img_path in img_path_list:
        img_id = img_path.split('.')[0].split('/')[-1]
        obj_path = img_path.split('.')[0] + '.xml'

        # Pass if the image have not been anotated 
        if os.path.exists(obj_path) == False:
            pass

        elif  os.path.exists(obj_path) == True:
    # -------------------------------------
    # OLD:
    # for filename in os.listdir(annotated_img_dir):
    #    if filename.split('.')[1] == 'xml': # only for annotated images. filename is now effectively annotationes.

    #     img_id = filename.split('.')[0]

    #     obj_path = os.path.join(annotated_img_dir, filename)
        
    # --------------------------------------------    
        
            tree = ElementTree.parse(obj_path)

            objs = []
            annotations = tree.findall('object')
            
            for i in annotations: # go through all annotated objs in a given image

                label = i.find('name').text # get the label            
                objs.append(label)

            img_feature_count = dict(Counter(objs))
            img_dict = {**{'img_id': img_id}, **img_feature_count} # like this, img_id is first by defult..

            annotation_count_list.append(img_dict)

    df_annotation = pd.DataFrame(annotation_count_list).fillna(0)
    df_annotation.iloc[:,1:] = df_annotation.iloc[:,1:].astype('int')

    # you also need the fix here:
    df_annotation.rename(columns={'flag_us':'flagUS'}, inplace= True)
    df_annotation.rename(columns={'flag_iraqi':'falgIRQ'}, inplace= True)
    df_annotation.rename(columns={'military_vehicle':'militaryVehicle'}, inplace= True)
    df_annotation.rename(columns={'religious_garment_female':'religiousGarmentFemale'}, inplace= True)
    df_annotation.rename(columns={'prayer_salah':'prayerSalah'}, inplace= True)
    df_annotation.rename(columns={'prayer_informal':'prayerInformal'}, inplace= True)
    df_annotation.rename(columns={'blooded_area':'bloodedArea'}, inplace= True)

    for old_name in df_annotation.columns[1:]: # we do not change name image ID

        new_name = f'{old_name}_annotated'
        df_annotation.rename(columns={old_name:new_name}, inplace= True)

    df_merged = pd.merge(df_merged, df_annotation,on=['img_id'], how='outer')

    return(df_merged)


def annotate_df(FULL = False, alt_threshold = False):

    """Simple a switch to add anotation if we are on the annotated set. Shitty name then..."""

    df_merged = make_df_merged(FULL, alt_threshold)
    
    if FULL == False:
    
        annotated_img_dir = '/home/simon/Documents/Bodies/data/jeppe/images' #'/home/projects/ku_00017/data/raw/bodies/OD_images_annotated
        df_merged = get_annotations(annotated_img_dir, df_merged)

    return(df_merged)


def get_meta_keys():

    """Returns the 'dict_keys' for the IPTCInfo"""
    
    dict_keys = ['object name', 
             'edit status', 
             'editorial update', 
             'urgency', 
             'subject reference', 
             'category', 
             'supplemental category', 
             'fixture identifier', 
             'keywords', 
             'content location code', 
             'content location name', 
             'release date', 
             'release time', 
             'expiration date', 
             'expiration time', 
             'special instructions', 
             'action advised', 
             'reference service', 
             'reference date', 
             'reference number', 
             'date created', 
             'time created', 
             'digital creation date', 
             'digital creation time', 
             'originating program', 
             'program version', 
             'object cycle', 
             'by-line', 
             'by-line title', 
             'city', 
             'sub-location', 
             'province/state', 
             'country/primary location code', 
             'country/primary location name', 
             'original transmission reference', 
             'headline', 
             'credit', 
             'source', 
             'copyright notice', 
             'contact', 
             'caption/abstract', 
             'local caption', 
             'writer/editor', 
             'image type', 
             'image orientation', 
             'language identifier', 
             'custom1', 
             'custom2', 
             'custom3', 
             'custom4', 
             'custom5', 
             'custom6', 
             'custom7', 
             'custom8', 
             'custom9', 
             'custom10', 
             'custom11', 
             'custom12', 
             'custom13', 
             'custom14', 
             'custom15', 
             'custom16', 
             'custom17', 
             'custom18', 
             'custom19', 
             'custom20']

    return(dict_keys)


def get_IPTC_data(path, filename):

    """Returns IPTC data given a path and a file name pertaining to one specific image."""

    file_path = os.path.join(path, filename)
    print(file_path)

    info = IPTCInfo(file_path, force=True)
    dict_keys = get_meta_keys()
    
    for i in dict_keys:
        if info[i] != None:
            if len(info[i]) > 0:
                print(f'key: {i}, info: {info[i]}\n')


def meta_to_df(df, img_dir = '/home/simon/Documents/Bodies/data/jeppe/images'):

    """Returns new df with meta data from images. Needs old df and path to images"""

    df_expanded = df.copy()
    dict_keys = get_meta_keys()

    # create empty columns
    for k in dict_keys:    
        df_expanded[k] = np.nan #pd.NA #None

    # get IPCT info fro each img_id
    for count, i in enumerate(df_expanded['img_id']):#[0:1]:

        print(f'img: {i}, {count}/{df_expanded.shape[0]+1}...', end='\r') # for debug

        filename = i + '.jpg'
        file_path = os.path.join(img_dir, filename)
        info = IPTCInfo(file_path, force=True)
        
        # Fill IPTC info into columns for i img_id
        for j in dict_keys:

            # print(j)# for debug

            if info[j] != None:
                if len(info[j]) > 0:

                    if type(info[j]) == bytes:
                        df_expanded.loc[df_expanded['img_id'] == i, j] = info[j].decode('utf-8', 'ignore') # error = ingnore/replace
                        
                    elif type(info[j]) == list:
                        temp_list = []
                        for n in info[j]:
                            temp_list.append(n.decode('utf-8', 'ignore'))
                        
                        # Make the list into a series of list fitting the size of the data frame slice
                        temp_list_series = pd.Series([temp_list] * df_expanded.loc[df_expanded['img_id'] == i, j].shape[0]) # it is a hack...
                        df_expanded.loc[df_expanded['img_id'] == i, j] = temp_list_series

                    else:
                        # just add
                        df_expanded.loc[df_expanded['img_id'] == i, j] = info[j]

    # remove columns with all NaNs
    df_cleaned = df_expanded.dropna(axis=1, how='all')
    return(df_cleaned)


def plot_corr(df, train_test_both = 'both'):

    """Corrolation plots across all models, ensemble estimates and annotations"""

    if train_test_both == 'both':

        front_str_list = []
        for i in df.columns:
            if '_' in i: # no meta features have this.
                front_str_list.append(i.split('_')[0])

        features = set(front_str_list)
        df_subset = df

    elif train_test_both == 'train':

        train_front_str_list = []
        for i in df[df['train']==1].columns:
            if '_' in i: # no meta features have this.
                train_front_str_list.append(i.split('_')[0])

        features = set(train_front_str_list)
        df_subset = df[df['train']==1]


    elif train_test_both == 'test':

        test_front_str_list = []
        for i in df[df['test']==1].columns:
            if '_' in i: # no meta features have this.
                test_front_str_list.append(i.split('_')[0])

        features = set(test_front_str_list)
        df_subset = df[df['test']==1]


    else:
        print('wrong input for train_test_both')


    # -------------------------

    for i in features:

        temp_df = df_subset.loc[:,df_subset.columns.str.startswith(i)]
        if temp_df.shape[1]>1: # more than one column; this solves a lot.

            title = f'{i}: {train_test_both}'
            print(f'plotting {title}. Dim: {df_subset.shape}')
            g =sns.pairplot(temp_df)
            g.fig.suptitle(f'{i}: {train_test_both}', y=1.08) # y= some height>1
            plt.show()



def feature_dist_plots(df, feature_version): # feature should be all and go inside.. but then you need mean first or it gets messig...

    """Feature version can be a model short, eg. 'fasterX101' or retinaR50. 
    It can also be a ensamble indication e.g. mean or median."""

    pub_status = df['custom2'].unique()

    # raw = df[df['custom2'] == 'Raw'][feature]
    # Published

    plt.figure(figsize= [15,15])
    plt.title(feature_version)

    for i in pub_status:

        ratio_list = []
        feature_list = []

        for j in df.columns[df.columns.str.endswith(feature_version)]:
            n = df[(df['custom2'] == i) & (df[j] >= 1)].shape[0]
            N = df[df['custom2'] == i][j].shape[0]
            ratio = n/N

            feature_list.append(j)
            ratio_list.append(ratio)
        
        plt.barh(np.arange(0, len(ratio_list), 1), ratio_list, alpha = 0.5, label = f'{i} (N = {N})')
        plt.yticks(np.arange(0,len(ratio_list),1), feature_list, fontsize = 16, label = f'{i} (N = {N})')
        plt.xlabel('Ratio of objects in subset of images (Raw vs. Published)', fontsize = 14)
        plt.legend(fontsize = 14, loc = 'lower right')

        # plt.savefig(f'fig_name.pdf', bbox_inches="tight")   

    plt.show()