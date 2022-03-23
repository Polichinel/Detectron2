
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import pickle
# from utils import *

from collections import Counter
# import pandas as pd

from xml.etree import ElementTree, ElementInclude


# Per default, this is done in the prediction script. But this might be used to change the threshold after training to avoid retraining.

# Shell input ----------------------------------------- #
model_option_dict = {'a': 'faster_rcnn_R_50_FPN_3x', 'b': 'faster_rcnn_R_101_FPN_3x', 'c': 'faster_rcnn_X_101_32x8d_FPN_3x', 'd': 'retinanet_R_50_FPN_3x', 'e': 'retinanet_R_101_FPN_3x'}# ----------------------

model_option = input(f"Choose model: \na) {model_option_dict['a']} \nb) {model_option_dict['b']} \nc) {model_option_dict['c']} \nd) {model_option_dict['d']} \ne) {model_option_dict['e']}\n")
full_option = input('Use full set of images? (y/n): ')

if full_option == 'y':
    model_name = model_option_dict[model_option] + '_FULL'
elif full_option == 'n':
    model_name = model_option_dict[model_option]

threshold = input('Enter threshold (0.1 - 0.9): ')

threshold = float(threshold)

print('Model: {}'.format(model_name))
print('Threshold: {}'.format(threshold))
# -------------------------------------------------------#

def get_classes(img_dir):
    """Creates a list of classes and corrosponding ints. also a dict to translate"""

    obj_name = []

    # Get all objects that have been annotated
    for filename in os.listdir(img_dir):
        if filename.split('.')[1] == 'xml':
            box_path = os.path.join(img_dir, filename)

            tree = ElementTree.parse(box_path)
            lst_obj = tree.findall('object')

            for j in lst_obj:
                obj_name.append(j.find('name').text)
    
    classes = list(sorted(set(obj_name))) # all labesl
    classes_int = list(np.arange(0,len(classes))) # corrospoding int
    class_to_int = dict(zip(classes,classes_int)) # a dict to translate between them

    return(classes, classes_int, class_to_int)

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

def get_int_to_class():
    # This is always the annotatedd folder..
    #annotated_img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated'  
    annotated_img_dir = '/home/simon/Documents/Bodies/data/jeppe/images'
    classes, classes_int, class_to_int = get_classes(annotated_img_dir)
    int_to_class = dict(zip(classes_int, classes))

    return int_to_class


def get_output_tX(model_name, threshold):

    # Get the instances_list
    #instances_list_dir = f'/home/projects/ku_00017/data/generated/bodies/detectron_outputs/{model}/instances_list.pkl'
    
    # you did the name thing to be more secure..
    if model_name.split('_')[-1] == 'FULL':
        instances_list_dir = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model_name}/instances_list_FULL.pkl'
        #img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' #full run!!!
        img_dir='/media/simon/Seagate Expansion Drive/images_spanner'

    else:
        instances_list_dir = f'/home/simon/Documents/Bodies/data/computerome_outputs/detectron_outputs/{model_name}/instances_list.pkl'
        img_dir = '/home/simon/Documents/Bodies/data/jeppe/images'


    with open(instances_list_dir, 'rb') as file:
        instances_list = pickle.load(file)


    # get images path and int to class dict.
    img_path_list = get_img_path(img_dir)
    int_to_class = get_int_to_class()

    # containers:
    output_list = []
    all_img_feature_list = [] # to create the slim df, its easier this way...

    # number of images to predict
    total_count = len(img_path_list)

    # prediction loop
    for count, img_path in enumerate(img_path_list[0:total_count]):

        instance = instances_list[count]

        img_id = img_path.split('/')[-1].split('.')[0]

        img_dict = {'img_id': img_id, 'scores': None , 'pred_classes': None}

        mask = instance.scores.numpy()>threshold # maybe these scores should be used for more stuff.. lot of information here
    
        # only save instances above the threshold
        img_dict['scores'] = instance.scores.numpy()[mask]
        img_dict['pred_classes'] = instance.pred_classes.numpy()[mask]

        img_feature_Int_count = dict(Counter(instance.pred_classes.numpy()[mask])) # counting the classes - int encoded
        img_dict = {**img_dict, **img_feature_Int_count} # merging counts with other info 
        
        img_feature_list = [int_to_class[i] for i in instance.pred_classes.numpy()[mask]] # convert from int encoded feature to str of feature name
        img_feature_count = dict(Counter(img_feature_list)) # count the actual feature name
        img_dict = {**img_dict, **img_feature_count} # merging counts name with other info - actual feature

        output_list.append(img_dict)
        all_img_feature_list += img_feature_list #this will just be a list of all encountered features..

        print(f'img id: {img_id}, {count+1} of {total_count} done...', end = '\r')  


    #print(f'\n {count} of {total_count} done...')  
    # get the unique set of features - nice for the thin df.
    all_img_feature_list = list(set(all_img_feature_list))    

    return output_list, all_img_feature_list

output_list, all_img_feature_list = get_output_tX(model_name, threshold)

print(f'\nOutput from {len(output_list)} images handled...')

# pickle configurations and save
#location = f'/home/projects/ku_00017/data/generated/bodies/detectron_outputs/{model_name}'
location =f'/home/simon/Documents/Bodies/data/computerome_outputs/alt_threshold_outputs/{model_name}'
os.makedirs(location, exist_ok = True)

with open(location + f'/output_list_t{int(threshold*100)}.pkl', 'wb') as file:
    pickle.dump(output_list, file)

with open(location + f'/all_img_feature_list_t{int(threshold*100)}.pkl', 'wb') as file:
    pickle.dump(all_img_feature_list, file)

print('Output pickled and saved')