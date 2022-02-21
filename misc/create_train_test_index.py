import os
import pickle
import copy
import numpy as np
from xml.etree import ElementTree, ElementInclude

# a small script to make sure all models use the same train/test set and that we can identify what images was use where afterwards.

np.random.seed(42)

img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated'

#img_dir = '/home/simon/Documents/Bodies/data/jeppe/images'

def get_annotation_path(img_dir):

    """Creates a list of all box paths."""

    annotation_list = []

    for filename in os.listdir(img_dir):
        if filename.split('.')[1] == 'xml':
            annotation_list.append(filename)

    return(annotation_list)


def get_train_test(annotation_list, train_ratio = 0.8):

    train_n = int(len(annotation_list) * train_ratio)
    train_set = np.random.choice(annotation_list, train_n, replace = False)# asshole....
    test_set = [i for i in annotation_list if i not in train_set]

    return(train_set, test_set)

annotation_list = get_annotation_path(img_dir)
train_set, test_set = get_train_test(annotation_list, train_ratio = 0.8)

train_test_index = {'train': train_set, 'test': test_set}

with open('train_test_index.pkl', 'wb') as file:
    pickle.dump(train_test_index, file)