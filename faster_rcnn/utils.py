import os
import cv2
import pickle
import numpy as np
from xml.etree import ElementTree, ElementInclude
from detectron2.structures import BoxMode

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


def get_img_dicts(img_dir):

    _, _, class_to_int = get_classes(img_dir) # only need the dict here.
    
    dataset_dicts = []
    idx = 0

    for filename in os.listdir(img_dir):
        if filename.split('.')[1] == 'xml': # only for annotated images. filename is now effectively annotationes.

            img_name = filename.split('.')[0] + '.jpg' # the image name w/ correct extension.
            
            record = {}
            img_path = os.path.join(img_dir, img_name)
 
            height, width = cv2.imread(img_path).shape[:2]

            record["file_name"] = img_path #  needs to be the full path to the image file acccording to docs.
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            obj_path = os.path.join(img_dir, filename)
            tree = ElementTree.parse(obj_path)

            annotations = tree.findall('object')

            for i in annotations: # go through all annotated objs in a given image

                label = i.find('name').text # get the label
                box = i.findall('bndbox') # find the box

                for j in box: # get the 4 measures from the box

                    xmin = float(j.find('xmin').text) 
                    xmax = float(j.find('xmax').text) 
                    ymin = float(j.find('ymin').text)
                    ymax = float(j.find('ymax').text) 

                obj = { 'bbox': [xmin, ymin, xmax, ymax],
                        'bbox_mode': BoxMode.XYXY_ABS, # remember to change!
                        'category_id': class_to_int[label],
                        'catagory_label': label,
                        'iscrowd' : 0}

                objs.append(obj)

            record["annotations"] = objs

            dataset_dicts.append(record)
            idx += 1
            print(idx, end="\r")
  
    return(dataset_dicts)

