import os
import cv2
import pickle
import numpy as np
from xml.etree import ElementTree, ElementInclude
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer


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

    # if you just want a list to go through, you cna generalizr the function below (get_img_path)... 
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

def viz_sample(img_dir, predictor, n, bodies_OD_metadata):

    """Vizualise a sample of images"""

    img_path_list = get_img_path(img_dir)
    sample_dir = os.path.join(os.getcwd(), 'sample_pred_img')

    # Check/create the sample dir 
    if os.path.isdir(sample_dir):
        print(sample_dir, "already exists.")
    
    else:
        os.mkdir(sample_dir)
        print(sample_dir, "is created.")

    for i in range(n):
        im_path = np.random.choice(img_path_list, 1, replace= False).item()
        print(im_path)

        im = cv2.imread(im_path)
        outputs = predictor(im)

        # create and save the image
        v = Visualizer(im[:, :, ::-1], metadata=bodies_OD_metadata, scale=1.2) # you have this from earlier
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        viz_img = out.get_image()[:, :, ::-1]
        viz_img_path = os.path.join(sample_dir, f'frcnn_test{i}.jpg')
        cv2.imwrite(viz_img_path, viz_img)
        print(f'{viz_img_path} saved')
    
    print('Sample .jpgs saved...')