import os
import cv2
import pickle
import numpy as np
from xml.etree import ElementTree, ElementInclude
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

np.random.seed(42) # see if this is the culprit.

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


def get_annotation_path(img_dir):

    """Creates a list of all box paths."""

    annotation_list = []

    for filename in os.listdir(img_dir):
        if filename.split('.')[1] == 'xml':
            annotation_list.append(filename)

    return(annotation_list)

def get_train_test(annotation_list, train_ratio = 0.8):

    train_n = int(len(annotation_list) * train_ratio)
    train_set = np.random.choice(annotation_list, train_n, replace = False)
    test_set = [i for i in annotation_list if i not in train_set]

    return(train_set, test_set)


def get_img_dicts(img_dir, train = True):

    _, _, class_to_int = get_classes(img_dir) # only need the dict here.
    annotation_list = get_annotation_path(img_dir) # new
    train_set, test_set = get_train_test(annotation_list) 

    dataset_dicts = []
    idx = 0

    # if you just want a list to go through, you cna generalizr the function below (get_img_path)... 
    # and if you had that function splitting into train and test would be simple.

    if train == True:
        subset = train_set
    
    elif train == False:
        subset = test_set

    for filename in subset:

    # for filename in os.listdir(img_dir):
    #    if filename.split('.')[1] == 'xml': # only for annotated images. filename is now effectively annotationes.

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

def viz_sample(img_dir, predictor, n, bodies_OD_metadata):

    """Vizualise a sample of images"""

    img_path_list = get_img_path(img_dir)
    sample_dir = os.path.join(os.getcwd(), 'sample_pred_img')

    os.makedirs(sample_dir, exist_ok = True)

    for i in range(n):
        im_path = np.random.choice(img_path_list, 1, replace= False).item()
        print(im_path)

        im = cv2.imread(im_path)
        outputs = predictor(im)

        # create and save the image
        # Some issue w/ retina image and the metadata... idx of of list regarding classe...
        v = Visualizer(im[:, :, ::-1], metadata=bodies_OD_metadata, scale=1.2) # you have this from earlier
        #v = Visualizer(im[:, :, ::-1], scale=1.2) # you have this from earlier

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        viz_img = out.get_image()[:, :, ::-1]
        viz_img_path = os.path.join(sample_dir, f'retina_test{i}.jpg')
        cv2.imwrite(viz_img_path, viz_img)
        print(f'{viz_img_path} saved')
    
    print('Sample .jpgs saved...')


def get_train_cfg(config_file_path, checkpoint_url, train_data, output_dir, num_worker, img_per_batch, learning_rate, decay_LR, max_iter, n_classes, device):
    """Returns the cfg opject"""

# also needs to take train_data and output_dir.

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo

    cfg.DATASETS.TRAIN = (train_data)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = num_worker
    cfg.SOLVER.IMS_PER_BATCH = img_per_batch #  Number of images per batch across all machines. If we have 16 GPUs and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
    cfg.SOLVER.BASE_LR = learning_rate  # base learning rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = decay_LR # do not decay learning rate

    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #  128 would be faster (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes  #note: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.MODEL.RETINANET.NUM_CLASSES = n_classes

    cfg.MODEL.DVICE = device

    cfg.OUTPUT_DIR = output_dir

    return(cfg)

def register_dataset(img_dir, train_data, test_data):

    classes, _ , _ = get_classes(img_dir) # need fot meta data

    DatasetCatalog.register(train_data, lambda: get_img_dicts(img_dir)) 
    DatasetCatalog.register(test_data, lambda: get_img_dicts(img_dir, train=False)) #new
    MetadataCatalog.get(train_data).thing_classes=classes #MetadataCatalog.get("my_data").set(thing_classes=classes) # alt
    MetadataCatalog.get(test_data).thing_classes=classes #MetadataCatalog.get("my_data").set(thing_classes=classes) # alt

    return(DatasetCatalog, MetadataCatalog)