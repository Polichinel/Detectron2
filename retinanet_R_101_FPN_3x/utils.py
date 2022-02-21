import os
import cv2
import pickle
import copy
import numpy as np
from xml.etree import ElementTree, ElementInclude
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from LossEvalHook import *

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

# DECAPRIATED
# def get_train_test(annotation_list, train_ratio = 0.8):

#     train_n = int(len(annotation_list) * train_ratio)
#     train_set = np.random.choice(annotation_list, train_n, replace = False)# asshole....
#     test_set = [i for i in annotation_list if i not in train_set]

#     return(train_set, test_set)

def get_train_test():
    # fixed 80/20 ratio. Change in "create_train_test_index.py" in under misc.

    train_test_index_path = '/home/projects/ku_00017/people/simpol/scripts/bodies/Detectron2/misc/train_test_index.pkl'

    with open(train_test_index_path, 'rb') as file:
        train_test_index = pickle.load(file)

    train_set = train_test_index['train']
    test_set = train_test_index['test']

    return(train_set, test_set)


def get_img_dicts(img_dir, train = True):

    _, _, class_to_int = get_classes(img_dir) # only need the dict here.
    annotation_list = get_annotation_path(img_dir) # new
    train_set, test_set = get_train_test() 

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

# -------------------------------------------------------------------
def viz_sample(model_name ,img_dir, predictor, n, bodies_OD_metadata):

    """Vizualise a sample of images"""

    img_path_list = get_img_path(img_dir)
    #sample_dir = os.path.join(os.getcwd(), 'sample_pred_img')
    sample_dir = f"/home/projects/ku_00017/data/generated/bodies/sample_pred_img/{model_name}"

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
        viz_img_path = os.path.join(sample_dir, f'retinanet_R_101_FPN_3x_test{i}.jpg')
        cv2.imwrite(viz_img_path, viz_img)
        print(f'{viz_img_path} saved')
    
    print('Sample .jpgs saved...')
# -------------------------------------------------------------------

def get_train_cfg(config_file_path, checkpoint_url, train_data, test_data, output_dir, num_worker, img_per_batch, learning_rate, decay_LR, max_iter, n_classes, device):
    """Returns the cfg opject"""

# also needs to take train_data and output_dir.

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo

    cfg.DATASETS.TRAIN = (train_data)

    # new ------------------
    #cfg.DATASETS.TEST = ()
    cfg.DATASETS.TEST = (test_data,) # test data needs to be input
    cfg.TEST.EVAL_PERIOD = 100
    # tjekc what this does before adding more jazz.

    # ---------------------------------

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



# ----------------------------------------------------------------------- NEW

class MyTrainer(DefaultTrainer):

    @classmethod # newest addition
    def build_train_loader(cls, cfg):
        
        # top transform is to avoid memory fail - it is default
        train_augmentations = [T.ResizeShortestEdge(short_edge_length=(672, 704, 736, 736, 800), max_size=1333, sample_style='choice'),
                               T.RandomBrightness(0.5, 2), 
                               T.RandomContrast(0.5, 2),
                               T.RandomSaturation(0.5, 2),
                               T.RandomLighting(0.7),
                               T.RandomFlip(prob=0.5, horizontal=True, vertical=False)]

        train_mapper = DatasetMapper(cfg, is_train=True, augmentations=train_augmentations)

        return build_detection_train_loader(cfg, mapper=train_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # return COCOEvaluator(dataset_name, cfg, True, output_folder) 
        # Regarding second argument, task = cfg:
        # COCO Evaluator instantiated using config, this is deprecated be havior. Please pass in explicit arguments instead.
        # Tasks (tuple[str]) – tasks that can be evaluated under the given configuration. A task is one of “bbox”, “segm”, “keypoints”.
        # Note: By default, will infer this automatically from predictions.
        return COCOEvaluator(dataset_name, ('bbox',), True, output_folder)
                     
    def build_hooks(self):

        # --------------------
        cfg_pkl_path = 'retinanet_R_101_FPN_3x.pkl' # path to the config file you just created

        with open(cfg_pkl_path, 'rb') as file:
            cfg = pickle.load(file)
        # -------------------

        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks