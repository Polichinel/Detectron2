import torch
print(f'torch/cuda: {torch.__version__}')

# Some basic setup:
# Setup detectron2 logger
import detectron2
print(f'Detectron: {detectron2.__version__}')

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

from utils import *

print('Libs imported with no errors')
print('starting retraining...')


# home dir
#img_dir = '/home/simon/Documents/Bodies/data/jeppe/images'

# computerome dir
img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated'

#dataset_dicts = get_img_dicts(img_dir) # you don't need to load it here.. I thinks..

classes, _ , _ = get_classes(img_dir) # need fot meta data
n_classes = len(classes) # you'll need this futher down

# -----------------------------------------------------------------------------
# you do still not have dedicated train and test set. is it about time you did?
# meybe also move it out of the jeppe dir..
# -----------------------------------------------------------------------------

DatasetCatalog.register("bodies_OD_data", lambda: get_img_dicts(img_dir))
#MetadataCatalog.get("my_data").set(thing_classes=classes) # alt
MetadataCatalog.get("bodies_OD_data").thing_classes=classes
bodies_OD_metadata = MetadataCatalog.get("bodies_OD_data") # needed below.

print('data registered')

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("bodies_OD_data",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") #is there a better one?
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" # alt. no diff.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2**9# 2**8 #2**10 # you will need to train longer than 300 for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #  128 would be faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes  #(see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
#note: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

print('modle loaded and hyper parameters set.')
print('beginning traning')


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# inference:

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# visualize prediction------------------------------------------------------
viz_sample(img_dir, predictor, 10, bodies_OD_metadata)
