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


classes, _ , _ = get_classes(img_dir) # need fot meta data
n_classes = len(classes) # you'll need this futher down

# -----------------------------------------------------------------------------
# you do still not have dedicated train and test set. is it about time you did?
# meybe also move it out of the jeppe dir..
# and this file should be called train. Maybe  train_somthing... 
# -----------------------------------------------------------------------------

#output_dir = 
train_data = "bodies_OD_data"
#test_data  = "./output/object_detection" # maybe here you can difference between the models...

DatasetCatalog.register(train_data, lambda: get_img_dicts(img_dir)) #MetadataCatalog.get("my_data").set(thing_classes=classes) # alt
MetadataCatalog.get(train_data).thing_classes=classes
bodies_OD_metadata = MetadataCatalog.get(train_data) # needed below.

print('data registered')

# choosing model 
# for more models, see: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# config_file_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

config_file_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
device = 'cuda'

print(f"Using model {config_file_path.split('/')[1][:-5]}")
print(f'On device = {device}')

num_worker = 2
img_per_batch = 2
learning_rate = 0.00025
decay_LR = []
max_iter =  2**8# 2**9# 2**8 #2**10 # you will need to train longer than 300 for a practical dataset
print(f'running for {max_iter} iterations. Learing rate: {learning_rate}, Image per batch: {img_per_batch}')

# goes into utils
def get_train_cfg(config_file_path, checkpoint_url, train_data, num_worker, img_per_batch, learning_rate, decay_LR, max_iter, n_classes, device):
# also needs to take train_data and output_dir.

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo

    cfg.DATASETS.TRAIN = (train_data)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = num_worker
    cfg.SOLVER.IMS_PER_BATCH = img_per_batch
    cfg.SOLVER.BASE_LR = learning_rate  # pick a good LR
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = decay_LR        # do not decay learning rate

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #  128 would be faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes  #note: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.MODEL.DVICE = device

    #cfg.OUTPUT_DIR = output_dir

    return(cfg)



cfg = get_train_cfg(config_file_path, checkpoint_url, train_data, num_worker, img_per_batch, learning_rate, decay_LR, max_iter, n_classes, device)

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
