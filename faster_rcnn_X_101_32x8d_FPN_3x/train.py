from typing import Protocol
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
import os, json, cv2, random, shutil, time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

from utils import *

print('Libs imported with no errors')  # -------------------------------------------------------------------------------

img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated' #  '/home/simon/Documents/Bodies/data/jeppe/images'

classes, _ , _ = get_classes(img_dir) # need fot meta data
n_classes = len(classes) # you'll need this futher down

# choosing model. for more models, see: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

# Faster RCNN
config_file_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
device = 'cuda'

model_name = config_file_path.split('/')[1][:-5]

print(f"Using model {model_name}")
print(f'On device = {device}')

num_worker = 2
img_per_batch = 2
learning_rate = 0.00025
decay_LR = []
max_iter =  100000# 2**17 # #2**8 # 2**12 # 2**15

print(f'running for {max_iter} iterations. Learing rate: {learning_rate}, Image per batch: {img_per_batch}')

#output_dir = "./output/frcnn" 
output_dir = f"/home/projects/ku_00017/people/simpol/scripts/bodies/Detectron2/output/{model_name}"

train_data = "bodies_OD_data"
test_data  = "bodies_OD_data_test"

print('hyper parameters and paths defined') 

DatasetCatalog, MetadataCatalog = register_dataset(img_dir, train_data, test_data)

print('data registered and catalog + meta pickled')  

def main():

    cfg = get_train_cfg(config_file_path, checkpoint_url, train_data, test_data, output_dir, num_worker, img_per_batch, learning_rate, decay_LR, max_iter, n_classes, device)

    # pickle configurations
    with open(f'{model_name}.pkl', 'wb') as file:
        pickle.dump(cfg, file, protocol = pickle.HIGHEST_PROTOCOL)

    print(f'cfg saved as {model_name}.pkl')
    print('model loaded and hyper parameters set.')
    print('beginning traning...')
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    #trainer = DefaultTrainer(cfg) 
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------
# running validaiton
# Also data agmentaiton...

# -----------------------------------------------------------------------------