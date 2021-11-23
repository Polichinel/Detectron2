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

print('Libs imported with no errors')

# Initiate model/predictor
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

print('Model loaded and initiated')

# find initial test image
im_path = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated/JS43733.jpg'
im = cv2.imread(im_path)
print(type(im)) # this will just be the array, but that should be enough for a check.
print(im.shape)

# predicting:
outputs = predictor(im)

# looking at the putputs
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# viz
# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
viz_im = out.get_image()[:, :, ::-1]
print(type(viz_im))
viz_im_path = './Turtorial1_test1.jpg'
cv2.imwrite(viz_im_path, viz_im)
