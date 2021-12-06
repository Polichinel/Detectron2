from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *

cfg_pkl_path = 'faster_rcnn_X_101_32x8d_FPN_3x.pkl' # path to the config file you just created

with open(cfg_pkl_path, 'rb') as file:
    cfg = pickle.load(file)
 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

# visualize prediction------------------------------------------------------
img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated' #  '/home/simon/Documents/Bodies/data/jeppe/images'

with open('bodies_OD_metadata.pkl', 'rb') as file:
    bodies_OD_metadata = pickle.load(file)

viz_sample(img_dir, predictor, 10, bodies_OD_metadata)

# more to come. 
# Run on test set exclusively. 
# get som metrics
# Run on unlabeled set - or that migt be a third script called inference.  
