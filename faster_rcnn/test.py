from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import os
import pickle
from utils import *

cfg_pkl_path = 'faster_rcnn_X_101_32x8d_FPN_3x.pkl' # path to the config file you just created

with open(cfg_pkl_path, 'rb') as file:
    cfg = pickle.load(file)
 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

# visualize prediction.------------------------------------------------------
# the img_dir and the open of the pikle could go inside the function viz_sample

img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated' #  '/home/simon/Documents/Bodies/data/jeppe/images'

# NEW ---------------------------------------------

with open('DatasetCatalog.pkl', 'rb') as file:
    DatasetCatalog = pickle.load(file)

with open('MetadataCatalog.pkl', 'rb') as file:
    MetadataCatalog = pickle.load(file)

train_data = "bodies_OD_data"
test_data  = "bodies_OD_data_test"

bodies_OD_metadata = MetadataCatalog.get(train_data) # is this used at all before test now??? 

# -------------------------------------------------

viz_sample(img_dir, predictor, 10, bodies_OD_metadata)

# Get AP: could be function in utils ------------------------------------------------------
#model_name = cfg_pkl_path.split('.')[0] # no this need to be "bodies_OD_data_test" and the dataset needs to be registrated...


evaluator = COCOEvaluator(test_data, output_dir = cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, test_data)
print(inference_on_dataset(predictor.model, val_loader, evaluator))

# more to come. 
# Run on test set exclusively. 
# get som metrics
# Run on unlabeled set - or that migt be a third script called inference.  
