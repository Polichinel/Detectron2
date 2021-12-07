from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog

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

train_data = "bodies_OD_data"
test_data  = "bodies_OD_data_test"

DatasetCatalog, MetadataCatalog = register_dataset(img_dir, train_data, test_data)
bodies_OD_metadata = MetadataCatalog.get(train_data) # is this used at all before test now??? 

# viz sample -------------------------------------------------

viz_sample(img_dir, predictor, 10, bodies_OD_metadata)

# Get AP ------------------------------------------------------

evaluator = COCOEvaluator(train_data, output_dir = cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, train_data)
#print(inference_on_dataset(predictor.model, val_loader, evaluator))

with open('train_data_results.txt', 'w') as file:
        file.write(str(inference_on_dataset(predictor.model, val_loader, evaluator)))


evaluator = COCOEvaluator(test_data, output_dir = cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, test_data)
#print(inference_on_dataset(predictor.model, val_loader, evaluator))

with open('test_results.txt', 'w') as file:
        file.write(str(inference_on_dataset(predictor.model, val_loader, evaluator)))

print('train and test- results saved')

# Run on unlabeled set - or that migt be a third script called inference.  
