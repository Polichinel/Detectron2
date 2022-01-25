from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import pickle
from utils import *

cfg_pkl_path = 'faster_rcnn_R_101_FPN_3x.pkl' # path to the config file you just created
model_name = "faster_rcnn_R_101_FPN_3x"

with open(cfg_pkl_path, 'rb') as file:
    cfg = pickle.load(file)
 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

# visualize prediction.------------------------------------------------------
# the img_dir and the open of the pikle could go inside the function viz_sample

img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated' #  '/home/simon/Documents/Bodies/data/jeppe/images'

# NEW --------------------------------------------- (needs copying to retina!)

train_data = "bodies_OD_data"
test_data  = "bodies_OD_data_test"

DatasetCatalog, MetadataCatalog = register_dataset(img_dir, train_data, test_data)
bodies_OD_metadata = MetadataCatalog.get(train_data) 

# viz sample -------------------------------------------------

viz_sample(model_name, img_dir, predictor, 10, bodies_OD_metadata)

# Get AP ------------------------------------------------------ (needs copying to retina!)

evaluator = COCOEvaluator(train_data, output_dir = cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, train_data)
#print(inference_on_dataset(predictor.model, val_loader, evaluator))

with open('train_results.txt', 'w') as file:
        file.write(str(inference_on_dataset(predictor.model, val_loader, evaluator)))


evaluator = COCOEvaluator(test_data, output_dir = cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, test_data)
#print(inference_on_dataset(predictor.model, val_loader, evaluator))

with open('test_results.txt', 'w') as file:
        file.write(str(inference_on_dataset(predictor.model, val_loader, evaluator)))

print('train and test- results saved')
