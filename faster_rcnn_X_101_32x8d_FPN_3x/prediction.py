from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import pickle
from utils import *

cfg_pkl_path = 'faster_rcnn_X_101_32x8d_FPN_3x.pkl' # path to the config file you just created
model_name = "faster_rcnn_X_101_32x8d_FPN_3x"

with open(cfg_pkl_path, 'rb') as file:
    cfg = pickle.load(file)
 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold

predictor = DefaultPredictor(cfg)

img_dir = '/home/projects/ku_00017/data/raw/bodies/images_johan'

img_path_list = get_img_path(img_dir)

outputs_list = []

for img_path in img_path_list[0:5]: #the five first as a test.

        im = cv2.imread(img_path)
        outputs = predictor(im)
        outputs['im_id'] = img_path.split('/')[-1].split('0')[0]
        outputs_list.append(outputs["instances"].to("cpu"))

# pickle configurations - where do you want to save this?
with open(f'outputs_list.pkl', 'wb') as file:
    pickle.dump(outputs_list, file, protocol = pickle.HIGHEST_PROTOCOL)