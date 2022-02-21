from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import pickle
from utils import *

from collections import Counter
# import pandas as pd

cfg_pkl_path = 'faster_rcnn_R_101_FPN_3x.pkl' # path to the config file you just created
model_name = "faster_rcnn_R_101_FPN_3x"

with open(cfg_pkl_path, 'rb') as file:
    cfg = pickle.load(file)
 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold

predictor = DefaultPredictor(cfg)

#img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' % for when you do the full run!!!
img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated'

img_path_list = get_img_path(img_dir)
#print(img_path_list[0:10])

# ---------------------------
annotated_img_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated'
classes, classes_int, class_to_int = get_classes(annotated_img_dir)
int_to_class = dict(zip(classes_int, classes)) # dict to translate from int encoding of feature to str of feature name.
# ---------------------------

# containers:
instances_list = []
output_list = []
all_img_feature_list = [] # to create the slim df, its easier this way...

# number of images to predict
total_count = len(img_path_list)

# prediction loop
for count, img_path in enumerate(img_path_list[0:total_count]): #10000


        im = cv2.imread(img_path)
        output = predictor(im)
        instance = output["instances"].to("cpu")
        instances_list.append(instance) # nice to have for backup and checks

        # more manageable output format ---------------------------------------------------------------------
        img_id = img_path.split('/')[-1].split('.')[0]
        img_dict = {'img_id': img_id, 'scores': None , 'pred_classes': None}
        img_dict['scores'] = instance.scores.numpy()
        img_dict['pred_classes'] = instance.pred_classes.numpy()

        img_feature_Int_count = dict(Counter(instance.pred_classes.numpy())) # counting the classes - int encoded

        img_dict = {**img_dict, **img_feature_Int_count} # merging counts with other info 

        img_feature_list = [int_to_class[i] for i in instance.pred_classes.numpy()] # convert from int encoded feature to str of feature name
        img_feature_count = dict(Counter(img_feature_list)) # count the actual feature name

        img_dict = {**img_dict, **img_feature_count} # merging counts name with other info - actual feature 
        
        output_list.append(img_dict)
        all_img_feature_list += img_feature_list #this will just be a list of all encountered features..

        print(f'img id: {img_id}, {count} of {total_count} done...', end = '\r')


# Outputs wo/ pandas:
all_img_feature_list = list(set(all_img_feature_list)) # get the unique set of features - nice for the thin df.

# pickle configurations and save
location = f'/home/projects/ku_00017/data/generated/bodies/detectron_outputs/{model_name}'
os.makedirs(location, exist_ok = True)

# A bit overkill, but nice for checking errors down the line.
with open(location + '/instances_list.pkl', 'wb') as file:
    pickle.dump(instances_list, file)

instances_list_location = location + 'instances_list.pkl'
with open(location + '/output_list.pkl', 'wb') as file:
    pickle.dump(output_list, file)

with open(location + '/all_img_feature_list.pkl', 'wb') as file:
    pickle.dump(all_img_feature_list, file)


# Outputs w/ pandas:
# all_img_feature_list = list(set(all_img_feature_list)) # get the unique set of features - nice for the thin df.

# df_thick = pd.DataFrame(output_list).fillna(0) # NaN countes to 0
# df_thick.iloc[:,3:] = df_thick.iloc[:,3:].astype('int') # counts as ints - not floats 
# df_thin = df_thick[['img_id'] + all_img_feature_list].copy() # making a df with just feature counts and img_id

# # pickle configurations and save
# location = f'/home/projects/ku_00017/data/generated/bodies/detectron_outputs/{model_name}'
# os.makedirs(location, exist_ok = True)

# # A bit overkill, but nice for checking errors down the line.
# with open(location + '/instances_list.pkl', 'wb') as file:
#     pickle.dump(instances_list, file)

# instances_list_location = location + 'instances_list.pkl'
# with open(location + '/df_thick.pkl', 'wb') as file:
#     pickle.dump(df_thick, file)

# with open(location + '/df_thin.pkl', 'wb') as file:
#     pickle.dump(df_thin, file)
