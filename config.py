# config.py
# Berisi semua konfigurasi, path, dan parameter aplikasi.

import os
import torch

# --- Path Aplikasi ---
APP_DATA_PATH = 'app_data'
SSD_MODEL_FILENAME = 'ssd_mobilenetv3_large_kotakkecil.pth'
CHAR_CLASSIFIER_MODEL_FILENAME = 'mobilenetv3_small_char_classifier_augmented.pth'
TEMP_PROCESSING_DIR = 'temp_streamlit_processing_files' 

# Membuat path lengkap
SSD_MODEL_PATH = os.path.join(APP_DATA_PATH, SSD_MODEL_FILENAME)
CHAR_CLASSIFIER_MODEL_PATH = os.path.join(APP_DATA_PATH, CHAR_CLASSIFIER_MODEL_FILENAME)
ANNOTATION_CROP_AREA_DIR = APP_DATA_PATH 

# --- Parameter Model dan Pemrosesan ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter Model Klasifikasi Karakter
CHAR_IMAGE_SIZE = (64, 64)
CHAR_TARGET_CLASSES_LIST = [str(i) for i in range(10)] + ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'X', 'Y', 'Z']
CHAR_NUM_CLASSES = len(CHAR_TARGET_CLASSES_LIST)
CHAR_IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CHAR_TARGET_CLASSES_LIST)}

# Parameter Model SSD
SSD_NUM_CLASSES = 2 
SSD_DETECTION_THRESHOLD = 0.3 
SSD_NMS_IOU_THRESHOLD = 0.3   
SSD_INPUT_SIZE = (320, 320) 

# Parameter Filter dan Klasifikasi
CHAR_CLASSIFICATION_THRESHOLD = 0.5 
MIN_CHAR_BOX_WIDTH = 5    
MIN_CHAR_BOX_HEIGHT = 10   
MAX_CHAR_BOX_ASPECT_RATIO = 2.5 
MIN_CHAR_AREA = MIN_CHAR_BOX_WIDTH * MIN_CHAR_BOX_HEIGHT

# Parameter UI
ITEMS_PER_PAGE = 30
