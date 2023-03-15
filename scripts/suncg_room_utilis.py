import numpy as np
import json
import csv
import glob
import yaml
from tqdm import tqdm
from scene_manager.tasks.psedo_task_gen import ROOM_CATEGORIES, OBJECT_CATEGORIES
from scene_manager.gen_scene_utils import MODEL_CATEGORY_FILE, SUNCG_PATH


def generate_model_category_mapping():
    # Generate dictionary from cad model id -> object category
    model_category_mapping = {}

    with open(MODEL_CATEGORY_FILE, 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            model_category_mapping[row[1]] = row[2]

    return model_category_mapping
