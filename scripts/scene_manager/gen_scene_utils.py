#!/usr/bin/env python3

import numpy as np
from scene_manager.common_utilis import *
from scene_manager.sdf_utilis import *
import csv
from scipy.spatial.transform import Rotation


with open('./configs/dataset_paths.yaml', 'r') as file:
    dataset_paths = yaml.safe_load(file)
    MODEL_CATEGORY_MAP_FILE = dataset_paths['MODEL_CATEGORY_MAP_FILE']
    MODEL_CATEGORY_FILE = dataset_paths['MODEL_CATEGORY_FILE']
    SUNCG_PATH = dataset_paths['SUNCG_PATH']

def convert2RoomInfo(room_data, seg_map):
    if room_data['up'] == [0, 1, 0]:
        # seg_map['boxes'][0][2] = abs(
        #     room_data['bbox'][2] - room_data['bbox'][5])
        # seg_map['boxes'][0][3] = abs(
        #     room_data['bbox'][0] - room_data['bbox'][3])

        seg_map['min'] = [room_data['bbox'][0],  -
                          room_data['bbox'][5],  room_data['bbox'][1]]
        seg_map['max'] = [room_data['bbox'][3],  -
                          room_data['bbox'][2],  room_data['bbox'][4]]
    else:
        print("this room's up vector is off")
        return False
    room_max = [room_data['bbox'][5],
                room_data['bbox'][3], room_data['bbox'][4]]
    return room_max



def imgbox2Bound(box, resol=None, map_size=None):
    x1, y1, z1, x0, y0, z0 = box
    return (x0, y0, x1, y1)


def detectBounds(furn_info, seg_map_dict):
    for key in furn_info.keys():
        x0, y0, x1, y1 = imgbox2Bound(
            seg_map_dict['max']+seg_map_dict['min'], seg_map_dict['resolution'], seg_map_dict['size'])

        if furn_info[key]['cat'] == 'door':
            furn_info[key]['bounds'] = [[0], [np.abs(2*(x1 - x0 + y1 - y0))]]
        else:

            furn_info[key]['bounds'] = [[x0, y0, 0],
                                        [x1, y1, 2 * np.pi]]  # lower , higher

def findDoorSem(furn_info):
    for id in furn_info.nodes():
        if furn_info.nodes()[id]['cat'] == 'door':
            return furn_info.nodes()[id]


def findDoorID(furn_info):
    for id in furn_info.nodes():
        if furn_info.nodes()[id]['cat'] == 'door':
            return id


def loadSuncgTypeMap():
    suncgType2Cat = {}
    with open(MODEL_CATEGORY_MAP_FILE, 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            suncgType2Cat[row[0]] = row[1]

    return suncgType2Cat
