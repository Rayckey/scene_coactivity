#!/usr/bin/env python3

import numpy as np
import copy
from concept_net.graph_utilis import findRootNodes
from scene_manager.common_utilis import *
from scene_manager.sdf_utilis import *
import csv
from scipy.spatial.transform import Rotation


with open('./configs/dataset_paths.yaml', 'r') as file:
    dataset_paths = yaml.safe_load(file)
    MODEL_CATEGORY_MAP_FILE = dataset_paths['MODEL_CATEGORY_MAP_FILE']
    MODEL_CATEGORY_FILE = dataset_paths['MODEL_CATEGORY_FILE']
    SUNCG_PATH = dataset_paths['SUNCG_PATH']

def room2networkx(room, model_bb_mapping, model_cat_mapping):

    furn_info = nx.DiGraph()

    idx2thing = {}
    thing2idx = {}
    idx = 0

    # add all the nodes
    for thing in room['node_list']:

        if room['node_list'][thing]['type'] != 'wall':
            furn_info.add_node(idx)
            furn_info.nodes()[idx]['type'] = room['node_list'][thing]['type']

        if room['node_list'][thing]['type'] != 'root' and room['node_list'][thing]['type'] != 'wall':
            # furn_info.nodes()[idx]['info'] = room['node_list'][thing]['self_info']
            furn_info.nodes()[
                idx]['id'] = room['node_list'][thing]['self_info']['node_model_id']
            furn_info.nodes()[
                idx]['cat'] = model_cat_mapping[furn_info.nodes()[idx]['id']]

            furn_info.nodes()[idx]['max'] = model_bb_mapping[furn_info.nodes()[
                idx]['id']][-1]
            furn_info.nodes()[idx]['min'] = model_bb_mapping[furn_info.nodes()[
                idx]['id']][-2]
            furn_info.nodes()[idx]['front'] = model_bb_mapping[furn_info.nodes()[
                idx]['id']][0]
            furn_info.nodes()[idx]['bb'] = np.array(
                model_bb_mapping[furn_info.nodes()[idx]['id']][1])/100.
            # t = np.array(room['node_list'][thing]['self_info']['transform']).reshape([4, 4], order='F')

            furn_info.nodes()[idx]['rotation'] = np.reshape(
                room['node_list'][thing]['self_info']['rotation'], (3, 3))

            furn_info.nodes()[idx]['pose'] = [room['node_list'][thing]['self_info']['translation'], list(
                Rotation.from_matrix(furn_info.nodes()[idx]['rotation']).as_euler('xyz'))]

        idx2thing[idx] = thing
        thing2idx[thing] = idx

        idx += 1

    idx = 0
    # add all the supporting edges
    for thing in room['node_list']:
        if 'support' in room['node_list'][thing]:
            for other_thing in room['node_list'][thing]['support']:
                furn_info.add_edge(
                    thing2idx[thing], thing2idx[other_thing], att='support')

    root_id = -1
    for id in furn_info.nodes():
        if furn_info.nodes()[id]['type'] == 'root':
            root_id = id
            break
    furn_info.remove_node(root_id)

    return furn_info


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
