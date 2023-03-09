#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from itertools import combinations
import yaml

from scene_manager.common_utilis import TaskConfig, TaskInfo

with open('./configs/categories.yaml', 'r') as file:
    categories = yaml.safe_load(file)
    TASK_RELATED = categories['TASK_RELATED']
    ROOM_CATEGORIES = categories['ROOM_CATEGORIES']
    OBJECT_CATEGORIES = categories['OBJECT_CATEGORIES']
    INTERACTABLE = categories['INTERACTABLE']
    
def getTables(furn_info):
    tables = []
    for id in furn_info.nodes:
        if 'cat' in furn_info.nodes()[id]:
            if 'table' in furn_info.nodes()[id]['cat'] or 'desk' in furn_info.nodes()[id]['cat']:
                tables.append(id)
    return tables


def getShelves(furn_info):
    shelves = []
    for id in furn_info.nodes:
        if 'cat' in furn_info.nodes()[id]:
            if 'dresser' in furn_info.nodes()[id]['cat'] or 'shel' in furn_info.nodes()[id]['cat'] or 'cabinet' in furn_info.nodes()[id]['cat'] or 'stand' in furn_info.nodes()[id]['cat']:
                shelves.append(id)
    return shelves


def getNotTools(furn_info):
    stuff = []
    for id in furn_info.nodes:
        if 'cat' in furn_info.nodes()[id]:
            if furn_info.nodes()[id]['cat'] in TASK_RELATED:
                stuff.append(id)
    return stuff


def genMixMatchTasks(furn_info):
    tables = getTables(furn_info)
    shelves = getShelves(furn_info)
    stuff = tables + shelves
    pairs = list(combinations(stuff, 2))

    tasks = []

    for pa in pairs:
        t0 = TaskInfo(task_type='pick', pose=np.array(
            [0]), item_id=pa[0])

        t1 = TaskInfo(task_type='place', pose=np.array(
            [0]), item_id=pa[1])

        tasks.append([t0, t1])

    t_config = TaskConfig(pose_init=None, tasks=tasks)
    return t_config


def genMoreTasks(furn_info):
    stuff = getNotTools(furn_info)
    pairs = list(combinations(stuff, 2))

    tasks = []

    for pa in pairs:
        t0 = TaskInfo(task_type='pick', pose=np.array(
            [0]), item_id=pa[0])

        t1 = TaskInfo(task_type='place', pose=np.array(
            [0]), item_id=pa[1])

        tasks.append([t0, t1])

    t_config = TaskConfig(pose_init=None, tasks=tasks)
    return t_config


def genLongTask(task_int):

    ts = TaskInfo(task_type='goto', pose=np.array(
        [0, -1.5,  0.]), item_id=None)

    tp = TaskInfo(task_type='goto', pose=np.array(
        [0, 1.5,  0.]), item_id=None)

    t0 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=0)

    t1 = TaskInfo(task_type='place', pose=np.array(
        [0]), item_id=1)

    t2 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=2)

    t3 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=3)

    t4 = TaskInfo(task_type='place', pose=np.array(
        [0]), item_id=4)

    t5 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=5)

    t6 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=6)

    t7 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=7)

    t8 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=8)
    t9 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=9)
    t10 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=10)
    t11 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=11)
    t12 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=12)
    t13 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=13)
    t14 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=14)
    t15 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=15)
    t16 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=16)
    t17 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=17)
    t18 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=18)
    t19 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=19)
    t20 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=20)

    tr = TaskInfo(task_type='return', pose=np.array(
        [0]), item_id=-1)

    if task_int == 0:
        # relax, organize, clean up
        t_config = TaskConfig(pose_init=None, tasks=[
            [t1, t14,  t15, t1], [t10, t16, t17, t10]])
    elif task_int == 1:
        t_config = TaskConfig(pose_init=None, tasks=[
            [t16], [t17], [t14],  [t15]])
    elif task_int == 2:
        t_config = TaskConfig(pose_init=None, tasks=[
            [t17, t4, t6, t8, t16, t8, t6, t4]])

    return t_config


def genLongOriginalTask(task_int):

    ts = TaskInfo(task_type='goto', pose=np.array(
        [0, -1.5,  0.]), item_id=None)

    tp = TaskInfo(task_type='goto', pose=np.array(
        [0, 1.5,  0.]), item_id=None)

    t0 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=0)

    t1 = TaskInfo(task_type='place', pose=np.array(
        [0]), item_id=1)

    t2 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=2)

    t3 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=3)

    t4 = TaskInfo(task_type='place', pose=np.array(
        [0]), item_id=4)

    t5 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=5)

    t6 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=6)

    t7 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=7)

    t8 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=8)
    t9 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=9)
    t10 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=10)
    t11 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=11)
    t12 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=12)
    t13 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=13)
    t14 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=14)
    t15 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=15)
    t16 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=16)
    t17 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=17)
    t18 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=18)
    t19 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=19)
    t20 = TaskInfo(task_type='pick', pose=np.array(
        [0]), item_id=20)

    tr = TaskInfo(task_type='return', pose=np.array(
        [0]), item_id=-1)

    if task_int == 0:
        # relax, organize, clean up
        t_config = TaskConfig(pose_init=None, tasks=[
            [t1, t14, t1, t15, t4, t6, t8, t16, t4, t6, t8, t17, t10, tr]])
    elif task_int == 1:
        t_config = TaskConfig(pose_init=None, tasks=[
            [t16, tr, t17, tr, t14, tr, t15]])
    elif task_int == 2:
        t_config = TaskConfig(pose_init=None, tasks=[
            [t17, t4, t6, t8, t16, t4, t6, t8,  t14, t1, t15, t1, t14, t10, t15, t10, tr]])

    return t_config


def genAllTasks(furn_info):
    stuff = [id for id in furn_info.nodes if furn_info.nodes()[id]
             ['cat'] != 'door']
    pairs = list(combinations(stuff, 2))

    tasks = []

    for pa in pairs:
        t0 = TaskInfo(task_type='pick', pose=np.array(
            [0]), item_id=pa[0])

        t1 = TaskInfo(task_type='place', pose=np.array(
            [0]), item_id=pa[1])

        tasks.append([t0, t1])

    t_config = TaskConfig(pose_init=None, tasks=tasks)
    return t_config
