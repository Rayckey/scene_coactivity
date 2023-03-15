#!/usr/bin/env python3

import numpy as np
from copy import deepcopy

import pickle
import os

from scene_manager.common_utilis import TaskConfig, TaskInfo


def main(display=True, teleport=False, partial=False):

    sdf_path = ('./scene_manager/sdfurniture')
    sdf_path = os.path.join(sdf_path, 'tasks')

    # all the tasks
    task_name = 'mock_2_4_0_task'
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



    # bedroom task
    task_name = 'sun_bedroom_0_task'

    t_config = TaskConfig(pose_init=None, tasks=[
                          [t0,t4, t3, t2,tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)


    task_name = 'sun_bedroom_1_task'

    t_config = TaskConfig(pose_init=None, tasks=[
                          [t2,t3, t4, t1,t0, tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)



    task_name = 'sun_bedroom_2_task'

    t_config = TaskConfig(pose_init=None, tasks=[
                          [t3, t4, t5, t2,t0, tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)


    task_name = 'sun_livingroom_0_task'

    t_config = TaskConfig(pose_init=None, tasks=[
                          [t0, t6, t2, t0, tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)


    task_name = 'sun_livingroom_1_task'

    t_config = TaskConfig(pose_init=None, tasks=[
                          [t0, t6, t5, t2, t1, tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)


    task_name = 'sun_studio_0_task'

    t_config = TaskConfig(pose_init=None, tasks=[
                          [t0, t1, t2, t4, tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)


    task_name = 'sun_medium_0_task'
    # relax, organize, clean up
    t_config = TaskConfig(pose_init=None, tasks=[
                          [t1, t14, t1 ,t15, t4, t6, t8, t16, t4, t6, t8, t17, t10, tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)


    task_name = 'sun_medium_1_task'
    # office restock
    t_config = TaskConfig(pose_init=None, tasks=[
                          [t16, tr, t17, tr, t14, tr, t15]])
    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)

    task_name = 'sun_medium_2_task'
    # distribute items
    t_config = TaskConfig(pose_init=None, tasks=[
                          [t17, t4, t6, t8, t16, t4, t6, t8,  t14, t1, t15, t1 , t14, t10, t15, t10 , tr]])

    output_path = os.path.join(sdf_path, task_name + '.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(t_config, f)

    print('finished')


if __name__ == '__main__':
    main()
