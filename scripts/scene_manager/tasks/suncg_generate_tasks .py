import os
import pickle
import yaml
from scene_manager.common_utilis import TaskConfig, TaskInfo, SceneInformation
from scene_manager.tasks.psedo_task_gen import INTERACTABLE


def creatTasks(si: SceneInformation):

    furn_info = si.furniture_info

    ids = list(furn_info.keys())
    tasks = []

    for idx1 in range(len(ids)):
        for idx2 in range(idx1 + 1, len(ids)):
            if furn_info[ids[idx1]]['type'] in INTERACTABLE and furn_info[ids[idx2]]['type'] in INTERACTABLE:
                tp = TaskInfo(task_type='travel', pose=None,
                              item_id=[ids[idx1], ids[idx2]])
                tasks.append([tp])

    return TaskConfig(pose_init=None, tasks=tasks)


if __name__ == '__main__':

    scene_path = './scene_manager'
    room_path = os.path.join(scene_path, 'rooms')
    task_path = os.path.join(scene_path, 'tasks')

    goodones = [
        # 256,
        # 320,
        # 321,
        # 352,
        # 471,
        # 281,
        # 347,
        # 469,
        # 329,
        # 536,
        # 630,
        # 637,
        # 798,
        # 1240,
        # 1241,
        # 1266,
        # 1275,
        # 1294,
        'sun_livingroom_0',
        'sun_livingroom_1',
        'sun_small_0',
        'sun_small_1',
        'sun_small_2',
        'sun_demo_0',
        # 1774
        'sun_pipeline_0',
        # 281,
        # 347,
        # 471
        'sun_oral_0',
        'sun_oral_1',
        'sun_oral_2'
        ]

    for idx in goodones:
        print(idx)
        p = os.path.join(room_path, str(idx) + '.pickle')
        with open(p, 'rb') as f:
            si = pickle.load(f)

        t_config = creatTasks(si)

        output_path = os.path.join(task_path, str(idx) + '_task' + '.pickle')
        with open(output_path, 'wb') as f:
            pickle.dump(t_config, f)
