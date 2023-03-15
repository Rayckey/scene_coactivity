import pybullet as p
import numpy as np
import time
import argparse
import os
from optimization_manager import LayeredManager
from scene_manager.common_utilis import center_furniture, filter_higher_up_furniture, filter_ignored_furniture, WEIGHTS
from sampling.sampling_utilis import *
from concept_net.graph_utilis import *
from scene_manager.tasks.psedo_task_gen import *
from scene_manager.common_utilis import ROBOTS

def from_furn_info(room_type, room_id, GUI, images, robot_name):
    np.random.seed(int(time.time()))

    scene_name = f'{room_type}_{room_id}'

    # room_type = names[0]
    processed_position_net = open(
        f'./suncg/data/{room_type}_position_net_processed_lognorm.bin', 'rb')
    position_net = pickle.load(processed_position_net)

    processed_occurrence_net = open(
        f'./suncg/data/{room_type}_occurrence_net_processed.bin', 'rb')
    occurrence_net = pickle.load(processed_occurrence_net)

    try:
        processed_furn_info = open(
            f'./rooms/suncg/{scene_name}', 'rb')
        furn_info, seg_map = pickle.load(processed_furn_info)
    except:
        return

    filter_higher_up_furniture(furn_info)
    filter_ignored_furniture(furn_info)

    robot = ROBOTS[robot_name]

    use_block = True
    use_3d = False

    t_config = genMoreTasks(furn_info)

    center_furniture(furn_info)
    manager = LayeredManager(scene_name=scene_name, furniture_info=furn_info, seg_map=seg_map, position_net=position_net, occurrence_net=occurrence_net, robot=robot,
                             use_gui=GUI, write_images=images, depth=2, breadth=10)

    randomize_init = True

    opt_configs = AdaptiveAnnealingConfig(max_iterations=500, epoch=500, temp_init=52.3, randomize=randomize_init,
                                          scene_name=scene_name,
                                          robot=manager.robot['name'], term_criteria=1e-3, step_scale=2.0, ns=5, nt=2,
                                          n_term=50, scaling_factors=2.0, vocal=False, log_every=100,
                                          desired_sdf_res=0.1, decay=0.85, write_images=images, weights=WEIGHTS)

    manager.storeTaskConfig(t_config)
    manager.storeOptimizationStates(opt_configs)

    # start_time = time.time()

    furn_info, seg_map = manager.solve()

    # print("--- %s seconds ---" % (time.time() - start_time))

    finihsed_furn_info = open(
        f'./results/{scene_name}', 'w+b')
    pickle.dump((furn_info, seg_map), finihsed_furn_info)

    p.disconnect()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for differnt options of SunCG data processing.')
    parser.add_argument('--room', type=str, default='living',
                        help='define room category.')
    parser.add_argument('--idx', type=int, default=None,
                        help='define the room ID from the selected room category.')
    parser.add_argument('--GUI', type=bool, default=False,
                        help='define whether the GUI powered by Pybullet will be displayed. This will show the optimization process in real time')
    parser.add_argument('--images', type=bool, default=False,
                        help='define whether debug images will be generated.')
    parser.add_argument('--robot', type=str, default='dingo',
                        help="define the robot's base size(default: dingo, options: dingo, husky)")
    args = parser.parse_args()

    from_furn_info(args.room, args.idx,
                args.GUI, args.images, args.robot)
