import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import os
import pickle
from scene_manager.scene_manager2 import *
from sampling.sampling_utilis import *
from scene_manager.tasks.psedo_task_gen import genLongOriginalTask, genMixMatchTasks, genAllTasks, genMoreTasks, genLongTask
from visualization.visual_utilis import *
from scene_manager.common_utilis import ROBOTS

def viewSeriesRooms(res_name, reach, app, path, color, resolution):
    fi = open(res_name, "rb")
    opt_config, task_config, stepping_config, scene_hist, hist = pickle.load(
        fi)

    head, tail = os.path.split(res_name)

    idx = 0
    for scene_info in scene_hist:
        furn_info, robot, seg_map, grouping_order, task_config = scene_info
        offcenter_furniture(furniture_info=furn_info)

        if not (reach or app or path or color):
            visualizeFurnitureInfo(
                furn_info, seg_map, grouping_order, use_walls=False, GUI=False,
                snip_shot=True, make_gif=False, name=f"./images/{tail}_{idx}", resolution=resolution)
        else:
            visualizeOverlayInfo(
                furn_info, opt_config, task_config, robot, seg_map,
                reach=reach, app=app, path=path, color=color,
                grouping_order=grouping_order,
                use_walls=False, GUI=False, snip_shot=True,
                name=f"./images/{tail}_{idx}", resolution=resolution)

        idx += 1


def viewSingleRoom(scene_name, gif, reach, app, path, color, robot_name, resolution):
    # res_path = os.path.join(folder_name, scene_name)
    head, tail = os.path.split(scene_name)

    with open(scene_name, "rb") as fi:
        furn_info, seg_map = pickle.load(fi)

    try:
        robot = ROBOTS[robot_name]
    except:
        print('Unknown option, please pick between dingo and husky for the service robot.')
        raise ValueError(robot_name)

    offcenter_furniture(furniture_info=furn_info)
    filter_higher_up_furniture(furn_info)
    filter_ignored_furniture(furn_info)
    
    if not (reach or app or path or color):
        visualizeFurnitureInfo(
            furn_info, seg_map, snip_shot=True, make_gif=gif, name=f'./images/{tail}', resolution=resolution)
    else:
        # generate a dummy config
        opt_config = AdaptiveAnnealingConfig(max_iterations=100, epoch=50, temp_init=52.3, randomize=False,
                                             scene_name=scene_name,
                                             robot=robot['name'], term_criteria=0.01, step_scale=2.0, ns=5, nt=1,
                                             n_term=10, scaling_factors=2.0, vocal=False, log_every=100,
                                             desired_sdf_res=0.01, decay=0.5, write_images=False, weights=WEIGHTS)

        task_config = genMoreTasks(furn_info)

        visualizeOverlayInfo(
            furn_info, opt_config, task_config, robot, seg_map,
            make_gif=gif, reach=reach, app=app, path=path, color=color,
            snip_shot=True, name=f'./images/{scene_name}', resolution=resolution)


def viewFinalRoom(res_name, gif, reach, app, path, color, resolution):

    with open(res_name, "rb") as fi:
        opt_config, task_config, stepping_config, scene_hist, hist = pickle.load(
            fi)
    head, tail = os.path.split(res_name)
    scene_info = scene_hist[-1]
    furn_info, robot, seg_map, grouping_order, task_config = scene_info
    offcenter_furniture(furniture_info=furn_info)

    if not (reach or app or path or color):
        visualizeFurnitureInfo(
            furn_info, seg_map, grouping_order, use_walls=False, GUI=False,
            snip_shot=True, make_gif=gif, name=f"./images/{tail}_{-1}", resolution=resolution)
    else:
        visualizeOverlayInfo(
            furn_info, opt_config, task_config, robot, seg_map,
            make_gif=gif, reach=reach, app=app, path=path, color=color,
            grouping_order=grouping_order,
            use_walls=False, GUI=False, snip_shot=True,
            name=f"./images/{tail}_{-1}", resolution=resolution)


def viewAllRooms(folder_name, overlay):
    dir_list = os.listdir(folder_name)
    for scene_name in dir_list:
        res_path = os.path.join(folder_name, scene_name)
        fi = open(res_path, "rb")
        furn_info, seg_map = pickle.load(fi)
        robot = ROBOTS['dingo']

        # offcenter_furniture(furniture_info=furn_info)
        if overlay is None:
            visualizeFurnitureInfo(
                furn_info, seg_map, None, False, False, True, f'./images/{scene_name}')
        else:
            opt_config = AdaptiveAnnealingConfig(max_iterations=100, epoch=50, temp_init=52.3, randomize=False,
                                                 scene_name=scene_name,
                                                 robot=robot['name'], term_criteria=0.01, step_scale=2.0, ns=5, nt=1,
                                                 n_term=10, scaling_factors=2.0, vocal=False, log_every=100,
                                                 desired_sdf_res=0.01, decay=0.5, write_images=False, weights=WEIGHTS)
            task_config = genAllTasks(furn_info)

            visualizeOverlayInfo(
                furn_info, opt_config, task_config, robot, seg_map, None, False, False, True, f'./images/{scene_name}', overlay)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments for visualizing scene/results')

    parser.add_argument('-i', '--room_id', type=str, default="",
                        help='the room_id to be shown, ignored if there is no input.')

    parser.add_argument('-r', '--results', type=str, default="",
                        help='the optimized result to be shown, ignored if there is no input.')

    parser.add_argument('-g', '--gif', help='generate rotating gifs for output.',
                        action='store_true')

    parser.add_argument('-s', '--sequence', help='generate optimization sequence image, only works with --results input, not compatible with --gif',
                        action='store_true')

    parser.add_argument('-p', '--resolution', type=int, default=700,
                        help='define the image resolution.')

    parser.add_argument('-b', '--accessible', help='overlay the output scene with (B)lue accessible space.',
                        action='store_true')

    parser.add_argument('-l', '--path', help='overlay the output scene with robot path (L)ine.',
                        action='store_true')

    parser.add_argument('-a', '--interaction', help='visualize the pseudo-inter(A)ction function for each object.',
                        action='store_true')

    parser.add_argument('-c', '--color', help='(C)olor-code the objects by whether they are accessible to the robot.',
                        action='store_true')

    parser.add_argument('--robot', type=str, default='dingo',
                        help="define the robot's base size (default: dingo, options: dingo, husky), not applicable when visualizing results")

    args = parser.parse_args()

    if args.gif and args.sequence:
        print("Gif and Sequence cannot be generated at the same time.")

    # view original room
    if len(args.room_id) > 0:
        room_name = args.room_id
        if not os.path.isfile(room_name):
            room_name = os.path.join('./rooms', args.room_id)

        if args.sequence:
            print("Original rooms cannot have sequence")

        else:
            viewSingleRoom(room_name, args.gif, args.accessible, args.interaction, args.path,
                           args.color, args.robot, args.resolution)

    # view results
    if len(args.results) > 0:
        res_name = args.results
        if not os.path.isfile(res_name):
            res_name = os.path.join('./results', args.results)

        if args.sequence:
            viewSeriesRooms(res_name, args.accessible, args.interaction, args.path,
                            args.color, args.resolution)

        else:
            viewFinalRoom(res_name, args.gif, args.accessible, args.interaction, args.path,
                          args.color, args.resolution)
