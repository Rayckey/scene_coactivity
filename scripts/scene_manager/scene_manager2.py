#!/usr/bin/env python3
import numpy as np
import pickle
from scipy import ndimage
from copy import deepcopy
import pybullet as p
from itertools import combinations

from scipy.stats import lognorm
from sampling.sampling_utilis import OptimizationConfig, TaskConfig
# from scene_manager.common_utilis import *

from scene_manager.sdf_utilis import *
from scene_manager.bullet.bullet_utilis import *
from scene_manager.common_utilis import *

from networkx import DiGraph
from scene_manager.gen_scene_utils import loadSuncgTypeMap


with open('./configs/exceptions.yaml', 'r') as file:
    exceptions = yaml.safe_load(file)
    DOORS = exceptions['DOORS']
    LOWERDOWN = exceptions['LOWERDOWN']


class SceneManager():

    def __init__(self, furn_info: DiGraph = None, position_net=None, seg_map=None, robot=None, use_gui=True) -> None:

        if isinstance(seg_map, str):
            with open(seg_map, 'rb') as f:
                seg_map = pickle.load(f)
        if isinstance(furn_info, str):
            with open(seg_map, 'rb') as f:
                seg_map = pickle.load(f)
        if isinstance(robot, str):
            with open(seg_map, 'rb') as f:
                seg_map = pickle.load(f)

        self.seg_map = seg_map
        self.furniture_info = furn_info
        self.robot = robot
        self.tasks = None
        self.objectives = []
        self.unpack_info = {}
        self.sdf_buffer = {}
        self.interact_buffer = {}
        self.anti_buffer = {}
        self.wall_buffer = None
        self.wall_buffer_enlarged = None
        self.interact = {}
        self.anti = {}
        self.write_images = False
        self.h = 0
        self.w = 0
        self.extent = []
        self.h_extended = 0
        self.w_extended = 0
        self.extended_extent = []
        self.padding = 0.
        self.extension = 0.
        self.use_gui = use_gui
        self.type_map = loadSuncgTypeMap()
        self.position_net = position_net

    def loadProcessedspatialnx(self, path: str):
        fi = open(path, "rb")
        self.position_net = pickle.load(fi)

    def repackage(self):
        return SceneInformation(self.furniture_info, self.robot, self.seg_map, self.grouping_order, self.task_config)

    def packID2States(self, grouping_order):
        # grouping_order = {lead_id: [all furn id in that group]} if used grouping
        # grouping_order = {'use': [all furn id]}  if used individually, ignore all not listed
        # grouping_order = {'use': [all furn id], 'fix':[all furn id]}  if used individually, fix other furn
        plot_ids = []
        state = []
        state_keys = []
        unpack_info = {}

        room_min = np.array(self.seg_map['min'])
        room_size = np.array(self.seg_map['max']) - room_min

        if grouping_order is None:
            for id in self.furniture_info.nodes:
                plot_ids += [id]
                state_keys += [id]
                ss = (np.array(self.furniture_info.nodes()[
                      id]['pose'][0]) - room_min)/room_size
                state += [ss[0], ss[2], 0.5]
                unpack_info[id] = {}

            state = np.array(state)
            bounds = [np.zeros_like(state).tolist(),
                      np.ones_like(state).tolist()]
            return plot_ids, state, state_keys, unpack_info, bounds
        for root_or_type in grouping_order:
            if root_or_type == 'use':
                plot_ids += grouping_order[root_or_type]
                state_keys += grouping_order[root_or_type]
                for _id in grouping_order[root_or_type]:
                    ss = (np.array(self.furniture_info.nodes()[
                          _id]['pose'][0]) - room_min)/room_size
                    state += [ss[0], ss[2], 0.5]
            elif root_or_type == 'fix':
                plot_ids += grouping_order[root_or_type]
            else:
                plot_ids += [root_or_type]
                plot_ids += grouping_order[root_or_type]
                state_keys += [root_or_type]
                ss = (np.array(self.furniture_info.nodes()[
                      root_or_type]['pose'][0]) - room_min)/room_size
                state += [ss[0], ss[2], 0.5]
                unpack_info[root_or_type] = {}

                for _id in grouping_order[root_or_type]:
                    unpack_info[root_or_type][_id] = self.get_relative_transform(
                        _id, root_or_type)

        state = np.array(state)
        bounds = [np.zeros_like(state).tolist(), np.ones_like(state).tolist()]
        return plot_ids, state, state_keys, unpack_info, bounds

    def unpackStates2ID(self, states):
        cur_idx = 0
        for sk in self.state_keys:
            state = states[cur_idx:(cur_idx+3)]
            ss = np.array([state[0], 0, state[1]])
            ss = ss * self.seg_map['size'] + self.seg_map['min']
            self.furniture_info.nodes()[sk]['pose'][0][0] = ss[0]
            self.furniture_info.nodes()[sk]['pose'][0][2] = ss[2]
            rr = R.from_matrix(self.furniture_info.nodes()[
                               sk]['rotation']) * R.from_euler('xyz', [0, (state[2]-0.5)*np.pi*2, 0])
            self.furniture_info.nodes()[sk]['pose'][1] = rr.as_euler(
                'xyz').tolist()

            if sk in self.unpack_info:
                for _id in self.unpack_info[sk]:
                    trans_p = get_transform_from_xyz_rpy(self.furniture_info.nodes(
                    )[sk]['pose'][0], self.furniture_info.nodes()[sk]['pose'][1])
                    trans_n = trans_p @ self.unpack_info[sk][_id]
                    self.furniture_info.nodes(
                    )[_id]['pose'][0][0] = trans_n[0, 3]
                    self.furniture_info.nodes(
                    )[_id]['pose'][0][1] = trans_n[1, 3]
                    self.furniture_info.nodes(
                    )[_id]['pose'][0][2] = trans_n[2, 3]
                    self.furniture_info.nodes()[_id]['pose'][1] = R.from_matrix(
                        trans_n[:3, :3]).as_euler('xyz').tolist()

            cur_idx += 3
        return

    def updateBulletFurniture(self, states):
        self.unpackStates2ID(states)
        self.updateBulletEnv()

    def setupOptimizationConfig(self, opt_config: OptimizationConfig):

        self.weights = opt_config.weights

        self.grouping_order = opt_config.grouping_order

        self.prepBulletEnv(use_gui=self.use_gui, use_floor=False)
        self.genBulletWalls()

        plot_ids, x, self.state_keys, self.unpack_info, self.bounds = self.packID2States(
            opt_config.grouping_order)

        self.plot_ids = plot_ids

        if opt_config.randomize:
            x = self.genRandZeta(x)

        self.clearBulletIDs(plot_ids)

        self.genBulletEnv(
            ids=plot_ids, use_block=True, use_3d=True)

        self.desired_sdf_res = opt_config.desired_sdf_res

        self.objectives = opt_config.objectives

        self.h = int(np.round(
            (self.seg_map['max'][0]-self.seg_map['min'][0])/opt_config.desired_sdf_res))
        self.w = int(np.round(
            (self.seg_map['max'][2]-self.seg_map['min'][2])/opt_config.desired_sdf_res))

        self.h_extended = int(
            np.round((self.seg_map['max'][0]-self.seg_map['min'][0] + 2 * opt_config.padding) / opt_config.desired_sdf_res))
        self.w_extended = int(
            np.round((self.seg_map['max'][2]-self.seg_map['min'][2] + 2 * opt_config.padding) / opt_config.desired_sdf_res))
        self.padding = opt_config.padding
        self.extension = opt_config.extension

        self.write_images = opt_config.write_images

        return x, self.bounds
        # return

    def setTaskConfig(self, task_config: TaskConfig):
        # self.pose_init = task_config.pose_init
        self.pose_init = self.getStartPose()
        self.tasks = task_config.tasks
        self.task_config = task_config

    def clearBulletIDs(self, ids=[]):
        if len(ids) > 0:
            for id in ids:
                self.furniture_info.nodes()[id]['bullet'] = None

    def genBulletEnv(self, ids=[], use_block=True, use_3d=True):

        if len(ids) == 0:
            ids = list(self.furniture_info.nodes())

        for id in ids:
            sem = self.furniture_info.nodes()[id]
            if sem['bullet'] is not None:
                continue

            self.furniture_info.nodes()[id]['bullet'] = gen_bullet_obj(sem, cat=self.type_map[self.furniture_info.nodes()[
                id]['cat']], use_block=use_block, use_3d=use_3d, color=np.array([0, 0, 0, 1]))
            self.furniture_info.nodes(
            )[id]['bullet_center'] = gen_bullet_center(sem)
            p.setCollisionFilterPair(self.furniture_info.nodes(
            )[id]['bullet'], self.furniture_info.nodes()[id]['bullet_center'], -1, -1, 0)

    def updateBulletEnv(self):
        for id in self.plot_ids:
            update_bullet_obj(self.furniture_info.nodes()[
                id]['bullet'], self.furniture_info.nodes()[id]['pose'])
            update_bullet_obj(self.furniture_info.nodes()[
                id]['bullet_center'], self.furniture_info.nodes()[id]['pose'])

    def prepBulletEnv(self, use_gui=False, use_floor=True):
        try:
            p.connect(p.GUI) if use_gui == True else p.connect(p.DIRECT)
        except:
            print("Bullet is already on")
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # optionally
        # """set Gravity"""
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # don't create a ground plane, to allow for gaps etc
        p.resetSimulation()
        if use_floor:
            p.createCollisionShape(p.GEOM_PLANE)
            p.createMultiBody(0, 0)
        if use_gui:
            p.resetDebugVisualizerCamera(5, 0, 180, [0, 10, 0])
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def genBulletWalls(self, thickness=20.0):
        upper = np.array(self.seg_map['max'])
        lower = np.array(self.seg_map['min'])
        box = upper - lower
        center = lower + box/2.

        color = [1, 1, 1, 1]

        linkPositions = [[upper[0]+thickness/2., center[1], center[2]], [center[0], center[1], upper[2]+thickness/2.],
                         [lower[0]-thickness/2., center[1], center[2]], [center[0], center[1], lower[2]-thickness/2.]]

        collisionShapeId = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                                       halfExtents=[[thickness/2., box[1]/2., box[2]/2. + thickness], [box[0]/2., box[1]/2., thickness/2.],
                                                                    [thickness/2., box[1]/2., box[2]/2. + thickness], [box[0]/2., box[1]/2., thickness/2.]],
                                                       collisionFramePositions=linkPositions)

        visualShapeId = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                                 halfExtents=[[thickness/2., box[1]/2., box[2]/2. + thickness], [box[0]/2., box[1]/2., thickness/2.],
                                                              [thickness/2., box[1]/2., box[2]/2. + thickness], [box[0]/2., box[1]/2., thickness/2.]],
                                                 rgbaColors=[color]*4,
                                                 visualFramePositions=linkPositions)

        obj_id = p.createMultiBody(baseMass=1,
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=[0, 0, 0])
        return obj_id

    def discritizeSDF(self, w=1024, h=1024, b=None, upper=None, lower=None, y=0):

        begin_point = (upper[0], lower[1])
        end__point = (lower[0], upper[1])
        if b is not None:
            img, extent, P = sample_slice_2d(
                b, w=w, h=h, bounds=(begin_point, end__point), y=y)
        else:
            img, extent, P = sample_slice_2d(
                self.sdf, w=w, h=h, bounds=(begin_point, end__point), y=y)

        return img, extent, P

    def genReachabilityCV(self, pose=None, w=1024, h=1024, write_images=False, return_none=False):
        pic, extent, coordinates = self.discritizeSDF(w=w, h=h)
        # pic = np.flip(pic, axis=0)

        if write_images:
            write_normalized_image(pic, '__temp__/savedImage.jpg')

        pic_sdf = pic - self.robot['size']
        mask = pic > self.robot['size']

        mask_b = mask.astype(np.uint8) * 255
        labeled, num_seg = label(mask_b, return_num=True)

        if pose is not None:
            pose_sdf = pos2SDF(pose[:2], extent, w, h)
            seg_id = labeled[pose_sdf[0], pose_sdf[1]]
            if seg_id > 0:
                return self.convertReachability2SDF(
                    labeled == seg_id, pic_sdf, extent, w, h), coordinates, pic
            else:
                if return_none:
                    return None
                mask = np.zeros_like(labeled)
                mask[pose_sdf[0], pose_sdf[1]] = 1
                mask = mask > 0
                return self.convertReachability2SDF(mask, pic_sdf, extent, w, h), coordinates, pic

    def convertReachability2SDF(self, mask, pic_sdf, extent, w, h):
        dist = self.dist2Reachability(mask, extent, w, h)
        return pic_sdf * mask + dist * np.logical_not(mask)

    def dist2Reachability(self, mask, extent, w, h):
        edt, inds = ndimage.distance_transform_edt(
            np.logical_not(mask), return_indices=True)
        dist_idx = inds.astype('float')
        idxes = np.indices(mask.shape)
        res = get_img_res(extent, w, h)
        dist_idx[0] = np.abs(dist_idx[0] - idxes[0]) * res[0]
        dist_idx[1] = np.abs(dist_idx[1] - idxes[1]) * res[1]
        return -np.sqrt(dist_idx[0] ** 2 + dist_idx[1] ** 2)

    def genRandZeta(self, x):
        return np.random.random(x.shape)

    def objectiveFunction(self, state=None):
        if state is not None:
            self.updateBulletFurniture(state)

        if np.any(np.isnan(state)):
            print(' NAN showed up in states')

        cost = 0

        use_approach = True
        use_oppose = True

        object_sdf_pad, _, _, app_pic_pad, opp_pic_pad = self.discritizeSDFfromBuffer(
            self.plot_ids, use_approach=use_approach, use_oppose=use_oppose)

        app_pic = getCenteredSDF(app_pic_pad, self.w, self.h)
        opp_pic = getCenteredSDF(opp_pic_pad, self.w, self.h)
        object_sdf = getCenteredSDF(object_sdf_pad, self.w, self.h)

        reachability_sdf = self.convertSDF2Reachability(object_sdf)

        if self.write_images:

            write_colorized_image(object_sdf, './buffer/images/obj_sdf.png',  use_transparent=True, cut_off_value=0.0, max_rgb=[
                0.06, 0.429, 0.853], min_rgb=[0.13, 0.73, 0.81])
            write_colorized_image(app_pic, './buffer/images/app.png',  use_transparent=True, cut_off_value=0.05, max_rgb=[
                0.05, 0.80, 0.32], min_rgb=[0.49, 0.8, 0.05])
            if reachability_sdf is not None:
                write_colorized_image(np.array(reachability_sdf > 0).astype('float'), './buffer/images/rea_sdf.png',  use_transparent=True, cut_off_value=0.0, max_rgb=[
                    0.06, 0.429, 0.853], min_rgb=[0.13, 0.73, 0.81])
            write_colorized_image(np.array(object_sdf > 0).astype('float'), './buffer/images/obj.png', use_transparent=True,
                                  cut_off_value=0.05, max_rgb=[0.8, 0.16, 0.05], min_rgb=[0.8, 0.67, 0.05])
            write_colorized_image(-opp_pic, './buffer/images/ant.png', use_transparent=True,
                                  cut_off_value=0.05, max_rgb=[0.8, 0.16, 0.05], min_rgb=[0.8, 0.67, 0.05])

        if 'collision' in self.objectives:
            p.performCollisionDetection()  # need this once for all collisions
            cp = p.getContactPoints()
            collision_dist = 0
            for c in cp:
                collision_dist += c[8]
            collision_cost = WEIGHTS['collision'] * \
                collision2energy(collision_dist)
            cost += collision_cost

        if 'interaction' in self.objectives:
            # not using the extended sdf to save memory, hope this works well
            aff = 0

            use_sdf = reachability_sdf
            # use_sdf = object_sdf
            for id in self.interact.keys():
                aff_total = self.interact_buffer[id]['best']

                aff = np.sum(getCenteredSDF(
                    self.interact[id], self.w, self.h) * use_sdf)

                aff_cost = WEIGHTS['interaction'] * \
                    (1.0 - low_satuating_cost(aff))*2
                cost += aff_cost

        if 'anti' in self.objectives:
            ant = 0
            # ant_total = 0
            use_sdf = self.wall_buffer_enlarged * \
                np.array(self.wall_buffer_enlarged < 0).astype('float')
            for id in self.anti.keys():
                ant_total = self.anti_buffer[id]['best']

                ant = np.sum(self.anti[id] * use_sdf)

                ant_cost = WEIGHTS['anti'] * (1.0 - low_satuating_cost(ant))*2
                cost += ant_cost

        if 'planning' in self.objectives and reachability_sdf is not None:
            coordinates = extent2Coordinates(self.extent, self.h, self.w)
            if reachability_sdf is not None and type(reachability_sdf) is not int:
                path_dist, paths, max_total = self.PlanningStarCost(
                    reachability_sdf, coordinates, self.w, self.h)

                for dist in path_dist:
                    # pln_cost = WEIGHTS['planning'] * \
                    #     ratio_cost(path_dist, max_total)  # TODO
                    pln_cost = WEIGHTS['planning'] * \
                        (lower_satuating_cost(dist)-0.5)*2
                    cost += pln_cost

            else:
                paths = []
                cost += WEIGHTS['planning'] * 10.0

            if self.write_images:
                write_image_path(
                    reachability_sdf, paths, './buffer/images/paths.png')

        if 'relation' in self.objectives:

            rel_list = []

            for relation in self.relations:
                id1, id2, re = relation
                if self.isInterrupted(id2, id1):
                    rel_list.append(1.0)
                    continue

                cat1 = self.furniture_info.nodes()[
                    id1]['cat'].replace("_", " ")
                cat2 = self.furniture_info.nodes()[
                    id2]['cat'].replace("_", " ")

                if re == 'NEXTO':
                    # distance betwwen two coordinates
                    NEXTTO = self.getNexttoCost(idx_n=id2, idx_p=id1)
                    NEXTTO_cost = 1.0 - \
                        sigmoid(10*lognorm.pdf(NEXTTO, *
                                self.position_net[cat1][cat2]['NEXTTO']))
                    rel_list.append(NEXTTO_cost)
                elif re == 'CLOSETO':
                    # distance betwwen two bounding box
                    CLOSETO = self.getClosetoCost(idx_n=id2, idx_p=id1)
                    CLOSETO_cost = 1.0 - \
                        sigmoid(10*lognorm.pdf(CLOSETO, *
                                self.position_net[cat1][cat2]['CLOSETO']))
                    rel_list.append(CLOSETO_cost)
                elif re == 'FRONTOF':
                    FRONTOF = self.getFrontCost(
                        idx_n=id2, idx_p=id1)  # on +z axis
                    FRONTOF_cost = 1.0 - \
                        sigmoid(lognorm.pdf(
                            FRONTOF, *self.position_net[cat1][cat2]['FRONTOF']))
                    rel_list.append(FRONTOF_cost)
                elif re == 'FACING':
                    FACING = self.getFacingCost(
                        idx_n=id2, idx_p=id1)  # id1 is facing id2
                    FACING_cost = 1.0 - \
                        sigmoid(lognorm.pdf(
                            FACING, *self.position_net[cat1][cat2]['FACING']))
                    rel_list.append(FACING_cost)
                elif re == 'PARALLEL':
                    # id1 facing same direction as id2
                    PARALLEL = self.getParallelCost(idx_n=id2, idx_p=id1)
                    PARALLEL_cost = 1.0 - \
                        sigmoid(lognorm.pdf(
                            PARALLEL, *self.position_net[cat1][cat2]['PARALLEL']))
                    rel_list.append(PARALLEL_cost)
                elif re == 'CLOSETOBOUND':
                    CLOSETOBOUND = self.getClosetoBoundCost(
                        idx_n=id2, idx_p=id1)
                    CLOSETOBOUND_cost = 1.0 - \
                        sigmoid(5*lognorm.pdf(CLOSETOBOUND, *
                                self.position_net[cat1][cat2]['CLOSETOBOUND']))
                    rel_list.append(CLOSETOBOUND_cost)
                elif re == 'FACINGBOUND':
                    FACINGBOUND = self.getFacingBoundCost(idx_n=id2, idx_p=id1)
                    FACINGBOUND_cost = 1.0 - \
                        sigmoid(lognorm.pdf(FACINGBOUND, *
                                self.position_net[cat1][cat2]['FACINGBOUND']))
                    rel_list.append(FACINGBOUND_cost)


            rel_cost = WEIGHTS['relation'] * np.sum(rel_list)
            cost += rel_cost


        if np.isnan(cost):
            print(' NAN showed up in cost')

        return cost

    def setOptimizeRelations(self, edges):
        self.relations = edges

    def generateOverlayImages(self):
        object_sdf_pad, _, _, app_pic_pad, opp_pic_pad = self.discritizeSDFfromBuffer(
            list(self.furniture_info.nodes), use_approach=True, use_oppose=True)

        app_pic = getCenteredSDF(app_pic_pad, self.w, self.h)
        opp_pic = getCenteredSDF(opp_pic_pad, self.w, self.h)
        object_sdf = getCenteredSDF(object_sdf_pad, self.w, self.h)

        reachability_sdf = self.convertSDF2Reachability(object_sdf)

        app_png = gen_colorized_image(app_pic,  use_transparent=True, cut_off_value=0.05, max_rgb=[
            0.76862745098, 0.63921568627, 0.69019607843], min_rgb=[0.76862745098, 0.63921568627, 0.69019607843])
        rea_png = gen_colorized_image(np.array(reachability_sdf > 0).astype('float'),  use_transparent=True, cut_off_value=0.05, max_rgb=[
            0.81176470588, 0.63921568627, 0.30196078431], min_rgb=[0.81176470588, 0.63921568627, 0.30196078431])
        rea_png[:, :, 3] = (rea_png[:, :, 3].astype(
            'float') * 0.5).astype('uint8')
        ant_png = gen_colorized_image(-opp_pic, use_transparent=True,
                                      cut_off_value=0.05, max_rgb=[0.85490196078, 0.85490196078, 0.85490196078], min_rgb=[0.85490196078, 0.85490196078, 0.85490196078])
        write_colorized_image(app_pic, 'app.png',  use_transparent=True, cut_off_value=0.05, max_rgb=[
            0.76862745098, 0.63921568627, 0.69019607843], min_rgb=[0.76862745098, 0.63921568627, 0.69019607843])
        write_colorized_image(np.array(object_sdf > 0).astype('float'), 'obj.png',  use_transparent=True, cut_off_value=0.0, max_rgb=[
            0.81176470588, 0.63921568627, 0.30196078431], min_rgb=[0.81176470588, 0.63921568627, 0.30196078431])
        write_colorized_image(np.array(reachability_sdf > 0).astype('float'), 'rea_sdf.png',  use_transparent=True, cut_off_value=0.0, max_rgb=[
            0.81176470588, 0.63921568627, 0.30196078431], min_rgb=[0.81176470588, 0.63921568627, 0.30196078431])
        write_colorized_image(-opp_pic, 'ant.png', use_transparent=True,
                              cut_off_value=0.05, max_rgb=[0.85490196078, 0.85490196078, 0.85490196078], min_rgb=[0.85490196078, 0.85490196078, 0.85490196078])
        coordinates = extent2Coordinates(self.extent, self.h, self.w)

        if reachability_sdf is not None and type(reachability_sdf) is not int:
            planned_back_up = {}
            start_id = 's'

            # task lists
            travel_map = (reachability_sdf < 0).astype('float')
            extend = [coordinates[0][0], coordinates[-1]
                      [0], coordinates[0][1], coordinates[-1][1]]
            solver = PathSolver(travel_map)

            dist = 0
            # max_possible_dist = 0
            paths = []
            suc = []
            fai = []
            task_related_ids = []
            for task in self.tasks:
                for target in task:
                    target_id = target.item_id
                    task_related_ids += [start_id, target_id]
            task_related_ids = [id for id in task_related_ids if id != 's']
            task_related_ids = np.unique(task_related_ids)

            for task in self.tasks:

                start_node = self.getStartIndex()

                for target in task:

                    _d = None
                    target_id = target.item_id
                    if target_id == -1:
                        goal_node = self.getStartIndex()
                    else:
                        if (start_id, target_id) in planned_back_up.keys():
                            _, goal_node, _d = planned_back_up[(
                                start_id, target_id)]

                        elif (target_id, start_id) in planned_back_up.keys():
                            _, goal_node, _d = planned_back_up[(
                                target_id, start_id)]
                        else:
                            goal_node, goal_coord = self.findInteractPose(
                                reachability_sdf, target_id, coordinates)

                    if goal_node is None or start_node is None:
                        dist += self.getMaxPlanningCost(coordinates)
                    elif _d is not None:
                        dist += _d
                    else:
                        path = solver.astar(
                            tuple(start_node), tuple(goal_node))
                        if path is not None:
                            foundPath = list(path)
                            paths += foundPath
                            _d = get_dist_from_path(
                                foundPath, extend, self.w, self.h)
                        else:
                            _d = self.getMaxPlanningCost(coordinates)
                        dist += _d
                        planned_back_up[(start_id, target_id)] = (
                            start_node, goal_node, _d)

                    start_node = goal_node

        for id in task_related_ids:
            goal_node, goal_coord = self.findInteractPose(
                reachability_sdf, id, coordinates)
            if goal_node is None:
                fai.append(id)
            else:
                suc.append(id)

        normalizedImg = np.zeros((rea_png.shape[0], rea_png.shape[1]), 'uint8')
        if len(paths) > 0:
            paths = np.array(paths)
            normalizedImg[paths[:, 0], paths[:, 1]] = 255

        path_png = dilation(normalizedImg, get_circular_kernel(2))
        # path_png = gen_colorized_image(np.array(path_png > 0).astype('float'),  use_transparent=True, cut_off_value=0.0, max_rgb=[
        #     0.95294117647, 0.9294117647, 0.87058823529], min_rgb=[0.95294117647, 0.9294117647, 0.87058823529])
        path_png = gen_colorized_image(np.array(path_png > 0).astype('float'),  use_transparent=True, cut_off_value=0.0, max_rgb=[
            0.0, 1.0, 1.0], min_rgb=[0.0, 1.0, 1.0])
        return app_png,  rea_png, ant_png, path_png, suc, fai

    def evaluateReachability(self):
        object_sdf_pad, _, _, app_pic_pad, opp_pic_pad = self.discritizeSDFfromBuffer(
            list(self.furniture_info.nodes), use_approach=True, use_oppose=True)

        app_pic = getCenteredSDF(app_pic_pad, self.w, self.h)
        opp_pic = getCenteredSDF(opp_pic_pad, self.w, self.h)
        object_sdf = getCenteredSDF(object_sdf_pad, self.w, self.h)

        reachability_sdf = self.convertSDF2Reachability(object_sdf)

        total_reach = np.sum((reachability_sdf > 0).astype('float'))

        total_aff = 0
        max_aff = 0
        for id in self.interact:
            temp_app = getCenteredSDF(self.interact[id], self.w, self.h)
            total_aff += np.sum((reachability_sdf > 0).astype('float') * (
                temp_app > hinge_loss(self.robot['reach'], self.robot['reach']).astype('float')))
            temp_max_app = self.interact_buffer[id]['pic']
            max_aff += np.sum(temp_max_app > hinge_loss(
                self.robot['reach'], self.robot['reach']).astype('float'))

        coordinates = extent2Coordinates(self.extent, self.h, self.w)

        if reachability_sdf is not None and type(reachability_sdf) is not int:
            planned_back_up = {}
            start_id = 's'

            # task lists
            travel_map = (reachability_sdf < 0).astype('float')
            extend = [coordinates[0][0], coordinates[-1]
                      [0], coordinates[0][1], coordinates[-1][1]]
            solver = PathSolver(travel_map)

            dist = 0
            # max_possible_dist = 0
            suc = []
            fai = []
            task_related_ids = []
            for task in self.tasks:
                for target in task:
                    target_id = target.item_id
                    task_related_ids += [start_id, target_id]
            task_related_ids = [id for id in task_related_ids if id != 's']
            task_related_ids = np.unique(task_related_ids)

            for task in self.tasks:
                has_failed = False
                start_node = self.getStartIndex()

                for target in task:

                    _d = None
                    target_id = target.item_id

                    if (start_id, target_id) in planned_back_up.keys():
                        _, goal_node, _d = planned_back_up[(
                            start_id, target_id)]

                    elif (target_id, start_id) in planned_back_up.keys():
                        _, goal_node, _d = planned_back_up[(
                            target_id, start_id)]
                    else:
                        goal_node, goal_coord = self.findInteractPose(
                            reachability_sdf, target_id, coordinates)

                    if goal_node is None or start_node is None:
                        dist += self.getMaxPlanningCost(coordinates)
                        has_failed = True
                    elif _d is not None:
                        dist += _d
                    else:
                        path = solver.astar(
                            tuple(start_node), tuple(goal_node))
                        if path is not None:
                            foundPath = list(path)
                            _d = get_dist_from_path(
                                foundPath, extend, self.w, self.h)
                        else:
                            _d = self.getMaxPlanningCost(coordinates)
                            has_failed = True
                        dist += _d
                        planned_back_up[(start_id, target_id)] = (
                            start_node, goal_node, _d)

                    start_node = goal_node

                # if not has_failed:
                #     suc += 1

        for id in task_related_ids:
            goal_node, goal_coord = self.findInteractPose(
                reachability_sdf, id, coordinates)
            if goal_node is None:
                fai.append(id)
            else:
                suc.append(id)

        area_size = np.array(
            self.seg_map['max']) - np.array(self.seg_map['min'])
        total_area = area_size[0] * area_size[2]
        return total_reach*self.desired_sdf_res*self.desired_sdf_res,  total_aff*self.desired_sdf_res*self.desired_sdf_res,  total_area, max_aff*self.desired_sdf_res*self.desired_sdf_res, len(suc), len(suc + fai)

    def findInteractPose(self, reachability_sdf, id, coords):
        if id not in self.interact.keys():
            return None, None

        weight = getCenteredSDF(self.interact[id], self.w, self.h)

        # weight = weight * (weight > np.exp(- self.robot['reach']))
        weight = weight * \
            (weight > hinge_loss(self.robot['reach'], self.robot['reach']))
        approachable_pix = np.where(
            (weight.flatten() * reachability_sdf.flatten()) > 0.0)
        approachable_coord = coords[approachable_pix]
        approachable_coord_diff = approachable_coord - \
            np.array(self.furniture_info.nodes()[id]['pose'][0])
        if len(approachable_coord_diff) == 0:
            return None, None
        approachable_idx = np.argmin(
            approachable_coord_diff[:, 0] ** 2 + approachable_coord_diff[:, 2]**2)
        # np.sum(approachable_coord_diff ** 2, axis=1))

        img_pose_idx = (int(approachable_pix[0][approachable_idx] / reachability_sdf.shape[1]),
                        approachable_pix[0][approachable_idx] % reachability_sdf.shape[1])

        if img_pose_idx[0] > reachability_sdf.shape[0] or img_pose_idx[1] > reachability_sdf.shape[1]:
            print('star planning wrong')
        return img_pose_idx, approachable_coord[approachable_idx]

    def getMaxPlanningCost(self, coordinates):
        x_extend = abs(coordinates[0][0] - coordinates[-1][0])
        y_extend = abs(coordinates[0][1] - coordinates[-1][1])
        return 2 * (x_extend + y_extend)

    def PlanningStarCost(self, sdf, coordinates, w, h):

        planned_back_up = {}
        start_id = 's'

        # task lists
        travel_map = (sdf < 0).astype('float')
        extend = [coordinates[0][0], coordinates[-1]
                  [0], coordinates[0][1], coordinates[-1][1]]
        solver = PathSolver(travel_map)

        dist = []
        max_possible_dist = 0
        paths = []
        for task in self.tasks:

            start_node = self.getStartIndex()

            for target in task:

                _d = None
                target_id = target.item_id

                if target_id not in self.plot_ids:
                    continue

                if (start_id, target_id) in planned_back_up.keys():
                    _, goal_node, _d = planned_back_up[(
                        start_id, target_id)]

                elif (target_id, start_id) in planned_back_up.keys():
                    _, goal_node, _d = planned_back_up[(
                        target_id, start_id)]
                else:
                    goal_node, goal_coord = self.findInteractPose(
                        sdf, target_id, coordinates)

                if goal_node is None or start_node is None:
                    dist += [self.getMaxPlanningCost(coordinates)]
                elif _d is not None:
                    dist += [_d]
                else:
                    path = solver.astar(tuple(start_node), tuple(goal_node))
                    if path is not None:
                        foundPath = list(path)

                        # debug
                        fpnp = np.array(foundPath)
                        if any(fpnp[:, 0] > sdf.shape[0]) or any(fpnp[:, 1] > sdf.shape[1]):
                            print('star planning exceeded boundary')
                        paths += foundPath
                        _d = get_dist_from_path(foundPath, extend, w, h)
                    else:
                        _d = self.getMaxPlanningCost(coordinates)
                    dist += [_d]

                    planned_back_up[(start_id, target_id)] = (
                        start_node, goal_node, _d)

                start_node = goal_node
                max_possible_dist += self.getMaxPlanningCost(coordinates)

        #       task_type: str
        #       pose: np.array
        #       item_id: int
        # path_dist, paths, max_total

        return dist, np.array(paths), max_possible_dist

    def evaluatePairwiseRelation(self):
        graph = self.furniture_info.copy()

        ids = list(self.furniture_info.nodes)

        # ids = list(self.interact.keys())

        pairs = list(combinations(list(self.furniture_info.nodes), 2))

        # for id1,id2  in pairs:
        for id1 in ids:
            for id2 in ids:
                if id1 == id2:
                    continue

                # distance betwwen two coordinates
                NEXTTO = self.getNexttoCost(idx_n=id2, idx_p=id1)
                # distance betwwen two bounding box
                CLOSETO = self.getClosetoCost(idx_n=id2, idx_p=id1)
                CLOSETOBOUND = self.getClosetoBoundCost(idx_n=id2, idx_p=id1)
                FRONTOF = self.getFrontCost(idx_n=id2, idx_p=id1)  # on +z axis
                # SIDEOF = self.getSideofCost(idx_n=id2, idx_p=id1, val=FRONTOF)  # on +- x axis
                # BACKOF = self.getBackofCost(idx_n=id2, idx_p=id1, val=FRONTOF)  # on -z axis
                FACING = self.getFacingCost(
                    idx_n=id2, idx_p=id1)  # id1 is facing id2
                FACINGBOUND = self.getFacingBoundCost(idx_n=id2, idx_p=id1)
                # AGAINST = self.getAgainstCost(idx_n=id2, idx_p=id1, val=FACING)
                # AWAY = self.getAwayCost(idx_n=id2, idx_p=id1, val=FACING)
                # id1 facing same direction as id2
                PARALLEL = self.getParallelCost(idx_n=id2, idx_p=id1)
                # OPPOSE = self.getOpposeCost(idx_n=id2, idx_p=id1, val=PARALLEL)
                # ORTHOGONAL = self.getOrthogonalCost(idx_n=id2, idx_p=id1, val=PARALLEL) # id1 facing opposite direction as id2

                # SHARED = self.getSharedCost(idx_n=id2, idx_p=id1) # id1 and id2 interaction overlay
                # SHARED_MAX = np.sum(
                # self.interact[id1]) if SHARED is not None else 0

                # distance
                graph.add_edge(id1, id2, NEXTTO=NEXTTO, CLOSETO=CLOSETO,
                               FRONTOF=FRONTOF, FACING=FACING, PARALLEL=PARALLEL,
                               CLOSETOBOUND=CLOSETOBOUND, FACINGBOUND=FACINGBOUND)

        return graph

    def getNexttoCost(self, idx_n, idx_p):
        pose_p = self.furniture_info.nodes()[idx_p]['pose']
        pose_n = self.furniture_info.nodes()[idx_n]['pose']
        diffs = math.sqrt((pose_p[0][0]-pose_n[0][0]) ** 2 +
                          (pose_p[0][2] - pose_n[0][2])**2)
        return diffs

    def getClosetoCost(self, idx_n, idx_p):
        cp = p.getClosestPoints(
            self.furniture_info.nodes()[idx_n]['bullet'], self.furniture_info.nodes()[idx_p]['bullet'], LARGE_OFFSET)
        for c in cp:
            return c[8]

    def getClosetoBoundCost(self, idx_n, idx_p):
        cp = p.getClosestPoints(
            self.furniture_info.nodes()[idx_n]['bullet'], self.furniture_info.nodes()[idx_p]['bullet_center'], LARGE_OFFSET)
        for c in cp:
            return c[8]

    def isInterrupted(self, idx_n, idx_p):
        pose_p = self.furniture_info.nodes()[idx_p]['pose']
        pose_n = self.furniture_info.nodes()[idx_n]['pose']
        res = p.rayTest(pose_p[0], pose_n[0])
        # id1 = res[0]
        # print(res)
        if res[0][0] == self.furniture_info.nodes()[idx_n]['bullet']:
            return False
        return True

    def get_relative_transform(self, idx_n, idx_p):
        pose_p = self.furniture_info.nodes()[idx_p]['pose']
        pose_n = self.furniture_info.nodes()[idx_n]['pose']

        trans_p = get_transform_from_xyz_rpy(
            pose_p[0], pose_p[1])

        trans_n = get_transform_from_xyz_rpy(
            pose_n[0], pose_n[1])
        return np.linalg.pinv(trans_p) @ trans_n

    def get_relative_transformFront(self, idx_n, idx_p, trans_p_n=None):
        if trans_p_n is None:
            trans_p_n = self. get_relative_transform(idx_n, idx_p)

        fa = math.atan2(self.furniture_info.nodes()[
                        idx_p]['front'][0], self.furniture_info.nodes()[idx_p]['front'][1])

        trans_f = get_transform_from_xyz_rpy(
            [0, 0, 0], [0, fa, 0])

        return np.linalg.pinv(trans_f) @ trans_p_n

    def getBoundingBox2Front(self, idx_n, idx_p, trans_f_n=None):
        if trans_f_n is None:
            trans_f_n = self. get_relative_transformFront(idx_n, idx_p)

        m_max = self.furniture_info.nodes(
        )[idx_n]['center'] + self.furniture_info.nodes()[idx_n]['bb'] / 2
        m_min = self.furniture_info.nodes(
        )[idx_n]['center'] - self.furniture_info.nodes()[idx_n]['bb'] / 2

        corners = np.array([[m_max[0], 0, m_max[2], 1], [m_min[0], 0, m_max[2], 1], [
                           m_max[0], 0, m_min[2], 1], [m_min[0], 0, m_min[2], 1]]).T

        return trans_f_n @ corners

    def getFacingBoundCost(self, idx_n, idx_p):

        trans_p_n = self.get_relative_transform(idx_n, idx_p)
        trans_f_n = self.get_relative_transformFront(idx_n, idx_p, trans_p_n)
        box = self.getBoundingBox2Front(idx_n, idx_p, trans_f_n)

        angs = np.arctan2(box[0, :], box[2, :])

        if np.all(np.abs(angs) < np.pi/2.) and np.any(angs < 0) and np.any(angs > 0):
            return 0
        else:
            return np.min(np.abs(angs))

    def getFacingDirection(self, idx_):
        pose_ = self.furniture_info.nodes()[idx_]['pose']

        rot_ = R.from_euler('xyz', pose_[1]).as_matrix()
        front = np.array(self.furniture_info.nodes()[idx_]['front'])

        return rot_ @ front.T

    def getParallelCost(self, idx_n, idx_p):
        # diffs = self.get_relative_transform(idx_n, idx_p)
        front_n = self.getFacingDirection(idx_n)
        front_p = self.getFacingDirection(idx_p)

        front_p = np.array([front_p[0], front_p[2]])
        # front_p /= np.linalg.norm(front_p)
        front_n = np.array([front_n[0], front_n[2]])
        # front_n /= np.linalg.norm(front_n)
        diffs = np.dot(front_p, front_n)
        diffs = 1.0 if diffs > 1.0 else diffs
        diffs = -1.0 if diffs < -1.0 else diffs
        _a = math.acos(diffs)

        return abs(_a)

    def getOpposeCost(self, idx_n, idx_p, val=None):
        if val is not None:
            _a = val
        else:
            _a = self.getParallelCost(idx_n, idx_p)
        return np.pi - _a

    def getOrthogonalCost(self, idx_n, idx_p, val=None):
        if val is not None:
            _a = val
        else:
            _a = self.getParallelCost(idx_n, idx_p)
        _a = abs(_a - np.pi/2)

        return _a

    def getFacingCost(self, idx_n, idx_p):

        diffs = self.get_relative_transform(idx_n, idx_p)

        # _a = math.atan2(diffs[2, 3], diffs[0, 3])

        front_n = np.array([diffs[0, 3], diffs[2, 3]])
        front_n /= np.linalg.norm(front_n)
        front_p = np.array([self.furniture_info.nodes()[
            idx_p]['front'][0], self.furniture_info.nodes()[idx_p]['front'][2]])
        diffs = np.dot(front_p, front_n)
        diffs = 1.0 if diffs > 1.0 else diffs
        diffs = -1.0 if diffs < -1.0 else diffs
        _a = math.acos(diffs)

        return _a

    def getFrontCost(self, idx_n, idx_p):

        diffs = self.get_relative_transform(
            idx_p, idx_n)  # this one is reversed

        front_p = np.array([diffs[0, 3], diffs[2, 3]])
        front_p /= np.linalg.norm(front_p)
        front_n = np.array([self.furniture_info.nodes()[
            idx_n]['front'][0], self.furniture_info.nodes()[idx_n]['front'][2]])
        diffs = np.dot(front_p, front_n)
        diffs = 1.0 if diffs > 1.0 else diffs
        diffs = -1.0 if diffs < -1.0 else diffs
        _a = math.acos(diffs)

        # _d = abs(math.atan2(diffs[2, 3], diffs[0, 3]))

        return _a

    def toFurnitureFrame(self, furn_pose, pose):
        # sem = self.furniture_info.nodes()[id]
        # furn_pose = get_transform_from_xyz_rpy(sem['pose'][0], sem['pose'][1])
        furn_inv = np.linalg.pinv(furn_pose)
        # if len(pose.shape) > 1:
        #     res = []
        #     for pos in pose:
        #         Pa = pos
        #         res.append(furn_inv @ Pa)
        #     return res

        return np.transpose(furn_inv @ np.transpose(pose))

    def toWorldFrame(self, furn_pose, pose):
        return np.transpose(furn_pose @ np.transpose(pose))

    def toFurnitureApproach(self, id, sdf, coordinates, ax=[]):

        sem = self.furniture_info.nodes()[id]

        axis = ax
        if sem['front'].tolist() == [0, 0, -1]:
            axis = [(_a-2) % 4 for _a in ax]

        elif sem['front'].tolist() == [1, 0, 0]:
            axis = [(_a+1) % 4 for _a in ax]

        h, w = sdf.shape
        coord_angles = np.arctan2(coordinates[:, 0], coordinates[:, 2])

        # ori_p = [0, pose_p[1][1] + math.atan2(self.furniture_info.nodes(
        #     )[idx_p]['front'][0], self.furniture_info.nodes()[idx_p]['front'][2]), 0]

        first_cutoff = math.atan2(
            sem['bb'][0]/2., sem['bb'][2]/2.)
        second_cutoff = math.atan2(
            sem['bb'][0]/2., -sem['bb'][2]/2.)

        not_furn_mask = sdf.flatten() >= 0.
        front_of_furn_mask = np.zeros((len(coordinates)), dtype=bool)
        dist_from_furn = deepcopy(sdf.flatten())

        # + z
        if 0 in axis:
            front_of_furn_mask = np.logical_or(np.logical_and(
                coord_angles <= first_cutoff, coord_angles >= -first_cutoff), front_of_furn_mask)

        # + x
        if 1 in axis:
            front_of_furn_mask = np.logical_or(np.logical_and(
                coord_angles > first_cutoff, coord_angles < second_cutoff), front_of_furn_mask)

        # - z
        if 2 in axis:
            front_of_furn_mask = np.logical_or(np.logical_or(
                coord_angles >= second_cutoff, coord_angles <= -second_cutoff), front_of_furn_mask)
        # - x
        if 3 in axis:
            front_of_furn_mask = np.logical_or(np.logical_and(
                coord_angles > -second_cutoff, coord_angles < -first_cutoff), front_of_furn_mask)

        useable_mask = np.logical_and(front_of_furn_mask, not_furn_mask)
        dist_from_furn_mask_invert = np.invert(useable_mask)
        dist_from_furn[np.where(dist_from_furn_mask_invert)] = np.zeros(
            np.sum(dist_from_furn_mask_invert))

        # dist_from_furn = pesudo_distribution(
        #     dist_from_furn, self.robot['reach'])

        dist_from_furn = hinge_loss(
            dist_from_furn, self.robot['reach'])

        dist_from_furn *= useable_mask

        return np.reshape(dist_from_furn, [h, w])

    def toFurnitureFront(self, id, sdf, coordinates, ax=[]):
        h, w = sdf.shape
        res = np.zeros_like(sdf)

        sem = self.furniture_info.nodes()[id]
        axis = ax
        if sem['front'].tolist() == [0, 0, -1]:
            axis = [(_a-2) % 4 for _a in ax]

        elif sem['front'].tolist() == [1, 0, 0]:
            axis = [(_a+1) % 4 for _a in ax]

        for a in axis:
            # + z
            if a == 0:
                front_of_furn_mask = - \
                    self.furniture_info.nodes(
                    )[id]['bb'][0]/2.0 < coordinates[:, 0]
                front_of_furn_mask *= coordinates[:,
                                                  0] < self.furniture_info.nodes()[id]['bb'][0]/2.0
                dist_from_furn = coordinates[:, 2] - \
                    self.furniture_info.nodes()[id]['bb'][2]/2.0
                dist_from_furn_mask = dist_from_furn > 0

        # + x
            elif a == 1:
                front_of_furn_mask = - \
                    self.furniture_info.nodes(
                    )[id]['bb'][2]/2.0 < coordinates[:, 2]
                front_of_furn_mask *= coordinates[:,
                                                  2] < self.furniture_info.nodes()[id]['bb'][2]/2.0
                dist_from_furn = coordinates[:, 0] - \
                    self.furniture_info.nodes()[id]['bb'][0]/2.0
                dist_from_furn_mask = dist_from_furn > 0
        # - z
            elif a == 2:
                front_of_furn_mask = - \
                    self.furniture_info.nodes(
                    )[id]['bb'][0]/2.0 < coordinates[:, 0]
                front_of_furn_mask *= coordinates[:,
                                                  0] < self.furniture_info.nodes()[id]['bb'][0]/2.0
                dist_from_furn = - \
                    (coordinates[:, 2] +
                     self.furniture_info.nodes()[id]['bb'][2]/2.0)
                dist_from_furn_mask = dist_from_furn > 0

        # - x
            else:

                front_of_furn_mask = - \
                    self.furniture_info.nodes(
                    )[id]['bb'][2]/2.0 < coordinates[:, 2]
                front_of_furn_mask *= coordinates[:,
                                                  2] < self.furniture_info.nodes()[id]['bb'][2]/2.0
                dist_from_furn = - \
                    (coordinates[:, 0] +
                     self.furniture_info.nodes()[id]['bb'][0]/2.0)
                dist_from_furn_mask = dist_from_furn > 0

            dist_from_furn_mask *= front_of_furn_mask
            dist_from_furn_mask_invert = np.invert(dist_from_furn_mask)
            dist_from_furn[np.where(dist_from_furn_mask_invert)] = np.zeros(
                np.sum(dist_from_furn_mask_invert))

            # dist_from_furn = pesudo_distribution(
            #     dist_from_furn, self.robot['reach'], self.robot['reach'])

            dist_from_furn = hinge_loss(
                dist_from_furn, self.robot['reach'])

            dist_from_furn *= dist_from_furn_mask

            res += np.reshape(dist_from_furn, [h, w])
        return res

    def generateApprochabilityBuffer(self, id, coordinates, include_positive=True, include_negative=False):

        coords = np.zeros([coordinates.shape[0], 4])
        coords[:, :3] = coordinates[:, :3]
        coords[:, 3] = np.ones(coordinates.shape[0])

        simplified_class = self.type_map[self.furniture_info.nodes()[
            id]['cat']]

        use_function = self.toFurnitureApproach
        # use_function = self.toFurnitureFront
        if simplified_class == 'table':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    # id, self.sdf_buffer[id]['pic'], coordinates, ax=[1, 3])
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 2, 3])
                # id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 2])
            if include_negative:
                res = None

        elif simplified_class == 'coffee_table':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 2, 3])
            if include_negative:
                res = None

        elif simplified_class == 'decoration':
            if include_positive:
                res = None
            if include_negative:
                res = None

        elif simplified_class == 'omnitool':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 2, 3])
            if include_negative:
                res = None

        elif simplified_class == 'bed':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
            if include_negative:
                # use_function = self.toFurnitureFront
                res = - use_function(
                    # id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[2])

        elif simplified_class == 'chair':
            if include_positive:
                # res = use_function(
                # # id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
                # id, self.sdf_buffer[id]['pic'], coordinates, ax=[2])
                res = None
            if include_negative:
                res = None

        elif simplified_class == 'armchair':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    # id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0])

            if include_negative:
                res = None

        elif simplified_class == 'sofa':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    # id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0])

            if include_negative:
                use_function = self.toFurnitureFront
                res = - use_function(
                    # id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[2])

        elif simplified_class == 'tool':
            if include_positive:
                # res = use_function(
                #     id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 3])
                res = None
            if include_negative:
                res = None

        elif simplified_class == 'ottoman':
            if include_positive:
                use_function = self.toFurnitureApproach
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 2, 3])
            if include_negative:
                res = None

        elif simplified_class == 'lamp':
            if include_positive:
                use_function = self.toFurnitureApproach
                # res = use_function(
                #     id, self.sdf_buffer[id]['pic'], coordinates, ax=[0, 1, 2, 3])
                res = None
            if include_negative:
                res = None

        elif simplified_class == 'door':
            use_function = self.toFurnitureFront
            if include_positive:
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0])
            if include_negative:
                res = None

        elif simplified_class == 'cabinet':
            use_function = self.toFurnitureFront
            if include_positive:
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0])
            if include_negative:
                res = - \
                    use_function(
                        id, self.sdf_buffer[id]['pic'], coordinates, ax=[2])

        elif simplified_class == 'television':
            use_function = self.toFurnitureFront
            if include_positive:
                res = use_function(
                    id, self.sdf_buffer[id]['pic'], coordinates, ax=[0])
            if include_negative:
                res = - \
                    use_function(
                        id, self.sdf_buffer[id]['pic'], coordinates, ax=[2])

        else:
            print(
                f'unknown object with type {simplified_class}, nothing generated')
            res = None

        return res

    def createIndividualSDFBuffer(self, id):

        sem = deepcopy(self.furniture_info.nodes()[id])
        sem['pose'] = [[0, 0, 0], [0, 0, 0]]
        b = convert_node2object(sem)

        box = sem['bb'] + np.ones_like(sem['min'])*self.extension*2

        upper = [box[0]/2., box[2]/2.]
        lower = [-box[0]/2., -box[2]/2.]

        if self.furniture_info.nodes()[id]['cat'] in HIGHUP or self.furniture_info.nodes()[id]['cat'] in LOWERDOWN or self.furniture_info.nodes()[id]['cat'] == 'door':
            pic, extent, coordinates = None, None, None

        else:
            pic, extent, coordinates = self.discritizeSDF(
                w=int(box[2]/self.desired_sdf_res), h=int(box[0]/self.desired_sdf_res), b=b, upper=upper, lower=lower)

            self.sdf_buffer[id] = {'pic': pic,
                                   'extent': extent, 'coordinates': coordinates}

            self.createIndividualApprochabilityBuffer(id, extent, coordinates)
            self.createIndividualOppositeBuffer(id, extent, coordinates)
            self.interact_buffer[id]['best'] = self.getBestCaseApprochability(
                id)
            self.anti_buffer[id]['best'] = self.getBestCaseOpposition(id)

    def createIndividualApprochabilityBuffer(self, id, extent, coordinates):
        pic = self.generateApprochabilityBuffer(
            id, coordinates, include_positive=True, include_negative=False)

        self.interact_buffer[id] = {'pic': pic,
                                   'extent': extent, 'coordinates': coordinates}

    def createIndividualOppositeBuffer(self, id, extent, coordinates):
        pic = self.generateApprochabilityBuffer(
            id, coordinates,  include_positive=False, include_negative=True)

        self.anti_buffer[id] = {'pic': pic,
                                    'extent': extent, 'coordinates': coordinates}

    def getBestCaseApprochability(self, id):
        if self.interact_buffer[id]['pic'] is None:
            return 0.
        return np.sum(self.interact_buffer[id]['pic'] * self.sdf_buffer[id]['pic'])

    def getBestCaseOpposition(self, id):
        if self.anti_buffer[id]['pic'] is None:
            return 0.
        return np.sum(- self.anti_buffer[id]['pic'] * self.sdf_buffer[id]['pic'])

    def genSolidRoom(self):
        upper = self.seg_map['max']
        lower = self.seg_map['min']

        return genRoomSDF(upper, lower)

    def createWallSDFBuffer(self):
        # generate a box, then flip the signs
        b = self.genSolidRoom()

        upper = np.array([self.seg_map['max'][0], self.seg_map['max'][2]])
        lower = np.array([self.seg_map['min'][0], self.seg_map['min'][2]])

        y = (self.seg_map['max'][1] + self.seg_map['min'][1])/2.
        self.wall_buffer, extent, coordinates = self.discritizeSDF(
            w=self.w, h=self.h, b=b, upper=upper, lower=lower, y=y)

        self.extent = extent

        self.wall_buffer *= -1.0

        self.wall_buffer_enlarged, extended_extent, coordinates = self.discritizeSDF(
            w=self.w_extended, h=self.h_extended, b=b, upper=upper + self.padding, lower=lower-self.padding, y=y)

        self.extended_extent = extent

        self.wall_buffer_enlarged *= -1.0

    def createCompleteSDFBuffer(self):
        for id in self.plot_ids:
            self.createIndividualSDFBuffer(id)
        self.createWallSDFBuffer()

    def discritizeSDFfromBuffer(self, include_ids=[], use_approach=False, use_oppose=False):
        upper = self.seg_map['max']
        lower = self.seg_map['min']

        begin_point = (upper[0]+self.padding, lower[2]-self.padding)
        lower_point = (lower[0]-self.padding, upper[2]+self.padding)

        world_extent, world_coord = getRoomExtent(
            bounds=(begin_point, lower_point), h=self.h_extended, w=self.w_extended)

        img = deepcopy(self.wall_buffer_enlarged)
        approach = np.zeros_like(img)
        oppose = np.zeros_like(img)
        self.interact.clear()
        self.anti.clear()

        for id in include_ids:

            if self.furniture_info.nodes()[id]['cat'] in HIGHUP or self.furniture_info.nodes()[id]['cat'] in LOWERDOWN or self.furniture_info.nodes()[id]['cat'] == 'door':
                continue

            sdf = self.sdf_buffer[id]
            _extend = sdf['extent']
            _coord = sdf['coordinates']
            furn_sdf = sdf['pic']

            _approah = self.interact_buffer[id]['pic']
            _oppose = self.anti_buffer[id]['pic']

            coords = np.zeros([world_coord.shape[0], 4])
            coords[:, :3] = world_coord[:, :3]
            coords[:, 3] = np.ones(world_coord.shape[0])
            sem = self.furniture_info.nodes()[id]
            # ori_pose = [0, sem['pose'][1][1], 0]
            furn_pose = get_transform_from_xyz_rpy(
                sem['pose'][0], sem['pose'][1])

            coord_in_furn_frame = self.toFurnitureFrame(furn_pose, coords)
            valid_coords_in_furn_img, logic_filter = self.toFurnitureSDFBuffer(
                _extend, sdf['pic'].shape[1], sdf['pic'].shape[0], coord_in_furn_frame)
            # coord_in_world_frame = self.toWorldFrame(
            #     furn_pose=furn_pose, pose=coords)
            # valid_coords_in_world_img, logic_filter = self.toWorldSDFBuffer(
            #     world_extent, self.w_extended, self.h_extended, coord_in_world_frame)
            logic_idx = np.unravel_index(logic_filter, img.shape)

            img[logic_idx[0], logic_idx[1]] = np.minimum(
                img[logic_idx[0], logic_idx[1]], furn_sdf[valid_coords_in_furn_img[:, 0], valid_coords_in_furn_img[:, 1]])

            # img = np.minimum(
            #     furn_sdf[coord_in_furn_frame[:, 0], coord_in_furn_frame[:, 1]].reshape(self.h_extended, self.w_extended), img)

            if use_approach and _approah is not None:
                self.interact[id] = np.zeros_like(img)
                self.interact[id][logic_idx[0], logic_idx[1]
                                 ] = _approah[valid_coords_in_furn_img[:, 0], valid_coords_in_furn_img[:, 1]]
                approach += self.interact[id]
            if use_oppose and _oppose is not None:
                self.anti[id] = np.zeros_like(img)
                self.anti[id][logic_idx[0], logic_idx[1]
                                ] = _oppose[valid_coords_in_furn_img[:, 0], valid_coords_in_furn_img[:, 1]]
                oppose += self.anti[id]

        return img, world_extent, world_coord, approach, oppose

    def convertSDF2Reachability(self, sdf, return_none=False):
        pic_sdf = sdf - self.robot['size']
        mask = sdf > self.robot['size']
        mask_b = mask.astype(np.uint8) * 255
        labeled, num_seg = label(mask_b, return_num=True)

        start_idx = self.getStartIndex()
        if start_idx is None:
            return None
        init_pose_is_free = mask_b[start_idx[0], start_idx[1]]
        seg_id = labeled[start_idx[0], start_idx[1]]
        if init_pose_is_free:
            return self.convertReachability2SDF(labeled == seg_id, pic_sdf, self.extent, w=self.w, h=self.h)
        else:
            if return_none:
                return None
            mask = np.zeros_like(labeled)
            mask[start_idx[0], start_idx[1]] = 1
            mask = mask > 0
            return self.convertReachability2SDF(mask, pic_sdf, self.extent, w=self.w, h=self.h)

    @staticmethod
    def findDoorSem(furn_info):
        for id in furn_info.nodes():
            if 'cat' in furn_info.nodes()[id] and furn_info.nodes()[id]['cat'] == 'door':
                return id, furn_info.nodes()[id]
        return None, None

    @staticmethod
    def convertPose2Index(pose, extent, res):
        res_a = np.array([res, res])
        idxes = np.array([extent[0]-pose[0], pose[2] - extent[2]]) / res_a
        return idxes.round().astype('int')

    def getStartPose(self):
        _, door_sem = SceneManager.findDoorSem(self.furniture_info)
        if door_sem is None:
            return None
        door_trans = get_transform_from_xyz_rpy(
            door_sem['pose'][0],  door_sem['pose'][1])

        relative_pose = np.array(
            door_sem['front']) * (self.desired_sdf_res + self.robot['size'] + door_sem['bb'][2]/2.)
        # door_sem['front']) * door_sem['bb'][2]/2.
        relative_pose = np.append(relative_pose, 1.)
        door_pose = door_trans @ relative_pose

        ma = np.array(self.seg_map['max'])
        mi = np.array(self.seg_map['min'])
        if np.all(ma[[0, 2]] > door_pose[[0, 2]]) and np.all(mi[[0, 2]] < door_pose[[0, 2]]):
            use_front = np.array(door_sem['front'])
        else:
            use_front = -np.array(door_sem['front'])

        relative_pose = use_front * door_sem['bb'][2]/2.
        relative_pose = np.append(relative_pose, 1.)
        door_pose = door_trans @ relative_pose

        d1 = np.abs(self.seg_map['max'] - door_pose[:3])
        d2 = np.abs(self.seg_map['min'] - door_pose[:3])
        d3 = [d1[0], d1[2], d2[0], d2[2]]
        min_d = np.min([d1[0], d1[2], d2[0], d2[2]])
        min_index = d3.index(min_d)
        if min_index == 0:
            door_trans[0, 3] = self.seg_map['max'][0]
        elif min_index == 1:
            door_trans[2, 3] = self.seg_map['max'][2]
        elif min_index == 2:
            door_trans[0, 3] = self.seg_map['min'][0]
        else:
            door_trans[2, 3] = self.seg_map['min'][2]

        relative_pose = use_front * \
            (self.desired_sdf_res*2 + self.robot['size'])

        relative_pose = np.append(relative_pose, 1.)
        door_pose = door_trans @ relative_pose

        start_pose = door_pose[:3]
        return start_pose

    def getStartIndex(self):
        # start_pose = self.getStartPose()
        start_pose = self.pose_init
        if start_pose is None:
            return None
        return SceneManager.convertPose2Index(start_pose, self.extent, self.desired_sdf_res)

    def toFurnitureSDFBuffer(self, img_extent, w, h, coord_in_furn_frame):
        max_sdf_coord = np.array([img_extent[0], img_extent[2]])
        coords_in_furn_img = np.around(
            (max_sdf_coord - coord_in_furn_frame[:, [0, 2]]) / get_img_res(img_extent, w, h))
        coords_in_furn_img = coords_in_furn_img.astype('int')
        logic_filter = np.where(np.logical_and(np.logical_and(coords_in_furn_img[:, 0] < h, coords_in_furn_img[:, 0] > 0), np.logical_and(
            coords_in_furn_img[:, 1] < w, coords_in_furn_img[:, 1] > 0)))
        valid_coords_in_furn_img = coords_in_furn_img[logic_filter]

        return valid_coords_in_furn_img, logic_filter

    def toWorldSDFBuffer(self, world_extent, w, h, coord_in_world_frame):
        max_sdf_coord = np.array([world_extent[0], world_extent[2]])
        coords_in_world_img = np.ceil(
            (max_sdf_coord - coord_in_world_frame[:, [0, 2]]) / get_img_res(world_extent, w, h))
        coords_in_world_img = coords_in_world_img.astype('int')

        logic_filter = np.where(np.logical_and(np.logical_and(coords_in_world_img[:, 0] < h, coords_in_world_img[:, 0] > 0), np.logical_and(
            coords_in_world_img[:, 1] < w, coords_in_world_img[:, 1] > 0)))
        valid_coords_in_world_img = coords_in_world_img[logic_filter]

        return valid_coords_in_world_img, logic_filter
