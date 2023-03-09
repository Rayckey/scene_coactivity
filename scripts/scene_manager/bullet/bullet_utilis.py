import pybullet as p
import numpy as np
import time
import pybullet_data
import math
from scipy.spatial.transform import Rotation as R
import os

SUPER_SMALL_OFFSET = 1e-5


def genBulletTable(size=(1, 1, 1), thickness=0.05, trans=(), d_type="square", support_thickness=0.2, use_block=False, use_3d=False, color=None):

    _size = np.array(size) / 2.0

    if color is None:
        color = [0, 0, 0, 1]
    if use_block:
        if d_type == "square":
            Id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[
                                        _size[0], _size[1], _size[2]])
            # Id_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[
            #                            _size[0], _size[1], _size[2]], rgbaColor=color)

            linkPositions = [[_size[0]-thickness, 0. ,_size[2]-thickness], [_size[0]-thickness,   0, -_size[2]+thickness],
                             [-_size[0]+thickness, 0., _size[2]-thickness], [-_size[0]+thickness, 0, -_size[2]+thickness], [0, _size[1]-thickness, 0]]

            Id_v = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                            halfExtents=[[thickness, _size[1], thickness], [thickness, _size[1], thickness],
                                                         [thickness, _size[1], thickness], [thickness, _size[1], thickness], [_size[0], thickness, _size[2]]],
                                            rgbaColors=[color]*5,
                                            visualFramePositions=linkPositions)

        elif d_type == "round":
            Id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=_size[0]/2., height=size[2])

            Id_v = p.createVisualShape(
                p.GEOM_CYLINDER, radius=_size[0]/2., length=size[2], rgbaColor=color)

            # linkPositions = [[0, 0, -_size[2]+thickness],
            #                  [0, 0, 0], [0, 0, _size[2]-thickness]]

            # Id_v = p.createVisualShapeArray(shapeTypes=[p.GEOM_CYLINDER, p.GEOM_CYLINDER, p.GEOM_CYLINDER],
            #                                 radii =[_size[0]/2., thickness, _size[0]/2.], heights=[thickness, size[2], thickness],
            #                                 rgbaColors=[color]*3,
            #                                 visualFramePositions=linkPositions)

    else:

        return None

    pose = trans[0]
    if not use_3d:
        pose[2] = 0
    obj_id = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=Id,
                               baseVisualShapeIndex=Id_v,
                               basePosition=pose,
                               baseOrientation=p.getQuaternionFromEuler(trans[1]))

    return obj_id


def genBulletCabinet(size=(1, 1, 1), drawers=["d", "l", "r"], thickness=0.01, bottom_thickness=None, trans=(), configuration=(), use_block=False, use_3d=False, color=None):

    _size = np.array(size) / 2.0
    if color is None:
        color = [0, 0, 0, 1]

    if use_block:
        Id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[
                                    _size[0], _size[1], _size[2]])

        # Id_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[
        #     _size[0], _size[1], _size[2]], rgbaColor=color)

        linkPositions = [[0, 0, _size[2]+thickness/2.], [0, _size[1], 0 ],
                         [0,  0., -_size[2]-thickness/2.], [0, -_size[1], 0], [-_size[0], 0, 0]]

        Id_v = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                        halfExtents=[[_size[0]+thickness, _size[1]+thickness , thickness], [_size[0]+thickness, thickness , _size[2]+thickness],
                                                     [_size[0]+thickness, _size[1]+thickness , thickness], [_size[0]+thickness, thickness , _size[2]+thickness], [thickness, _size[1]+thickness, _size[2]+thickness]],
                                        rgbaColors=[color]*5,
                                        visualFramePositions=linkPositions)

    else:

        return 0

    pose = trans[0]
    if not use_3d:
        pose[2] = 0
    obj_id = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=Id,
                               baseVisualShapeIndex=Id_v,
                               basePosition=pose,
                               baseOrientation=p.getQuaternionFromEuler(trans[1]))

    return obj_id


def genBulletShelf(size=(1, 1, 1), num_block=1, thickness=0.01, trans=(), d_type="open", use_block=False, use_3d=False, color=None):

    _size = np.array(size) / 2.0

    if color is None:
        color = [0, 0, 0, 1]

    if use_block:
        # if d_type == "square":
        #     b = box(size)
        # elif d_type == "round":
        #     b = capped_cylinder(-Z * size[2], Z * size[2], size[0]/2.0)

        Id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[
                                    _size[0], _size[1], _size[2]])

        # Id_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[
        #     _size[0], _size[1], _size[2]], rgbaColor=color)

        linkPositions = [[0, _size[1]+thickness/2., 0.], [0, 0, _size[2]],
                         [0, -_size[1]-thickness/2., 0.], [0, 0, -_size[2]], [-_size[0], 0, 0]]

        Id_v = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                        halfExtents=[[_size[0]+thickness, thickness, _size[2]+thickness], [_size[0]+thickness, _size[1]+thickness, thickness],
                                                     [_size[0]+thickness, thickness, _size[2]+thickness], [_size[0]+thickness, _size[1]+thickness, thickness], [thickness, _size[1]+thickness, _size[2]+thickness]],
                                        rgbaColors=[color]*5,
                                        visualFramePositions=linkPositions)

    else:

        return -1

    pose = trans[0]
    if not use_3d:
        pose[2] = 0

    obj_id = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=Id,
                               baseVisualShapeIndex=Id_v,
                               basePosition=pose,
                               baseOrientation=p.getQuaternionFromEuler(trans[1]))

    return obj_id

def genBulletBall(trans=()):

    color = [0, 0, 0, 1]

    Id = p.createCollisionShape(p.GEOM_SPHERE, radius=SUPER_SMALL_OFFSET)
    Id_v = p.createVisualShape(p.GEOM_SPHERE, radius=SUPER_SMALL_OFFSET, rgbaColor=color)

    pose = trans[0]

    obj_id = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=Id,
                                baseVisualShapeIndex=Id_v,
                                basePosition=pose,
                                baseOrientation=p.getQuaternionFromEuler(trans[1]))

    return obj_id


def genBulletBlock(size=(1, 1, 1),  trans=(), use_block=False, use_3d=False, color=None):

    _size = np.array(size) / 2.0

    if color is None:
        color = [0, 0, 0, 1]

    if use_block:
        Id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[
                                    _size[0], _size[1], _size[2]])

        Id_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[
            _size[0], _size[1], _size[2]], rgbaColor=color)

        pose = trans[0]
        if not use_3d:
            pose[2] = 0

        obj_id = p.createMultiBody(baseMass=1,
                                   baseCollisionShapeIndex=Id,
                                   baseVisualShapeIndex=Id_v,
                                   basePosition=pose,
                                   baseOrientation=p.getQuaternionFromEuler(trans[1]))

        return obj_id
    else:

        return None


def genBulletChair(size=(1, 1, 1), d_type="square", back_ratio=0.5, back_thickness=0.05, seat_thickness=0.02, trans=(), support_thickness=0.02, use_block=False, use_3d=False, color=None):

    _size = np.array(size) / 2.0

    if color is None:
        color = [0, 0, 0, 1]
    if use_block:
        if d_type == "square":
            Id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[
                                        _size[0], _size[1], _size[2]])

            # Id_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[
            #                            _size[0], _size[1], _size[2]], rgbaColor=color)

            seat_height = (1.-back_ratio) * _size[2]
            leg_pos =  - (_size[2] - seat_height)

            linkPositions = [[_size[0]-support_thickness, leg_pos, _size[2]-support_thickness  ], 
                            [_size[0]-support_thickness,  leg_pos, -_size[2]+support_thickness ], 
                            [-_size[0]+support_thickness, leg_pos, _size[2]-support_thickness  ], 
                            [-_size[0]+support_thickness, leg_pos, -_size[2]+support_thickness ],
                            [0, seat_height* ( back_ratio- 0.5), 0], 
                            [0, _size[1]-seat_height, -_size[2]+back_thickness]]

            Id_v = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                            halfExtents=[[support_thickness, seat_height, support_thickness ], [support_thickness, seat_height, support_thickness],
                                                         [support_thickness, seat_height, support_thickness ], [support_thickness, seat_height, support_thickness], [_size[0], seat_thickness, _size[2]], [_size[0],  _size[1]-seat_height, back_thickness ]],
                                            rgbaColors=[color]*6,
                                            visualFramePositions=linkPositions)

        elif d_type == "round":
            Id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=_size[0]/2., height=size[2])

            # Id_v = p.createVisualShape(
            #     p.GEOM_CYLINDER, radius=_size[0]/2., length=size[2], rgbaColor=color)

            linkPositions = [[0, 0, -_size[2]+support_thickness],
                             [0, 0, 0], [0, 0, _size[2]-support_thickness]]

            Id_v = p.createVisualShapeArray(shapeTypes=[p.GEOM_CYLINDER, p.GEOM_CYLINDER, p.GEOM_CYLINDER],
                                            radius=[_size[0]/2., support_thickness, _size[0]/2.], height=[support_thickness, size[2], support_thickness],
                                            rgbaColors=[color]*3,
                                            visualFramePositions=linkPositions)

    else:

        return None

    pose = trans[0]
    if not use_3d:
        pose[2] = 0
    obj_id = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=Id,
                               baseVisualShapeIndex=Id_v,
                               basePosition=pose,
                               baseOrientation=p.getQuaternionFromEuler(trans[1]))

    return obj_id


def genBulletLamp(size=(1, 1, 1), thickness=0.01, trans=(), support_thickness=0.2, use_block=False, use_3d=False, color=None):
    return genBulletTable(size=size, thickness=thickness, trans=trans, d_type="round", support_thickness=support_thickness, use_block=use_block, use_3d=use_3d, color=color)


def genBulletBed(size=(1, 1, 1), thickness=0.3, trans=(), support_thickness=0.3, use_block=False, use_3d=False, color=None):
    return genBulletTable(size=size, thickness=thickness, trans=trans, d_type="square", support_thickness=support_thickness, use_block=use_block, use_3d=use_3d, color=color)


def genBulletRoom(bboxes, thickness=0.01):

    obj_id = []

    color = [1, 1, 1, 1]
    for bbox, tf in bboxes:

        box = np.array(bbox) / 2.0
        # b.append(p.createCollisionShape(
        #     p.GEOM_BOX, halfExtents=[box[0], thickness/2., box[2]]))
        # b.append(p.createCollisionShape(
        #     p.GEOM_BOX, halfExtents=[thickness/2., box[1], box[2]]))
        linkPositions = [[0, box[1]+thickness/2., 0], [box[0]+thickness/2., 0, 0],
                         [0, -box[1]-thickness/2., 0], [-box[0]-thickness/2., 0, 0]]

        collisionShapeId = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                                       halfExtents=[[box[0]+thickness, thickness/2., box[2]], [thickness/2., box[1]+thickness, box[2]],
                                                                    [box[0]+thickness, thickness/2., box[2]], [thickness/2., box[1]+thickness, box[2]]],
                                                       collisionFramePositions=linkPositions)

        visualShapeId = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX],
                                                 halfExtents=[[box[0]+thickness, thickness/2., box[2]], [thickness/2., box[1]+thickness, box[2]],
                                                              [box[0]+thickness, thickness/2., box[2]], [thickness/2., box[1]+thickness, box[2]]],
                                                 rgbaColors=[color]*4,
                                                 visualFramePositions=linkPositions)

        obj_id = p.createMultiBody(baseMass=1,
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=tf)
    return obj_id
