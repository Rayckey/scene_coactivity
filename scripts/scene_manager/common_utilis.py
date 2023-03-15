#!/usr/bin/env python3
import pickle
import numpy as np
from PIL import Image, ImageDraw

from transforms3d import quaternions
from scipy.spatial.transform import Rotation as R

from copy import deepcopy
import scipy
import yaml
from packaging import version
import networkx as nx
from typing import NamedTuple
from scene_manager.bullet.bullet_utilis import *
from scene_manager.sdf_utilis import *
from skimage.morphology import erosion, dilation, closing, opening, area_closing, area_opening
from skimage.measure import label
from scene_manager.astar.astar import *

scipy_version = version.parse(scipy.version.version)


with open('./configs/exceptions.yaml', 'r') as file:
    exceptions = yaml.safe_load(file)
    HIGHUP = exceptions['HIGHUP']
    IGNORED = exceptions['IGNORED']
    # BROKEN = exceptions['BROKEN']
    BROKEN = []
    ISOLATED = exceptions['ISOLATED']

with open('./configs/weights.yaml', 'r') as file:
    weights = yaml.safe_load(file)
    CONSTANTS = weights['CONSTANTS']
    CLEARANCE = weights['CONSTANTS']['CLEARANCE']
    SMALL_OFFSET = weights['CONSTANTS']['SMALL_OFFSET']
    LARGE_OFFSET = weights['CONSTANTS']['LARGE_OFFSET']
    WEIGHTS = weights['WEIGHTS']
    MAXIMUN_RETRIES = weights['CONSTANTS']['MAXIMUN_RETRIES']

with open('./configs/robots.yaml', 'r') as file:
    ROBOTS = yaml.safe_load(file)


def quat_to_xyzw(orn, seq):
    """Convert quaternion from arbitrary sequence to XYZW (pybullet convention)."""
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = [seq.index(axis) for axis in 'xyzw']
    return orn[inds]


def quat_from_xyzw(xyzw, seq):
    """Convert quaternion from XYZW (pybullet convention) to arbitrary sequence."""
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = ['xyzw'.index(axis) for axis in seq]
    return xyzw[inds]


def quat_xyzw_from_rot_mat(rot_mat):
    """Convert quaternion from rotation matrix"""
    quatWXYZ = quaternions.mat2quat(rot_mat)
    quatXYZW = quat_to_xyzw(quatWXYZ, 'wxyz')
    return quatXYZW


def get_transform_from_xyz_rpy(xyz, rpy):
    """
    Returns a homogeneous transformation matrix (numpy array 4x4)
    for the given translation and rotation in roll,pitch,yaw
    xyz = Array of the translation
    rpy = Array with roll, pitch, yaw rotations
    """
    if scipy_version >= version.parse("1.4"):
        rotation = R.from_euler('xyz', [rpy[0], rpy[1], rpy[2]]).as_matrix()
    else:
        rotation = R.from_euler('xyz', [rpy[0], rpy[1], rpy[2]]).as_dcm()
    transformation = np.eye(4)
    transformation[0:3, 0:3] = rotation
    transformation[0:3, 3] = xyz
    return transformation


def get_xyz_rpy_from_transform(transformation):

    rotation = R.from_matrix(transformation[0:3, 0:3]).as_euler('xyz')

    xyz = transformation[0:3, 3]
    return [xyz, rotation]


def get_possible_group_from_central(cat, functional_setup={}):
    # groups = []
    for id, sem in functional_setup.items():
        for group in sem:
            if group['central'] == cat:
                return deepcopy(group)
                # groups.append(group)
    # return groups


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = max(0, 1.0*(1 - ratio))
    r = max(0, 1.0*(ratio - 1))
    g = 1.0 - b - r
    return np.array([r, g, b])


def assign_color(groups):
    for idx in range(len(groups)):
        groups[idx]['color'] = rgb(0, len(groups), idx)


def get_relative_transform(pose_a, pose_b):
    Ta = get_transform_from_xyz_rpy(pose_a[0], pose_a[1])
    Tb = get_transform_from_xyz_rpy(pose_b[0], pose_b[1])
    return np.linalg.pinv(Tb) @ Ta


def genereate_pairwise_relation(states, id_keys):
    pair_relation = {}
    for idx in range(len(id_keys)):
        state = states[idx*3:(idx*3+3)]
        pair_relation[id_keys[idx]] = {}
        pair_relation[id_keys[idx]]['state'] = state
        for jdx in range(len(id_keys)):
            if jdx == idx:
                continue
            state = states[jdx*3:(jdx*3+3)]
            diff_state = pair_relation[id_keys[idx]]['state'] - state
            dst = np.sqrt(np.sum(diff_state[:2]**2))
            pair_relation[id_keys[idx]][id_keys[jdx]] = np.array(
                [dst, np.abs(diff_state[2])])
    return pair_relation


# class SceneInformation(NamedTuple):
#     furniture_info: dict
#     robot: dict
#     seg_map: dict
#     groups: list
#     poses: dict

class TaskInfo(NamedTuple):
    task_type: str
    pose: np.array
    item_id: int


class TaskConfig(NamedTuple):
    tasks: list
    pose_init: np.array


class SceneInformation(NamedTuple):
    furniture_info: nx.DiGraph
    robot: dict
    seg_map: dict
    grouping_order: dict
    task_config: TaskConfig


class PathSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a accessible position"""

    def __init__(self, map: np.array):
        self.map = map
        self.height, self.width = map.shape
        # super.__init__(data= map)

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """same as the direct distance"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        return[(nx, ny) for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y), (x-1, y - 1), (x+1, y + 1), (x - 1, y+1), (x + 1, y-1)]if 0 <= ny < self.width and 0 <= nx < self.height and self.map[nx][ny] == 0]


def render_scene():
    while (1):
        camData = p.getDebugVisualizerCamera()
        viewMat = camData[2]
        projMat = camData[3]
        p.getCameraImage(256,
                         256,
                         viewMatrix=viewMat,
                         projectionMatrix=projMat,
                         renderer=p.ER_BULLET_HARDWARE_OPENGL)
        time.sleep(0.01)


def render_scene_once():
    camData = p.getDebugVisualizerCamera()
    viewMat = camData[2]
    projMat = camData[3]
    p.getCameraImage(256,
                     256,
                     viewMatrix=viewMat,
                     projectionMatrix=projMat,
                     renderer=p.ER_BULLET_HARDWARE_OPENGL)


def pos2SDF(pos, extend, w, h):
    # loc_x = pos
    res = get_img_res(extend, w, h)
    idxes = (np.array([extend[0], extend[2]]) - pos) / res
    return idxes.round().astype('int')


def get_img_res(extent, w, h):
    return (np.array([extent[0], extent[2]]) -
            np.array([extent[1], extent[3]])) / np.array([h, w])


def get_dist_from_path(path, extend, w, h):
    # NO_PATH_PENALTY = 100

    # if len(path) < 1:
    #     return NO_PATH_PENALTY

    res = get_img_res(extend, w, h)

    _path = np.array(path)
    _path = np.diff(_path, axis=0)
    dist = _path * res
    dist = np.sum(np.sqrt(np.sum(dist**2, axis=1)))
    return dist


def val2heat(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = np.maximum(0, 1.0*(1 - ratio))
    r = np.maximum(0, 1.0*(ratio - 1))
    g = 1.0 - b - r
    return r, g, b


def val2rgb(minimum, maximum, value, max_rgb, min_rgb):
    if maximum - minimum == 0:
        return np.zeros_like(value), np.zeros_like(value), np.zeros_like(value)
    ratio = (value-minimum) / (maximum - minimum)

    r = (max_rgb[0]-min_rgb[0]) * ratio + min_rgb[0]
    g = (max_rgb[1]-min_rgb[1]) * ratio + min_rgb[1]
    b = (max_rgb[2]-min_rgb[2]) * ratio + min_rgb[2]
    return r, g, b


def val2transparent(minimum, maximum, value, maximal_trans_ratio, minimal_trans_ratio):
    # minimum, maximum = float(minimum), float(maximum)
    # ratio = (value-minimum) / (maximum - minimum)
    ratio = (value-minimum) / (maximum - minimum)
    ratio = ratio * (maximal_trans_ratio -
                     minimal_trans_ratio) + minimal_trans_ratio
    ratio = ratio * 255
    ratio[np.where(value == 0.)] = 0

    # ratio = np.ones_like(value) * (value > minimum) * 255
    return ratio


def get_circular_kernel(radius):
    ker = np.zeros((radius*2+1, radius*2+1), 'uint8')
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    ker[mask] = 1
    return ker


def write_dilated_image(img, name, radius, rgb=[1.0, 1.0, 1.0]):

    # r, g, b = val2rgb(np.min(img_o), np.max(img_o), img_o, max_rgb, min_rgb)
    # colored = np.array([r,g,b])
    if radius > 0:
        img_erosion = dilation(img, get_circular_kernel(radius))
    else:
        img_erosion = img

    colored = np.zeros((img.shape[0], img.shape[1], 4))
    occupied = np.where(img_erosion > 0)
    temp = np.zeros((img.shape[0], img.shape[1]))
    temp[occupied] = 255/2
    colored[:, :, 3] = temp

    temp[occupied] = rgb[2]*255
    colored[:, :, 0] = temp

    temp[occupied] = rgb[1]*255
    colored[:, :, 1] = temp

    temp[occupied] = rgb[0]*255
    colored[:, :, 2] = temp

    skimage.io.imsave(name, colored)

    return img_erosion > 0


def write_colorized_image(img, name, use_transparent=True, cut_off_value=-10.0, max_rgb=[1.0, 1.0, 1.0], min_rgb=[1.0, 1.0, 1.0]):
    return skimage.io.imsave(name, gen_colorized_image(img, use_transparent, cut_off_value, max_rgb, min_rgb))


def gen_colorized_image(img, use_transparent=True, cut_off_value=-10.0, max_rgb=[1.0, 1.0, 1.0], min_rgb=[1.0, 1.0, 1.0]):
    img_o = img * (img > cut_off_value)
    r, g, b = val2rgb(np.min(img_o), np.max(img_o), img_o, max_rgb, min_rgb)
    # colored = np.array([r,g,b])
    if use_transparent:
        colored = np.zeros((img.shape[0], img.shape[1], 4))
        t = val2transparent(np.min(img_o), np.max(img_o), img_o, 1.0, 0.1)
        colored[:, :, 3] = t
    else:
        colored = np.zeros((img.shape[0], img.shape[1], 3))
    colored[:, :, 0] = b*255
    colored[:, :, 1] = g*255
    colored[:, :, 2] = r*255
    # colored.astype('uint8')
    # img = Image.fromarray(colored, 'RGBA')
    # img.save(name, 'PNG')
    return colored.astype('uint8')


def write_normalized_image(img, name):
    normalizedImg = np.zeros_like(img)
    normalizedImg = (np.zeros_like(img) - np.min(img)) / \
        (np.min(img)+np.max(img))
    if np.isnan(np.sum(normalizedImg)):
        return False
    return skimage.io.imsave(name, normalizedImg)


def write_normalized_imagePath(img, path, name):
    normalizedImg = np.zeros_like(img)
    normalizedImg = (np.zeros_like(img) - np.min(img)) / \
        (np.min(img)+np.max(img))
    result = np.zeros((*normalizedImg.shape, 3), dtype=np.uint8)
    result[:, :, 0] = normalizedImg
    result[:, :, 1] = normalizedImg
    result[:, :, 2] = normalizedImg
    if len(path) > 1:
        result[path[:, 0], path[:, 1], 2] = 255
    return skimage.io.imsave(name, result.astype('uint8'))


def draw_ellipse(image, bounds, width=1, outline='white', antialias=4):
    """Improved ellipse drawing function, based on PIL.ImageDraw."""

    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results.
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')
    draw = ImageDraw.Draw(mask)

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.ellipse([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)


def overlayRGB(r, g, b, a, use_add=False):
    result = np.zeros((*r.shape, 4), dtype=np.uint8)
    result[:, :, 3] = a
    result[:, :, 0] = r
    result[:, :, 1] = g
    result[:, :, 2] = b
    if use_add:
        return result
    temp = np.where(g)
    rr = result[:, :, 0]
    gg = result[:, :, 1]
    rr[temp] = 0
    temp = np.where(b)
    rr[temp] = 0
    gg[temp] = 0
    result[:, :, 0] = rr
    result[:, :, 1] = gg
    return result


def overlay_images(img0, img1):
    if img0 is None:
        return img1
    img2 = deepcopy(img0)
    has_stuff = np.where(img1[:, :, 3] > 0)
    img2[has_stuff] = img1[has_stuff]
    return img2


def overlay_transparant_images(img0, img1):
    # Extract the RGB channels
    srcRGB = img0[..., :3]
    dstRGB = img1[..., :3]

    # Extract the alpha channels and normalise to range 0..1
    srcA = img0[..., 3]/255.0
    dstA = img1[..., 3]/255.0

    # Work out resultant alpha channel
    outA = srcA + dstA*(1-srcA)

    # Work out resultant RGB
    outRGB = (srcRGB*srcA[..., np.newaxis] + dstRGB*dstA[...,
              np.newaxis]*(1-srcA[..., np.newaxis])) / outA[..., np.newaxis]

    # Merge RGB and alpha (scaled back up to 0..255) back into single image
    outRGBA = np.dstack((outRGB, outA*255)).astype(np.uint8)

    return outRGBA


def write_image_path(img, paths, name):
    normalizedImg = np.zeros_like(img)
    result = np.zeros((*normalizedImg.shape, 4), dtype=np.uint8)
    result_red = np.zeros(normalizedImg.shape, dtype=np.uint8)
    result_grn = np.zeros(normalizedImg.shape, dtype=np.uint8)
    result_blu = np.zeros(normalizedImg.shape, dtype=np.uint8)

    if len(paths) > 0:
        result_grn[paths[:, 0], paths[:, 1]] = 255

    result_red = dilation(result_red, get_circular_kernel(1))
    result_grn = dilation(result_grn, get_circular_kernel(1))
    result_blu = dilation(result_blu, get_circular_kernel(1))

    result_alp = np.bitwise_or(result_red > 0, result_grn > 0)
    result_alp = np.bitwise_or(
        result_alp > 0, result_blu > 0).astype('uint8')*255

    result = overlayRGB(result_red, result_grn, result_blu,
                        result_alp, use_add=True)

    img = Image.fromarray(result, 'RGBA')
    img.save(name, 'PNG')


def is_involved(id, combined_ids):
    ids = combined_ids.split("+")
    if len(ids) > 1:
        return str(id) == ids[0] or str(id) == ids[1]
    else:
        return str(id) == combined_ids


def minimum(a):
    m = np.min(a)
    minpos = a.tolist().index(m)
    return m, minpos


def convert_node2object(obj_info, use_3d=True):
    cat = obj_info['cat']
    if cat in ['wall', 'floors', 'ceilings']:
        return None

    bb = obj_info['bb']
    b = box(bb)
    return b


def center_furniture(furniture_info):
    for id in list(furniture_info.nodes):
        sem = furniture_info.nodes()[id]
        if 'centered' not in sem or not sem['centered']:
            sem['center'] = sem['bb']/2. + np.array(sem['min'])
            trans = get_transform_from_xyz_rpy(sem['pose'][0], sem['pose'][1])
            pt = np.concatenate([sem['center'], [1]]).T
            new_pt = trans @ pt
            sem['pose'][0] = new_pt[:3].tolist()
            sem['centered'] = True


def offcenter_furniture(furniture_info):
    for id in list(furniture_info.nodes):
        sem = furniture_info.nodes()[id]
        if 'centered' in sem and sem['centered']:
            trans = get_transform_from_xyz_rpy(sem['pose'][0], sem['pose'][1])
            pt = np.concatenate([-1.0 * sem['center'], [1]]).T
            new_pt = trans @ pt
            sem['pose'][0] = new_pt[:3].tolist()
            sem['centered'] = False


def filter_higher_up_furniture(furniture_info):
    remove_id = []
    for id in list(furniture_info.nodes):
        sem = furniture_info.nodes()[id]
        if sem['cat'] in HIGHUP:
            remove_id.append(id)
    furniture_info.remove_nodes_from(remove_id)


def filter_ignored_furniture(furniture_info):
    remove_id = []
    for id in list(furniture_info.nodes):
        sem = furniture_info.nodes()[id]
        if sem['cat'] in IGNORED:
            remove_id.append(id)
    furniture_info.remove_nodes_from(remove_id)


def gen_bullet_center(obj_info):
    pose = obj_info['pose']
    b = genBulletBall(trans=pose)

    return b


def gen_bullet_obj(obj_info, cat, use_block=False, use_3d=True, color=None):

    bb = obj_info['bb']
    pose = obj_info['pose']
    opt = []

    if cat == 'cabinet':
        b = genBulletCabinet(size=bb, drawers=opt, thickness=0.01, bottom_thickness=None,
                             trans=pose, configuration=np.zeros(len(opt)), use_block=use_block, use_3d=use_3d, color=color)
    elif cat == 'table':
        b = genBulletTable(size=bb, thickness=0.1, trans=pose,
                           d_type='square', support_thickness=0.02, use_block=use_block, use_3d=use_3d, color=color)
    elif cat == 'chair':
        b = genBulletChair(size=bb, back_ratio=0.5, back_thickness=0.02,
                           seat_thickness=0.02, trans=pose, support_thickness=0.02, use_block=use_block, use_3d=use_3d, color=color)

    elif cat == 'sofa':
        b = genBulletChair(size=bb, back_ratio=0.7, back_thickness=0.1,
                           seat_thickness=0.2, trans=pose, support_thickness=0.1, use_block=use_block, use_3d=use_3d, color=color)

    elif cat == 'lamp':
        # b = genBulletLamp(size=bb, thickness=0.3, support_thickness=0.05,
        #                   trans=pose, use_block=use_block, use_3d=use_3d, color=color)
        b = genBulletBlock(size=bb, trans=pose,
                           use_block=use_block, use_3d=use_3d, color=color)
    elif cat == 'bed':
        b = genBulletChair(size=bb, back_ratio=0.6, back_thickness=0.03,
                           seat_thickness=bb[2]*0.15, trans=pose, support_thickness=0.1, use_block=use_block, use_3d=use_3d, color=color)
    elif cat == 'armchair':
        b = genBulletChair(size=bb, back_ratio=0.7, back_thickness=0.1,
                           seat_thickness=0.2, trans=pose, support_thickness=0.1, use_block=use_block, use_3d=use_3d, color=color)

    else:
        b = genBulletBlock(size=bb, trans=pose,
                           use_block=use_block, use_3d=use_3d, color=color)

    return b


def update_bullet_obj(bullet_id, pose):
    p.resetBasePositionAndOrientation(
        bullet_id, posObj=pose[0], ornObj=p.getQuaternionFromEuler(pose[1]))


def sample_pose_in_image(box):
    upper = np.array([box[1]+box[3], box[0]+box[2]])
    lower = np.array([box[1], box[0]])
    return lower + np.random.random(2) * (upper-lower)
    # ed = samplePose(np.array(random_entry[1]), use_rand=True)


def collision2energy(dis):
    if dis < 0:
        return CLEARANCE*0.5 - dis
    elif dis <= CLEARANCE:
        return 1./(2*CLEARANCE) * ((dis - CLEARANCE) ** 2)
    else:
        return 0


def penalty_second_order_cost(state):
    c = np.sum(state ** 2)
    return c


def pesudo_distribution(x, std):
    # return np.exp(( - (x-offset)**2 / std))
    return np.exp((- (x)**2 / std))


def hinge_loss(x, m):
    return np.maximum(0, - x/m + 1)

# def ratio_cost(score, high):
#     if high == 0:
#         return 0
#     return 1.0 - math.exp(-score/high)


def ratio_cost(score, high):
    if high == 0:
        return 0
    # return math.tanh(score/high)
    return 1.0/(1 + math.exp(-9.0 * (score/high - 0.5)))


def low_satuating_cost(score):
    return 1.0/(1 + math.exp(-CONSTANTS['low'] * score))


def lower_satuating_cost(score):
    return 1.0/(1 + math.exp(-CONSTANTS['lower'] * score))


def satuating_cost(score):
    return math.tanh(score)


def ratio2Energy(ratio):
    return -np.log(ratio + SMALL_OFFSET)


def sigmoid(x, order=1):
    return x ** order / (1 + x**order)


def linear(x, m=np.pi):
    return (m-x)/m


def hyperbolic_tangent(x):
    return np.tanh(x)


def find_root(G: nx.DiGraph, child):
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        print(f"found root: {child}")
        return child
    else:
        return find_root(G, parent[0])


def get_room_diagonal(seg_map):
    return np.linalg.norm(seg_map['size'][[0, 2]])


def in_broken(thing1, thing2):
    combo = (thing1, thing2), (thing2, thing1)
    if thing1 in ISOLATED or thing2 in ISOLATED:
        return True
    if combo[0] in BROKEN or combo[1] in BROKEN:
        return True
    return False
