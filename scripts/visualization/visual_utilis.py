import os
import trimesh
import imageio
import pyvista as pv
import csv
import skimage
from skimage.transform import rescale
from scene_manager.gen_scene_utils import findDoorSem, room2networkx
from concept_net.graph_utilis import *
from sampling.sampling_utilis import AdaptiveAnnealingConfig, ResultHistory
from scipy.spatial.transform import Rotation
from scene_manager.scene_manager2 import *
from PIL import Image, ImageDraw

with open('./configs/dataset_paths.yaml', 'r') as file:
    dataset_paths = yaml.safe_load(file)
    MODEL_PATH = dataset_paths['MODEL_PATH']
    MODELS_FILE = dataset_paths['MODELS_FILE']


def generate_model_bb():
    model_bb_mapping = {}

    with open(MODELS_FILE, 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            if row[1] == 'front':
                continue
            model_bb_mapping[row[0]] = []
            model_bb_mapping[row[0]].append(
                list(map(float, row[1].split(','))))
            model_bb_mapping[row[0]].append(
                list(map(float, row[5].split(','))))
            model_bb_mapping[row[0]].append(
                list(map(float, row[3].split(','))))
            model_bb_mapping[row[0]].append(
                list(map(float, row[4].split(','))))

    return model_bb_mapping


def spawnRobot(door_sem, robot):
    tm = trimesh.load('robots_model/' + robot['name'] + '.obj', force='scene')

    door_trans = get_transform_from_xyz_rpy(
        door_sem['pose'][0], door_sem['pose'][1])
    relative_pose = np.array(
        [robot['size']/2 + door_sem['bb'][0]/2.+0.1, 0, 0, 1])
    door_pose = door_trans @ relative_pose
    start_coord = door_pose[:2]

    center = tm.centroid
    trans = np.eye(4)
    trans[:3, 3] = [start_coord[0], start_coord[1], 0]
    r = Rotation.from_euler('xyz', [0, 0, math.atan2(
        start_coord[1] - door_sem['pose'][0][1], start_coord[0]-door_sem['pose'][0][0])]).as_matrix()
    print(math.atan2(start_coord[1] - door_sem['pose']
          [0][1], start_coord[0]-door_sem['pose'][0][0]))
    trans[:3, :3] = r
    tm.apply_transform(trans)

    return tm


def size2Extent(box_size):
    return [-box_size[0]/2., box_size[0]/2., -box_size[1]/2.,
            box_size[1]/2., -box_size[2]/2., box_size[2]/2.]


def genWallWithTexture(Extents, linkPositions, rotaion, texture_direction=0):
    tex_len = 1.5
    m = genBoxMesh(Extents, linkPositions, rotaion)
    ss = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, tex_len)
    m = trimesh.Trimesh(vertices=ss[0], faces=ss[1])

    im = Image.open('textures/beige_wall_001_disp.png')
    kwargs = {'Kd': [1.0, 1.0, 1.0], 'Ka': [0.0, 0.0, 0.0], 'Ks': [
        0.4, 0.4, 0.4], 'Ke': ['0', '0', '0'], 'd': ['1'], 'Ns': 10.0, 'illum': ['2']}
    material = trimesh.visual.texture.SimpleMaterial(image=im, diffuse=[255, 255, 255, 255], ambient=[
                                                     0, 0, 0, 255], specular=[102, 102, 102, 255], glossiness=10.0)

    material.kwargs = kwargs

    uv = []
    if texture_direction == 0:
        for v in m.vertices:
            uv.append([(v[0]/tex_len), (v[1])/tex_len, (v[2])/tex_len])

    if texture_direction == 1:
        for v in m.vertices:
            uv.append([(v[2]/tex_len), (v[1])/tex_len, (v[0])/tex_len])

    tex = trimesh.visual.TextureVisuals(uv, image=im, material=material)

    mm = trimesh.Trimesh(vertices=m.vertices, faces=m.faces,
                         visual=tex, validate=True, process=False)
    # validate=True, process=False)

    mm.visual = tex

    return mm


def genWallWithoutTexture(extents):
    tex_len = 1.5
    box = pv.Box(bounds=size2Extent(extents))
    # .subdivide_adaptive(max_edge_len=tex_len)
    box.texture_map_to_plane(inplace=True)
    return box


def cropTexture(plane_size, image, texture_len=1.5):
    new_img_size = (plane_size/texture_len *
                    np.array((image.shape[0], image.shape[1]))).astype('int')

    looped_idx = np.indices((new_img_size[0], new_img_size[1]))
    looped_idx[0, :, :] = looped_idx[0, :, :] % image.shape[0]
    looped_idx[1, :, :] = looped_idx[1, :, :] % image.shape[1]
    looped_idx = looped_idx.reshape((2, new_img_size[0] * new_img_size[1]))

    return image[looped_idx[0, :], looped_idx[1, :], :].reshape((new_img_size[0], new_img_size[1], image.shape[2]))


def findDoorWall(bbox, center=[], door_sem={}):

    dist_to_2all = [door_sem['pose'][0][2] - (center[2]+bbox[2]/2), door_sem['pose'][0][0] - (center[0] +
                    bbox[0]/2), door_sem['pose'][0][2] - (center[2]+bbox[2]/2), door_sem['pose'][0][0] - (center[0]-bbox[0]/2)]
    dist_to_2all = [abs(ele) for ele in dist_to_2all]
    min_value = min(dist_to_2all)
    min_index = dist_to_2all.index(min_value)
    return min_index


def gen_room_meshes(bbox, center=[], thickness=0.01, mins=[], maxs=[],  door_sem={}, use_walls=False, floor_overlay=None):

    if use_walls:
        linkPositions = [[center[0], center[1], maxs[2]], [maxs[0], center[1], center[2]],
                         [center[0], center[1], mins[2]], [mins[0], center[1], center[2]]]

        linkRotations = [[0, 0, 0], [0,  np.pi/2, 0],
                         [0, 0, 0], [0,  np.pi/2, 0]]

        Extents = [[bbox[0], bbox[1], thickness],
                   [bbox[2], bbox[1], thickness]]
        meshes = []
        textures = []

        meshes.append(genWallWithoutTexture(
            Extents[0]))
        meshes.append(genWallWithoutTexture(
            Extents[1]))
        meshes.append(genWallWithoutTexture(
            Extents[0]))
        meshes.append(genWallWithoutTexture(
            Extents[1]))

        im = skimage.io.imread('textures/beige_wall_001_disp.png')
        textures.append(cropTexture(
            np.array([Extents[0][0], Extents[0][1]]), deepcopy(im)))
        textures.append(cropTexture(
            np.array([Extents[1][0], Extents[1][1]]), deepcopy(im)))
        textures.append(cropTexture(
            np.array([Extents[0][0], Extents[0][1]]), deepcopy(im)))
        textures.append(cropTexture(
            np.array([Extents[1][0], Extents[1][1]]), deepcopy(im)))

        if door_sem is not None:
            wall_idx = findDoorWall(bbox, center, door_sem)
            del meshes[wall_idx]
            del linkPositions[wall_idx]
            del linkRotations[wall_idx]
            del textures[wall_idx]

            # texture_direction = 0 if wall_idx in [0, 2] else 1

            z = door_sem['bb'][1]
            y = door_sem['bb'][1]
            x = door_sem['bb'][0]

            if wall_idx == 0 or wall_idx == 2:
                extents_ex = [[maxs[0] - door_sem['pose'][0][0] - x/2, bbox[1], thickness],
                              [door_sem['pose'][0][0] - mins[0] - x/2, bbox[1], thickness]]
                # [door_sem['pose'][0][0] - center[0]+bbox[0]/2 - y/2, z, thickness]]
            elif wall_idx == 1 or wall_idx == 3:
                extents_ex = [[maxs[2] - door_sem['pose'][0][2] - x/2, bbox[1], thickness],
                              [door_sem['pose'][0][2] - mins[2] - x/2, bbox[1], thickness]]
                # [z, door_sem['pose'][0][1] - center[1]+bbox[1]/2 - y/2, thickness]]

            meshes.append(genWallWithoutTexture(extents_ex[0]))
            meshes.append(genWallWithoutTexture(extents_ex[1]))
            textures.append(cropTexture(
                np.array([extents_ex[0][0], extents_ex[0][1]]), deepcopy(im)))
            textures.append(cropTexture(
                np.array([extents_ex[1][0], extents_ex[1][1]]), deepcopy(im)))
            # meshes.append(genWallWithTexture(extents_ex[2], [0, 0, 0], [0, 0, 0]), texture_direction)

            if wall_idx == 0:
                linkPositions += [[maxs[0] - extents_ex[0][0]/2., center[1], maxs[2]],
                                  [mins[0] + extents_ex[1][0]/2., center[1], maxs[2]]]
                # [center[0]-bbox[0]/2 + extents_ex[2][0]/2, center[1] + bbox[1]/2, z/2]]
                linkRotations += [[0, 0, 0],
                                  [0, 0, 0]]
            if wall_idx == 1:
                linkPositions += [[maxs[0], center[1], maxs[2] - extents_ex[0][0]/2.],
                                  [maxs[0], center[1],  mins[2] + extents_ex[1][0]/2.]]
                # [center[0]+bbox[0]/2, center[1]-bbox[1]/2 + extents_ex[2][1]/2, z/2]]
                linkRotations += [[0, np.pi/2, 0],
                                  [0, np.pi/2, 0]]
            if wall_idx == 2:
                linkPositions += [[maxs[0] - extents_ex[0][0]/2., center[1], mins[2]],
                                  [mins[0] + extents_ex[1][0]/2., center[1], mins[2]]]
                # [center[0]-bbox[0]/2 + extents_ex[2][0]/2, center[1] - bbox[1]/2, z/2]]
                linkRotations += [[0, 0, 0],
                                  [0, 0, 0]]
            if wall_idx == 3:
                linkPositions += [[mins[0], center[1], maxs[2] - extents_ex[0][0]/2.],
                                  [mins[0], center[1],  mins[2] + extents_ex[1][0]/2.]]
                # [center[0]-bbox[0]/2, center[1]-bbox[1]/2 + extents_ex[2][1]/2, z/2]]
                linkRotations += [[0, np.pi/2, 0],
                                  [0, np.pi/2, 0]]
    else:
        linkPositions = []
        linkRotations = []
        textures = []
        meshes = []

    floor = genWallWithoutTexture([bbox[0], 0, bbox[2]])

    im = skimage.io.imread('textures/floor-2.jpg')
    im = cropTexture(
        np.array([bbox[2], bbox[0]]), deepcopy(im), texture_len=1.0)
    im = np.append(im, np.ones(
        (im.shape[0], im.shape[1], 1), 'uint8')*255, axis=2)
    if floor_overlay is not None:
        floor_overlay = np.flip(
            np.flip(floor_overlay.transpose(1, 0, 2), axis=0), axis=1)
        new_floor_overlay = rescale(floor_overlay, [
                                    im.shape[0]/floor_overlay.shape[0], im.shape[1]/floor_overlay.shape[1], 1])
        floor_jpg = overlay_transparant_images(
            (new_floor_overlay*255).astype('uint8'), im)
        textures.append(floor_jpg)
    else:
        textures.append(im)

    linkPositions.append([center[0], mins[1], center[2]])
    linkRotations.append([0, 0, 0])

    return meshes + [floor], linkPositions, linkRotations, textures


def genBoxMesh(bb, linkPosition, linkRotation):
    trans = np.eye(4)
    trans[:3, 3] = linkPosition
    trans[:3, :3] = Rotation.from_euler('xyz', linkRotation).as_matrix()
    m = trimesh.creation.box(extents=bb, transform=trans)
    color = [255, 255, 255, 255]
    m.visual.face_colors = color
    # convert color visual to texture
    m.visual = m.visual.to_texture()
    # # convert back to color
    # m.visual = m.visual.to_color()
    return m


def spawnObj(sem):

    obj_path = os.path.join(MODEL_PATH, sem['id'])
    obj_path = os.path.join(obj_path, sem['id']+'.obj')

    tm = trimesh.load(obj_path, force='scene')

    rot = Rotation.from_euler(
        'xyz', [sem['pose'][1][0], sem['pose'][1][1], sem['pose'][1][2]]).as_matrix()
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = [sem['pose'][0][0], sem['pose'][0][1], sem['pose'][0][2]]
    tm.apply_transform(trans)

    return tm


def gen_furn_meshes(sem, with_texture=True):

    thing_path = os.path.join(MODEL_PATH, sem['id'])
    obj_path = os.path.join(thing_path, sem['id']+'.obj')
    mtl_path = os.path.join(thing_path, sem['id']+'.mtl')
    if not os.path.exists(mtl_path):
        mtl_path = None

    # tm = trimesh.load(obj_path, force='scene')

    rot = Rotation.from_euler(
        'xyz', [sem['pose'][1][0], sem['pose'][1][1], sem['pose'][1][2]]).as_matrix()
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = [sem['pose'][0][0], sem['pose'][0][1], sem['pose'][0][2]]

    if with_texture:
        meshes, texture_dict = plot_obj_with_multiple_textures(
            obj_path, mtl_path)
        for mesh in meshes:
            mesh.transform(trans)
        return meshes, texture_dict
    else:
        mesh = pv.read(obj_path)
        mesh.transform(trans)
        return mesh


def plot_obj_with_multiple_textures(obj_path, mtl_path=None):
    obj_mesh = pv.read(obj_path)
    if mtl_path is None:
        # parse the obj file for mtl_path if the mtl_path is not set.
        pass

    texture_dir = os.path.dirname(obj_path)

    texture_dict = {}
    mtl_names = []

    # parse the mtl file
    with open(mtl_path) as mtl_file:
        current_key = -1
        for line in mtl_file.readlines():
            parts = line.strip().split()
            if len(parts) == 2:
                if parts[0] == 'newmtl':
                    current_key += 1
                    mtl_names.append(parts[1])
                    texture_dict[current_key] = {}
                elif parts[0] == 'map_Kd':
                    texture_dict[current_key][parts[0]] = os.path.join(
                        texture_dir, parts[1])
                else:
                    texture_dict[current_key][parts[0]] = parts[1]
            elif len(parts) == 4:
                texture_dict[current_key][parts[0]] = [
                    float(parts[1]), float(parts[2]), float(parts[3])]

    material_ids = obj_mesh.cell_data['MaterialIds']

    # This one do.
    mesh_parts = []
    for i in np.unique(material_ids):
        mesh_part = obj_mesh.extract_cells(material_ids == i)
        # if 'map_Kd' in texture_dict[i]:
        #     mesh_part.textures[mtl_names[i]] = pv.read_texture(os.path.join(texture_dir, texture_dict[i]['map_Kd']))
        mesh_parts.append(mesh_part)

    return mesh_parts, texture_dict


def loadFromJson(room):
    scene_id, furn_dict, furn_list, seg_map = room
    furn_info = nx.DiGraph()
    for idx in furn_dict:
        furn_info.add_node(int(idx), **furn_dict[idx])
    for idx in furn_info.nodes():
        furn_info.nodes()[idx]['max'] = np.array(furn_info.nodes()[idx]['max'])
        furn_info.nodes()[idx]['min'] = np.array(furn_info.nodes()[idx]['min'])
        furn_info.nodes()[idx]['front'] = np.array(
            furn_info.nodes()[idx]['front'])
        furn_info.nodes()[idx]['bb'] = np.array(furn_info.nodes()[idx]['bb'])
        furn_info.nodes()[idx]['rotation'] = np.array(
            furn_info.nodes()[idx]['rotation'])
        furn_info.nodes()[idx]['pose'][0] = np.array(
            furn_info.nodes()[idx]['pose'][0])
        furn_info.nodes()[idx]['pose'][1] = np.array(
            furn_info.nodes()[idx]['pose'][1])
    furn_info.add_edges_from(furn_list)
    seg_map['size'] = np.array(seg_map['size'])
    seg_map['max'] = np.array(seg_map['max'])
    seg_map['min'] = np.array(seg_map['min'])
    seg_map['center'] = np.array(seg_map['center'])
    return furn_info, seg_map


def add_furn_meshes(furn_sem, plotter):
    meshes, textures_dict = gen_furn_meshes(furn_sem)
    for idx in range(len(meshes)):
        mesh_part = meshes[idx]
        if 'map_Kd' in textures_dict[idx]:
            texture = pv.read_texture(textures_dict[idx]['map_Kd'])
            plotter.add_mesh(mesh_part, texture=texture, specular=textures_dict[idx]
                             ['Ks'][0], opacity=float(textures_dict[idx]['d']))
        else:
            plotter.add_mesh(mesh_part, color=textures_dict[idx]['Ka'], specular=textures_dict[idx]
                             ['Ks'][0], opacity=float(textures_dict[idx]['d']))


def add_room_meshes(plotter, bbox, center, seg_map, door_sem, use_walls, overlay=None):
    meshes, link_pos, link_rot, textures = gen_room_meshes(bbox, center, mins=np.array(
        seg_map['min']), maxs=np.array(seg_map['max']), door_sem=door_sem, use_walls=use_walls, floor_overlay=overlay)
    for idx in range(len(link_pos)):
        trans = np.eye(4)
        trans[:3, 3] = link_pos[idx]
        trans[:3, :3] = Rotation.from_euler(
            'xyz', link_rot[idx]).as_matrix()
        mesh = meshes[idx]
        mesh.transform(trans)
        plotter.add_mesh(mesh, smooth_shading=True,
                         texture=pv.numpy_to_texture(textures[idx]))


def visualizeOverlayInfo(furn_info, opt_config: AdaptiveAnnealingConfig, task_config: TaskConfig, robot: dict, seg_map=None,
                         make_gif = False, reach=False, app=False, path=False, color=False,
                         grouping_order=None, use_walls=True, GUI=False, snip_shot=False, name='', resolution=2000):
    scene_name = opt_config.scene_name
    room_type = scene_name.split('_')[0]
    # processed_position_net = open(
    #     f'./suncg/data/{room_type}_position_net_processed_lognorm.bin', 'rb')
    # position_net = pickle.load(processed_position_net)
    if seg_map is None:
        center, bbox, mins, maxs = SceneManager.getRoomDimension(furn_info)
        seg_map = {'resolution': 1.0, 'size': maxs-mins,
                   'max': maxs.tolist(), 'min': mins.tolist()}

    manager = SceneManager(furn_info, None,
                           seg_map, robot, use_gui=True)
    manager.setOptimizeRelations([])
    opt_config.write_images = False
    opt_config.desired_sdf_res = 0.01
    manager.setupOptimizationConfig(opt_config)
    manager.setTaskConfig(task_config)
    manager.createCompleteSDFBuffer()
    app_png,  rea_png, ant_png, path_png, suc, fai = manager.generateOverlayImages()

    overlayed_image = None
    if reach:
        overlayed_image = rea_png
    if path:
        overlayed_image = overlay_images(overlayed_image, path_png)
    if app:
        new_overlay = overlay_images(ant_png, app_png)
        overlayed_image = overlay_images(overlayed_image, new_overlay)

    plotter = pv.Plotter(
        window_size=[resolution, resolution], off_screen=not GUI)
    plotter.store_image = snip_shot
    # plotter.enable_eye_dome_lighting()

    bbox = seg_map['size']
    center = np.array(seg_map['min']) + seg_map['size']/2.
    door_sem = findDoorSem(furn_info)

    add_room_meshes(plotter, bbox, center, seg_map,
                    door_sem, use_walls, overlay=overlayed_image)

    plot_ids = []
    if grouping_order is None:
        plot_ids = list(furn_info.nodes)
    else:
        for lead_id, subs in grouping_order.items():
            combined_ids = [lead_id] + subs if lead_id != 'fix' else subs
            plot_ids += combined_ids

    offcenter_furniture(furniture_info=furn_info)
    for id in plot_ids:

        sem = furn_info.nodes()[id]
        if id in suc and color:
            mesh = gen_furn_meshes(sem, with_texture=False)
            plotter.add_mesh(
                mesh, color=[0.78039215686, 0.83529411764, 0.63137254902])
        elif id in fai and color:
            mesh = gen_furn_meshes(sem, with_texture=False)
            plotter.add_mesh(
                mesh, color=[0.70196078431, 0.34117647058, 0.31764705882])
        else:
            add_furn_meshes(sem, plotter)

    plotter.camera_position = 'zx'
    # plotter.camera.azimuth = 30
    plotter.camera.roll += 90
    plotter.camera.elevation -= 30
    plotter.reset_camera_clipping_range()

    plotter.remove_all_lights()
    light = pv.Light(light_type='headlight', intensity=1.5)
    plotter.add_light(light)
    # plotter.enable_shadows()

    if GUI:
        # plotter.show_axes()
        plotter.show()
    if snip_shot:
        # skimage.io.imsave(name+'.png', plotter.image)
        plotter.set_background('white')
        plotter.screenshot(
            name+'.png', transparent_background=False, window_size=[resolution, resolution])

    if make_gif:
        plotter.set_background('white')
        images = []
        for roll in range(0, 360, 4):
            plotter.camera_position = 'zx'
            plotter.camera.roll = roll
            plotter.camera.elevation -= 45
            # plotter.camera.azimuth = azimuth
            plotter.reset_camera_clipping_range()
            plotter.render()
            images.append(plotter.screenshot(
                None, transparent_background=False, window_size=[resolution, resolution]))

        imageio.mimsave(name+'.gif', images)


def visualizeFurnitureInfo(furn_info, seg_map=None, grouping_order=None, use_walls=False, 
        GUI=False, snip_shot=False, make_gif=False, name='', resolution=2000):
    if seg_map is None:
        center, bbox, mins, maxs = SceneManager.getRoomDimension(furn_info)
        seg_map = {'resolution': 1.0, 'size': maxs-mins,
                   'max': maxs.tolist(), 'min': mins.tolist()}

    plotter = pv.Plotter(
        window_size=[resolution, resolution], off_screen=not GUI)
    plotter.store_image = snip_shot
    # plotter.enable_eye_dome_lighting()

    bbox = seg_map['size']
    center = np.array(seg_map['min']) + seg_map['size']/2.

    door_sem = findDoorSem(furn_info)

    add_room_meshes(plotter, bbox, center, seg_map,
                    door_sem, use_walls, overlay=None)

    plot_ids = []
    if grouping_order is None:
        plot_ids = list(furn_info.nodes)
    else:
        for lead_id, subs in grouping_order.items():
            combined_ids = [lead_id] + subs if lead_id != 'fix' else subs
            plot_ids += combined_ids

    for id in plot_ids:
        sem = furn_info.nodes()[id]
        add_furn_meshes(sem, plotter)

    plotter.camera_position = 'zx'
    # plotter.camera.azimuth = 30
    plotter.camera.roll += 90
    plotter.camera.elevation -= 30
    plotter.reset_camera_clipping_range()

    plotter.remove_all_lights()
    light = pv.Light(light_type='headlight', intensity=1.5)
    plotter.add_light(light)

    # plotter.enable_shadows()
    # - A headlight is attached to the camera, looking at its
    #     focal point along the axis of the camera.

    # - A camera light also moves with the camera, but it can
    #     occupy a general position with respect to it.

    # - A scene light is stationary with respect to the scene,
    #     as it does not follow the camera. This is the default.

    if GUI:
        plotter.show()
    if snip_shot:
        # skimage.io.imsave(name+'.png', plotter.image)
        plotter.set_background('white')
        plotter.screenshot(
            name+'.png', transparent_background=False, window_size=[resolution, resolution])
    #     pyrender.Viewer(scene, use_perspective_cam=True,
    #                     shadows=True, use_raymond_lighting=True)
    if make_gif:
        plotter.set_background('white')
        images = []
        for roll in range(0, 360, 4):
            plotter.camera_position = 'zx'
            plotter.camera.roll = roll
            plotter.camera.elevation -= 45
            # plotter.camera.azimuth = azimuth
            plotter.reset_camera_clipping_range()
            plotter.render()
            images.append(plotter.screenshot(
                None, transparent_background=False, window_size=[resolution, resolution]))

        imageio.mimsave(name+'.gif', images)
    pv.close_all()
    return
