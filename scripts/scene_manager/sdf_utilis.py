# from sdf import *
from scene_manager.sdf_lib.sdf import *
from scene_manager.sdf_lib.sdf.mesh import _cartesian_product
import numpy as np
from copy import deepcopy
import skimage.io

DUMMY_HEIGHT = 20


def getCenteredSDF(padded_sdf, w, h):
    s = padded_sdf.shape
    dw = int((s[1]-w)/2)
    dh = int((s[0]-h)/2)
    return padded_sdf[dh:(dh+h), dw:(dw+w)]


def getPaddedImg(padded_img, centered_img):
    b = padded_img.shape
    s = centered_img.shape
    dw = int((b[1]-s[1])/2)
    dh = int((b[0]-s[0])/2)

    pp = deepcopy(padded_img)
    pp[dh:(dh+s[0]), dw:(dw + s[1])] = centered_img

    return pp


def getPaddedSDF(padded_sdf, centered_sdf):
    pp = getPaddedImg(padded_img=padded_sdf, centered_img=centered_sdf)
    return np.minimum(padded_sdf, pp)


def getRoomExtent(bounds, h, w, y=0):
    (x0, z0), (x1, z1) = bounds

    X = np.linspace(x0, x1, h)
    Z = np.linspace(z0, z1, w)

    extend = (X[0], X[-1], Z[0], Z[-1])

    Y = np.array([y])

    P = _cartesian_product(X, Y, Z)
    return extend, P


def extent2Coordinates(extent, h, w):
    X = np.linspace(extent[0], extent[1], h)
    Z = np.linspace(extent[2], extent[3], w)
    Y = np.array([0])
    return _cartesian_product(X, Y, Z)


def sample_slice_2d(sdf, bounds, w=1024, h=1024, y=0):
    extend, P = getRoomExtent(bounds, h, w, y=y)
    return sdf(P).reshape((h, w)), extend, P


def placeOnGround(furn, box=None):
    # if box is None:
    p1, p2 = mesh._estimate_bounds(furn)
    return furn.translate((0, 0, -p1[2]))


def genDrawerSingle(size=(1, 1, 1), thickness=0.01, bottom_thickness=None, trans=(), joint_angle=0):
    b = box(size, np.array((0, 0, 0)))
    bottom_thickness = thickness if bottom_thickness is None else bottom_thickness
    ss = slab(x1=size[0]/2.-thickness, y0=-size[1]/2.+thickness,
              y1=size[1]/2.-thickness, z0=-size[2]/2.+bottom_thickness)
    b -= b & ss
    if len(trans):
        b = b.translate(trans)
    b = b.translate((joint_angle, 0, 0))
    return b


def genDivider(size=(1, 1, 1), thickness=0.01, num_compartments=2):
    length_ = size[2]/num_compartments
    c = box((size[0], size[1], thickness), (0, 0, -size[2]/2.0))
    d = c
    for dum in range(num_compartments-1):
        c = c.translate((0, 0, length_))
        d = d | c
    return d


def genDoorSingle(size=(1, 1, 1), thickness=0.01, trans=(), joint_angle=0, d_type="l"):
    door_size = (thickness, size[1], size[2])
    b = box(door_size, np.array((0, 0, 0)))

    if d_type == "l":
        b = b.rotate(joint_angle, Z)
        if len(trans):
            b = b.translate(trans)
        b = b.translate((size[0]/2.0, 0, 0))
        b = b.translate((np.sin(joint_angle)*size[1]/2.0,
                        (1.0-np.cos(joint_angle))*size[1]/2.0, 0))

    else:
        b = b.rotate(-joint_angle, Z)
        if len(trans):
            b = b.translate(trans)
        b = b.translate((size[0]/2.0, 0, 0))
        b = b.translate((np.sin(joint_angle)*size[1]/2.0,
                        -(1.0-np.cos(joint_angle))*size[1]/2.0, 0))

    return b


def genTable(size=(1, 1, 1), thickness=0.1, trans=(), d_type="round", support_thickness=0.2, use_block=False, use_3d=False):

    if not use_3d and not use_block:
        _trans = (- size[0]/2. + support_thickness/2.,
                  size[1]/2. - support_thickness/2.)
        b = rectangle((support_thickness, support_thickness), _trans)
        _trans = (- size[0]/2. + support_thickness/2.,
                  - size[1]/2. + support_thickness/2.)
        b = b | rectangle((support_thickness, support_thickness), _trans)
        _trans = (size[0]/2. - support_thickness/2.,
                  size[1]/2. - support_thickness/2.)
        b = b | rectangle((support_thickness, support_thickness), _trans)
        _trans = (size[0]/2. - support_thickness/2.,
                  - size[1]/2. + support_thickness/2.)
        b = b | rectangle((support_thickness, support_thickness), _trans)

        b = b.rotate(trans[1])
        b = b.translate(trans[0])
        return b

    if not use_3d and use_block:
        # b = rectangle(size)
        # b = b.rotate(trans[1])
        # b = b.translate(trans[0])
        if d_type == "square":
            b = box([size[0], size[1], DUMMY_HEIGHT])
            b = b.rotate(trans[1], Z)
            b = b.translate((trans[0][0], trans[0][1], 0))
        elif d_type == "round":
            b = capped_cylinder(-Z * DUMMY_HEIGHT, Z *
                                DUMMY_HEIGHT, (size[0]+size[1])/4.0)
            b = b.rotate(trans[1], Z)
            b = b.translate((trans[0][0], trans[0][1], 0))
        return b

    if use_block:
        if d_type == "square":
            b = box(size)
        elif d_type == "round":
            b = capped_cylinder(-Z * size[2], Z * size[2], size[0]/2.0)

    else:

        if d_type == "square":
            b = rounded_box(size,  0.025)
            ss = slab(x0=-size[0]/2.+support_thickness, x1=size[0] /
                      2.-support_thickness, z1=size[2]/2.-thickness)
            b -= b & ss
            ss = slab(y0=-size[1]/2.+support_thickness,
                      y1=size[1]/2.-support_thickness, z1=size[2]/2.-thickness)
            b -= b & ss
        elif d_type == "round":
            b = rounded_cylinder(size[0]/2.0, 0.025, thickness)
            b = b.translate((0, 0, size[2]/2. - thickness/2.))
            b = b | rounded_cylinder(
                size[0]/4.0, 0.025, thickness).translate((0, 0, -size[2]/2. + thickness/2.))
            b = b | capped_cylinder(-Z*size[2]/2.,
                                    Z*size[2]/2., support_thickness)

    b = placeOnGround(b)
    if len(trans):
        b = b.rotate(trans[1][0], X)
        b = b.rotate(trans[1][1], Y)
        b = b.rotate(trans[1][2], Z)
        b = b.translate(trans[0])
    return b


def genCabinet(size=(1, 1, 1), drawers=["d", "l", "r"], thickness=0.01, bottom_thickness=None, trans=(), configuration=(), use_block=False, use_3d=False):

    if not use_3d:
        # _size = list(size)[:2]
        # b = rectangle(_size)
        # b = b.rotate(trans[1])
        # b = b.translate(trans[0])
        b = box([size[0], size[1], DUMMY_HEIGHT])
        b = b.rotate(trans[1], Z)
        b = b.translate((trans[0][0], trans[0][1], 0))
        return b

    if use_block:
        base = box(size)
    else:

        base = box(size, np.array((0, 0, 0)))
        base_s = slab(x0=-size[0]/2.+thickness, y0=-size[1]/2.+thickness,
                      y1=size[1]/2.-thickness, z0=-size[2]/2.+thickness, z1=size[2]/2.-thickness)
        base -= base & base_s
        height = -size[2]/2
        if len(drawers) > 0:
            drawer_height = size[2]/len(drawers)
            for idx in range(len(drawers)):
                s_type = drawers[idx]
                d_size = (size[0], size[1], drawer_height)
                if s_type == "d":
                    d = genDrawerSingle(size=d_size, thickness=thickness,
                                        bottom_thickness=bottom_thickness, trans=(0, 0, height + drawer_height/2.0), joint_angle=configuration[idx])
                else:
                    d = genDoorSingle(size=d_size, thickness=thickness, trans=(
                        0, 0, height + drawer_height/2.0), joint_angle=configuration[idx], d_type=s_type)
                height += drawer_height
                base = base | d

            base = base | genDivider(
                size=size, thickness=thickness, num_compartments=len(drawers))

    b = placeOnGround(base)

    if len(trans):
        b = b.rotate(trans[1][0], X)
        b = b.rotate(trans[1][1], Y)
        b = b.rotate(trans[1][2], Z)
        b = b.translate(trans[0])
    return b


def genBlock(size=(1, 1, 1), trans=()):
    b = box([size[0], DUMMY_HEIGHT, size[2]])
    b = b.rotate(trans[0][4], Y)
    b = b.translate((trans[0][0], 0, trans[0][2]))
    return b


def genShelf(size=(1, 1, 1), num_block=1, thickness=0.01, trans=(), d_type="open", use_block=False, use_3d=False):

    if not use_3d:
        # b = rectangle(size)
        # b = b.rotate(trans[1])
        # b = b.translate(trans[0])
        # return b
        b = box([size[0], size[1], DUMMY_HEIGHT])
        b = b.rotate(trans[1], Z)
        b = b.translate((trans[0][0], trans[0][1], 0))
        return b

    if use_block:
        b = box(size)
    else:
        base = box(size, np.array((0, 0, 0)))
        if d_type == "open":
            base_s = slab(x0=-size[0]/2.+thickness, y0=-size[1]/2.+thickness,
                          y1=size[1]/2.-thickness, z0=-size[2]/2.+thickness)
            base -= base & base_s
        elif d_type == "closed":
            base_s = slab(x0=-size[0]/2.+thickness, y0=-size[1]/2.+thickness,
                          y1=size[1]/2.-thickness, z0=-size[2]/2.+thickness, z1=size[2]/2.-thickness)
            base -= base & base_s
        if num_block > 0:
            base = base | genDivider(
                size=size, thickness=thickness, num_compartments=num_block)
        b = placeOnGround(base)
    if len(trans):
        b = b.rotate(trans[1][0], X)
        b = b.rotate(trans[1][1], Y)
        b = b.rotate(trans[1][2], Z)
        b = b.translate(trans[0])
    return b


def genChair(size=(1, 1, 1), back_ratio=0.5, back_thickness=0.05, seat_thickness=0.1, trans=(), support_thickness=0.1, use_block=False, use_3d=False):

    if not use_3d:
        return genTable(size=size, support_thickness=support_thickness, d_type="square", trans=trans, use_block=use_block, use_3d=use_3d)

    if use_block:
        b = box(size)
    else:
        table_size = (size[0], size[1], size[2] * (1-back_ratio))
        b = genTable(table_size, thickness=seat_thickness,
                     d_type="square", support_thickness=support_thickness).translate((0, 0, -table_size[2]/2.0))

        back_size = (back_thickness, size[1], size[2] * back_ratio)
        s = box(back_size).translate(
            (-size[0]/2. + back_thickness/2.0, 0, size[2] * back_ratio))
        b = b | s
        b = placeOnGround(b)
    if len(trans):
        b = b.rotate(trans[1][0], X)
        b = b.rotate(trans[1][1], Y)
        b = b.rotate(trans[1][2], Z)
        b = b.translate(trans[0])
    return b


def genLamp(size=(1, 1, 1), thickness=0.01, trans=(), support_thickness=0.2, use_block=False, use_3d=False):
    return genTable(size=size, thickness=thickness, trans=trans, d_type="round", support_thickness=support_thickness, use_block=use_block, use_3d=use_3d)


def genBed(size=(1, 1, 1), thickness=0.3, trans=(), support_thickness=0.3, use_block=False, use_3d=False):
    return genTable(size=size, thickness=thickness, trans=trans, d_type="square", support_thickness=support_thickness, use_block=use_block, use_3d=use_3d)


def genRoom(bboxes, thickness=2.0, use_block=False, use_3d=True):

    if not use_3d:
        b = None
        for bbox, tf in bboxes:

            _bbox1 = [bbox[0]+thickness, thickness, DUMMY_HEIGHT]
            _bbox2 = [thickness, bbox[1]+thickness, DUMMY_HEIGHT]
            _tf1 = tf + np.array([0, bbox[1]/2.0 + thickness/2., 0])
            _tf2 = tf + np.array([bbox[0]/2.0 + thickness/2., 0, 0])
            _tf3 = tf + np.array([0, -bbox[1]/2.0 - thickness/2., 0])
            _tf4 = tf + np.array([-bbox[0]/2.0 - thickness/2., 0, 0])
            # _bbox = bbox[:2]
            # _tf = tf[:2]
            if b is None:
                # b = rectangle(_bbox, _tf)

                b = box(_bbox1, _tf1)
                b = b | box(_bbox2, _tf2)
                b = b | box(_bbox1, _tf3)
                b = b | box(_bbox2, _tf4)
            else:
                f = box(_bbox1, _tf1)
                f = f | box(_bbox2, _tf2)
                f = f | box(_bbox1, _tf3)
                f = f | box(_bbox2, _tf4)
                b = b | f
            # f = sphere().shell(0.05) & plane(-Z)
        # b = b.shell(thickness)
        return b

    b = None
    for bbox, tf in bboxes:
        if b is None:
            b = box(bbox, tf)
        else:
            f = box(bbox, tf)
            b = b | f
        # f = sphere().shell(0.05) & plane(-Z)
    b = b.shell(thickness)

    b = b & plane(-Z, (0, 0, bbox[2]/2.-2*thickness))
    b = b & plane(Z, (0, 0, -bbox[2]/2.+2*thickness))
    if not use_block:
        b = b.translate((0, 0, bbox[2]/2-2*thickness))

    return b


def genRoomSDF(upper, lower):
    box_size = np.array(upper) - np.array(lower)
    center = lower + box_size/2.
    box_size[1] = box_size[1] * 2.
    b = box(box_size, center)

    return b


if __name__ == '__main__':
    # Cabinet_size = (1, 1, 1.5)
    # Shelf_size = (1, 1, 1)
    # Table_size = (1, 1, 1)
    # Chair_size = (1, 1, 1)
    # rooms = [((6, 6, 3), (0, 0, 0)), ((4, 4, 3), (3, 3, 0))]

    # b = genCabinet(size=Cabinet_size, drawers=["d", "l", "r"], thickness=0.05, bottom_thickness=None, trans=(
    #     (0, 0, 0), (0, 0, 0)), configuration=(0.2, 0.5, 0.6), use_3d=True)

    # b = b | genShelf(size=Shelf_size, num_block=3, trans=((0, 2, 0), (0, 0, np.pi/2)),
    #                  thickness=0.05, d_type="open", use_3d=True)

    # b = b | genShelf(size=Shelf_size, num_block=3, trans=((-2, -2, 0), (0, 0, -np.pi/2)),
    #                  thickness=0.05, d_type="closed", use_3d=True)

    # b = b | genTable(size=Table_size, thickness=0.2, trans=(
    #     (0, -2, 0), (0, 0, 0)), d_type="square", support_thickness=0.1, use_3d=True)

    # b = b | genTable(size=Table_size, thickness=0.2, trans=(
    #     (2, -2, 0), (0, 0, 0)), d_type="round", support_thickness=0.1, use_3d=True)

    # b = b | genChair(size=Chair_size, back_ratio=0.5, back_thickness=0.05,
    #                  seat_thickness=0.1, trans=((2, 2, 0), (0, 0, -np.pi/2.)), support_thickness=0.1, use_3d=True)

    # b = b | genRoom(rooms, thickness=0.1)

    # b.show_slice(z=0.0)
    # b.save('out.stl')

    Cabinet_size = (1, 1)
    Shelf_size = (1, 1)
    Table_size = (1, 1)
    Chair_size = (1, 1)
    rooms = [((6, 6), (0, 0))]

    b = genCabinet(size=Cabinet_size, thickness=0.05, bottom_thickness=None, trans=(
        (0, 0), 0), use_3d=False)

    b = b | genShelf(size=Shelf_size, num_block=3, trans=((0, 2), np.pi/2),
                     thickness=0.05, d_type="open", use_3d=False)

    b = b | genShelf(size=Shelf_size, num_block=3, trans=((-2, -2),  -np.pi/2),
                     thickness=0.05, d_type="closed", use_3d=False)

    b = b | genTable(size=Table_size, thickness=0.2, trans=(
        (0, -2), 0), d_type="square", support_thickness=0.1, use_3d=False)

    b = b | genTable(size=Table_size, thickness=0.2, trans=(
        (2, -2),  0), d_type="round", support_thickness=0.1, use_3d=False)

    b = b | genChair(size=Chair_size, back_ratio=0.5, back_thickness=0.05,
                     seat_thickness=0.1, trans=((2, 2),  -np.pi/2.), support_thickness=0.1, use_3d=False)

    b = b | genRoom(rooms, thickness=0.1, use_3d=False)

    bounds = (rooms[0][1][0] - rooms[0][0][0]/2., rooms[0][1][1] - rooms[0][0][1] /
              2.), (rooms[0][1][0] + rooms[0][0][0]/2., rooms[0][1][1] + rooms[0][0][1]/2.)

    imgs = sample_slice_2d(b, bounds, w=1024, h=1024)

    normalizedImg = (np.zeros_like(imgs) - np.min(imgs)) / \
        (np.min(imgs)+np.max(imgs))
    # normalizedImg = cv.normalize(imgs,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    skimage.io.imsave('savedImage.jpg', normalizedImg)
