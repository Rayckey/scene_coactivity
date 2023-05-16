import argparse
import json
from concept_net.conceptnet_utilis import plot
import os
import pickle
import csv
from concept_net.graph_utilis import *
from suncg_room_utilis import generate_model_category_mapping
from visualization.visual_utilis import *
from scipy import stats
from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min,
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant,
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell,
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice,
    semicircular, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow,
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
)


def gather_spatialnx(room_type, GUI, gen_images, gen_graph, collect):
    test_rooms = []
    with open('selected_rooms.csv', 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            test_rooms.append(row[0])

    rooms_file = open(
        f'./suncg/data/{room_type}')
    valid_rooms = json.load(rooms_file)

    cat_path = "./suncg"
    cat_path = os.path.join(cat_path, 'category.txt')

    suncg_object_types = []
    with open(cat_path, 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            suncg_object_types.append(row[0])

    # scene groups
    # add all nodes
    spatialnx = DiGraph()
    for cat in suncg_object_types:
        spatialnx.add_node(cat.replace("_", " "))

    for cat1 in spatialnx:
        for cat2 in spatialnx:
            spatialnx.add_edge(cat1, cat2, NEXTTO=[], CLOSETO=[], CLOSETOBOUND=[],
                                 FRONTOF=[], FACING=[], FACINGBOUND=[], PARALLEL=[])

    idx = -1
    for room in valid_rooms:
        idx += 1
        if f'{room_type}_{idx}' in test_rooms:
            continue

        if idx > len(valid_rooms) * 0.9:
            break

        if os.path.exists(f'./suncg/data/graphs/{room_type}_{idx}.bin'):
            fi = open(
                f'./suncg/data/graphs/{room_type}_{idx}.bin', "rb")
            furn_info, seg_map = pickle.load(fi)

        else:
            furn_info, seg_map = loadFromJson(room)
            visualizeFurnitureInfo(furn_info, seg_map, None, use_walls=False, GUI=GUI, snip_shot=gen_images,
                                   name=f'./suncg/data/graphs/{room_type}_{idx}_rgb.png')
            fi = open(
                f'./suncg/data/graphs/{room_type}_{idx}.bin', "w+b")
            pickle.dump([furn_info, seg_map], fi)

        if collect:

            if os.path.exists(f'./suncg/data/graphs/{room_type}_{idx}.pos'):
                fi = open(f'./suncg/data/graphs/{room_type}_{idx}.pos', "rb")
                pos = pickle.load(fi)
            else:
                robot = {'name': 'husky', 'size': 0.595, 'reach': 0.85}
                manager = SceneManager(furn_info=furn_info, robot=robot,
                                       seg_map=seg_map, use_gui=GUI)
                opt_config = OptimizationConfig(scene_name=f"{room_type}_{idx}", robot="",
                                                randomize=False, desired_sdf_res=0.1, write_images=gen_images, padding=2)
                center_furniture(manager.furniture_info)
                manager.setupOptimizationConfig(opt_config)
                pos = manager.evaluatePairwiseRelation()
                fi = open(
                    f'./suncg/data/graphs/{room_type}_{idx}.pos', "w+b")
                pickle.dump(pos, fi)

            # room_diagonal = get_room_diagonal(seg_map)

            for id1 in pos:
                for id2 in pos:
                    if id1 == id2:
                        continue
                    # spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                    #     id2]['cat'].replace("_", " ")]['NEXTTO'] .append(pos[id1][id2]['NEXTTO'])
                    # spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                    #     id2]['cat'].replace("_", " ")]['CLOSETO'] .append(pos[id1][id2]['CLOSETO'])
                    spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                        id2]['cat'].replace("_", " ")]['FRONTOF'] .append(pos[id1][id2]['FRONTOF'])
                    spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                        id2]['cat'].replace("_", " ")]['FACING'] .append(pos[id1][id2]['FACING'])
                    spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                        id2]['cat'].replace("_", " ")]['PARALLEL'] .append(pos[id1][id2]['PARALLEL'])
                    spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                        id2]['cat'].replace("_", " ")]['CLOSETOBOUND'] .append(pos[id1][id2]['CLOSETOBOUND'])
                    spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                        id2]['cat'].replace("_", " ")]['FACINGBOUND'] .append(pos[id1][id2]['FACINGBOUND'])
                    # spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                    #     id2]['cat'].replace("_", " ")]['SHARED'] .append(pos[id1][id2]['SHARED'])
                    # spatialnx[pos.nodes()[id1]['cat'].replace("_", " ")][pos.nodes()[
                    #     id2]['cat'].replace("_", " ")]['SHARED_MAX'] .append(pos[id1][id2]['SHARED_MAX'])

            if gen_graph:
                plot(
                    pos, f'./suncg/data/graphs/{room_type}_{idx}_pos.png')

    fi = open(
        f'./suncg/data/{room_type}_position_net.bin', "w+b")
    pickle.dump(spatialnx, fi)
    print('Finished! ')


def fit_spatialnx(room_type):
    fi = open(f'./suncg/data/{room_type}_position_net.bin', "rb")
    position_net = pickle.load(fi)
    res = DiGraph()
    res.add_nodes_from(position_net)
    for cat1 in res:
        for cat2 in res:
            if cat1 == 'root' or cat2 == 'root' or len(position_net[cat1][cat2]['FACING']) < 5:
                continue

            # shared_normalized = np.array([position_net[cat1][cat2]["SHARED"][idx]/position_net[cat1]
            #                             [cat2]["SHARED_MAX"][idx] for idx in range(len(position_net[cat1][cat2]["SHARED"]))])
            front_normalized = np.array(
                position_net[cat1][cat2]["FRONTOF"])
            facing_normalized = np.array(
                position_net[cat1][cat2]["FACING"])
            parallel_normalized = np.array(
                position_net[cat1][cat2]["PARALLEL"])

            close2bound_normalized = np.array(
                position_net[cat1][cat2]["CLOSETOBOUND"])
            facingbound_normalized = np.array(
                position_net[cat1][cat2]["FACINGBOUND"])

            # shared_normalized = shared_normalized[~np.isnan(shared_normalized)]
            front_normalized = front_normalized[~np.isnan(
                front_normalized)]
            facing_normalized = facing_normalized[~np.isnan(
                facing_normalized)]
            parallel_normalized = parallel_normalized[~np.isnan(
                parallel_normalized)]
            close2bound_normalized = close2bound_normalized[~np.isnan(
                close2bound_normalized)]
            facingbound_normalized = facingbound_normalized[~np.isnan(
                facingbound_normalized)]

            # nextto_fitted = stats.distributions.lognorm.fit(
            #     position_net[cat1][cat2]["NEXTTO"])
            # closeto_fitted = stats.distributions.lognorm.fit(
            #     position_net[cat1][cat2]["CLOSETO"])
            frontof_fitted = stats.distributions.lognorm.fit(
                front_normalized)
            facing_fitted = stats.distributions.lognorm.fit(
                facing_normalized)
            parallel_fitted = stats.distributions.lognorm.fit(
                parallel_normalized)
            closetobound_fitted = stats.distributions.lognorm.fit(
                close2bound_normalized)
            facingbound_fitted = stats.distributions.lognorm.fit(
                facingbound_normalized)
            # , floc=-0.1, fscale=1.2
            # if len(shared_normalized) < 10:
            #     shared_fitted = None
            # else:
            #     shared_fitted = stats.distributions.beta.fit(shared_normalized)

            # TODO: get distribution with cat
            # fine in code, coarse in paper

            res.add_edge(cat1, cat2,
                         # NEXTTO=nextto_fitted,
                         # CLOSETO=closeto_fitted,
                         FRONTOF=frontof_fitted,
                         FACING=facing_fitted,
                         PARALLEL=parallel_fitted,
                         CLOSETOBOUND=closetobound_fitted,
                         FACINGBOUND=facingbound_fitted)
            # SHARED=shared_fitted)
    fi = open(
        f'./suncg/data/{room_type}_position_net_processed_lognorm.bin', "w+b")
    pickle.dump(res, fi)


def fit_occurrencenet(room_type):
    with open(f'./suncg/data/{room_type}_position_net.bin', "rb") as fi:
        position_net = pickle.load(fi)

    res = DiGraph()
    res.add_nodes_from(position_net)

    res1 = DiGraph()
    res1.add_nodes_from(position_net)

    for cat1 in res:
        for cat2 in res:
            if cat1 == 'root' or cat2 == 'root' or len(position_net[cat1][cat2]['FACING']) < 5:
                continue

            res1.add_edge(cat1, cat2, LEN=len(
                position_net[cat1][cat2]["FACING"]))

    for cat1 in res1:
        top = 0
        for cat2 in res1[cat1]:
            if cat1 == 'root' or cat2 == 'root':
                continue
            top = np.maximum(res1[cat1][cat2]['LEN'], top)

        ss = deepcopy(res1[cat1])
        for cat2 in ss:
            if cat1 == 'root' or cat2 == 'root':
                continue
            l = res1[cat1][cat2]['LEN']
            res1.remove_edge(cat1, cat2)
            if top > 0:
                res.add_edge(cat1, cat2, CO=float(l)/top)

    fi = open(
        f'./suncg/data/{room_type}_occurrence_net_processed.bin', "w+b")
    pickle.dump(res, fi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for differnt options of SunCG data processing.')

    room_options = ["living", "office", "bedroom"]
    parser.add_argument('-r', '--room', type=str, default='bedroom', choices=room_options,
                        help='define room category to be iterated (options: living, office, bedroom)')

    parser.add_argument('-i', '--images', help='generate top-down images for the rooms as it collects data.',
                        action='store_true')

    parser.add_argument('-g', '--graph', help='generate scene gragh visualization as it collects data.',
                        action='store_true')

    parser.add_argument('-c', '--collect', help='collect object spatial information from SUNCG, needed for -p and -o',
                        action='store_true')

    parser.add_argument('-p', '--pos', help="learn the distribution of the objects' relative poses.",
                        action='store_true')

    parser.add_argument('-o', '--occ', help="learn the objects' spatial co-occurrence",
                        action='store_true')

    args = parser.parse_args()

    if args.collect or args.images or args.graph:
        gather_spatialnx(args.room, False, args.images,
                           args.graph, args.collect)

    if True or args.images or args.graph:
        gather_spatialnx(args.room, False, args.images,
                           args.graph, args.collect)

    if args.pos:
        fit_spatialnx(args.room)

    if args.occ:
        fit_occurrencenet(args.room)

