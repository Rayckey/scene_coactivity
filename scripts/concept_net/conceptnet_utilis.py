#!/usr/bin/env python3
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx import DiGraph, Graph
from concept_net.graph_utilis import genConceptNetGraph, find_root
import pickle
import json
import requests
import yaml

with open('./configs/exceptions.yaml', 'r') as file:
    exceptions = yaml.safe_load(file)
    STATIC_CAT = exceptions['STATIC_CAT']
    FIXED = exceptions['FIXED']


def plot(graph, save_to_path: str = None):
    # edge width is proportional to sum of weight
    edgewidth = []
    for u, v in graph.edges():
        weight_sum = 0
        ds = graph.get_edge_data(u, v)
        for d in ds:
            if d == 'att' or d == 'SHARED_MAX':
                continue
            weight_sum += ds[d]
        edgewidth.append(weight_sum)

    edgewidth = np.array(edgewidth) * len(edgewidth) / \
        np.sum(np.array(edgewidth))
    edgewidth = edgewidth.tolist()

    # node size is different for each type of node
    nodesize = [10 for v in graph.nodes()]

    # Generate layout for visualization
    pos = nx.kamada_kawai_layout(graph)
    fig, ax = plt.subplots(figsize=(12, 12))
    # Visualize graph components
    nx.draw_networkx_edges(graph, pos, alpha=0.3,
                           width=edgewidth, edge_color="m")
    nx.draw_networkx_nodes(
        graph, pos, node_size=nodesize, node_color="#210070", alpha=0.9)
    label_options = {"ec": "k", "fc": "white", "alpha": 0.6}
    nx.draw_networkx_labels(
        graph, pos, font_size=9, bbox=label_options)

    fig.tight_layout()
    plt.axis("off")

    if save_to_path is not None:
        path_name = save_to_path if save_to_path.endswith(
            '.png') else save_to_path + '.png'
        plt.savefig(path_name)
    else:
        plt.show()


def save(G: Graph, fname):
    json.dump(dict(nodes=[[n, G.nodes[n]] for n in G.nodes()],
                   edges=[[u, v, G.get_edge_data(u, v)['weight']] for u, v in G.edges()]),
              open(fname, 'w'), indent=2)


def load(fname):
    G = nx.DiGraph()
    d = json.load(open(fname))
    G.add_nodes_from(d['nodes'])
    for edge in d['edges']:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    return G


def extract_cat_from_furniture(furn_info):
    cat_list = []
    for idx in furn_info.nodes:
        if 'cat' in furn_info.nodes[idx].keys():
            if furn_info.nodes[idx]['cat'] in STATIC_CAT:
                continue
            else:
                cat_list.append(furn_info.nodes[idx]['cat'])
    return cat_list


def get_adjective_noun(string):
    if string.find('_'):
        idx = string.find('_')
        noun = string[idx+1: len(string)]
        adjective = string[0: idx]
    else:
        noun = string
        adjective = ''

    return adjective, noun


def high_low_level_cat(cat_list):
    high_level_cat = []
    high_low_level_cat = []
    for cat in cat_list:
        _, cat_adj_removed = get_adjective_noun(cat)
        if cat_adj_removed not in high_level_cat:
            high_level_cat.append(cat_adj_removed)
            high_low_level_cat.append(
                {'high_level': cat_adj_removed, 'low_level': [cat]})
        else:
            for dict in high_low_level_cat:
                if dict['high_level'] == cat_adj_removed and cat not in dict['low_level']:
                    dict['low_level'].append(cat)
    return high_level_cat, high_low_level_cat


def proposal_generation(furniture_graph: Graph, high_level_cat, high_low_level_cat, weight_threshold):
    Proposals = []
    for u, v in furniture_graph.edges():
        if furniture_graph[u][v]['weight'] > weight_threshold and u in high_level_cat and v in high_level_cat:
            index_1 = high_level_cat.index(u)
            index_2 = high_level_cat.index(v)
            u_low_cats = high_low_level_cat[index_1]['low_level']
            v_low_cats = high_low_level_cat[index_2]['low_level']
            for u_low_cat in u_low_cats:
                for v_low_cat in v_low_cats:
                    Proposals.append(
                        [v_low_cat, u_low_cat, {'WEIGHT': furniture_graph[u][v]['weight']}])

    return Proposals


def calc_dist_in_conceptnet(high_level_cat, depth, breadth, scene_name):
    file_name = scene_name + '_' + \
        str(depth)+'_'+str(breadth)+'.json'
    file_path = os.path.join(
        './buffer/conceptnet/high_level_cat_relation_graph')
    file_list = os.listdir(file_path)
    if file_name in file_list:
        furniture_graph_high_level = load(file_path+'/'+file_name)
    else:
        g = DiGraph()
        for cat in high_level_cat:
            g.add_node(cat, cat=cat)

        full_graph = genConceptNetGraph(
            g, depth=depth, breadth=breadth, limit_first_layer=False)
        furniture_graph_high_level = full_graph.toFurnitureNetwork()
        save(furniture_graph_high_level, file_path+'/'+file_name)

    # plotGraph(furniture_graph_high_level)
    print(furniture_graph_high_level.nodes())
    return furniture_graph_high_level


def refine_proposal(furniture_graph: Graph, high_level_cat, high_low_level_cat, weight_threshold, proposals):
    Proposals = proposals
    connected_nodes = []
    low_level_cat = []
    for dict in high_low_level_cat:
        for cat in dict['low_level']:
            low_level_cat.append(cat)

    for u, v in furniture_graph.edges():
        connected_nodes.append(u)
        connected_nodes.append(v)

    for node in furniture_graph.nodes():
        if node not in connected_nodes:
            index = high_level_cat.index(node)
            node_low = high_low_level_cat[index]['low_level']
            for node1 in node_low:
                for node2 in low_level_cat:
                    if node1 != node2:
                        file_name = str(node1)+'_'+str(node2) + '.json'
                        filepath = os.path.join(
                            './buffer/conceptnet/conceptnet_json')
                        list = os.listdir(filepath)
                        if file_name in list:
                            contentJson = None
                            with open(filepath+'/' + file_name) as f:
                                contentJson = f.readlines()
                                contentJson = json.loads(contentJson[0])
                                Content = contentJson
                        else:
                            Content = requests.get(
                                'https://api.conceptnet.io/query?node=/c/en/'+str(node1)+'&other=/c/en/'+str(node2)).json()
                            with open(filepath+'/' + file_name, 'w') as f:
                                f.writelines(json.dumps(Content))
                        if Content["edges"] is not None:
                            weight = 0
                            for edge in Content['edges']:
                                weight += edge['weight']
                            if weight >= weight_threshold:
                                Proposals.append(
                                    [node1, node2, {'WEIGHT': weight}])

    return Proposals


def identifyFixedObjects(furn_info: DiGraph, ids=[]):
    fixed = []
    if len(ids) == 0:
        for id in furn_info.nodes:
            if furn_info.nodes()[id]['cat'] in FIXED:
                fixed.append(id)
    else:
        for id in ids:
            if furn_info.nodes()[id]['cat'] in FIXED:
                fixed.append(id)
                for ed in furn_info.edges():
                    if id == ed[0]:
                        fixed.append(ed[1])

    for id in furn_info.nodes:
        if find_root(furn_info, id) in fixed and find_root(furn_info, id) != id:
            fixed.append(id)

    return fixed
