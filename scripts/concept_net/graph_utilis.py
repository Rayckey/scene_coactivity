#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
import os
import networkx as nx
from networkx import DiGraph, Graph, MultiGraph, MultiDiGraph
import matplotlib.pyplot as plt
import json
import requests
import yaml

with open('./configs/relations.yaml', 'r') as file:
    relations = yaml.safe_load(file)
    VALID_RELATIONS = relations['VALID_RELATIONS']
    NOT_VALID_RELATIONS = relations['NOT_VALID_RELATIONS']


def find_root(G: MultiDiGraph, child):
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        print(f"found root: {child}")
        return child
    else:
        return find_root(G, parent[0])


class ConceptGraph(object):
    def __init__(self, depth=2, breadth=15) -> None:
        self.depth = depth
        self.breadth = breadth
        self.graph = MultiGraph()

    @staticmethod
    def isCompleted(pending_nodes):
        lens = 0
        for pd in pending_nodes:
            lens += len(pd)
        return lens == 0

    def populateGraph(self, thing, limit_first_layer):
        pending_nodes = [[] for idx in range(self.depth)]
        finished = False
        current_thing = thing
        current_depth = -1
        while not finished:
            new_nodes = self.addChildren(current_thing, limit_first_layer)

            if current_depth < self.depth-1:
                pending_nodes[current_depth + 1] += new_nodes
                pending_nodes[current_depth +
                              1] = list(dict.fromkeys(pending_nodes[current_depth + 1]))

            if current_depth == -1 or len(pending_nodes[current_depth]) == 0:
                current_depth += 1

            try:
                current_thing = pending_nodes[current_depth].pop()
            except:
                finished = True

            finished = ConceptGraph.isCompleted(pending_nodes)

    def addChildren(self, thing, limit_first_layer):
        if thing not in self.graph.nodes():
            self.graph.add_node(thing, node_type='furniture')
        else:
            self.graph.nodes()[thing]['node_type'] = 'furniture'
        jason = ConceptNet.loookupNode(thing=thing)
        _edges = ConceptNet.filterEdgeCredibility(jason['edges'])
        _edges = ConceptNet.filterRedundantEdge(_edges, thing)
        _edges = ConceptNet.filterIsAEdge(_edges)
        if limit_first_layer:
            edges = ConceptNet.filterEdgeValidRelations(_edges)
        else:
            edges = deepcopy(_edges)

        while len(edges) < self.breadth and 'view' in jason and 'nextPage' in jason['view']:
            jason = ConceptNet.query4More(jason)
            _edges = ConceptNet.filterEdgeCredibility(jason['edges'])
            _edges = ConceptNet.filterRedundantEdge(_edges, thing)
            _edges = ConceptNet.filterIsAEdge(_edges)
            # edges += ConceptNet.filterEdgeValidRelations(_edges)
            # if len(_edges) >= self.breadth:
            #     edges += _edges[:self.breadth]
            # else:
            edges += _edges

        if len(edges) > self.breadth:
            edges = edges[:self.breadth]

        new_nodes = []
        for edge in edges:
            relation, weight, tail, sources = ConceptGraph.extractEdgeInfo(
                edge)

            if tail not in self.graph.nodes():
                self.graph.add_node(tail, node_type='com')

            if tail not in new_nodes:
                new_nodes.append(tail)

            if thing != tail:
                self.graph.add_edge(thing, tail, relation,
                                    weight=weight, sources=sources)

        return new_nodes

    def toFurnitureNetwork(self):

        ng = Graph()
        ng.add_nodes_from(self.graph)

        for u, v in self.graph.edges():
            weight_sum = 0
            ds = self.graph.get_edge_data(u, v)
            for d in ds:
                weight_sum += ds[d]['weight']
            if weight_sum == 0:
                continue
            ng.add_edge(u, v, weight=1./weight_sum)

        g = Graph()
        g.add_nodes_from([n for n in self.graph.nodes() if self.graph.nodes()[
                         n]["node_type"] == 'furniture'])
        # weights will act like springs, parallel is add, series is reciprocal

        for furn1 in g.nodes():
            for furn2 in g.nodes():
                if furn1 == furn2:
                    continue
                # paths = nx.all_simple_paths(ng, furn1, furn2)

                if nx.has_path(ng, furn1, furn2):
                    paths = nx.all_shortest_paths(
                        ng, furn1, furn2, weight='weight')
                    # paths = nx.all_simple_paths(ng, furn1, furn2)
                    weight_sum = 0
                    for path in paths:
                        weight_sum += 1.0 / \
                            nx.path_weight(ng, path=path, weight='weight')

                    if weight_sum == 0:
                        continue
                    g.add_edge(furn1, furn2, weight=weight_sum)

        return g

    @staticmethod
    def extractEdgeInfo(edge):
        relation = ConceptNet.extractEdgeRelationLabel(edge)
        weight = ConceptNet.extractEdgeWeight(edge)
        tail = ConceptNet.extractEdgeEndLabel(edge)
        sources = ConceptNet.extractEdgeSources(edge)
        return relation, weight, tail, sources

    def trimSingleDegreeNodes(self):
        to_be_removed = [
            x for x in self.graph.nodes() if self.graph.degree(x) <= 1]
        for x in to_be_removed:
            self.graph.remove_node(x)

    def plot(self, save_to_path: str = None):
        # edge width is proportional to sum of weight
        edgewidth = []
        for u, v in self.graph.edges():
            weight_sum = 0
            ds = self.graph.get_edge_data(u, v)
            for d in ds:
                weight_sum += ds[d]['weight']
            edgewidth.append(weight_sum)

        edgewidth = np.array(edgewidth) * len(edgewidth) / \
            np.sum(np.array(edgewidth))
        edgewidth = edgewidth.tolist()

        # node size is different for each type of node
        nodesize = [len(self.graph.nodes()[v]['node_type'])
                    ** 3 for v in self.graph.nodes()]

        # Generate layout for visualization
        pos = nx.kamada_kawai_layout(self.graph)
        fig, ax = plt.subplots(figsize=(12, 12))
        # Visualize graph components
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3,
                               width=edgewidth, edge_color="m")
        nx.draw_networkx_nodes(
            self.graph, pos, node_size=nodesize, node_color="#210070", alpha=0.9)
        label_options = {"ec": "k", "fc": "white", "alpha": 0.6}
        nx.draw_networkx_labels(
            self.graph, pos, font_size=9, bbox=label_options)

        fig.tight_layout()
        plt.axis("off")

        if save_to_path is not None:
            path_name = save_to_path if save_to_path.endswith(
                '.png') else save_to_path + '.png'
            plt.imsave(path_name)
        plt.show()


class ConceptNet(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def plotGraph(graph, save_to_path: str = None):
        # edge width is proportional to sum of weight
        edgewidth = []
        for u, v in graph.edges():
            ds = graph.get_edge_data(u, v)['weight']
            edgewidth.append(ds)

        # node size is different for each type of node
        nodesize = [30 for v in graph.nodes()]

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
            plt.imsave(path_name)
        plt.show()

    @staticmethod
    def filterEdgeCredibility(edges):
        new_edges = {}
        for edge in edges:
            if edge['@id'] in new_edges and ConceptNet.extractEdgeSources(edge) < ConceptNet.extractEdgeSources(new_edges[edge['@id']]):
                continue
            new_edges[edge['@id']] = edge

        return list(new_edges.values())

    @staticmethod
    def filterEdgeValidRelations(edges):
        new_edges = {}
        for edge in edges:
            if edge['rel']['label'] in VALID_RELATIONS:
                new_edges[edge['@id']] = edge

        return list(new_edges.values())

    @staticmethod
    def filterRedundantEdge(edges, name):
        new_edges = {}
        for edge in edges:
            if edge['rel']['label'] != name:
                new_edges[edge['@id']] = edge

        return list(new_edges.values())

    def filterIsAEdge(edges):
        new_edges = {}
        for edge in edges:
            okay = True
            for not_valid in NOT_VALID_RELATIONS:
                if not_valid in edge['rel']['@id']:
                    okay = False
            if okay:
                new_edges[edge['@id']] = edge

        return list(new_edges.values())

    @staticmethod
    def extractEdgeRelationLabel(edge):
        if edge['rel']['@type'] == 'Relation':
            return edge['rel']['label']
        return None

    @staticmethod
    def extractEdgeRelation(edge):
        if edge['rel']['@type'] == 'Relation':
            return edge['rel']['@id']
        return None

    @staticmethod
    def extractEdgeWeight(edge):
        if edge['rel']['@type'] == 'Relation':
            return edge['weight']
        return None

    @staticmethod
    def extractEdgeEndLabel(edge):
        if edge['rel']['@type'] == 'Relation':
            return edge['end']['label']
        return None

    @staticmethod
    def extractEdgeEnd(edge):
        if edge['rel']['@type'] == 'Relation':
            return edge['end']['@id']
        return None

    @staticmethod
    def extractEdgeSources(edge):
        if edge['rel']['@type'] == 'Relation':
            return len(edge['sources'])
        return None

    @staticmethod
    def removewired(string: str):
        string_new = ''
        parts = string.split('_')
        for part in parts:
            if part == 'a' or part == 'an' or part == 'the' or part == 'of' or part == 'piece' or part == 'any' or part == 'you' or part == 'your':
                continue
            string_new += part + '_'
        string_new = string_new[0: len(string_new)-1]
        return string_new

    @staticmethod
    def loookupNode(thing: str):
        file_name = ConceptNet.removewired(thing.replace(' ', '_')) + '.json'
        filepath = os.path.join('./buffer/conceptnet/conceptnet_json_2')
        list = os.listdir(filepath)
        if file_name in list:
            contentJson = None
            with open(filepath+'/' + file_name) as f:
                contentJson = f.readlines()
                contentJson = json.loads(contentJson[0])
                Content = contentJson
        else:
            Content = requests.get('https://api.conceptnet.io/query?node=/c/en/' +
                                   ConceptNet.removewired(thing.replace(' ', '_'))).json()
            with open(filepath+'/' + file_name, 'w') as f:
                f.writelines(json.dumps(Content))
        return Content

    @staticmethod
    def query4More(jason):
        file_name = jason['view']['nextPage'] + '.json'
        filepath = os.path.join('./buffer/conceptnet/conceptnet_json_2')
        list = os.listdir(filepath)
        if file_name in list:
            contentJson = None
            with open(filepath+'/' + file_name) as f:
                contentJson = f.readlines()
                contentJson = json.loads(contentJson[0])
                Content = contentJson
        else:
            Content = requests.get(
                f"http://api.conceptnet.io/{jason['view']['nextPage']}").json()
            # with open(filepath+'/' + file_name, 'w+') as f:
            #     f.writelines(json.dumps(Content))
        return Content

    @staticmethod
    def convert2WeightDict(res: list, edge_type: str = None):
        edges_weight = {}
        edges_sources = {}
        for ed in res['edges']:
            relation = ConceptNet.extractEdgeRelationLabel(ed)
            if edge_type is None or edge_type == relation:
                if relation not in edges_weight:
                    edges_weight[relation] = ConceptNet.extractEdgeWeight(ed)
                    edges_sources[relation] = ConceptNet.extractEdgeSources(ed)
                elif edges_sources[relation] < ConceptNet.extractEdgeSources(ed):
                    edges_weight[relation] = ConceptNet.extractEdgeWeight(ed)
                    edges_sources[relation] = ConceptNet.extractEdgeSources(ed)

        return edges_weight, edges_sources

    @staticmethod
    def convert2EndDict(res: list, edge_type: str = None):
        edges_weight = {}
        edges_sources = {}
        edges_ends = {}
        for ed in res['edges']:
            relation = ConceptNet.extractEdgeRelationLabel(ed)
            if edge_type is None or edge_type == relation:
                if relation not in edges_weight:
                    edges_weight[relation] = ConceptNet.extractEdgeWeight(ed)
                    edges_sources[relation] = ConceptNet.extractEdgeSources(ed)
                    edges_ends[relation] = ConceptNet.extractEdgeEndLabel(ed)
                elif edges_sources[relation] < ConceptNet.extractEdgeSources(ed):
                    edges_weight[relation] = ConceptNet.extractEdgeWeight(ed)
                    edges_sources[relation] = ConceptNet.extractEdgeSources(ed)
                    edges_ends[relation] = ConceptNet.extractEdgeEndLabel(ed)

        return edges_weight, edges_sources, edges_ends

    @staticmethod
    def loookupNodeRelation(thing: str, edge_type: str = None):
        return requests.get(f"http://api.conceptnet.io/r/{edge_type}/c/en/{thing.replace(' ', '_')}").json()

    @staticmethod
    def loookupRelations(thing1: str, thing2: str, edge_type: str = None):
        name1 = thing1.lower()
        name2 = thing2.lower()
        res = requests.get(
            f"https://api.conceptnet.io/query?node=/c/en/{name1.replace(' ', '_')}&other=/c/en/{name2.replace(' ', '_')}").json()

        edges_weight, edges_sources = ConceptNet.convert2WeightDict(
            res, edge_type)

        return edges_weight

    @staticmethod
    def lookupCommonType(thing1: str, thing2: str, common_thing: str):
        name1 = thing1.lower()
        name2 = thing2.lower()
        res1 = ConceptNet.loookupNodeRelation(name1, edge_type=common_thing)
        res2 = ConceptNet.loookupNodeRelation(name2, edge_type=common_thing)

        edges_weight1, edges_sources1 = ConceptNet.convert2WeightDict(res1)
        edges_weight2, edges_sources2 = ConceptNet.convert2WeightDict(res2)

        shared_common = [a for a in edges_weight1 if a in edges_weight2]
        shared_edges_weight = {}

        for a in shared_common:
            shared_edges_weight[a] = edges_weight1[a] * edges_weight2[a]

        return shared_edges_weight

    @staticmethod
    def lookupTrivialTags(thing1: str, thing2: str, query_relation: str):
        # I don't know if this is actually worth it
        name1 = thing1.lower()
        name2 = thing2.lower()
        res1 = ConceptNet.loookupNodeRelation(name1, edge_type=query_relation)

        edges_weight, edges_sources, edges_ends = ConceptNet.convert2EndDict(
            res1)

        related_weight = {}
        for r, e in edges_ends.items():
            if name2 in e:
                related_weight[r] = edges_weight[r]

        return related_weight


def genConceptNetGraph(g: DiGraph, depth=2, breadth=15, limit_first_layer=False):

    ccg = ConceptGraph(depth=depth, breadth=breadth)
    # for idx in range(len(g.nodes())

    unique_cats = [g.nodes()[id1]['cat'].replace("_", " ")
                   for id1 in g.nodes()]
    unique_cats = list(dict.fromkeys(unique_cats))

    for cat in unique_cats:
        ccg.populateGraph(cat, limit_first_layer=limit_first_layer)
        print(f"Completed node for {cat} ")

    return ccg


def findEndNodes(g: DiGraph):
    return list(np.where([x == 0 for x in g.degree(mode='out')])[0])


def findRootNodes(g: DiGraph):
    return list(np.where([x == 0 for x in g.degree(mode='in')])[0])


def findCentralFurniture(g: DiGraph):
    # idx = findEndNodes(g)
    return findEndNodes(g)


def isEndNode(g: DiGraph, idx: int):
    return g.degree(idx, mode='out') == 0


