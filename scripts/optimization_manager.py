"""Group furniture by preference relations and optimize each group and scene."""

import time
import warnings

import networkx as nx
import numpy as np
import pybullet as p

try:
    from cma.optimization_tools import EvalParallel2
except ImportError:
    from cma.interfaces import EvalParallel as EvalParallel2

from copy import deepcopy

import cma
from concept_net.conceptnet_utilis import (
    calc_dist_in_conceptnet,
    extract_cat_from_furniture,
    high_low_level_cat,
    identifyFixedObjects,
    proposal_generation,
    refine_proposal,
)
from networkx import DiGraph, Graph, MultiDiGraph, stoer_wagner
from sampling.sampling_utilis import (
    AdaptiveSimulatedAnnealing,
    OptimizationConfig,
    OptimizationResult,
    TakeStepManual,
    createSteppingConfigByVariable,
    dumpSceneHistory,
)
from scene_manager.common_utilis import (
    ISOLATED,
    find_root,
    get_room_diagonal,
    in_broken,
    linear,
    sigmoid,
)
from scene_manager.scene_manager2 import DOORS, SceneManager
from scipy.stats import lognorm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def convertCMAResults(res):
    """Adapt CMA's best solution and objective value to the shared result record."""
    xbest = res.xbest
    xfave = res.xfavorite
    hist = [xfave, xbest]
    return OptimizationResult(
        best=xbest, best_eval=res.fbest, scores=[res.fbest], hist=hist, steps=[]
    )


def select_strongest_relations(edges):
    """Select the highest-weight cluster, including sparse and uniform inputs."""
    edges = sorted(edges, key=lambda edge: edge[2]["WEIGHT"])
    if not edges:
        return []
    weights = np.array([edge[2]["WEIGHT"] for edge in edges]).reshape(-1, 1)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Spatial relation weights must be finite")
    unique_count = len(np.unique(weights))
    if unique_count == 1:
        return edges

    # Silhouette scoring requires 2 <= number of labels < number of samples.
    best_count, best_score = None, -np.inf
    for count in range(2, min(10, len(edges) - 1, unique_count) + 1):
        labels = KMeans(n_clusters=count, n_init=10, random_state=0).fit_predict(
            weights
        )
        if not 1 < len(np.unique(labels)) < len(edges):
            continue
        score = silhouette_score(weights, labels)
        if score > best_score:
            best_count, best_score = count, score
    if best_count is None:
        return [edge for edge in edges if edge[2]["WEIGHT"] == weights[-1, 0]]
    labels = GaussianMixture(n_components=best_count, random_state=0).fit_predict(
        weights
    )
    return [edge for edge, label in zip(edges, labels) if label == labels[-1]]


class OptimizationManager(object):
    def __init__(
        self,
        scene_name,
        furniture_info,
        seg_map,
        position_net,
        occurrence_net,
        robot=None,
        use_gui=True,
        write_images=False,
        depth=2,
        breadth=5,
    ) -> None:
        self.scene_name = scene_name
        self.seg_map = seg_map
        self.furniture_info = furniture_info
        self.robot = {} if robot is None else robot
        self.write_images = write_images
        self.use_gui = use_gui
        self.groups = []
        self.opt_config = None
        self.t_config = None
        self.step_size = 1.0
        self.connected_to_bullet = False
        self.depth = depth
        self.breadth = breadth
        self.position_net = position_net
        self.occurrence_net = occurrence_net

        self.createGroupRelation()

    def storeOptimizationStates(self, opt_config: OptimizationConfig):
        self.opt_config = opt_config

    def storeTaskConfig(self, t_config):
        self.t_config = t_config

    def divideSubGroups(self, edge_net: Graph, groups: list):
        """Split groups larger than five when a minimum cut crosses fewer than three edges."""
        need_more_adjustment = False
        more_groups = []
        remove_groups = []
        for group in groups:
            if len(group) <= 5:
                continue
            else:
                ss = stoer_wagner(edge_net.subgraph(group).copy(), weight="WEIGHT")
                if nx.cut_size(edge_net, ss[1][0], ss[1][1]) < 3:
                    more_groups.append(ss[1][0])
                    more_groups.append(ss[1][1])
                    remove_groups.append(group)
                    need_more_adjustment = True

        if need_more_adjustment:
            for re in remove_groups:
                groups.remove(re)
            groups += more_groups
            return self.divideSubGroups(edge_net, groups)

        return groups

    def createGroupRelation(self):
        """Cluster strong relations into groups, then keep supported objects together."""
        con_net = self.createGroupRelationFromConcept()
        pos_weights, _ = self.addSpatialRelation(con_net)

        pos_net = MultiDiGraph()
        pos_net.add_nodes_from(self.furniture_info)
        group_net = MultiDiGraph()
        group_net.add_nodes_from(self.furniture_info)

        top_edges = select_strongest_relations(pos_weights)

        for edge in top_edges:
            pos_net.add_edge(
                edge[0],
                edge[1],
                WEIGHT=edge[2]["WEIGHT"],
                CAT=edge[2]["CAT"],
                TYPE=edge[2]["TYPE"],
            )

        group_edges = [
            ed
            for ed in top_edges
            if not in_broken(
                self.furniture_info.nodes()[ed[0]]["cat"],
                self.furniture_info.nodes()[ed[1]]["cat"],
            )
        ]
        for edge in group_edges:
            group_net.add_edge(
                edge[0],
                edge[1],
                WEIGHT=edge[2]["WEIGHT"],
                CAT=edge[2]["CAT"],
                TYPE=edge[2]["TYPE"],
            )

        self.groups = [
            list(group) for group in nx.connected_components(group_net.to_undirected())
        ]
        self.pos_net = pos_net
        self.adjustGroups4Support()
        self.groups = [gr for gr in self.groups if len(gr) > 0]

    def findDoorSem(self):
        for id in self.furniture_info.nodes:
            if self.furniture_info.nodes()[id]["cat"] in DOORS:
                return id, self.furniture_info.nodes()[id]

        return None, None

    def adjustGroups4Support(self):
        """Move each supported object into its parent's group.

        Furniture edges must form an acyclic support hierarchy with at most one
        parent per object. Repeat until changes propagate through every level.
        """
        ss = self.furniture_info.edges()
        mismatched = False
        for group in self.groups:
            for s in ss:
                if s[0] in group and s[1] in group:
                    pass
                elif s[0] in group and s[1] not in group:
                    group.append(s[1])
                    mismatched = True
                elif s[0] not in group and s[1] in group:
                    group.remove(s[1])
                    mismatched = True
        if mismatched:
            self.adjustGroups4Support()

    def rearrangeGroup4Support(self, group):
        """Map each support root to its descendants within a complete support group."""
        lookup_dict = {i: [] for i in group}

        for i in group:
            rt = find_root(self.furniture_info, i)
            if rt == i:
                pass
            else:
                lookup_dict[rt].append(i)
                lookup_dict.pop(i, None)
        return lookup_dict

    def addSpatialRelation(self, concept_edges: Graph):
        """Score ordered object pairs using learned spatial distributions.

        ConceptNet relations select eligible category pairs. Each spatial score
        is weighted by the stronger of their two directional co-occurrence
        weights. Return individual relation edges and aggregate pair scores.
        """
        manager = SceneManager(
            furn_info=self.furniture_info,
            position_net=self.position_net,
            seg_map=self.seg_map,
            robot=self.robot,
            use_gui=self.use_gui,
        )
        grouping_order = {"use": [n for n in list(self.furniture_info.nodes)]}
        opt_config = OptimizationConfig(
            scene_name="temp",
            robot="temp",
            grouping_order=grouping_order,
            randomize=False,
            desired_sdf_res=0.1,
            write_images=False,
            padding=2,
        )
        manager.setupOptimizationConfig(opt_config)

        manager.createCompleteSDFBuffer()
        this_pos = manager.evaluatePairwiseRelation()

        final_pos = DiGraph()
        final_pos.add_nodes_from(self.furniture_info)

        edges = []

        room_diagonal = get_room_diagonal(self.seg_map)

        for id1 in self.furniture_info.nodes():
            for id2 in self.furniture_info.nodes():
                if id1 == id2:
                    continue
                cat1 = self.furniture_info.nodes()[id1]["cat"].replace("_", " ")
                cat2 = self.furniture_info.nodes()[id2]["cat"].replace("_", " ")

                if (
                    not manager.isInterrupted(id2, id1)
                    and self.position_net.has_edge(cat1, cat2)
                    and concept_edges.has_edge(
                        self.furniture_info.nodes()[id1]["cat"],
                        self.furniture_info.nodes()[id2]["cat"],
                    )
                    and self.furniture_info.nodes()[id1]["cat"] not in ISOLATED
                    and self.furniture_info.nodes()[id2]["cat"] not in ISOLATED
                ):

                    if cat2 not in self.position_net[cat1]:
                        continue
                    FRONTOF = sigmoid(
                        lognorm.pdf(
                            this_pos[id1][id2]["FRONTOF"],
                            *self.position_net[cat1][cat2]["FRONTOF"],
                        )
                    )
                    FACING = sigmoid(
                        lognorm.pdf(
                            this_pos[id1][id2]["FACING"],
                            *self.position_net[cat1][cat2]["FACING"],
                        )
                    )
                    PARALLEL = sigmoid(
                        lognorm.pdf(
                            this_pos[id1][id2]["PARALLEL"],
                            *self.position_net[cat1][cat2]["PARALLEL"],
                        )
                    )

                    FACINGBOUND = sigmoid(
                        lognorm.pdf(
                            this_pos[id1][id2]["FACINGBOUND"],
                            *self.position_net[cat1][cat2]["FACINGBOUND"],
                        )
                    )
                    CLOSETOBOUND = linear(
                        this_pos[id1][id2]["CLOSETOBOUND"] / room_diagonal, 1.0
                    ) * sigmoid(
                        10
                        * lognorm.pdf(
                            this_pos[id1][id2]["CLOSETOBOUND"],
                            *self.position_net[cat1][cat2]["CLOSETOBOUND"],
                        )
                    )

                    use_co = np.maximum(
                        self.occurrence_net[cat1][cat2]["CO"],
                        self.occurrence_net[cat2][cat1]["CO"],
                    )

                    final_pos.add_edge(
                        id1,
                        id2,
                        WEIGHT=(FACINGBOUND + FRONTOF + PARALLEL + CLOSETOBOUND)
                        * use_co,
                        CAT=f"{cat1} + {cat2}",
                    )

                    edges.append(
                        (
                            id1,
                            id2,
                            {
                                "TYPE": "CLOSETOBOUND",
                                "CAT": f"{cat1} + {cat2}",
                                "WEIGHT": CLOSETOBOUND * use_co,
                            },
                        )
                    )
                    edges.append(
                        (
                            id1,
                            id2,
                            {
                                "TYPE": "FRONTOF",
                                "CAT": f"{cat1} + {cat2}",
                                "WEIGHT": FRONTOF * use_co,
                            },
                        )
                    )
                    if FACINGBOUND > FACING:
                        edges.append(
                            (
                                id1,
                                id2,
                                {
                                    "TYPE": "FACINGBOUND",
                                    "CAT": f"{cat1} + {cat2}",
                                    "WEIGHT": FACINGBOUND * use_co,
                                },
                            )
                        )
                    else:
                        edges.append(
                            (
                                id1,
                                id2,
                                {
                                    "TYPE": "FACING",
                                    "CAT": f"{cat1} + {cat2}",
                                    "WEIGHT": FACING * use_co,
                                },
                            )
                        )
                    edges.append(
                        (
                            id1,
                            id2,
                            {
                                "TYPE": "PARALLEL",
                                "CAT": f"{cat1} + {cat2}",
                                "WEIGHT": PARALLEL * use_co,
                            },
                        )
                    )

        return edges, final_pos

    def createGroupRelationFromConcept(self):
        depth = self.depth
        breadth = self.breadth
        weight_threshold = 0.0
        cat_list = extract_cat_from_furniture(self.furniture_info)
        high_level, high_low_level = high_low_level_cat(cat_list)
        furniture_graph = calc_dist_in_conceptnet(
            high_level, depth, breadth, self.scene_name
        )
        Proposals = proposal_generation(
            furniture_graph, high_level, high_low_level, weight_threshold
        )
        Proposals = refine_proposal(
            furniture_graph, high_level, high_low_level, weight_threshold, Proposals
        )
        final_concept = Graph()
        final_concept.add_edges_from(Proposals)
        return final_concept

    def adoptState(self):
        for id in self.manager.furniture_info.nodes():
            self.furniture_info.nodes()[id][
                "pose"
            ] = self.manager.furniture_info.nodes()[id]["pose"]

    def solve(self):
        raise NotImplementedError

    def getCentrolNode(self, group):
        cen = np.zeros((3))
        for id in group:
            cen += np.array(self.furniture_info.nodes()[id]["pose"][0])
        cen /= len(group)
        min_id = -1
        min_dist = 1000
        for id in group:
            dist = np.linalg.norm(
                cen - np.array(self.furniture_info.nodes()[id]["pose"][0])
            )
            if dist < min_dist:
                min_id = id
                min_dist = dist
        return min_id

    def genreateSubgroupProblem(self, group=None, objectives=None):
        """Build a movable subgroup, or the whole scene when group is empty.

        Return None when no stage is needed. A singleton skips the subgroup
        stage but can still move during whole-scene optimization. The method
        retains its legacy spelling for existing callers.
        """
        group = [] if group is None else list(group)
        if len(group) == 1:
            self.manager = None
            return None

        fixed_ids = identifyFixedObjects(self.furniture_info, [])
        door_id, _ = self.findDoorSem()
        if door_id is not None and door_id not in fixed_ids:
            fixed_ids.append(door_id)
        movable = [node for node in group if node not in fixed_ids]

        if group:
            grouping_order = self.rearrangeGroup4Support(movable)
        else:
            grouping_order = {}
            for functional_group in self.groups:
                members = [node for node in functional_group if node not in fixed_ids]
                if members:
                    center = self.getCentrolNode(members)
                    grouping_order[center] = [
                        node for node in members if node != center
                    ]

        if not grouping_order:
            self.manager = None
            return None
        grouping_order["fix"] = fixed_ids

        # Preserve multigraph edge keys when consuming subgroup relations.
        selected_edges = [
            (source, target, key, data)
            for source, target, key, data in self.pos_net.edges(keys=True, data=True)
            if group and source in group and target in group
        ]
        relations = [
            (source, target, data["TYPE"]) for source, target, _, data in selected_edges
        ]

        if p.isConnected():
            p.disconnect()
        self.manager = SceneManager(
            furn_info=self.furniture_info,
            position_net=self.position_net,
            seg_map=self.seg_map,
            robot=self.robot,
            use_gui=self.use_gui,
        )
        self.connected_to_bullet = True
        opt_config = deepcopy(self.opt_config)
        opt_config.objectives = [] if objectives is None else list(objectives)
        opt_config.randomize = False
        opt_config.grouping_order = grouping_order
        self.manager.setOptimizeRelations(relations)
        self.states, self.bounds = self.manager.setupOptimizationConfig(opt_config)
        self.manager.setTaskConfig(self.t_config)
        self.manager.createCompleteSDFBuffer()
        self.pos_net.remove_edges_from(
            (source, target, key) for source, target, key, _ in selected_edges
        )
        return opt_config

    def genreateLastProblem(self, objectives=None):
        """Optimize the whole scene with each functional group kept rigid."""
        return self.genreateSubgroupProblem(objectives=objectives)


class LayeredManager(OptimizationManager):
    """Optimize each functional group, then move the groups within the room."""

    def _optimize_current_problem(self, opt_config, group_index, cma_iterations):
        """Run annealing and CMA, then adopt and save the better final state."""
        stepping_config = createSteppingConfigByVariable(
            self.states,
            self.step_size,
            bounds=self.bounds,
        )
        take_step = TakeStepManual(processFun=None, stepping_config=stepping_config)
        pipeline_name = f"{self.scene_name}_layered_{group_index}"
        optimizer = AdaptiveSimulatedAnnealing()
        optimizer.setCustomOptimizationConfiguration(opt_config)
        optimizer.setCustomTakeStepFunction(take_step)
        optimizer.setCustomTasksConfiguration(self.t_config)
        optimizer.setPackagingFunction(self.manager.repackage)
        optimizer.setPipelineName(pipeline_name)
        annealing = optimizer(self.manager.objectiveFunction, self.states)
        annealing_cost = self.manager.objectiveFunction(annealing.best)

        options = cma.CMAOptions()
        options["tolfun"] = 1e-3
        options["tolx"] = 1e-3
        seed = getattr(opt_config, "seed", None)
        if seed is not None:
            # CMA uses zero to request a time-based seed.
            options["seed"] = seed or 1
        strategy = cma.CMAEvolutionStrategy(annealing.best, 0.05, options)
        strategy, history_states = cmaes_optimize(
            strategy,
            self.manager.objectiveFunction,
            iterations=cma_iterations,
        )
        cma_best = strategy.result.xbest
        cma_cost = (
            self.manager.objectiveFunction(cma_best)
            if cma_best is not None and np.all(np.isfinite(cma_best))
            else np.inf
        )
        use_cma = np.isfinite(cma_cost) and cma_cost <= annealing_cost
        best = cma_best if use_cma else annealing.best

        history = []
        for state in history_states:
            self.manager.objectiveFunction(state)
            history.append(deepcopy(self.manager.repackage()))
        # Evaluating history states mutates the scene; restore the selected state
        # before saving the final snapshot and adopting its furniture poses.
        self.manager.objectiveFunction(best)
        history.append(deepcopy(self.manager.repackage()))
        history_states.append(np.array(best, copy=True))
        dumpSceneHistory(
            history_states,
            opt_config,
            stepping_config,
            self.t_config,
            history,
            f"./results/{pipeline_name}_cma",
        )
        self.adoptState()

    def solve(self):
        if self.opt_config is None or self.t_config is None:
            raise ValueError(
                "Store optimization and task configurations before solving"
            )
        objectives = ["collision", "interaction", "anti", "planning", "relation"]
        start_time = time.time()
        for index, group in enumerate(self.groups):
            opt_config = self.genreateSubgroupProblem(group, objectives)
            if opt_config is not None:
                self._optimize_current_problem(opt_config, index, cma_iterations=200)

        opt_config = self.genreateLastProblem(objectives)
        if opt_config is not None:
            self._optimize_current_problem(opt_config, -1, cma_iterations=400)
        print(f"--- {time.time() - start_time:.2f} seconds ---")
        return self.furniture_info, self.seg_map


def cmaes_optimize(
    cmaes: cma.CMAEvolutionStrategy,
    objective_fct,
    maxfun=None,
    iterations=None,
    min_iterations=1,
    args=(),
    verb_disp=None,
    n_jobs=0,
    **kwargs,
):
    """Run CMA generations and snapshot the best state every ten generations.

    ``n_jobs=0`` evaluates sequentially; use it for stateful scene objectives.
    ``maxfun`` is checked between generations, so a population can exceed it.
    ``min_iterations`` takes precedence over an earlier CMA stop or iteration cap.
    """
    if kwargs:
        message = "ignoring unknown argument%s %s in cmaes_optimize" % (
            "s" if len(kwargs) > 1 else "",
            str(kwargs),
        )
        warnings.warn(message)

    if iterations is not None and min_iterations > iterations:
        warnings.warn(
            "doing min_iterations = %d > %d = iterations" % (min_iterations, iterations)
        )
        iterations = min_iterations

    citer, cevals = 0, 0
    hist = []
    with EvalParallel2(objective_fct, None if n_jobs == -1 else n_jobs) as eval_all:
        while not cmaes.stop() or citer < min_iterations:
            if (maxfun and cevals >= maxfun) or (iterations and citer >= iterations):
                return cmaes, hist
            citer += 1

            X = cmaes.ask()
            fitvals = eval_all(X, args=args)
            cevals += len(fitvals)
            cmaes.tell(X, fitvals)
            cmaes.disp(verb_disp)

            if citer % 10 == 0:
                hist.append(deepcopy(cmaes.result.xbest))

    if verb_disp:
        cmaes.disp(1)
        print("termination by", cmaes.stop())
        print("best f-value =", cmaes.result[1])
        print("solution =", cmaes.result[0])

    return cmaes, hist
