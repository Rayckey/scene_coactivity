import numpy as np
import warnings
try: from cma.optimization_tools import EvalParallel2
except: EvalParallel2 = None
from cma.interfaces import EvalParallel

from copy import deepcopy
from networkx import stoer_wagner
from itertools import compress
from scipy.stats import lognorm, norm, truncnorm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sampling.sampling_utilis import *
from concept_net.graph_utilis import *
from concept_net.conceptnet_utilis import identifyFixedObjects
from scene_manager.scene_manager2 import *
from concept_net.conceptnet_utilis import extract_cat_from_furniture, high_low_level_cat, calc_dist_in_conceptnet, proposal_generation, refine_proposal
from scene_manager.common_utilis import *
import cma


def convertCMAResults(res):
    xbest = res.xbest
    xfave = res.xfavorite
    scores = [res.evals_best]
    hist = [xfave, xbest]
    return OptimizationResult(best=xbest, best_eval=res.evals_best, scores=scores, hist=hist, steps=[])


class OptimizationManager(object):
    def __init__(self, scene_name, furniture_info, seg_map, position_net, occurrence_net, robot={}, use_gui=True, write_images=False, depth=2, breadth = 5) -> None:
        self.scene_name = scene_name
        self.seg_map = seg_map
        self.furniture_info = furniture_info
        self.robot = robot
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
        need_more_adjustment = False
        more_groups = []
        remove_groups = []
        # highest_weight = sorted(list(nx.get_edge_attributes(edge_net, 'WEIGHT').items()), key=lambda x: x[1])[-1][1]
        for group in groups:
            if len(group) <=5:
                continue
            else:
                ss = stoer_wagner(edge_net.subgraph(group).copy(), weight = 'WEIGHT')
                # if ss[0] <2:
                # if ss[0] < highest_weight:
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
        con_net = self.createGroupRelationFromConcept()
        # sorted_edges = sorted(list(con_edge.items()), key=lambda x: x[1])
        # # cat_list = extract_cat_from_furniture(self.furniture_info)
        # top_edges = sorted_edges[-len(list(self.furniture_info.nodes)):-1]
        # con_net = Graph()
        # for edge in top_edges:
        #     con_net.add_edge(edge[0][0], edge[0][1], WEIGHT=edge[1])

        pos_weights, pos_sum = self.addSpatialRelation(con_net)

        expected_num_pairs = len([id for id in self.furniture_info.nodes if id not in ISOLATED])

        pos_net = MultiDiGraph()
        pos_net.add_nodes_from(self.furniture_info)
        group_net = MultiDiGraph()
        group_net.add_nodes_from(self.furniture_info)

        # sort by weight, then pick the top cluster
        sorted_edges = sorted(pos_weights, key=lambda x: x[2]['WEIGHT'])
        # top_edges = sorted_edges[-expected_num_pairs*4:]

        data_ = [x[2]['WEIGHT'] for x in sorted_edges]
        X = np.array(data_).reshape(-1,1)

        # find the best number of clusters
        best_n_cluster = 0
        best_fit_score = 0
        for n_cluster in range(2,11,1):
            model = KMeans(n_cluster)
            model.fit(X)
            labels = model.predict(X)
            score = silhouette_score(X, labels)
            if score > best_fit_score:
                best_n_cluster = n_cluster
                best_fit_score = score 

        model = GaussianMixture(n_components=best_n_cluster)
        model.fit(X)
        prediction = model.predict(X)
        top_edges = list(compress(sorted_edges, [m==prediction[-1] for m in prediction]))

        # or thresholding
        # top_edges = [ed for ed in top_edges if ed[2]['WEIGHT']>0.5]

        for edge in top_edges:
            pos_net.add_edge(edge[0], edge[1], WEIGHT=edge[2]['WEIGHT'], CAT=edge[2]['CAT'], TYPE = edge[2]['TYPE'])

        group_edges = [ed for ed in top_edges if not in_broken(self.furniture_info.nodes()[ed[0]]['cat'], self.furniture_info.nodes()[ed[1]]['cat'])]
        # group_edges = top_edges
        for edge in group_edges:
            group_net.add_edge(edge[0], edge[1], WEIGHT=edge[2]['WEIGHT'], CAT=edge[2]['CAT'], TYPE = edge[2]['TYPE'])

        groups_gen = nx.connected_components(group_net.to_undirected())

        groups = []
        for gg in groups_gen:
            groups.append(list(gg))
                
        self.groups = groups
        self.pos_net = pos_net
        self.adjustGroups4Support()
        self.groups = [gr for gr in self.groups if len(gr)>0 ]
        # self.edge_net = edge_net


    def findDoorSem(self):
        for id in self.furniture_info.nodes:
            if self.furniture_info.nodes()[id]['cat'] in DOORS:
                return id, self.furniture_info.nodes()[id]
        
        return None, None

    def adjustGroups4Support(self):
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
        ss = self.furniture_info.edges()
        lookup_dict = { i : [] for i in group }

        for i in group:
            rt = find_root(self.furniture_info, i)
            if rt == i:
                pass
            else:
                lookup_dict[rt].append(i)
                lookup_dict.pop(i, None)
        return lookup_dict

    def addSpatialRelation(self, concept_edges: Graph):
        manager = SceneManager(furn_info=self.furniture_info, position_net=self.position_net,
                               seg_map=self.seg_map, robot=self.robot, use_gui=self.use_gui)
        grouping_order = {'use': [n for n in list(self.furniture_info.nodes)]}
        opt_config = OptimizationConfig(scene_name=f"temp", robot="temp", grouping_order=grouping_order,
                                        randomize=False, desired_sdf_res=0.1, write_images=False, padding=2)
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
                cat1 = self.furniture_info.nodes()[
                    id1]['cat'].replace("_", " ")
                cat2 = self.furniture_info.nodes()[
                    id2]['cat'].replace("_", " ")

                

                if not manager.isInterrupted(id2, id1) and self.position_net.has_edge(cat1, cat2) and concept_edges.has_edge(self.furniture_info.nodes()[id1]['cat'], self.furniture_info.nodes()[id2]['cat']) and self.furniture_info.nodes()[id1]['cat'] not in ISOLATED and self.furniture_info.nodes()[id2]['cat'] not in ISOLATED:
                        
                    if cat2 not in self.position_net[cat1]:
                        continue
                    FRONTOF =  sigmoid(lognorm.pdf(this_pos[id1][id2]['FRONTOF'], *self.position_net[cat1][cat2]['FRONTOF']) )
                    FACING =  sigmoid(lognorm.pdf(this_pos[id1][id2]['FACING'], *self.position_net[cat1][cat2]['FACING']) )
                    PARALLEL =  sigmoid(lognorm.pdf(this_pos[id1][id2]['PARALLEL'], *self.position_net[cat1][cat2]['PARALLEL']) )

                    FACINGBOUND =  sigmoid(lognorm.pdf(this_pos[id1][id2]['FACINGBOUND'], *self.position_net[cat1][cat2]['FACINGBOUND']) )
                    CLOSETOBOUND =  linear(this_pos[id1][id2]['CLOSETOBOUND']/room_diagonal, 1.0) * sigmoid(10*lognorm.pdf(this_pos[id1][id2]['CLOSETOBOUND'], *self.position_net[cat1][cat2]['CLOSETOBOUND']) )

                    use_co = np.maximum(self.occurrence_net[cat1][cat2]['CO'], self.occurrence_net[cat2][cat1]['CO'])

                    final_pos.add_edge(id1, id2, WEIGHT=(FACINGBOUND+ FRONTOF+
                                        PARALLEL+CLOSETOBOUND)* use_co, CAT = f'{cat1} + {cat2}')

                    edges.append((id1, id2, {'TYPE': 'CLOSETOBOUND',  'CAT': f'{cat1} + {cat2}','WEIGHT': CLOSETOBOUND* use_co}))
                    edges.append((id1, id2, {'TYPE': 'FRONTOF',  'CAT': f'{cat1} + {cat2}','WEIGHT': FRONTOF* use_co}))
                    if FACINGBOUND > FACING:
                        edges.append((id1, id2, {'TYPE': 'FACINGBOUND',   'CAT': f'{cat1} + {cat2}','WEIGHT': FACINGBOUND* use_co}))
                    else:
                        edges.append((id1, id2, {'TYPE': 'FACING',   'CAT': f'{cat1} + {cat2}','WEIGHT': FACING* use_co}))
                    edges.append((id1, id2, {'TYPE': 'PARALLEL', 'CAT': f'{cat1} + {cat2}','WEIGHT': PARALLEL* use_co}))

        return edges, final_pos

    def createGroupRelationFromConcept(self):
        depth = self.depth
        breadth = self.breadth
        weight_threshold = 0.0
        cat_list = extract_cat_from_furniture(self.furniture_info)
        high_level, high_low_level = high_low_level_cat(cat_list)
        furniture_graph = calc_dist_in_conceptnet(
            high_level, depth, breadth, self.scene_name)
        Proposals = proposal_generation(
            furniture_graph, high_level, high_low_level, weight_threshold)
        Proposals = refine_proposal(
            furniture_graph, high_level, high_low_level, weight_threshold, Proposals)
        final_concept = Graph()
        final_concept.add_edges_from(Proposals)
        # return nx.get_edge_attributes(final_concept, 'WEIGHT')
        return final_concept

    def adoptState(self):
        for id in self.manager.furniture_info.nodes():
            self.furniture_info.nodes()[id]['pose'] = self.manager.furniture_info.nodes()[
                id]['pose']

    def solve(self):
        NotImplementedError()

    def getCentrolNode(self, group):
        cen = np.zeros((3))
        for id in group:
            cen += np.array(self.furniture_info.nodes()[id]['pose'][0])
        cen /= len(group)
        min_id = -1
        min_dist = 1000
        for id in group:
            dist = np.linalg.norm(cen -np.array(self.furniture_info.nodes()[id]['pose'][0])) 
            if dist < min_dist:
                min_id = id
                min_dist = dist
        return min_id

    def genreateSubgroupProblem(self, group=[], objectives=[]):
        if len(group) == 1: # this is a single thing, not a group
            self.manager = None
            return
        try: 
            p.disconnect()
        except:
            pass

        use_ids = deepcopy(group)
        fix_ids = []
        self.manager = SceneManager(furn_info=self.furniture_info,position_net=self.position_net,seg_map=self.seg_map,
                                    robot=self.robot, use_gui=self.use_gui)
        self.connected_to_bullet = True

        opt_config = deepcopy(self.opt_config)
        opt_config.objectives = objectives
        if len(group) == 0:
            # use all of the groups
            grouping_order = {}
            fix_ids = identifyFixedObjects(self.furniture_info, [])
            for gr in self.groups:
                gg = [x for x in gr if x not in fix_ids]
                if len(gg) > 0:
                    center_node = self.getCentrolNode(gg)
                    grouping_order[center_node] = []
                    if len(gg) > 1:
                        grouping_order[center_node] = [g for g in gg if g != center_node]

            edges = []

            self.manager.setOptimizeRelations(edges)

            grouping_order['fix'] = fix_ids
            opt_config.randomize = False

        else:

            grouping_order = self.rearrangeGroup4Support(use_ids)
            fix_ids = identifyFixedObjects(self.furniture_info, [])
            use_ids = [x for x in use_ids if x not in fix_ids]

            grouping_order['fix'] = fix_ids
            opt_config.randomize = False

            edges = []
            edges_to_remove = []
            for ed in self.pos_net.edges(data=True):
                if ed[0] in group and ed[1] in group:
                    edges.append( (ed[0], ed[1], ed[2]['TYPE']) )
                    edges_to_remove.append( (ed[0], ed[1]))

            self.manager.setOptimizeRelations(edges)
            self.pos_net.remove_edges_from(edges_to_remove)

        if len(grouping_order) == 1:
            # no need to optimize a single thing or all fixed thing
            return None

        door_id, door_sem = self.findDoorSem()
        if door_id is not None and door_id not in grouping_order['fix'] :
            grouping_order['fix'].append(door_id)

        opt_config.grouping_order = grouping_order
        self.states, self.bounds = self.manager.setupOptimizationConfig(
            opt_config)

        self.manager.setTaskConfig(self.t_config)
        self.manager.createCompleteSDFBuffer()

        return opt_config

    def genreateLastProblem(self, objectives=[]):
        try: 
            p.disconnect()
        except:
            pass

        fix_ids = []
        self.manager = SceneManager(furn_info=self.furniture_info,position_net=self.position_net,seg_map=self.seg_map,
                                    robot=self.robot, use_gui=self.use_gui)
        self.connected_to_bullet = True

        opt_config = deepcopy(self.opt_config)
        opt_config.objectives = objectives

        # use all of the groups
        grouping_order = {}
        fix_ids = identifyFixedObjects(self.furniture_info, [])
        for gr in self.groups:
            gg = [x for x in gr if x not in fix_ids]
            if len(gg) > 0:
                center_node = self.getCentrolNode(gg)
                grouping_order[center_node] = []
                if len(gg) > 1:
                    grouping_order[center_node] = [g for g in gg if g != center_node]

        edges = []

        self.manager.setOptimizeRelations(edges)

        grouping_order['fix'] = fix_ids
        opt_config.randomize = False

        if len(grouping_order) == 1:
            # no need to optimize a single thing or all fixed thing
            return None

        door_id, door_sem = self.findDoorSem()
        if door_id is not None and door_id not in grouping_order['fix'] :
            grouping_order['fix'].append(door_id)

        opt_config.grouping_order = grouping_order
        self.states, self.bounds = self.manager.setupOptimizationConfig(
            opt_config)

        self.manager.setTaskConfig(self.t_config)
        self.manager.createCompleteSDFBuffer()

        return opt_config



class LayeredManager(OptimizationManager):
    def __init__(self, scene_name, furniture_info, seg_map, position_net, occurrence_net, robot={}, use_gui=True,  write_images=False, depth=2, breadth=5) -> None:
        super().__init__(scene_name, furniture_info, seg_map, position_net, occurrence_net, 
                         robot, use_gui, write_images, depth=depth, breadth=breadth)

    def solve(self):
        objectives = []
        objectives += ['collision']
        objectives += ['interaction']
        objectives += ['anti']
        objectives += ['planning']
        objectives += ['relation']

        start_time = time.time()

        # for individual functional group
        for idx in range(len(self.groups)):

            group = self.groups[idx]

            # this creates the mini manager
            opt_config = self.genreateSubgroupProblem(group, objectives)

            if opt_config is None:
                continue

            stepping_config = createSteppingConfigByVariable(
                self.states,  self.step_size, None, self.bounds)

            take_step = TakeStepManual(
                processFun=None, stepping_config=stepping_config)

            opt_fun = AdaptiveSimulatedAnnealing()
            opt_fun.setCustomOptimizationConfiguration(opt_config)
            opt_fun.setCustomTakeStepFunction(take_step)
            opt_fun.setCustomTasksConfiguration(self.t_config)
            opt_fun.setPackagingFunction(self.manager.repackage)
            opt_fun.setPipelineName(self.scene_name + '_layered_' + str(idx))

            ret = opt_fun(
                self.manager.objectiveFunction, self.states)

            annealing_best = self.manager.objectiveFunction(
                ret.best)

            # continue to CMA-ES
            opts = cma.CMAOptions()
            opts.set('tolfun', 1e-3)
            opts['tolx'] = 1e-3

            es = cma.CMAEvolutionStrategy(ret.best, 0.05, opts)
            es, hist_states = cmaes_optimize(es, self.manager.objectiveFunction, iterations=200)

            es_best = self.manager.objectiveFunction(es.result.xbest)

            if annealing_best < es_best:
                print('Reverting to annealing results')
                self.manager.objectiveFunction(ret.best)

            else:
                hist = []
                for res in hist_states:
                    self.manager.objectiveFunction(res)
                    hist.append(deepcopy(self.manager.repackage()))
                self.manager.objectiveFunction(es.result.xbest)
                hist.append(deepcopy(self.manager.repackage()))
                dumpSceneHistory(hist_states, opt_config, stepping_config,
                                self.t_config, hist, './results/'+ self.scene_name + '_layered_' + str(idx) + '_cma')
                self.manager.objectiveFunction(es.result.xbest)

            self.adoptState()

            del opt_fun
            del take_step
            del stepping_config

        # for the whole scene
        # each funtional group is fixed
        opt_config = self.genreateLastProblem(objectives)
        stepping_config = createSteppingConfigByVariable(
            self.states,  self.step_size, None, self.bounds)

        take_step = TakeStepManual(
            processFun=None, stepping_config=stepping_config)

        # ===================================
        # opt_config.max_iterations=1000
        # opt_config.temp_init=523.
        # opt_config.decay=0.9
        # ===================================

        opt_fun = AdaptiveSimulatedAnnealing()
        opt_fun.setCustomOptimizationConfiguration(opt_config)
        opt_fun.setCustomTakeStepFunction(take_step)
        opt_fun.setCustomTasksConfiguration(self.t_config)
        opt_fun.setPackagingFunction(self.manager.repackage)
        opt_fun.setPipelineName(self.scene_name + '_layered_-1')

        ret = opt_fun(self.manager.objectiveFunction, self.states)

        annealing_best = self.manager.objectiveFunction(ret.best)

        opts = cma.CMAOptions()
        opts.set('tolfun', 1e-3)
        opts['tolx'] = 1e-3

        es = cma.CMAEvolutionStrategy(ret.best, 0.05, opts)
        es, hist_states = cmaes_optimize(es, self.manager.objectiveFunction, iterations=400)
        es_best = self.manager.objectiveFunction(es.result.xbest)

        hist = []
        for res in hist_states:
            self.manager.objectiveFunction(res)
            hist.append(deepcopy(self.manager.repackage()))

        self.manager.objectiveFunction(es.result.xbest)
        hist.append(deepcopy(self.manager.repackage()))
        dumpSceneHistory(hist_states, opt_config, stepping_config,
                            self.t_config, hist, './results/'+ self.scene_name + '_layered_-1'+ '_cma')


        if annealing_best < es_best:
            print("--- %s seconds ---" % (time.time() - start_time))
            print('CMA result is sub optimal, please refer to annealing results')
            return self.manager.furniture_info, self.manager.seg_map
            
        else:
            print("--- %s seconds ---" % (time.time() - start_time))

            return self.manager.furniture_info, self.manager.seg_map


def cmaes_optimize(cmaes:cma.CMAEvolutionStrategy, objective_fct,
                 maxfun=None, iterations=None, min_iterations=1,
                 args=(),
                 verb_disp=None,
                 n_jobs=0,
                 **kwargs):
        if kwargs:
            message = "ignoring unkown argument%s %s in OOOptimizer.optimize" % (
                's' if len(kwargs) > 1 else '', str(kwargs))
            warnings.warn(
                message)  # warnings.simplefilter('ignore', lineno=186) suppresses this warning

        if iterations is not None and min_iterations > iterations:
            warnings.warn("doing min_iterations = %d > %d = iterations"
                  % (min_iterations, iterations))
            iterations = min_iterations

        citer, cevals = 0, 0
        hist = []
        with (EvalParallel2 or EvalParallel)(objective_fct,
                            None if n_jobs == -1 else n_jobs) as eval_all:
            while not cmaes.stop() or citer < min_iterations:
                if (maxfun and cevals >= maxfun) or (
                    iterations and citer >= iterations):
                    return cmaes , hist
                citer += 1

                X = cmaes.ask()  # deliver candidate solutions
                # fitvals = [objective_fct(x, *args) for x in X]
                fitvals = eval_all(X, args=args)
                cevals += len(fitvals)
                cmaes.tell(X, fitvals)  # all the work is done here
                cmaes.disp(verb_disp)  # disp does nothing if not overwritten

                if citer % 10 == 0:
                    hist.append(deepcopy(cmaes.result.xbest) ) 

        if verb_disp:  # do not print by default to allow silent verbosity
            cmaes.disp(1)
            print('termination by', cmaes.stop())
            print('best f-value =', cmaes.result[1])
            print('solution =', cmaes.result[0])

        return cmaes , hist
