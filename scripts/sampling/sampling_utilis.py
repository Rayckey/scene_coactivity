#!/usr/bin/env python3
import numpy as np
from numpy.core.numeric import zeros_like
from copy import deepcopy
from typing import NamedTuple
import pickle
from datetime import datetime
from scene_manager.common_utilis import TaskConfig, MAXIMUN_RETRIES


class AnnealingResult(NamedTuple):
    best: np.array
    best_eval: float
    scores: list
    hist: list
    steps: list


class OptimizationResult(NamedTuple):
    best: np.array
    best_eval: float
    scores: list
    hist: list
    steps: list


class OptimizationConfig(object):
    def __init__(self, scene_name, robot, randomize, desired_sdf_res, write_images, padding=3.0, extension=3.0, grouping_order=None, weights=[]) -> None:
        super().__init__()
        self.scene_name = scene_name
        self.robot = robot
        self.objectives = []
        self.randomize = randomize
        self.desired_sdf_res = desired_sdf_res
        self.write_images = write_images
        self.padding = padding
        self.grouping_order = grouping_order
        self.weights = weights
        self.extension = extension


class AdaptiveAnnealingConfig(OptimizationConfig):
    def __init__(self, scene_name, robot, randomize, desired_sdf_res, write_images, max_iterations, epoch, temp_init, term_criteria, step_scale, ns, nt, n_term, decay, vocal, log_every, scaling_factors, weights) -> None:
        super().__init__(scene_name, robot,
                         randomize, desired_sdf_res, write_images)
        self.max_iterations = max_iterations
        self.epoch = epoch
        self.temp_init = temp_init
        self.term_criteria = term_criteria
        self.step_scale = step_scale
        self.vocal = vocal
        self.log_every = log_every
        self.ns = ns
        self.nt = nt
        self.n_term = n_term
        self.decay = decay
        self.scaling_factors = scaling_factors
        self.weights = weights


class SteppingConfig(object):
    def __init__(self, initial, min=None, clipping=None, discretization=None, state_length=None, bounds=[]) -> None:
        super().__init__()
        self.initial = initial
        self.min = min
        self.clipping = clipping
        self.discretization = discretization
        self.state_length = state_length
        self.bounds = bounds


class ResultHistory(NamedTuple):
    config: AdaptiveAnnealingConfig
    tasks: TaskConfig
    step: SteppingConfig
    scenes: list
    result: AnnealingResult


def createSteppingConfigByVariable(states, init_bounds, discrete_type=None, bounds=[]):
    init = []
    clip = []
    discrete = []
    for __ in states:
        init += [init_bounds]
        clip += [1.0]
        if discrete_type is None:
            discrete += [None]
        else:
            discrete += discrete_type[0]

    return SteppingConfig(initial=init, min=None, clipping=clip, discretization=discrete, bounds=bounds)


class TakeStepBase(object):
    def __init__(self, object_info, num_objs, processFun, validFun, stepping_config: SteppingConfig, vocal=False) -> None:
        super().__init__()
        self.initial = stepping_config.initial
        self.stepping_config = stepping_config
        self.step_scale = 0.
        self.num_objs = num_objs
        self.move_this_idx = 0
        self.state_slicing_idx = 0
        self.object_ids = list(
            object_info.keys()) if object_info is not None else None
        self.object_info = object_info
        self.init_info = deepcopy(object_info)
        self.vocal = vocal
        self.validFun = validFun
        self.processFun = processFun

    def __call__(self, *args: any, **kwds: any) -> any:
        return super().__call__(*args, **kwds)


# manually define the variables that can be adjusted
class TakeStepManual(TakeStepBase):
    def __init__(self, processFun, stepping_config: SteppingConfig, vocal=False):
        super().__init__(None, None, processFun,
                         None, stepping_config, vocal)

    def isValid(ols, state, state_idx, bounds):
        # l_b = bounds[0][state_idx]
        # u_b = bounds[1][state_idx]
        # if state[state_idx] < l_b or state[state_idx] > u_b:
        #     return False
        if state[state_idx] < 0.0 or state[state_idx] > 1.0:
            return False
        # applyEnv2D(object_info, obj_id)
        return True

    def __call__(self, x, state_idx):

        _step_size = self.step_scale * self.initial[state_idx]

        has_valid_step = False
        retries = 0

        next_state = deepcopy(x)

        while not has_valid_step:

            this_step = np.random.random() * 2.*_step_size - _step_size

            next_state[state_idx] = x[state_idx] + this_step

            # check if steps need to be clipped, like np.pi
            if self.stepping_config.clipping[state_idx] is not None:
                next_state[state_idx] = next_state[state_idx] % self.stepping_config.clipping[state_idx]

            # check if the state is valid
            has_valid_step = self.isValid(
                next_state, state_idx, self.stepping_config.bounds)

            retries += 1

            if not has_valid_step and retries > MAXIMUN_RETRIES:
                if self.vocal:
                    print('failed to get new config')

                return x, False

        # process the state and store it if need to
        if self.processFun is not None:
            self.processFun(next_state, state_idx)

        return next_state, True


def dumpSceneHistory(ret, configs, stepping_config, task_configs, scene_infos, file_name):
    # dumps all saved scene instances
    if len(file_name) == 0:
        return
    dat = ResultHistory(config=configs, result=ret,
                        step=stepping_config, tasks=task_configs, scenes=scene_infos)

    with open(file_name, 'wb') as f:
        pickle.dump(dat, f)


def corana_update(step_sizes, accepted_steps, scaling_factors, ns):
    # update function for adaptive simulaed annealing's ransom search corana
    for i in range(len(step_sizes)):
        ai, ci = accepted_steps[i], scaling_factors[i]
        if ai > 0.6 * ns:
            step_sizes[i] *= 1 + ci * (ai/ns - 0.6) / 0.4
            if step_sizes[i] > 10.0:
                step_sizes[i] = 10.0
        elif ai < 0.4*ns:
            step_sizes[i] /= 1 + ci*(0.4 - ai/ns)/0.4
    return step_sizes


class OptimizationFunction():
    def __init__(self) -> None:
        # custom functions
        self.take_step = None
        self.configs = None
        self.task_configs = None
        self.packaging_fun = None
        self.pipeline_name = None

        # trackers
        self._total = 0
        self._scene_history = []

    def setPipelineName(self, name):
        self.pipeline_name = name

    def setCustomTakeStepFunction(self, take_step: TakeStepBase):
        assert issubclass(type(take_step), TakeStepBase)
        self.take_step = take_step

    def setCustomOptimizationConfiguration(self, configs: OptimizationConfig):
        assert issubclass(type(configs), OptimizationConfig)
        self.configs = configs

    def setCustomTasksConfiguration(self, task_configs: TaskConfig):
        assert issubclass(type(task_configs), TaskConfig)
        self.task_configs = task_configs

    def setPackagingFunction(self, packaging_fun):
        self.packaging_fun = packaging_fun

    def checkTerminatingCondition(self):
        NotImplementedError()

    def isTerminating(self):
        return self.checkTerminatingCondition()

    def isReturningtoBestCandidate(self):
        if self.configs.epoch > 0:
            return self._total % self.configs.epoch == 0

    def updateTemperature(self):
        NotImplementedError()

    def updateNextStep(self, next, curr):
        NotImplementedError()

    def __call__(self):
        NotImplementedError()

    def saveResults(self, ret, file_name):
        self._scene_history.append(deepcopy(self.packaging_fun()))
        dumpSceneHistory(ret, self.configs, self.take_step.stepping_config,
                         self.task_configs, self._scene_history, file_name)

    def genFileName(self, opt_configs: OptimizationConfig, opt_name=''):
        res = './results/'
        res += opt_name
        # res += '_'
        # res += opt_configs.grouping_order
        res += '_'
        res += opt_configs.scene_name
        res += '_'
        res += opt_configs.robot
        res += '_'
        res += str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        return res


class SimulatedAnnleaing(OptimizationFunction):
    def __init__(self) -> None:
        super().__init__()

    def updateTemperature(self):
        return self.configs.temp_min + (self.configs.temp_max - self.configs.temp_min) * ((float(self.configs.epoch - (self._total % self.configs.epoch))/self.configs.epoch)**2)

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, next, curr, t):
        next, full_round = self.take_step(next)
        if full_round:
            self.take_step.step_scale = self.configs.step_scale * t/self.configs.temp_max

    def __call__(self, objective, x0):
        # generate an initial point
        best = x0
        # evaluate the best point
        best_eval = objective(best)
        # current working solution
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self.scene_history = [deepcopy(self.packaging_fun())]

        while not self.isTerminating():

            if self.isReturningtoBestCandidate():
                self.take_step.step_scale = self.configs.step_scale
                curr, curr_eval = best, best_eval

            next = deepcopy(curr)

            t = self.updateTemperature()

            self.updateNextStep(next, curr, t)

            # evaluate candidate point
            next_eval = objective(next)

            if next_eval < best_eval:
                # store new best point
                best, best_eval = next, next_eval
                # keep track of scores
                scores.append(best_eval)
                hist.append(best)
                steps.append(self._total)
                # report progress
                print('>%d f(%s) = %.5f' % (self._total, best, best_eval))

                ret = AnnealingResult(best, best_eval, scores, hist, steps)
                self.saveResults(self, ret, self.genFileName())

            # difference between candidate and current point evaluation
            diff = next_eval - curr_eval
            # diff = next_eval - best_eval
            # calculate metropolis acceptance criterion
            metropolis = np.exp(-diff / t)
            # check if we should keep the new point
            if diff < 0 or np.random.rand() < metropolis:
                # store the new current point
                curr, curr_eval = next, next_eval

            self._total += 1
            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        print("Optimization exits")

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        return super().genFileName(self.configs, opt_name=self.pipeline_name + '_annealing')


class RandomWalk(OptimizationFunction):
    def __init__(self) -> None:
        super().__init__()

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, next, curr, t):
        next, full_round = self.take_step(next)
        if full_round:
            self.take_step.step_scale *= self.configs.decay_rate

    def __call__(self, objective, x0):
        # generate an initial point
        best = x0
        # evaluate the best point
        best_eval = objective(best)
        # current working solution
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self.scene_history = [deepcopy(self.packaging_fun())]

        result_file_name = self.genFileName()

        while not self.isTerminating():

            if self.isReturningtoBestCandidate():
                self.take_step.step_scale = self.configs.step_scale
                curr, curr_eval = best, best_eval

            next = deepcopy(curr)

            self.updateNextStep(next, curr)

            # evaluate candidate point
            next_eval = objective(next)

            if next_eval < best_eval:
                # store new best point
                best, best_eval = next, next_eval
                # keep track of scores
                scores.append(best_eval)
                hist.append(best)
                steps.append(self._total)
                # report progress
                print('>%d f(%s) = %.5f' % (self._total, best, best_eval))

                ret = AnnealingResult(best, best_eval, scores, hist, steps)
                self.saveResults(self, ret, result_file_name)

            # difference between candidate and current point evaluation
            diff = next_eval - curr_eval
            # check if we should keep the new point
            if diff < 0:
                # store the new current point
                curr, curr_eval = next, next_eval

            self._total += 1
            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        print("Optimization exits")

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        return super().genFileName(self.configs, opt_name=self.pipeline_name + '_randomwalk')


class RandomImprove(OptimizationFunction):
    def __init__(self) -> None:
        super().__init__()

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, next):
        next, full_round = self.take_step(next)
        if full_round:
            self.take_step.step_scale *= self.configs.decay_rate

    def __call__(self, objective, x0):
        # generate an initial point
        best = x0
        # evaluate the best point
        best_eval = objective(best)
        # current working solution
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self.scene_history = [deepcopy(self.packaging_fun())]

        result_file_name = self.genFileName()

        self.take_step.step_scale = self.configs.step_scale
        curr, curr_eval = best, best_eval
        while not self.isTerminating():

            next = deepcopy(curr)

            self.updateNextStep(next)
            # evaluate candidate point
            next_eval = objective(next)
            if next_eval < best_eval:
                # store new best point
                best, best_eval = next, next_eval
                # keep track of scores
                scores.append(best_eval)
                hist.append(best)
                steps.append(self._total)
                # report progress
                print('>%d f(%s) = %.5f' % (self._total, best, best_eval))

                ret = AnnealingResult(best, best_eval, scores, hist, steps)
                self.saveResults(ret, result_file_name)

            self._total += 1
            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        print("Optimization exits")

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        return super().genFileName(self.configs, opt_name=self.pipeline_name + '_randomimprove')


class AdaptiveSimulatedAnnealing(OptimizationFunction):
    def __init__(self) -> None:
        super().__init__()

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, curr, idx):
        return self.take_step(curr, idx)

    def updateTemperature(self, t):
        return t * self.configs.decay

    def isTerminating(self, step_sizes, scores):
        if np.sum(step_sizes) < self.configs.term_criteria:
            # if self._total > self.configs.epoch+100:
            print('terminating due to small step size')
            return True
            # else:
            #     self._total = self.configs.epoch
            #     return False

        # if len(scores) > self.configs.n_term and np.all(np.array(scores[-self.configs.n_term:-1]) - scores[-1] < self.configs.term_criteria):
        #     print('terminating due to stopping criterion')
        #     return True

        if self._total >= self.configs.max_iterations:
            print('terminating due to max iteration reached')
            return True

    def __call__(self, objective, x0):
        # generate an initial point
        best = x0
        t = self.configs.temp_init
        epoch = self.configs.epoch
        n_term = self.configs.n_term
        # evaluate the best point
        best_eval = objective(best)
        # current working solution
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self.scene_history = [deepcopy(self.packaging_fun())]
        nt = self.configs.nt

        self._total = 0
        counts_resets = 0

        self.take_step.step_scale = self.configs.step_scale
        step_sizes = np.array([self.configs.step_scale] * len(x0))
        scaling_factors = np.array([self.configs.scaling_factors] * len(x0))
        accepted_steps = zeros_like(x0)

        result_file_name = self.genFileName()

        while not self.isTerminating(step_sizes, scores):

            # if self._total % epoch == 0:
            # curr, curr_eval = best, best_eval
            # t = self.configs.temp_init
            if self._total % n_term == 0:
                curr, curr_eval = best, best_eval
                # t = self.configs.temp_init

            for idx in range(len(x0)):
                self.take_step.step_scale = step_sizes[idx]
                next, NEW_STEP = self.updateNextStep(curr, idx)
                # evaluate candidate point
                next_eval = objective(next)

                # difference between candidate and current point evaluation
                diff = next_eval - curr_eval

                if diff < 0 or np.random.rand() < np.exp(-diff / t):
                    # store the new current point
                    curr, curr_eval = next, next_eval

                    accepted_steps[idx] += 1

                if next_eval < best_eval:
                    # store new best point
                    best, best_eval = next, next_eval
                    # keep track of scores
                    scores.append(best_eval)
                    hist.append(best)
                    steps.append(self._total)
                    # report progress
                    print('>%d f(%s) = %.5f' % (self._total, best, best_eval))
                    ret = AnnealingResult(best, best_eval, scores, hist, steps)
                    self.saveResults(ret, result_file_name)

            self._total += 1

            if self._total % self.configs.ns == 0:
                corana_update(step_sizes, accepted_steps,
                              scaling_factors, self.configs.ns)
                accepted_steps = zeros_like(x0)
                counts_resets += 1

            if counts_resets >= nt:
                t = self.updateTemperature(t)
                if self.configs.vocal:
                    print('temperture is now : ', t)
                counts_resets = 0

            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        # return super().genFileName(self.configs, opt_name=self.pipeline_name)
        return './results/' + self.pipeline_name
