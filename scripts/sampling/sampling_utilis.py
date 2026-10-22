#!/usr/bin/env python3
"""Sampling strategies, optimizer configuration, and serialized scene histories.

Layered scene rearrangement uses AdaptiveSimulatedAnnealing. The other strategies
retain their legacy public names for existing callers and saved configurations.
"""

import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np
from scene_manager.config import MAXIMUN_RETRIES
from scene_manager.task_types import TaskConfig


class AnnealingResult(NamedTuple):
    """Best state and the sequence of improvements encountered during sampling."""

    best: np.ndarray
    best_eval: float
    scores: list
    hist: list
    steps: list


class OptimizationResult(NamedTuple):
    best: np.ndarray
    best_eval: float
    scores: list
    hist: list
    steps: list


class OptimizationConfig(object):
    def __init__(
        self,
        scene_name,
        robot,
        randomize,
        desired_sdf_res,
        write_images,
        padding=3.0,
        extension=3.0,
        grouping_order=None,
        weights=None,
    ) -> None:
        super().__init__()
        self.scene_name = scene_name
        self.robot = robot
        self.objectives = []
        self.randomize = randomize
        self.desired_sdf_res = desired_sdf_res
        self.write_images = write_images
        self.padding = padding
        self.grouping_order = grouping_order
        self.weights = {} if weights is None else deepcopy(weights)
        self.extension = extension


class AdaptiveAnnealingConfig(OptimizationConfig):
    """Configure coordinate sweeps, step adaptation, cooling, and restarts.

    A sweep proposes one move per coordinate. Step sizes adapt every ``ns``
    sweeps; temperature decays after ``nt`` adaptations. Every ``n_term`` sweeps,
    sampling restarts from the best state. ``epoch`` and ``log_every`` remain
    available for compatibility but are not used by AdaptiveSimulatedAnnealing.
    """

    def __init__(
        self,
        scene_name,
        robot,
        randomize,
        desired_sdf_res,
        write_images,
        max_iterations,
        epoch,
        temp_init,
        term_criteria,
        step_scale,
        ns,
        nt,
        n_term,
        decay,
        vocal,
        log_every,
        scaling_factors,
        weights,
        seed=None,
    ) -> None:
        super().__init__(
            scene_name, robot, randomize, desired_sdf_res, write_images, weights=weights
        )
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
        self.seed = seed


class SteppingConfig(object):
    """Per-coordinate step scales, optional wrap periods, and validity bounds.

    The legacy ``clipping`` field specifies modulo periods, not clamping limits.
    """

    def __init__(
        self,
        initial,
        min=None,
        clipping=None,
        discretization=None,
        state_length=None,
        bounds=None,
    ) -> None:
        super().__init__()
        self.initial = initial
        self.min = min
        self.clipping = clipping
        self.discretization = discretization
        self.state_length = state_length
        self.bounds = [] if bounds is None else bounds


class ResultHistory(NamedTuple):
    """Scene snapshots plus an annealing result or a list of CMA states."""

    config: AdaptiveAnnealingConfig
    tasks: TaskConfig
    step: SteppingConfig
    scenes: list
    result: Union[AnnealingResult, list]


def createSteppingConfigByVariable(
    states, init_bounds, discrete_type=None, bounds=None
):
    """Create equal step scales and unit wrap periods for normalized states."""
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

    return SteppingConfig(
        initial=init, min=None, clipping=clip, discretization=discrete, bounds=bounds
    )


class TakeStepBase(object):
    def __init__(
        self,
        object_info,
        num_objs,
        processFun,
        validFun,
        stepping_config: SteppingConfig,
        vocal=False,
    ) -> None:
        super().__init__()
        self.initial = stepping_config.initial
        self.stepping_config = stepping_config
        self.step_scale = 0.0
        self.num_objs = num_objs
        self.move_this_idx = 0
        self.state_slicing_idx = 0
        self.object_ids = list(object_info.keys()) if object_info is not None else None
        self.object_info = object_info
        self.init_info = deepcopy(object_info)
        self.vocal = vocal
        self.validFun = validFun
        self.processFun = processFun

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class TakeStepManual(TakeStepBase):
    """Propose a bounded move for one coordinate without changing the input."""

    def __init__(self, processFun, stepping_config: SteppingConfig, vocal=False):
        super().__init__(None, None, processFun, None, stepping_config, vocal)

    def isValid(self, state, state_idx, bounds):
        """States are normalized to [0, 1] unless explicit bounds are supplied."""
        lower, upper = (
            (bounds[0][state_idx], bounds[1][state_idx]) if len(bounds) else (0.0, 1.0)
        )
        return np.isfinite(state[state_idx]) and lower <= state[state_idx] <= upper

    def __call__(self, x, state_idx):

        _step_size = self.step_scale * self.initial[state_idx]

        has_valid_step = False
        retries = 0

        next_state = deepcopy(x)

        while not has_valid_step:

            this_step = np.random.random() * 2.0 * _step_size - _step_size

            next_state[state_idx] = x[state_idx] + this_step

            # Wrap periodic coordinates before checking their allowed bounds.
            if self.stepping_config.clipping[state_idx] is not None:
                next_state[state_idx] = (
                    next_state[state_idx] % self.stepping_config.clipping[state_idx]
                )

            has_valid_step = self.isValid(
                next_state, state_idx, self.stepping_config.bounds
            )

            retries += 1

            if not has_valid_step and retries > MAXIMUN_RETRIES:
                if self.vocal:
                    print("failed to get new config")

                return x, False

        # Notify the caller only after a valid proposal has been found.
        if self.processFun is not None:
            self.processFun(next_state, state_idx)

        return next_state, True


def dumpSceneHistory(
    ret, configs, stepping_config, task_configs, scene_infos, file_name
):
    """Write a ResultHistory pickle; an empty filename disables saving."""
    if len(file_name) == 0:
        return
    dat = ResultHistory(
        config=configs,
        result=ret,
        step=stepping_config,
        tasks=task_configs,
        scenes=scene_infos,
    )

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(dat, f)


def corana_update(step_sizes, accepted_steps, scaling_factors, ns):
    """Adapt each step size toward an acceptance rate between 40% and 60%."""
    for i in range(len(step_sizes)):
        ai, ci = accepted_steps[i], scaling_factors[i]
        if ai > 0.6 * ns:
            step_sizes[i] *= 1 + ci * (ai / ns - 0.6) / 0.4
            if step_sizes[i] > 10.0:
                step_sizes[i] = 10.0
        elif ai < 0.4 * ns:
            step_sizes[i] /= 1 + ci * (0.4 - ai / ns) / 0.4
    return step_sizes


class OptimizationFunction:
    def __init__(self) -> None:
        self.take_step = None
        self.configs = None
        self.task_configs = None
        self.packaging_fun = None
        self.pipeline_name = None

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
        raise NotImplementedError

    def isTerminating(self):
        return self.checkTerminatingCondition()

    def isReturningtoBestCandidate(self):
        if self.configs.epoch > 0:
            return self._total % self.configs.epoch == 0

    def updateTemperature(self):
        raise NotImplementedError

    def updateNextStep(self, next, curr):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def saveResults(self, ret, file_name):
        self._scene_history.append(deepcopy(self.packaging_fun()))
        dumpSceneHistory(
            ret,
            self.configs,
            self.take_step.stepping_config,
            self.task_configs,
            self._scene_history,
            file_name,
        )

    def genFileName(self, opt_configs: OptimizationConfig, opt_name=""):
        res = "./results/"
        res += opt_name
        res += "_"
        res += opt_configs.scene_name
        res += "_"
        res += opt_configs.robot
        res += "_"
        res += str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        return res


class SimulatedAnnleaing(OptimizationFunction):
    """Legacy annealing with a quadratic temperature schedule within each epoch."""

    def __init__(self) -> None:
        super().__init__()

    def updateTemperature(self):
        return self.configs.temp_min + (
            self.configs.temp_max - self.configs.temp_min
        ) * (
            (
                float(self.configs.epoch - (self._total % self.configs.epoch))
                / self.configs.epoch
            )
            ** 2
        )

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, next, curr, t):
        next, full_round = self.take_step(next)
        if full_round:
            self.take_step.step_scale = (
                self.configs.step_scale * t / self.configs.temp_max
            )
        return next

    def __call__(self, objective, x0):
        best = x0
        best_eval = objective(best)
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self._scene_history = [deepcopy(self.packaging_fun())]

        while not self.isTerminating():

            if self.isReturningtoBestCandidate():
                self.take_step.step_scale = self.configs.step_scale
                curr, curr_eval = best, best_eval

            next = deepcopy(curr)

            t = self.updateTemperature()

            next = self.updateNextStep(next, curr, t)

            next_eval = objective(next)

            if next_eval < best_eval:
                best, best_eval = next, next_eval
                scores.append(best_eval)
                hist.append(best)
                steps.append(self._total)
                print(">%d f(%s) = %.5f" % (self._total, best, best_eval))

                ret = AnnealingResult(best, best_eval, scores, hist, steps)
                self.saveResults(ret, self.genFileName())

            diff = next_eval - curr_eval
            metropolis = np.exp(-diff / t)
            if diff < 0 or np.random.rand() < metropolis:
                curr, curr_eval = next, next_eval

            self._total += 1
            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        print("Optimization exits")

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        return super().genFileName(
            self.configs, opt_name=self.pipeline_name + "_annealing"
        )


class RandomWalk(OptimizationFunction):
    """Legacy greedy walk: accept improvements and decay the proposal scale."""

    def __init__(self) -> None:
        super().__init__()

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, next, curr, t=None):
        next, full_round = self.take_step(next)
        if full_round:
            self.take_step.step_scale *= self.configs.decay_rate
        return next

    def __call__(self, objective, x0):
        best = x0
        best_eval = objective(best)
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self._scene_history = [deepcopy(self.packaging_fun())]

        result_file_name = self.genFileName()

        while not self.isTerminating():

            if self.isReturningtoBestCandidate():
                self.take_step.step_scale = self.configs.step_scale
                curr, curr_eval = best, best_eval

            next = deepcopy(curr)

            next = self.updateNextStep(next, curr)

            next_eval = objective(next)

            if next_eval < best_eval:
                best, best_eval = next, next_eval
                scores.append(best_eval)
                hist.append(best)
                steps.append(self._total)
                print(">%d f(%s) = %.5f" % (self._total, best, best_eval))

                ret = AnnealingResult(best, best_eval, scores, hist, steps)
                self.saveResults(ret, result_file_name)

            diff = next_eval - curr_eval
            if diff < 0:
                curr, curr_eval = next, next_eval

            self._total += 1
            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        print("Optimization exits")

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        return super().genFileName(
            self.configs, opt_name=self.pipeline_name + "_randomwalk"
        )


class RandomImprove(OptimizationFunction):
    """Legacy search that samples around the initial state and keeps improvements."""

    def __init__(self) -> None:
        super().__init__()

    def isTerminating(self):
        return self._total > self.configs.n_iterations

    def updateNextStep(self, next):
        next, full_round = self.take_step(next)
        if full_round:
            self.take_step.step_scale *= self.configs.decay_rate
        return next

    def __call__(self, objective, x0):
        best = x0
        best_eval = objective(best)
        curr, curr_eval = best, best_eval
        hist = [best]
        steps = []
        scores = [curr_eval]
        self._total = 0
        self._scene_history = [deepcopy(self.packaging_fun())]

        result_file_name = self.genFileName()

        self.take_step.step_scale = self.configs.step_scale
        curr, curr_eval = best, best_eval
        while not self.isTerminating():

            next = deepcopy(curr)

            next = self.updateNextStep(next)
            next_eval = objective(next)
            if next_eval < best_eval:
                best, best_eval = next, next_eval
                scores.append(best_eval)
                hist.append(best)
                steps.append(self._total)
                print(">%d f(%s) = %.5f" % (self._total, best, best_eval))

                ret = AnnealingResult(best, best_eval, scores, hist, steps)
                self.saveResults(ret, result_file_name)

            self._total += 1
            if self.configs.vocal:
                print("at step ", self._total, " score is ", next_eval)

        print("Optimization exits")

        ret = AnnealingResult(best, best_eval, scores, hist, steps)
        return ret

    def genFileName(self):
        return super().genFileName(
            self.configs, opt_name=self.pipeline_name + "_randomimprove"
        )


class AdaptiveSimulatedAnnealing(OptimizationFunction):
    """Coordinate-wise annealing with Corana step-size adaptation.

    One iteration proposes one move per coordinate. Failed proposals and
    non-finite costs are rejected. The objective is restored to the best state
    before returning because scene objectives mutate the simulation.
    """

    def updateNextStep(self, curr, idx):
        return self.take_step(curr, idx)

    def updateTemperature(self, temperature):
        return temperature * self.configs.decay

    def isTerminating(self, step_sizes, scores):
        return bool(
            np.sum(step_sizes) < self.configs.term_criteria
            or self._total >= self.configs.max_iterations
        )

    def __call__(self, objective, x0):
        config = self.configs
        for name in ("ns", "nt", "n_term"):
            value = getattr(config, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(config.max_iterations, (int, np.integer))
            or config.max_iterations < 0
        ):
            raise ValueError("max_iterations must be a non-negative integer")
        for name in ("temp_init", "step_scale"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("term_criteria", "scaling_factors"):
            value = getattr(config, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < config.decay <= 1:
            raise ValueError("decay must be in (0, 1]")

        best = np.asarray(x0, dtype=float).copy()
        if best.ndim != 1 or not np.all(np.isfinite(best)):
            raise ValueError("x0 must be a finite one-dimensional state")
        best_eval = objective(best)
        if not np.isfinite(best_eval):
            raise ValueError("The initial objective value must be finite")

        curr, curr_eval = best.copy(), best_eval
        hist, scores, steps = [best.copy()], [best_eval], []
        self._total = 0
        self._scene_history = []
        if self.packaging_fun is not None:
            self._scene_history.append(deepcopy(self.packaging_fun()))
        temperature = config.temp_init
        step_sizes = np.full(best.size, config.step_scale, dtype=float)
        scaling_factors = np.full(best.size, config.scaling_factors, dtype=float)
        accepted_steps = np.zeros(best.size, dtype=int)
        adaptation_count = 0
        result_file_name = self.genFileName()

        while best.size and not self.isTerminating(step_sizes, scores):
            if self._total % config.n_term == 0:
                curr, curr_eval = best.copy(), best_eval

            for idx in range(best.size):
                self.take_step.step_scale = step_sizes[idx]
                candidate, valid = self.updateNextStep(curr.copy(), idx)
                if not valid:
                    continue
                candidate_eval = objective(candidate)
                if not np.isfinite(candidate_eval):
                    continue

                difference = candidate_eval - curr_eval
                if difference <= 0 or np.random.rand() < np.exp(
                    -difference / temperature
                ):
                    curr, curr_eval = candidate.copy(), candidate_eval
                    accepted_steps[idx] += 1

                if candidate_eval < best_eval:
                    best, best_eval = candidate.copy(), candidate_eval
                    scores.append(best_eval)
                    hist.append(best.copy())
                    steps.append(self._total)
                    if config.vocal:
                        print(f">{self._total} f({best}) = {best_eval:.5f}")
                    if self.packaging_fun is not None:
                        self.saveResults(
                            AnnealingResult(best, best_eval, scores, hist, steps),
                            result_file_name,
                        )

            self._total += 1
            if self._total % config.ns == 0:
                corana_update(step_sizes, accepted_steps, scaling_factors, config.ns)
                accepted_steps.fill(0)
                adaptation_count += 1
            if adaptation_count >= config.nt:
                # Keep the Metropolis denominator positive after many decays.
                temperature = max(
                    self.updateTemperature(temperature), np.finfo(float).tiny
                )
                adaptation_count = 0
            if config.vocal:
                print(f"at step {self._total}, best score is {best_eval}")

        objective(best)
        result = AnnealingResult(best, best_eval, scores, hist, steps)
        if self.packaging_fun is not None:
            self.saveResults(result, result_file_name)
        return result

    def genFileName(self):
        return "./results/" + (self.pipeline_name or "annealing")
