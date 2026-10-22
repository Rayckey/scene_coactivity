"""Rearrange one precompiled SUNCG room. Run from the project root."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scene_manager.config import ROBOTS, WEIGHTS


def parse_bool(value):
    """Accept legacy '--GUI True/False' as well as bare boolean flags."""
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def load_room_inputs(room_type, room_id):
    """Load the scene and preference graphs, reporting missing input paths."""
    paths = (
        Path("rooms/suncg") / f"{room_type}_{room_id}",
        Path("suncg/data") / f"{room_type}_position_net_processed_lognorm.bin",
        Path("suncg/data") / f"{room_type}_occurrence_net_processed.bin",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input: {path}. Run from the scene_coactivity project root."
            )
    loaded = []
    for path in paths:
        with path.open("rb") as file:
            loaded.append(pickle.load(file))
    (furn_info, seg_map), position_net, occurrence_net = loaded
    return furn_info, seg_map, position_net, occurrence_net


def from_furn_info(room_type, room_id, GUI, images, robot_name, seed=None):
    """Load, optimize, and save a room; close the simulator on success or failure."""
    if robot_name not in ROBOTS:
        raise ValueError(
            f"Unknown robot {robot_name!r}; choose from {', '.join(ROBOTS)}"
        )
    furn_info, seg_map, position_net, occurrence_net = load_room_inputs(
        room_type, room_id
    )

    # Delay the simulation imports so --help and input validation stay usable
    # without a graphics/physics installation.
    import pybullet as p
    from optimization_manager import LayeredManager
    from sampling.sampling_utilis import AdaptiveAnnealingConfig
    from scene_manager.common_utilis import (
        center_furniture,
        filter_higher_up_furniture,
        filter_ignored_furniture,
    )
    from scene_manager.tasks.psedo_task_gen import genMoreTasks

    if seed is not None:
        np.random.seed(seed)
    scene_name = f"{room_type}_{room_id}"

    filter_higher_up_furniture(furn_info)
    filter_ignored_furniture(furn_info)

    robot = ROBOTS[robot_name]

    t_config = genMoreTasks(furn_info)

    center_furniture(furn_info)
    Path("results").mkdir(parents=True, exist_ok=True)
    try:
        manager = LayeredManager(
            scene_name=scene_name,
            furniture_info=furn_info,
            seg_map=seg_map,
            position_net=position_net,
            occurrence_net=occurrence_net,
            robot=robot,
            use_gui=GUI,
            write_images=images,
            depth=2,
            breadth=10,
        )
        opt_configs = AdaptiveAnnealingConfig(
            max_iterations=500,
            epoch=500,
            temp_init=52.3,
            randomize=True,
            scene_name=scene_name,
            robot=manager.robot["name"],
            term_criteria=1e-3,
            step_scale=2.0,
            ns=5,
            nt=2,
            n_term=50,
            scaling_factors=2.0,
            vocal=False,
            log_every=100,
            desired_sdf_res=0.1,
            decay=0.85,
            write_images=images,
            weights=WEIGHTS,
            seed=seed,
        )
        manager.storeTaskConfig(t_config)
        manager.storeOptimizationStates(opt_configs)
        furn_info, seg_map = manager.solve()
        with (Path("results") / scene_name).open("wb") as file:
            pickle.dump((furn_info, seg_map), file)
        return furn_info, seg_map
    finally:
        if p.isConnected():
            p.disconnect()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room",
        type=str,
        default="living",
        choices=("living", "office", "bedroom"),
        help="room category",
    )
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
        help="room ID within the selected category",
    )
    parser.add_argument(
        "--GUI",
        "--gui",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="display the optimization process in the PyBullet GUI",
    )
    parser.add_argument(
        "--images",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="generate debug images",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="dingo",
        choices=sorted(ROBOTS),
        help="robot dimensions to use (default: dingo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for repeatable optimization runs",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.idx < 0:
        parser.error("--idx must be non-negative")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    try:
        from_furn_info(
            args.room, args.idx, args.GUI, args.images, args.robot, args.seed
        )
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
