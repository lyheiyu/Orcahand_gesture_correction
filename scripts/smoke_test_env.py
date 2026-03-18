import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from orca_sim import (
        OrcaHandCombined,
        OrcaHandCombinedExtended,
        OrcaHandLeft,
        OrcaHandLeftExtended,
        OrcaHandRight,
        OrcaHandRightExtended,
        latest_version,
        list_versions,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"gymnasium", "mujoco", "numpy"}:
        missing = exc.name
        raise SystemExit(
            "Missing runtime dependency "
            f"'{missing}'. Activate the 'orca' environment and run "
            "'uv pip install -e .'."
        ) from exc
    raise


ENV_BUILDERS = {
    "left": OrcaHandLeft,
    "left_extended": OrcaHandLeftExtended,
    "right": OrcaHandRight,
    "right_extended": OrcaHandRightExtended,
    "combined": OrcaHandCombined,
    "combined_extended": OrcaHandCombinedExtended,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Orcasim environments.")
    parser.add_argument(
        "--env",
        choices=sorted(ENV_BUILDERS),
        default="right",
        help="Environment variant to load.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["rgb_array", "human"],
        default="rgb_array",
        help="Render mode to test.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of environment steps to run. Use 0 to run forever.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Embodiment version folder to load, for example 'v1'. Defaults to the latest.",
    )
    args = parser.parse_args()

    env_cls = ENV_BUILDERS[args.env]
    env = env_cls(render_mode=args.render_mode, version=args.version)
    obs, info = env.reset()

    print(f"loaded={args.env}")
    print(f"available_versions={list_versions()}")
    print(f"version={env.version}")
    print(f"latest_version={latest_version()}")
    print(f"obs_shape={obs.shape}")
    print(f"action_shape={env.action_space.shape}")
    print(f"action_low={env.action_space.low}")
    print(f"action_high={env.action_space.high}")
    print(f"info={info}")

    step = 0
    try:
        while True:
            if args.steps and step >= args.steps:
                break

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if args.render_mode == "rgb_array":
                frame = env.render()
                print(
                    f"step={step} frame_shape={None if frame is None else frame.shape} "
                    f"reward={reward} terminated={terminated} truncated={truncated}"
                )
            else:
                print(
                    f"step={step} reward={reward} "
                    f"terminated={terminated} truncated={truncated}"
                )
                time.sleep(1.0 / env.metadata["render_fps"])

            if terminated or truncated:
                obs, info = env.reset()
                print(f"reset step={step} info={info}")
            step += 1
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
