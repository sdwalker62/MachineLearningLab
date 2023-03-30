import argparse
import sys

from subprocess import run
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from utils.docker_utils import get_containers_by_author

run_cmd = partial(run, shell=True, capture_output=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively call the docker-stacks tests"
    )
    parser.add_argument("--cuda_ver", default="11.8.0", type=str)
    parser.add_argument("--containers", default="all", type=str)
    args = parser.parse_args()

    if args.containers == "all":
        containers = get_containers_by_author('samuel62')
    else:
        containers = args.containers.split(',')

    docker_stacks_path = Path("docker-stacks/").resolve()

    for image_prefix in containers:
        query_dir = image_prefix + "-notebook"
        test_dir = docker_stacks_path / query_dir
        cmd = f"""TEST_IMAGE="samuel62/machine_learning_lab:{image_prefix}_cuda_{args.cuda_ver}" """
        if test_dir.is_dir():
            cmd += (
                f"""pytest -x -m "not info" {docker_stacks_path}/test {test_dir}/test"""
            )

        run_cmd(cmd)
