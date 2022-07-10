import sys
import yaml

from subprocess import run
from functools import partial
from pathlib import Path


sys.path.append("..")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f.read())

cuda_ver = cfg["meta"]["cuda_ver"]
containers = cfg["meta"]["containers"]

run_cmd = partial(run, shell=True, capture_output=False)


if __name__ == "__main__":
    docker_stacks_path = Path("docker-stacks/").resolve()

    for image_prefix in containers:
        query_dir = image_prefix + "-notebook"
        test_dir = docker_stacks_path / query_dir
        cmd = f"""TEST_IMAGE="samuel62/machine_learning_lab:{image_prefix}_cuda_{cuda_ver}" """
        if test_dir.is_dir():
            cmd += (
                f"""pytest -x -m "not info" {docker_stacks_path}/test {test_dir}/test"""
            )

        run_cmd(cmd)
