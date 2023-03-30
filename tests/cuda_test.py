import sys
import argparse

from pathlib import Path
cwd = str(Path.cwd())
sys.path.insert(0, cwd)
sys.path.insert(0, cwd + "/docker-stacks")
from utils.docker_utils import get_containers_by_author
from tagging.docker_runner import DockerRunner


def drop_minor_ver(ver: str) -> str:
    sub_tokens = ver.split(".")
    return sub_tokens[0][1]


def test_containers_for_cuda():
    parser = argparse.ArgumentParser(
        description="Recursively call the cuda tests"
    )
    parser.add_argument("--cuda_ver", default="11.8.0", type=str)
    parser.add_argument("--containers", default="all", type=str)
    args = parser.parse_args()

    if args.containers == "all":
        containers = get_containers_by_author('samuel62')
    else:
        containers = args.containers.split(',')

    usr = "samuel62"
    repo = "machine_learning_lab"
    cmd = "ls -l /usr/local"
    for container in containers:
        with DockerRunner(f"{usr}/{repo}:{container}_cuda_{args.cuda_ver}") as c:
            content = DockerRunner.run_simple_command(
                c, cmd=cmd, print_result=True
            )
            assert f"cuda-{drop_minor_ver(args.cuda_ver)}" in content


if __name__ == "__main__":
    test_containers_for_cuda()
