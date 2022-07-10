import sys
import os
import yaml


sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "docker-stacks"))
from tagging.docker_runner import DockerRunner

sys.path.append("..")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f.read())

cuda_ver = cfg["meta"]["cuda_ver"]
containers = cfg["meta"]["containers"]


def drop_minor_ver(cuda_ver: str) -> str:
    subtokens = cuda_ver.split(".")
    return subtokens[0][1]


def test_containers_for_cuda():
    cmd = "ls -l /usr/local"
    for container in containers:
        container_name = f"samuel62/machine_learning_lab:{container}_cuda_{cuda_ver}"
        with DockerRunner(container_name) as container:
            content = DockerRunner.run_simple_command(
                container, cmd=cmd, print_result=True
            )
            assert f"cuda-{drop_minor_ver(cuda_ver)}" in content


if __name__ == "__main__":
    test_containers_for_cuda()
