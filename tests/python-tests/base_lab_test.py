import logging
import sys
import re
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "../docker-stacks"))
from tagging.docker_runner import DockerRunner


def test_cuda():
    container_name = "samuel62/machine_learning_lab:base_cuda_11.3.1"

    cmd = "ls -l /usr/local"

    with DockerRunner(container_name) as container:
        content = DockerRunner.run_simple_command(container, cmd=cmd, print_result=True)

    assert "cuda-11.3" in content


# if __name__ == "__main__":
#     test_cuda()
