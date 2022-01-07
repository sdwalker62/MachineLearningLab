import logging
import sys
import os
import tarfile
import docker

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "docker-stacks"))
from tagging.docker_runner import DockerRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = docker.from_env()
container = 'samuel62/machine_learning_lab:rll_cuda_11.3.1'

def run_cmd(container_name: str, cmd: str) -> str:
    logger.info(f"Executing command {cmd} on {container_name}...")
    with DockerRunner(container_name) as container:
        content = DockerRunner.run_simple_command(
            container, cmd=cmd, print_result=False
        )
    logger.info("... complete")
    return content


def run_tests():

    logger.info("Testing garage ...")
    
    
if __name__ == "__main__":
    run_tests()