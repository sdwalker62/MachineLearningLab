import sys
import logging
from subprocess import run
from functools import partial
import docker
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

run_cmd = partial(run, shell=True, capture_output=False)

docker_client = docker.from_env()


def tag_docker_image(image: str, target: str, tag: str):
    """re-tag a docker image on the host's machine"""
    logging.info(f"attempting to retag image with tag: {tag}")
    try:
        image = docker_client.images.get(image)
        image.tag(target, tag=tag)
    except docker.errors.ImageNotFound:
        logging.warn(f"Image: {image} not found!")


if __name__ == "__main__":
    docker_stacks_path = Path('docker-stacks/')

    custom_list = ['mll']

    image_prefix = sys.argv[1]
    owner = sys.argv[2]
    cuda_ver = sys.argv[3]

    if image_prefix not in custom_list:
        query_dir = image_prefix + '-notebook'
        test_dir = docker_stacks_path / query_dir
        cmd = f"""TEST_IMAGE="{owner}/machine_learning_lab:{image_prefix}_cuda_{cuda_ver}" """
        if test_dir.is_dir():
            cmd += f"""pytest -m "not info" docker-stacks/test {test_dir}/test"""
        else:
            cmd += f"""pytest -m "not info" docker-stacks/test"""

        run_cmd(cmd)

    # if image_prefix in custom_list:
    #     print(image_prefix)
    # else:
    #     owner = "samuel62"
    #     cmd = f"make -C docker-stacks OWNER={owner} "
    #     run_cmd
	# @echo "::group::test/$(OWNER)/$(notdir $@)"
	# @if [ ! -d "$(notdir $@)/test" ]; then TEST_IMAGE="$(OWNER)/$(notdir $@)" pytest -m "not info" test; \
	# else TEST_IMAGE="$(OWNER)/$(notdir $@)" pytest -m "not info" test $(notdir $@)/test; fi
	# @echo "::endgroup::"
    # image = "samuel62/machine_learning_lab:base_cuda_11.3.1"
    # target = "samuel62/machine_learning_lab"
    # tag = "base_cuda_11.3.1"
    # tag_docker_image(image, target, tag)
