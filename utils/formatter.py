#!/usr/bin/env python3

import logging
import sys
import re
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'docker-stacks'))
from tagging.docker_runner import DockerRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_installed_pkgs(container_name: str, cmd: str) -> str:
    logger.info(f'Executing command {cmd} on {container_name}...')
    with DockerRunner(container_name) as container:
        content = DockerRunner.run_simple_command(container, cmd=cmd, print_result=False)
    logger.info('... complete')
    return content


def add_pip_pkgs(container_name: str) -> str:    
    content = get_installed_pkgs(container_name, 'pip list')

    logger.info('Formatting pip packages into tabular format ...')
    lines = content.split('\n')
    for i in range(len(lines)):
        if i == 1:
            lines[i] = re.sub(r"\s", " | ", lines[i])
        else:
            lines[i] = re.sub(r"\s{3,}", " | ", lines[i])
        lines[i] = "| " + lines[i] + " |"

    lines.insert(0, "#1. `pip` Packages")
    logger.info('... complete')

    write_content = '\n'.join(lines)
    return write_content



def add_conda_pkgs(container_name: str) -> str:
    content = get_installed_pkgs(container_name, 'conda list')

    logger.info('Formatting conda packages into tabular format ...')
    lines = content.split('\n')
    for i in range(len(lines)):
        if i == 2:
            # the third line contains the headers, get rid of the comments
            headers = re.sub(r"\s{1,}", " | ", lines[i][2:])
            lines[i] = headers
        elif i > 2:
            lines[i] = re.sub(r"\s{1,}", " | ", lines[i])
        
        lines[i] = "| " + lines[i] + " |"

    # the first two line are comments, delete them
    for _ in range(2): del lines[0]
        
    lines.insert(0, '#2. `conda` Packages')
    lines.insert(2, '| --- | --- | --- | ---|')
    logger.info('... complete')

    write_content = '\n'.join(lines)
    return write_content
    

if __name__ == "__main__":
    container_name = 'samuel62/machine_learning_lab:base_cuda_11.3.1'

    pip_content = add_pip_pkgs(container_name)
    conda_content = add_conda_pkgs(container_name)

    tag = container_name.split(':')[1]
    image_type = tag.split('_')[0]
    image = f'docs/{image_type}_lab.md'

    toc = list()
    toc.append('# Table of Contents')
    toc.append('1. [`pip` Packages](#1-pip-packages)')
    toc.append('2. [`conda` Packages](#2-conda-packages)')
    toc.append('---')
    toc.append('\n')

    logger.info('Writing content to documents ...')
    with open(image, 'w') as w:
        w.write('\n'.join(toc))
        w.write(pip_content)
        w.write('\n')
        w.write('---')
        w.write('\n')
        w.write(conda_content)
    logger.info('...complete')