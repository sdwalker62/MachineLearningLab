#!/usr/bin/env python3

import logging
import sys
import re
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'docker-stacks'))
from tagging.docker_runner import DockerRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_pip_pkgs(container):
    logger.info(f'Executing container cmd on {container}...')
    with DockerRunner(container) as container:
        content = DockerRunner.run_simple_command(container, cmd="pip list", print_result=False)
    logger.info('... complete')

    logger.info('Formatting pip packages into tabular format ...')
    lines = content.split('\n')
    for i in range(len(lines)):
        if i == 1:
            lines[i] = re.sub(r"\s", " | ", lines[i])
        else:
            lines[i] = re.sub(r"\s{3,}", " | ", lines[i])
        lines[i] = "| " + lines[i] + " |"
    logger.info('... complete')

    # write_content = '\n'.join(lines)
    # with open('pip_table.txt', 'w') as w:
    #     w.write(write_content)
    
if __name__ == "__main__":
    container = 'samuel62/machine_learning_lab:base_cuda_11.3.1'

    add_pip_pkgs(container)