#!/usr/bin/env python3

import re 
import sys

replace_string = sys.argv[1]
reg_exp = r"(?<=ARG ROOT_CONTAINER=).*"

with open('Dockerfile', 'r+') as f:
    content = f.read()
    content = re.sub(reg_exp, replace_string, content)
    f.seek(0)
    f.write(content)
    f.truncate()
    f.close()