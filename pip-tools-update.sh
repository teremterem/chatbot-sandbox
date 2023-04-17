#!/bin/sh
. venv/bin/activate
pip-compile --upgrade --resolver=backtracking requirements.in
pip-compile --upgrade --resolver=backtracking dev-requirements.in
pip-sync requirements.txt dev-requirements.txt
