#!/usr/bin/env bash

set -e
set -x

uv run coverage run --source=atria_models -m pytest $@ 
uv run coverage report --show-missing
