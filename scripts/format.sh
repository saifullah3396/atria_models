#!/usr/bin/env bash

set -e
set -x

uv run ruff check src --fix  $@    # linter
uv run ruff format src --check  # formatter
