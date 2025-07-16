#!/usr/bin/env bash

set -e
set -x

# Use provided directory or default to 'src'
DIR=${1:-src}

uv run mypy "$DIR"           # type check
uv run ruff check "$DIR"     # linter
uv run ruff format "$DIR" --check # formatter
