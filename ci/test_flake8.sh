#!/usr/bin/env bash

. tools/activate_python.sh

set -euo pipefail

flake8_black_list=""


n_blacklist=$(wc -l <<< "${flake8_black_list}")
n_all=$(find SVS -name "*.py" | wc -l)
n_ok=$((n_all - n_blacklist))
cov=$(echo "scale = 4; 100 * ${n_ok} / ${n_all}" | bc)
echo "flake8-docstrings ready files coverage: ${n_ok} / ${n_all} = ${cov}%"

# --extend-ignore for wip files for flake8-docstrings
flake8 --show-source --extend-ignore=D utils ${flake8_black_list} SVS egs/*/*/local/*.py

# white list of files that should support flake8-docstrings
flake8 --show-source SVS --exclude=${flake8_black_list//$'\n'/,}
