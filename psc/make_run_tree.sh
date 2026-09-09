#!/bin/bash
# Build $PROJECT/pengu_hw on a Bridges-2 LOGIN node: an isolated, non-git run tree for the
# realism filter and the hardware_c1 sweep, in the same spirit as $PROJECT/pengu
# (PSC_ISOLATED_TREE, 2026-08-31): nothing here is a git checkout, so no reviver, watchdog
# or accidental commit can touch it.
#
# The login node has internet; compute nodes do not. Fetches a shallow, sparse clone of
# the branch from GitHub, copies only what the two scripts import, and deletes the clone.
#
#   bash make_run_tree.sh <branch>
#
# Costs zero SU.

set -euo pipefail
BRANCH="${1:?branch}"
REPO="https://github.com/robomechanics/pengu.git"
TREE="${PROJECT}/pengu_hw"
TMP="${PROJECT}/.pengu_hw_clone"

rm -rf "${TMP}"
git clone --quiet --depth 1 --branch "${BRANCH}" --filter=blob:none --sparse "${REPO}" "${TMP}"
git -C "${TMP}" sparse-checkout set --no-cone \
    'pengu_mujoco/*.py' 'pengu_mujoco/grid6/*.py' \
    'models/pengu1_05_hw_updated' 'models/pengu1_20_hw_updated' 'models/pengu1_31' \
    'pengu_mujoco/results/grid6_hw/filter_cells_c6.csv' \
    'pengu_mujoco/results/grid6_hw/c1/*.csv' 'pengu_mujoco/results/grid6_hw/c2/*.csv' \
    'pengu_mujoco/results/grid6_hw/c5/*.csv' 'pengu_mujoco/results/grid6_hw/c6/*.csv' \
    'pengu_mujoco/results/grid6_hw/c3/*.csv' 'pengu_mujoco/results/grid6_hw/c4/*.csv' \
    'psc/*.slurm' 'psc/make_run_tree.sh'
SHA=$(git -C "${TMP}" rev-parse --short HEAD)

mkdir -p "${TREE}/pengu_mujoco/grid6" "${TREE}/pengu_mujoco/results/grid6_hw" \
         "${TREE}/pengu_mujoco/results/grid6_report" "${TREE}/models" "${TREE}/psc" "${TREE}/logs"
cp "${TMP}"/pengu_mujoco/*.py            "${TREE}/pengu_mujoco/"
cp "${TMP}"/pengu_mujoco/grid6/*.py      "${TREE}/pengu_mujoco/grid6/"
cp -r "${TMP}"/models/pengu1_05_hw_updated "${TREE}/models/"
cp -r "${TMP}"/models/pengu1_20_hw_updated "${TREE}/models/"
cp -r "${TMP}"/models/pengu1_31            "${TREE}/models/"
cp "${TMP}"/pengu_mujoco/results/grid6_hw/filter_cells_c6.csv "${TREE}/pengu_mujoco/results/grid6_hw/"
for c in c1 c2 c3 c4 c5 c6; do
  mkdir -p "${TREE}/pengu_mujoco/results/grid6_hw/${c}"
  cp "${TMP}"/pengu_mujoco/results/grid6_hw/${c}/cells_*.csv "${TREE}/pengu_mujoco/results/grid6_hw/${c}/" 2>/dev/null || true
done
cp "${TMP}"/psc/*.slurm "${TMP}"/psc/make_run_tree.sh "${TREE}/psc/"
rm -rf "${TMP}"

cat > "${TREE}/PSC_ISOLATED_TREE" <<EOF
Isolated run tree for grid6/realism_check.py and grid6/hw_sweep.py, built $(date -u +%F) from
${REPO} branch ${BRANCH} @ ${SHA} by psc/make_run_tree.sh. Not a git checkout on purpose.
Outputs: pengu_mujoco/results/grid6_report/realism_c6.*.csv, pengu_mujoco/results/grid6_hw/<cfg>/hwact_*.csv
EOF

echo "tree ${TREE} @ ${SHA}"
find "${TREE}" -type f | wc -l | xargs echo "files:"
du -sh "${TREE}"
