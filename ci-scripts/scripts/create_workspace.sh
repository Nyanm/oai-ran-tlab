#!/bin/bash
# SPDX-License-Identifier: MIT

function die() {
  echo $@
  exit 1
}

[ $# -eq 3 ] || die "usage: $0 <directory> <repo> <branch>"

set -ex

dir=$1
repo=$2
branch=$3

rm -rf "${dir}"
git clone --filter=blob:none --depth=1 --branch "${branch}" "${repo}" "${dir}"
cd "${dir}"

# Ensure we are exactly at the latest remote branch state
git fetch origin "${branch}"
git checkout -B "${branch}" "origin/${branch}"
git reset --hard "origin/${branch}"

mkdir -p cmake_targets/log
exit 0
