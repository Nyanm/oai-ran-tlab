#!/bin/bash
# SPDX-License-Identifier: MIT

function die() {
  echo $@
  exit 1
}

[ $# -eq 3 ] || die "usage: $0 <directory> <branch> <commit>"

set -ex

dir=$1
branch=$2
commit=$3
repo="git@asterix:/home/git/openairinterface5g.git"

rm -rf "${dir}"
git clone --filter=blob:none --branch "${branch}" "${repo}" "${dir}"
cd "${dir}"
git config user.email "jenkins@openairinterface.org"
git config user.name "OAI Jenkins"
git config advice.detachedHead false
mkdir -p cmake_targets/log
git checkout -f "${commit}"
exit 0
