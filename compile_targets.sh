#!/bin/bash

function die() { echo $@ 1>&2; exit 1; }

ALL_TARGETS=$(cat ninja-targets) # default targets if none provided
[ $# -gt 0 ] && ALL_TARGETS=$@

rm results.txt
mkdir results

function build_target() {
  [ $# -eq 1 ] || die "need target"
  local target=$1
  local dir=$(mktemp -p. -d)
  CMAKE_OPT="-DENABLE_TESTS=ON -DENABLE_TELNETSRV=ON -DENABLE_UESCOPE=ON -DENABLE_NRSCOPE=ON -DENABLE_ENBSCOPE=ON -DENABLE_IMSCOPE_RECORD=ON -DENABLE_IMSCOPE=ON"
  echo cmake -GNinja ${CMAKE_OPT} -B ${dir} > results/${target}.log
  ( cmake -GNinja ${CMAKE_OPT} -B ${dir} && ninja -j4 -C ${dir} ${target} ) >> results/${target}.log 2>&1
  local result=$([ $? -eq 0 ] && echo built || echo failed)
  echo -e "${result}\t${target}" | tee -a results.txt
  rm -rf ${dir}
}
export -f build_target

parallel -j12 build_target ::: ${ALL_TARGETS}
