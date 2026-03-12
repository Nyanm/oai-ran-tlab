#!/bin/bash
#/*
# * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The OpenAirInterface Software Alliance licenses this file to You under
# * the OAI Public License, Version 1.1  (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.openairinterface.org/?page_id=698
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about the OpenAirInterface (OAI) Software Alliance:
# *      contact@openairinterface.org
# */

function usage {
    echo "OAI GitLab merge request applying script"
    echo "   Original Author: Raphael Defosseux"
    echo ""
    echo "Usage:"
    echo "------"
    echo ""
    echo "    doGitLabMerge.sh [OPTIONS] [MANDATORY_OPTIONS]"
    echo ""
    echo "Mandatory Options:"
    echo "------------------"
    echo ""
    echo "    --src-repo"
    echo "    Specify the source (head) repository of the pull request."
    echo ""
    echo "    --src-branch #### OR -sb ####"
    echo "    Specify the source branch of the merge request."
    echo ""
    echo "    --src-commit #### OR -sc ####"
    echo "    Specify the source commit ID (SHA-1) of the merge request."
    echo ""
    echo "    --target-repo"
    echo "    Specify the target (base) repo of the pull request (usually develop)."
    echo ""
    echo "    --target-branch #### OR -tb ####"
    echo "    Specify the target branch of the merge request (usually develop)."
    echo ""
    echo "    --target-commit #### OR -tc ####"
    echo "    Specify the target commit ID (SHA-1) of the merge request."
    echo ""
    echo "Options:"
    echo "--------"
    echo "    --help OR -h"
    echo "    Print this help message."
    echo ""
}

if [ $# -ne 12 ] && [ $# -ne 1 ]
then
    echo "Syntax Error: not the correct number of arguments"
    echo ""
    usage
    exit 1
fi

checker=0
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
    shift
    usage
    exit 0
    ;;
    -sr|--src-repo)
    SOURCE_REPO="$2"
    let "checker|=0x1"
    shift 2
    ;;
    -sb|--src-branch)
    SOURCE_BRANCH="$2"
    let "checker|=0x2"
    shift
    shift
    ;;
    -sc|--src-commit)
    SOURCE_COMMIT_ID="$2"
    let "checker|=0x4"
    shift
    shift
    ;;
    -tr|--target-repo)
    TARGET_REPO="$2"
    let "checker|=0x8"
    shift 2
    ;;
    -tb|--target-branch)
    TARGET_BRANCH="$2"
    let "checker|=0x10"
    shift
    shift
    ;;
    -tc|--target-commit)
    TARGET_COMMIT_ID="$2"
    let "checker|=0x20"
    shift
    shift
    ;;
    *)
    echo "Syntax Error: unknown option: $key"
    echo ""
    usage
    exit 1
esac

done

if [[ $TARGET_COMMIT_ID == "latest" ]]
then
    TARGET_COMMIT_ID=`git log -n1 --pretty=format:%H origin/$TARGET_BRANCH`
fi

echo "Source Repo is      : $SOURCE_REPO"
echo "Source Branch is    : $SOURCE_BRANCH"
echo "Source Commit ID is : $SOURCE_COMMIT_ID"
echo "Target Repo is      : $TARGET_REPO"
echo "Target Branch is    : $TARGET_BRANCH"
echo "Target Commit ID is : $TARGET_COMMIT_ID"

if [ $checker -ne 63 ]
then
    echo ""
    echo "Syntax Error: missing option"
    echo ""
    usage
    exit 1
fi

git config user.email "jenkins@openairinterface.org"
git config user.name "OAI Jenkins"

WORKDIR=$(pwd)
echo "Working directory: $WORKDIR"

if ! git fetch origin "$TARGET_BRANCH" || ! git checkout -f "origin/$TARGET_BRANCH"; then
    echo "fatal: target branch $TARGET_BRANCH does not exist in target repo"
    exit 1
fi

# Add source repo as remote and fetch branch
git remote add src $SOURCE_REPO
git fetch src $SOURCE_BRANCH

# Merge source branch tip
echo "Merging $SOURCE_REPO/$SOURCE_BRANCH into $TARGET_REPO/$TARGET_BRANCH"

if ! git merge --ff "src/$SOURCE_BRANCH" -m "Temporary merge from $SOURCE_REPO/$SOURCE_BRANCH for CI"; then
    echo "Merge conflicts detected. Aborting."
    exit 1
fi

# Clean up
git remote remove src

echo "Merge successful!"
exit 0
