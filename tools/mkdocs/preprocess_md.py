#!/usr/bin/env python3

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
#---------------------------------------------------------------------
# Python for MkDocs
#
#   Required Python Version
#     Python 3.x
# *-------------------------------------------------------------------------------

# Preprocess Markdown files for MkDocs HTML build:
# 1. Rewrite links in all Markdown files to correct relative URLs.
# 2. Remove [[_TOC_]] markers from all Markdown files.
# 3. In doc/README.md, remove '../' from links so they work at the folder root.


import pathlib
import logging

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Link Rewrite Functions
# -----------------------------
def rewrite_doc_links(content: str) -> str:
    """Rewrite doc links in general markdown files."""
    content = content.replace("](doc/", "](")
    content = content.replace("../doc/", "../")
    return content

def rewrite_readme_links(content: str) -> str:
    """Rewrite '../' links in README.md only."""
    return content.replace("../", "")

def remove_toc_markers(content: str) -> str:
    """Remove the [[_TOC_]] Markers from the markdown files."""
    return content.replace("[[_TOC_]]", "")

# -----------------------------
# File Processing
# -----------------------------
def process_file(path: pathlib.Path, rewrite_func) -> None:
    """Read, rewrite, and save file if modified."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return

    original = path.read_text(encoding="utf-8")
    updated = rewrite_func(original)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        logger.info(f"Updated: {path}")
