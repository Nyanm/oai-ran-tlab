
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
#---------------------------------------------------------------------

#!/usr/bin/env python3

import re
import yaml
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def configure_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


logger = logging.getLogger("mkdocs-nav-generator")


# -----------------------------------------------------------------------------
# Repository Root Detection
# -----------------------------------------------------------------------------

def get_repo_root() -> Path:
    """
    Detect repository root dynamically.
    """
    return Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def clean_path(path: str) -> str:
    """
    Normalize Markdown link paths for MkDocs usage.
    mkdocs-simple-plugin resolves paths relative to docs_dir.
    """
    return path.replace("../", "").replace("./", "")


# -----------------------------------------------------------------------------
# Markdown Parser
# -----------------------------------------------------------------------------

class MarkdownParser:
    HEADING_RE = re.compile(r"^(#{2,6})\s+(.*)")
    LINK_RE = re.compile(r"^- \[(.+?)\]\((.+?)\)")

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def parse(self) -> List[Dict[str, Any]]:
        logger.info(f"Parsing markdown file: {self.filepath}")

        if not self.filepath.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.filepath}")

        # Root of navigation tree
        root: List[Dict[str, Any]] = []
        # Key = heading level, Value = list of children
        current_nodes: Dict[int, List] = {1: root}

        with self.filepath.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()

                heading_match = self.HEADING_RE.match(line)
                if heading_match:
                    hashes, title = heading_match.groups()
                    level = len(hashes)

                    node = {title: []}

                    parent_level = level - 1
                    while parent_level > 1 and parent_level not in current_nodes:
                        parent_level -= 1

                    if parent_level in current_nodes:
                        current_nodes[parent_level].append(node)
                    else:
                        root.append(node)

                    current_nodes[level] = node[title]

                    for deeper in list(current_nodes.keys()):
                        if deeper > level:
                            del current_nodes[deeper]

                    continue

                # Detect markdown bullet links
                link_match = self.LINK_RE.match(line)
                if link_match:
                    title, path = link_match.groups()
                    entry = {title: clean_path(path)}

                    deepest_level = max(current_nodes.keys())
                    current_nodes[deepest_level].append(entry)

        logger.info("Markdown parsing complete.")
        return root


# -----------------------------------------------------------------------------
# Nav Builder
# -----------------------------------------------------------------------------

class MkDocsNavBuilder:
    """
    Wraps parsed markdown tree into MkDocs-compatible structure
    """
    def __init__(self, markdown_tree: List[Dict[str, Any]]):
        self.markdown_tree = markdown_tree

    def build(self) -> str:
        logger.info("Building mkdocs nav YAML block.")

        nav_structure = {
            "nav": [
                {"Home": "README.md"},
                {"RAN": self.markdown_tree},
            ]
        }

        yaml_block = yaml.dump(
            nav_structure,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        )

        return yaml_block


# -----------------------------------------------------------------------------
# YAML Injection Logic
# -----------------------------------------------------------------------------

class MkDocsInjector:
    """
    Injects generated navigation into existing mkdocs.yml.
    Only modifies content between defined navigation markers.
    """

    NAV_START = "## Navigation Entries Block Starts"
    NAV_END = "## Navigation Entries Block Ends"

    def __init__(self, mkdocs_path: Path):
        self.mkdocs_path = mkdocs_path

    def inject(self, nav_yaml: str):
        """
        Replace everything between NAV_START and NAV_END
        with the newly generated nav_yaml block.
        """
        logger.info(f"Injecting nav into: {self.mkdocs_path}")

        content = self.mkdocs_path.read_text(encoding="utf-8")

        if self.NAV_START not in content or self.NAV_END not in content:
            raise RuntimeError(
                f"Markers '{self.NAV_START}' or '{self.NAV_END}' not found in mkdocs.yml"
            )

        # Regex to match everything between the two markers (including newlines)
        pattern = re.compile(
            rf"({re.escape(self.NAV_START)})(.*)({re.escape(self.NAV_END)})",
            re.DOTALL,
        )

        def replacer(match):
            return (
                f"{match.group(1)}\n\n"
                f"{nav_yaml.strip()}\n\n"
                f"{match.group(3)}"
            )

        updated_content = re.sub(pattern, replacer, content)

        self.mkdocs_path.write_text(updated_content, encoding="utf-8")

        logger.info("Navigation successfully injected.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    repo_root = get_repo_root()

    default_input = repo_root / "doc" / "README.md"
    default_mkdocs = repo_root / "mkdocs.yml"

    parser = argparse.ArgumentParser(
        description="Inject generated navigation into mkdocs.yml"
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--mkdocs", type=Path, default=default_mkdocs)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    configure_logging(args.debug)

    logger.info("Starting navigation generation + injection")

    parsed_tree = MarkdownParser(args.input).parse()
    nav_yaml = MkDocsNavBuilder(parsed_tree).build()

    injector = MkDocsInjector(args.mkdocs)
    injector.inject(nav_yaml)


if __name__ == "__main__":
    main()
