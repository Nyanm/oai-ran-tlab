#!/usr/bin/env python3

# * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The OpenAirInterface Software Alliance licenses this file to You under
# * the OAI Public License, Version 1.1  (the "License"); you may not use this file
# * except in compliance with the License.
# *
# *      http://www.openairinterface.org/?page_id=698
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */
#---------------------------------------------------------------------
# Python for MkDocs
#
#   Required Python Version
#     Python 3.x
# *-------------------------------------------------------------------------------

from pathlib import Path
import logging
import argparse
from preprocess_md import (
    process_file,
    rewrite_doc_links,
    rewrite_readme_links,
    remove_toc_markers
)
from mkdocs_nav_generator import (
    configure_logging,
    get_repo_root,
    MarkdownParser,
    MkDocsNavBuilder,
    MkDocsInjector
)

logger = logging.getLogger(__name__)

def generate_and_inject_nav(debug: bool = False) -> None:
    """Generate MkDocs navigation from README.md and inject into mkdocs.yml."""
    configure_logging(debug)
    logger.info("Starting navigation generation + injection")

    repo_root = get_repo_root()
    input_md = repo_root / "doc" / "README.md"
    mkdocs_yaml = repo_root / "mkdocs.yml"

    parsed_tree = MarkdownParser(input_md).parse()
    nav_yaml = MkDocsNavBuilder(parsed_tree).build()

    injector = MkDocsInjector(mkdocs_yaml)
    injector.inject(nav_yaml)

    logger.info("MkDocs navigation generation completed.")

def rewrite_markdown_links() -> None:
    """Rewrite Markdown links and remove TOC markers in all .md files."""
    logger.info("Starting Markdown link rewrite.")

    root = Path(".")
    for md_file in root.rglob("*.md"):
        process_file(md_file, rewrite_doc_links)
        process_file(md_file, remove_toc_markers)

    # Special handling for doc/README.md
    readme_file = root / "doc" / "README.md"
    process_file(readme_file, rewrite_readme_links)

    logger.info("Markdown link rewrite completed.")

def main() -> None:
    # -----------------------------
    # CLI argument parsing
    # -----------------------------
    parser = argparse.ArgumentParser(description="Process Markdown and generate MkDocs navigation.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s | %(message)s")

    # -----------------------------
    # Step 1: Generate navigation
    # -----------------------------
    generate_and_inject_nav(debug=args.debug)

    # -----------------------------
    # Step 2: Rewrite Markdown links
    # -----------------------------
    rewrite_markdown_links()

if __name__ == "__main__":
    main()
