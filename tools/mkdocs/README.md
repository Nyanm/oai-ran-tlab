# OAI Documentation using MkDocs

<h1 align="center">
    <a href="https://openairinterface.org/">
        <img src="https://openairinterface.org/wp-content/uploads/2015/06/cropped-oai_final_logo.png" alt="OAI" width="550">
    </a>
</h1>

<p align="center">
    <a href="https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed/-/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-OAI--Public--V1.1-blue" alt="License">
    </a>
    <a href="https://www.mkdocs.org/">
        <img src="https://img.shields.io/badge/Documentation_Using-MkDocs-20c997?style=flat&logo=mkdocs&logoColor=white" alt="MkDocs">
    </a>
    <a href="https://squidfunk.github.io/mkdocs-material/getting-started/">
        <img src="https://img.shields.io/badge/Theme-Material_for_MkDocs-ffc107" alt="Material for MkDocs">
    </a>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-fd7e14?style=flat&logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://althack.dev/mkdocs-simple-plugin/v3.2.4/">
        <img src="https://img.shields.io/badge/mkdocs--simple--plugin-6f42c1" alt="mkdocs-simple-plugin v3.2.4">
    </a>
</p>


---

## Overview

This is a guide to build and serve the `openairinterface5g` documentation using [MkDocs](https://www.mkdocs.org/)
with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

---

## Prerequisites

### Python

Make sure `Python 3` is installed:

```bash
python3 --version
# Example output: Python 3.10.12
```

### Install Dependencies

Install all required Python packages using `pip` and the provided [requirements.txt](requirements.txt) file:

```bash
pip install -r requirements.txt
```

---


## MkDocs Files

Here’s a brief overview of the files related to MkDocs:

| Folder / File                               | Description                                             |
| --------------------------------------------------- | ------------------------------------------------------- |
| `mkdocs.yml`                    | Main MkDocs configuration file                           |
| `tools/mkdocs/requirements.txt`         | Python dependencies required to build the documentation |
| `doc/`                                  | Main folder containing markdown documentation files    |
| `doc/overrides/`             | Custom theme templates or overrides for Material theme |
| `doc/assets/images/`         | Logo and favicon images                                 |
| `doc/assets/stylesheets/extra.css` | Extra custom CSS for styling                            |
| `tools/mkdocs/mkdocs_nav_generator.py` | Script to automatically generate the navigation in mkdocs.yml |

---

## Generating Navigation

The navigation for MkDocs is generated automatically using the custom Python tool:

```bash
python3 tools/mkdocs/mkdocs_nav_generator.py
```

* This script scans the documentation files listed in [doc/README.md](../../doc/README.md)
* It generates the navigation entries for your [mkdocs.yml](../../mkdocs.yml)

---

## Building the Documentation

To generate the static site:

```bash
mkdocs build 2>&1 | tee mkdocs_build.log
```

The output will be in the `site/` directory.


---

## Serving Locally

To preview the documentation locally, run the below command and open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

```bash
mkdocs serve
```

---

## Versioned Documentation with Mike

[Mike](https://github.com/jimporter/mike/blob/master/README.md) allows to maintain multiple versions of the documentation.

**1. Deploy a New Version**

```bash
# Deploy documentation as version 1.0 and assign it the alias 'latest'
mike deploy 1.0 latest
```

This command generates a versioned snapshot of your documentation for `1.0` and links the alias latest to it for easy access.

**2. Set the default version**

```bash
mike set-default latest
```

This version will be served as the default when users visit the documentation.

**3. List All Deployed Versions**

```bash
mike list
```

Displays all deployed versions along with their aliases.

**4. Preview versioned documentation**

```bash
mike serve
```

You can view the site on [http://localhost:8000/](http://localhost:8000/).

---
