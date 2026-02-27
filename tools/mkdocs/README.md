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

Make sure `Python 3.10` or higher is installed:

```bash
python3 --version
```

### Python Dependencies

Install the Python packages required to build the documentation:

```bash
pip install -r tools/mkdocs/requirements.txt
```

### Docker

If you want to build and serve the documentation using Docker, install docker following the [official instructions](https://docs.docker.com/engine/install/).

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
| `tools/mkdocs/main.py` | Script to generate the navigation in `mkdocs.yml` and pre-process the markdown files |
| `tools/mkdocs/Dockerfile` | Build and run MkDocs using Docker |

---


## Building the Documentation

### Native MkDocs

#### Generate the navigation and preprocess Markdown


```bash
python3 tools/mkdocs/main.py
```

#### Build the static site

To generate the static site:

```bash
mkdocs build 2>&1 | tee mkdocs_build.log
```

The output will be in the `site/` directory.


#### Serve locally to preview

To preview the documentation locally, run the below command and open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

```bash
mkdocs serve
```

---

### Using Docker

#### Build the Docker image

```bash
docker build --no-cache -f tools/mkdocs/Dockerfile -t oai-mkdocs:latest .
```

- This installs dependencies from `tools/mkdocs/requirements.txt`.
- Runs `main.py` to generate the MkDocs navigation and preprocess Markdown so it renders correctly.
- Builds the MkDocs static site.

#### Serve the documentation using Docker

```bash
docker run -d   --name oai-mkdocs   -p 8000:8000   oai-mkdocs:latest
```

To preview the documentation, open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

#### View Docker logs

```bash
docker logs oai-mkdocs > mkdocs-docker.log 2>&1
```

To stop and remove the container, you can run below commands:

```bash
docker stop oai-mkdocs
docker rm oai-mkdocs
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
