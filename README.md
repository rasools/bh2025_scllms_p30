# bh2025_scllms_p30

Monorepo layout for scGPT, Cancer Foundation, and Nextflow pipelines.

## Structure

- external/scgpt: your fork of scGPT as a git submodule
- external/cancerfoundation: your fork of Cancer Foundation as a git submodule
- pipelines: Nextflow pipelines and configs

## Setup

- Use gh to fork upstream repos and clone this repo, then add submodules under external/

### prerequisite

- docker

## usage

```bash
cd pipelines
nextflow run main.nf
```

```bash
make run
```
