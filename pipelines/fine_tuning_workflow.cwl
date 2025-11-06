#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow

label: scGPT Fine-tuning Workflow
doc: Workflow for fine-tuning scGPT model with data and model inputs

inputs:
  script:
    type: File
    default:
      class: File
      path: scripts/scgpt/test_scGPT.py
    doc: Python script to execute
  
  config:
    type: File
    default:
      class: File
      path: scripts/scgpt/config.yml
    doc: YAML configuration file
  
  data_dir:
    type: Directory
    default:
      class: Directory
      path: data
    doc: Directory containing data files
  
  model_dir:
    type: Directory
    default:
      class: Directory
      path: model/pan_cancer
    doc: Directory containing model files

outputs:
  output_files:
    type: File[]
    outputSource: fine_tuning/output_files
    doc: Output files from fine-tuning

steps:
  fine_tuning:
    run: fine_tuning_tool.cwl
    in:
      script: script
      config: config
      data_dir: data_dir
      model_dir: model_dir
    out: [output_files]

