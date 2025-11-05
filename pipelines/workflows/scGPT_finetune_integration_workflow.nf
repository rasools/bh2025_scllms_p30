nextflow.enable.dsl=2

// 1. Import the process from the modules/ directory
include { scGPT_finetune_integration } from '../modules/scGPT_finetune_integration.nf'

// 2. Define the main workflow
workflow {
    scGPT_finetune_integration()
}