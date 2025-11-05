// Default parameters
params.config = "${projectDir}/scripts/scgpt/config.yml"
params.data_dir = "${projectDir}/data"
params.model_dir = "${projectDir}/model/pan_cancer"

// Python processing process
process fine_tuning {
    publishDir "./save"
    // can be a local or remote docker tag
    container "bh2025scllmsp30-scgpt:latest"
    tag "${script.baseName}"
    
    input:
    path script
    path data_dir
    path model_dir
    
    output:
    path "output_*.txt"
    
    script:
    """
    export CONFIG="${params.config}"
    python3 ${script}
    """
}

// Workflow block
workflow {
    ch_data_dir = channel.of(params.data_dir)
    ch_model_dir = channel.of(params.model_dir)
    ch_script = file("${projectDir}/scripts/scgpt/test_scGPT.py")
    fine_tuning(ch_script, ch_data_dir, ch_model_dir)
}