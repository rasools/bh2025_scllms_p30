// Default parameters
params.config = "${projectDir}/../scripts/scgpt/config.yml"
params.data_dir = "${projectDir}/../data"
params.model_dir = "${projectDir}/../model/pan_cancer"
params.container_image = "bh2025scllmsp30-scgpt:latest"

// Python processing process
process fine_tuning_cell_types {
    publishDir "./save"
    // can be a local or remote docker tag
    container "${params.container_image}"
    tag "fine_tuning_cell_types"
    
    input:
    path data_dir
    path model_dir
    
    output:
    path "output_*.txt"
    
    script:
    """
    export CONFIG="${params.config}"
    echo "projectDir: ${projectDir}"
    python3 "${projectDir}/../scripts/scgpt/fine_tuning_cell_types.py"
    """
}

// Workflow block
workflow {
    ch_data_dir = channel.of(params.data_dir)
    ch_model_dir = channel.of(params.model_dir)
    fine_tuning_cell_types(ch_data_dir, ch_model_dir)
}