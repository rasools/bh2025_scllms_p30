nextflow.enable.dsl=2

process fine_tuning_cell_types {
    // relative to the launch directory. 
    // outputs copied here after success
    publishDir "./output"

    tag "fine_tuning_cell_types"
    // can be a local or remote docker tag
    // Note: container must use params (evaluated at definition time, not execution time)
    container "${params.container_image}"
        
    input:
    path data_dir
    path model_dir
    path config

    output:
    path "save/*"

    script:
    """
    export CONFIG="${config}"
    python3 "${projectDir}/../scripts/scgpt/fine_tuning_cell_types.py"
    """
}

