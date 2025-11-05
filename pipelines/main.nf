// Default parameters
params.input_value = "test"
params.processing_mode = "production"

// Python processing process
process run_python_script {
    publishDir "../results"
    container "bh2025scllmsp30:latest"
    tag "$input_val"
    
    input:
    val input_val
    path script
    
    output:
    path "output_*.txt"
    
    script:
    """
    export PROCESSING_MODE="${params.processing_mode}"
    pip show numba
    python3 ${script} ${input_val}
    """
}

// Workflow block
workflow {
    ch_input = channel.of(params.input_value)
    ch_script = file("${projectDir}/scripts/dummy_script.py")
    run_python_script(ch_input, ch_script)
}