nextflow.enable.dsl=2

process scGPT_finetune_integration {
    tag "scGPT finetune on ${params.adata_name}"
    label 'withGpu' // We use this label to assign GPU resources in the config

    publishDir "${params.output_dir}", mode: 'copy', overwrite: true

    output:
    path "results/**", emit: fine_tune_results

    // This process has no explicit 'input:' channels
    // It relies entirely on 'params' from the config file

    script:
    """
    #!/bin/bash
    
    # ====================== ENVIRONMENT SETUP ======================
    ml purge
    ml apptainer cuda

    # WARNING: This pip install runs on the host, not in the container.
    # It will not affect the container's environment.
    # All packages should be in your SIF_FILE_NF.
    pip install --upgrade scib scanpy pandas scvi-tools
    
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
    
    # Define paths from global params.
    DATA_DIR_NF="${params.data_dir}"
    MODEL_DIR_NF="${params.model_dir}"
    SCRIPT_PATH_NF="${params.script_path}"
    SIF_FILE_NF="${params.sif_file}"

    # 1. Create the *internal* output directory where the Python script will write its results.
    mkdir -p ./results 

    echo "================================================================="
    echo "Starting scGPT fine-tuning job via Nextflow"
    echo "Data dir:          \$DATA_DIR_NF"
    echo "Model dir:         \$MODEL_DIR_NF"
    echo "Internal Output:   ./results"
    echo "Final Publish Dir: ${params.output_dir}" 
    echo "Script path:       \$SCRIPT_PATH_NF"
    echo "================================================================="

    # ====================== RUN TRAINING (The Core Command) ======================
    
    apptainer exec --nv \\
        --bind \$DATA_DIR_NF:/data \\
        --bind \$MODEL_DIR_NF:/models \\
        --bind ./results:/output \\
        --bind \$(dirname \$SCRIPT_PATH_NF):/script_dir \\
        "\$SIF_FILE_NF" \\
        python /script_dir/\$(basename \$SCRIPT_PATH_NF) \\
            --adata_path /data/${params.adata_name} \\
            --load_model /models \\
            --vocab_path /models/vocab.json \\
            --output_dir /output \\
            --data_is_raw True \\
            --epochs ${params.epochs} \\
            --batch_size ${params.batch_size} \\
            --fast_transformer ${params.fast_transformer} \\
            --do_train True
    
    STATUS=\$?
    
    echo "================================================================="
    if [ \$STATUS -eq 0 ]; then
        echo "✅ Job completed successfully. Files being published to ${params.output_dir}"
    else
        echo "❌ Job failed with exit code \$STATUS"
    fi
    echo "Finished at: \$(date)"
    echo "================================================================="
    """
}