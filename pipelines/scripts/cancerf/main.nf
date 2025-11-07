nextflow.enable.dsl=2

params.input = params.input ?: null            // Path to input .h5ad
params.outdir = params.outdir ?: 'results'     // Output directory
params.model_dir = params.model_dir ?: 'model/assets'
params.batch_key = params.batch_key ?: null
params.batch_size = params.batch_size ?: 64
params.bio_key = params.bio_key ?: null
params.umap = params.umap ?: false

process EMBED_ADATA {
    tag { sample_id }
    conda "${projectDir}/environment.yml"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(sample_id), path(h5ad)

    output:
    tuple val(sample_id), path("${sample_id}.embedded.h5ad")
    path("${sample_id}.umap.png"), optional: true

    script:
    def umapFlag = params.umap ? "--umap --umap_png ${sample_id}.umap.png --bio_key ${params.bio_key ?: ''}" : ""
    def batchKeyArg = params.batch_key ? "--batch_key ${params.batch_key}" : ""
    """
    export PYTHONPATH=${projectDir}:${'$'}{PYTHONPATH:-}
    python ${projectDir}/zero_shot_batch_integration/embed_and_umap.py \
      --input ${h5ad} \
      --output_h5ad ${sample_id}.embedded.h5ad \
      --model_dir ${projectDir}/${params.model_dir} \
      --batch_size ${params.batch_size} \
      ${batchKeyArg} \
      ${umapFlag}
    """
}

workflow {
    if (!params.input) {
        exit 1, 'Please provide --input pointing to an .h5ad file or pattern'
    }

    Channel
        .fromPath(params.input)
        .ifEmpty { exit 1, "No input files found for pattern: ${params.input}" }
        .map { file ->
            def base = file.baseName
            tuple(base, file)
        }
        | EMBED_ADATA
}
