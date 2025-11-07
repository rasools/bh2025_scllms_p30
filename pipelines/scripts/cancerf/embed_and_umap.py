import argparse
import os
from typing import Optional

import scanpy as sc

from model.embedding import embed


def run_embedding(
    input_h5ad_path: str,
    output_h5ad_path: str,
    model_dir: str,
    batch_key: Optional[str],
    batch_size: int,
    compute_umap: bool,
    bio_key: Optional[str],
    umap_png_path: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(output_h5ad_path) or ".", exist_ok=True)
    adata = sc.read_h5ad(input_h5ad_path)

    embedded_adata = embed(
        adata_or_file=adata,
        model_dir=model_dir,
        batch_key=batch_key,
        batch_size=batch_size,
    )

    # Save immediately after embedding (critical output)
    embedded_adata.write_h5ad(output_h5ad_path)
    
    # Cleanup after save
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if compute_umap:
        import sys
        import numpy as np
        
        # Ensure embedding array is clean and C-contiguous
        if "CancerGPT" in embedded_adata.obsm:
            embedded_adata.obsm["CancerGPT"] = np.ascontiguousarray(
                embedded_adata.obsm["CancerGPT"], dtype=np.float32
            )
        
        gc.collect()
        
        try:
            sc.pp.neighbors(embedded_adata, use_rep="CancerGPT")
            sc.tl.umap(embedded_adata)
            if umap_png_path is not None:
                os.makedirs(os.path.dirname(umap_png_path) or ".", exist_ok=True)
                fig = sc.pl.umap(
                    embedded_adata,
                    color=[bio_key] if bio_key else None,
                    frameon=False,
                    palette=sc.pl.palettes.default_20,
                    legend_loc=None,
                    return_fig=True,
                    title=["UMAP"] if bio_key else ["UMAP"],
                )
                fig.savefig(umap_png_path, bbox_inches="tight", dpi=200)
        except Exception as e:
            print(f"Warning: UMAP computation failed: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed anndata with CancerFoundation and optionally compute UMAP plot.")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output_h5ad", required=True, help="Output path for embedded .h5ad")
    parser.add_argument("--model_dir", default="model/assets", help="Path to model assets directory")
    parser.add_argument("--batch_key", default=None, help="Batch key in .obs for HVG selection")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--umap", action="store_true", help="Compute UMAP and save PNG if --umap_png provided")
    parser.add_argument("--bio_key", default=None, help="Obs key used for UMAP coloring")
    parser.add_argument("--umap_png", default=None, help="Output path for UMAP PNG plot")

    args = parser.parse_args()

    run_embedding(
        input_h5ad_path=args.input,
        output_h5ad_path=args.output_h5ad,
        model_dir=args.model_dir,
        batch_key=args.batch_key,
        batch_size=args.batch_size,
        compute_umap=args.umap,
        bio_key=args.bio_key,
        umap_png_path=args.umap_png,
    )


if __name__ == "__main__":
    main()


