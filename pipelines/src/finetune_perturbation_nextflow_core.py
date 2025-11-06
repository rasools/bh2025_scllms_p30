import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import argparse

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)

from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")


# ===========================================================
# Argument parsing
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune scGPT on a single-cell dataset for perturbation task."
    )
    parser.add_argument("--dataset_name", type=str, default="adamson", help="Dataset name")
    parser.add_argument("--split", type=str, default="simulation")
    parser.add_argument("--load_model", type=str, default=None, help="Path to pretrained model directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--n_hvg", type=int, default=1200, help="number of highly variable genes")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layer_size", type=int, default=128)
    parser.add_argument("--fast_transformer", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--GEPC", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--ecs_thres", type=float, default=0.8)
    parser.add_argument("--schedule_ratio", type=float, default=0.9)
    parser.add_argument("--amp", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--do_train", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--MLM", type=bool, default=True)
    parser.add_argument("--CLS", type=bool, default=False)
    parser.add_argument("--CCE", type=bool, default=False)
    parser.add_argument("--MVC", type=bool, default=False)
    return parser.parse_args()


args = parse_args()
set_seed(args.seed)

# ===========================================================
# Configuration from args 
# ===========================================================

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536  # Specific to perturbation task

# settings for training
ECS = args.ecs_thres > 0  # Elastic cell similarity objective
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
epochs = args.epochs
early_stop = 10
schedule_interval = 1

# settings for the model
embsize = args.layer_size  # embedding dimension
d_hid = args.layer_size  # dimension of the feedforward network model in nn.TransformerEncoder
n_layers_cls = 3  

# logging
log_interval = args.log_interval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(f"./save/dev_perturb_{args.dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

pert_data = PertData("./data")
print("args.dataset_name = ", args.dataset_name)
pert_data.load(data_name=args.dataset_name)
pert_data.prepare_split(split=args.split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)


if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    # Overwrite model settings with loaded config
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    use_fast_transformer = model_configs.get("use_fast_transformer", False)
else:
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
)
n_genes = len(genes)


# ===========================================================
# Create and train scGpt
# ===========================================================

ntokens = len(vocab)  # size of vocabulary
model = TransformerGenerator(
    ntokens,
    embsize,
    args.nhead,
    d_hid,
    args.nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=args.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    use_fast_transformer=args.fast_transformer,
)
if load_param_prefixs is not None and args.load_model is not None:
    # only load params that start with the prefix
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
elif args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
model.to(device)

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# Use args.schedule_ratio
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=args.CLS,
                CCE=args.CCE,
                MVC=args.MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(float)
    results["truth"] = truth.detach().cpu().numpy().astype(float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

    return results

best_val_loss = float("inf")
best_val_corr = 0
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    if args.do_train:
        train(
            model,
            train_loader,
        )

    val_res = eval_perturb(valid_loader, model, device)
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    logger.info(f"val_metrics at epoch {epoch}: ")
    logger.info(val_metrics)

    elapsed = time.time() - epoch_start_time
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

    val_score = val_metrics["pearson"]
    if val_score > best_val_corr:
        best_val_corr = val_score
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {val_score:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    scheduler.step()

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")

# ===========================================================
# Evaluations
# ===========================================================

# def predict(
#     model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
# ) -> Dict:
#     """
#     Predict the gene expression values for the given perturbations.

#     Args:
#         model (:class:`torch.nn.Module`): The model to use for prediction.
#         pert_list (:obj:`List[str]`): The list of perturbations to predict.
#         pool_size (:obj:`int`, optional): For each perturbation, use this number
#             of cells in the control and predict their perturbation results. Report
#             the stats of these predictions. If `None`, use all control cells.
#     """
#     adata = pert_data.adata
#     ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
#     if pool_size is None:
#         pool_size = len(ctrl_adata.obs)
#     gene_list = pert_data.gene_names.values.tolist()
#     for pert in pert_list:
#         for i in pert:
#             if i not in gene_list:
#                 raise ValueError(
#                     "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
#                 )

#     model.eval()
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         results_pred = {}
#         for pert in pert_list:
#             cell_graphs = create_cell_graph_dataset_for_prediction(
#                 pert, ctrl_adata, gene_list, device, num_samples=pool_size
#             )
#             loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
#             preds = []
#             for batch_data in loader:
#                 pred_gene_values = model.pred_perturb(
#                     batch_data, include_zero_gene, gene_ids=gene_ids, amp=args.amp
#                 )
#                 preds.append(pred_gene_values)
#             preds = torch.cat(preds, dim=0)
#             results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

#     return results_pred

# test_loader = pert_data.dataloader["test_loader"]
# test_res = eval_perturb(test_loader, best_model, device)
# test_metrics = compute_perturbation_metrics(
#     test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
# )
# print(test_metrics)

# # save the dicts in json
# with open(f"{save_dir}/test_metrics.json", "w") as f:
#     json.dump(test_metrics, f)

# deeper_res = deeper_analysis(pert_data.adata, test_res)
# non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

# metrics = ["pearson_delta", "pearson_delta_de"]
# metrics_non_dropout = [
#     "pearson_delta_top20_de_non_dropout",
#     "pearson_top20_de_non_dropout",
# ]
# subgroup_analysis = {}
# for name in pert_data.subgroup["test_subgroup"].keys():
#     subgroup_analysis[name] = {}
#     for m in metrics:
#         subgroup_analysis[name][m] = []

#     for m in metrics_non_dropout:
#         subgroup_analysis[name][m] = []

# for name, pert_list in pert_data.subgroup["test_subgroup"].items():
#     for pert in pert_list:
#         for m in metrics:
#             subgroup_analysis[name][m].append(deeper_res[pert][m])

#         for m in metrics_non_dropout:
#             subgroup_analysis[name][m].append(non_dropout_res[pert][m])

# for name, result in subgroup_analysis.items():
#     for m in result.keys():
#         mean_value = np.mean(subgroup_analysis[name][m])
#         logger.info("test_" + name + "_" + m + ": " + str(mean_value))