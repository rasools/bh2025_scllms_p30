import argparse
import yaml
import os
import json
import pickle
import scanpy as sc
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
import pandas as pd
from pathlib import Path

# --- YAML argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
args = parser.parse_args()

with open(args.config, "r") as f:
    CFG = yaml.safe_load(f)

# --- Paths from config ---
model_path = CFG["model_path"]
vocab_json = CFG["vocab_json"]
args_json = CFG["args_json"]
adata_path = CFG["adata_path"]
adata_filtered_path = CFG["adata_filtered_path"]
output_dir = Path(CFG["output_dir"])
wandb_api_key = CFG.get("wandb_api_key", "")
wandb_project = CFG.get("wandb_project", "scGPT")

# --- Generic keys for AnnData ---
celltype_key = CFG.get("celltype_key", "celltype")
batch_key = CFG.get("batch_key", "batch")
gene_name_key = CFG.get("gene_name_key", "gene_name")
pred_key = CFG.get("pred_key", "predictions")
label_id_key = CFG.get("label_id_key", f"{celltype_key}_id")

output_dir.mkdir(parents=True, exist_ok=True)

# --- Type safety patch for numeric fields ---
def ensure_numeric(cfg, key, cast_type=float):
    """Ensure config value is of correct numeric type."""
    if key in cfg and isinstance(cfg[key], str):
        try:
            cfg[key] = cast_type(eval(cfg[key], {"__builtins__": {}}))  # support '1e-4'
        except Exception:
            try:
                cfg[key] = cast_type(cfg[key])
            except Exception:
                raise ValueError(f"Invalid format for {key}: {cfg[key]}")

# Apply for numeric hyperparameters
for numeric_key in ["lr", "mask_ratio", "ecs_thres", "dab_weight",
                    "dropout", "schedule_ratio"]:
    ensure_numeric(CFG, numeric_key, float)

for numeric_key in ["seed", "epochs", "n_bins", "batch_size",
                    "layer_size", "nlayers", "nhead", "save_eval_interval"]:
    ensure_numeric(CFG, numeric_key, int)

# --- Imports identical to original ---
import copy
import gc
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

from anndata import AnnData
import scvi
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# --- Hyperparameters from config ---
hyperparameter_defaults = dict(
    seed=CFG["seed"],
    dataset_name=CFG["dataset_name"],
    do_train=CFG["do_train"],
    mask_ratio=CFG["mask_ratio"],
    epochs=CFG["epochs"],
    n_bins=CFG["n_bins"],
    MVC=CFG["MVC"],
    ecs_thres=CFG["ecs_thres"],
    dab_weight=CFG["dab_weight"],
    lr=CFG["lr"],
    batch_size=CFG["batch_size"],
    layer_size=CFG["layer_size"],
    nlayers=CFG["nlayers"],
    nhead=CFG["nhead"],
    dropout=CFG["dropout"],
    schedule_ratio=CFG["schedule_ratio"],
    save_eval_interval=CFG["save_eval_interval"],
    fast_transformer=CFG["fast_transformer"],
    pre_norm=CFG["pre_norm"],
    amp=CFG["amp"],
    include_zero_gene=CFG["include_zero_gene"],
    freeze=CFG["freeze"],
    DSBN=CFG["DSBN"],
)

run = wandb.init(
    config=hyperparameter_defaults,
    project=wandb_project,
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
    mode="offline"
)
config = wandb.config
print(dict(config))

set_seed(int(config.seed))

# --- Define constants ---
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = float(config.mask_ratio)
mask_value = "auto"

include_zero_gene = bool(config.include_zero_gene)
max_seq_len = 3001
n_bins = int(config.n_bins)

input_style = "binned"
output_style = "binned"

MLM = False
CLS = True
ADV = False
CCE = False
MVC = bool(config.MVC)
ECS = float(config.ecs_thres) > 0
DAB = False
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "cls"
adv_E_delay_epochs = 0
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = float(config.ecs_thres)
dab_weight = float(config.dab_weight)

explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False

lr = float(config.lr)
lr_ADV = 1e-3
batch_size = int(config.batch_size)
eval_batch_size = int(config.batch_size)
epochs = int(config.epochs)
schedule_interval = 1

fast_transformer = bool(config.fast_transformer)
fast_transformer_backend = "flash"
embsize = int(config.layer_size)
d_hid = int(config.layer_size)
nlayers = int(config.nlayers)
nhead = int(config.nhead)
dropout = float(config.dropout)

log_interval = 100
save_eval_interval = int(config.save_eval_interval)
do_eval_scib_metrics = True

# --- Pad and mask value definitions (identical to original script) ---
if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

# --- Data loading (genericized to keys from YAML) ---
dataset_name = config.dataset_name
save_dir = output_dir
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

adata = sc.read(adata_path)
adata_test = sc.read(adata_filtered_path)

# Map celltype + batch using provided keys
if celltype_key not in adata.obs.columns:
    raise KeyError(f"celltype_key '{celltype_key}' not found in adata.obs")
if celltype_key not in adata_test.obs.columns:
    raise KeyError(f"celltype_key '{celltype_key}' not found in adata_test.obs")

adata.obs["celltype"] = adata.obs[celltype_key].astype("category")
adata_test.obs["celltype"] = adata_test.obs[celltype_key].astype("category")

if batch_key not in adata.obs.columns:
    adata.obs[batch_key] = "0"
if batch_key not in adata_test.obs.columns:
    adata_test.obs[batch_key] = "1"

adata.obs["batch_id"] = adata.obs[batch_key]
adata_test.obs["batch_id"] = adata_test.obs[batch_key]

# Gene name handling: set index from gene_name_key if present, else keep index
if (gene_name_key in adata.var.columns) and (adata.var.index.name != gene_name_key):
    adata.var.set_index(adata.var[gene_name_key].astype(str), inplace=True)
if (gene_name_key in adata_test.var.columns) and (adata_test.var.index.name != gene_name_key):
    adata_test.var.set_index(adata_test.var[gene_name_key].astype(str), inplace=True)

adata.var["gene_name"] = adata.var.index.astype(str).tolist()
adata_test.var["gene_name"] = adata_test.var.index.astype(str).tolist()

data_is_raw = False
filter_gene_by_counts = False
adata_test_raw = adata_test.copy()

# Concatenate (original logic)
adata = adata.concatenate(adata_test, batch_key="batch_id")

# --- Labels ---
batch_id_labels = adata.obs["batch_id"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes_cat = adata.obs["celltype"].astype("category")
celltypes = celltypes_cat.cat.categories.tolist()
num_types = len(celltypes)
id2type = dict(enumerate(celltypes))
adata.obs[label_id_key] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.astype(str).tolist()

# --- Model, vocab, and config loading ---
model_file = model_path
vocab_file = vocab_json
model_config_file = args_json

vocab = GeneVocab.from_file(vocab_file)
shutil.copy(vocab_file, save_dir / "vocab.json")
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
logger.info(
    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}."
)
adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

# Model config
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
logger.info(f"Resume model from {model_file}, the model args will override {model_config_file}.")

embsize = int(model_configs["embsize"])
nhead = int(model_configs["nheads"])
d_hid = int(model_configs["d_hid"])
nlayers = int(model_configs["nlayers"])
n_layers_cls = int(model_configs.get("n_layers_cls", 3))

# Preprocessor identical to original
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=filter_gene_by_counts,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=data_is_raw,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    hvg_flavor="cell_ranger",
    binning=int(n_bins),
    result_binned_key="X_binned",
)

# Split back train/test using encoded batch ids (0,1)
adata_test = adata[adata.obs["batch_id"] == 1].copy()
adata = adata[adata.obs["batch_id"] == 0].copy()

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

# --- Remaining pipeline identical to your logic ---
input_layer_key = {
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]

all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = np.array(adata.obs[label_id_key].tolist())  # start from 0
batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(set(batch_ids))

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

vocab.set_default_index(vocab[pad_token])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)

def prepare_data(sort_seq_batch: bool = False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }
    return train_data_pt, valid_data_pt

# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), max(1, batch_size // 2))
    dataset = SeqDataset(data_pt)
    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=bool(config.DSBN),
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=bool(config.pre_norm),
)
try:
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    logger.info(f"Loading all model params from {model_file}")
except Exception:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location="cpu")
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

pre_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)

# Freeze all pre-decoder weights (same logic as original; controlled by CFG.freeze)
for name, para in model.named_parameters():
    print("-" * 20)
    print(f"name: {name}")
    if bool(config.freeze) and "encoder" in name and "transformer_encoder" not in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(
    dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
)

logger.info(f"Total Pre freeze Params {pre_freeze_param_count}")
logger.info(f"Total Post freeze Params {post_freeze_param_count}")
wandb.log(
    {
        "info/pre_freeze_param_count": pre_freeze_param_count,
        "info/post_freeze_param_count": post_freeze_param_count,
    },
)

model.to(device)
wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if (DAB and (DAB > 1)) else False

# --- ensure lr is float even if wandb stored it as string ---
try:
    lr_float = float(lr)
except Exception:
    lr_float = float(str(lr).replace("'", "").replace('"', ""))
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr_float, eps=1e-4 if bool(config.amp) else 1e-8
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=float(config.schedule_ratio)
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr_float)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=float(config.schedule_ratio)
    )
if ADV:
    criterion_adv = nn.CrossEntropyLoss()
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=float(config.schedule_ratio)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=float(config.schedule_ratio)
    )

scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp))

def train(model: nn.Module, loader: DataLoader) -> None:
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=bool(config.amp)):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or bool(config.DSBN) else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
            )

            masked_positions = input_values.eq(mask_value)  # positions to predict
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})
                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if DAB:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        total_loss += float(loss.item())
        total_cls += float(loss_cls.item()) if CLS else 0.0
        total_error += float(error_rate) if CLS else 0.0

        if batch % log_interval == 0 and batch > 0:
            lr_cur = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_err = total_error / log_interval if CLS else 0.0
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr_cur:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"cls {cur_cls:5.2f} | err {cur_err:5.2f} | " if CLS else "")
            )
            total_loss = 0.0
            total_cls = 0.0
            total_error = 0.0
            start_time = time.time()

def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/err", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")

def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=bool(config.amp)):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or bool(config.DSBN) else None,
                    CLS=CLS,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                )
                output_values = output_dict["cls_output"]
                loss = nn.CrossEntropyLoss()(output_values, celltype_labels)

            total_loss += float(loss.item()) * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_err += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_err / total_num,
            "epoch": epoch,
        },
    )
    if return_raw:
        return np.concatenate(predictions, axis=0)
    return total_loss / total_num, total_err / total_num

best_val_loss = float("inf")
best_model = None
define_wandb_metrcis()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if bool(config.do_train):
        train(model, loader=train_loader)

    val_loss, val_err = evaluate(model, loader=valid_loader)
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    scheduler.step()
    if DAB_separate_optim:
        scheduler_dab.step()

# %% inference
def test(model: nn.Module, adata: AnnData):
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = np.array(adata.obs[label_id_key].tolist())
    batch_ids = np.array(adata.obs["batch_id"].tolist())

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), max(1, eval_batch_size // 2)),
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(model, loader=test_loader, return_raw=True)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }
    return predictions, celltypes_labels, results

predictions, labels, results = test(best_model, adata_test)
adata_test_raw.obs[pred_key] = [id2type[p] for p in predictions]

# plot
palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_colors = palette_colors * 3
palette_map = {c: palette_colors[i] for i, c in enumerate(celltypes)}

with plt.rc_context({"figure.figsize": (8, 12), "figure.dpi": (300)}):
    fig, axs = plt.subplots(2, 1)
    sc.pl.umap(
        adata_test_raw,
        color="celltype",
        palette=palette_map,
        show=False,
        ax=axs[0],
        legend_loc="right margin",
    )
    axs[0].set_title("True celltype")
    sc.pl.umap(
        adata_test_raw,
        color=pred_key,
        palette=palette_map,
        show=False,
        ax=axs[1],
        legend_loc="right margin",
    )
    axs[1].set_title("Predicted celltype")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_dir / "results.png", dpi=300)

save_dict = {
    "predictions": predictions,
    "labels": labels,
    "results": results,
    "id_maps": id2type
}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)

results["test/cell_umap"] = wandb.Image(
    str(save_dir / "results.png"),
    caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
)
wandb.log(results)

# Confusion matrix
celltypes_display = list(celltypes)
for i in set([id2type[p] for p in predictions]):
    if i not in celltypes_display:
        celltypes_display.remove(i)
cm = confusion_matrix(labels, predictions)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cm = pd.DataFrame(cm, index=celltypes_display[:cm.shape[0]], columns=celltypes_display[:cm.shape[1]])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

results["test/confusion_matrix"] = wandb.Image(
    str(save_dir / "confusion_matrix.png"),
    caption=f"confusion matrix",
)
wandb.log(results)
