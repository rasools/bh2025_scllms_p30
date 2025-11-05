.PHONY: help run clean test clean-old

# Default parameters (can be overridden)
INPUT_VALUE ?= test2
PROCESSING_MODE ?= production

# Pipeline directory
PIPELINE_DIR = pipelines
MAIN_NF = $(PIPELINE_DIR)/main.nf
DATA_DIR = $(PIPELINE_DIR)/data
MODEL_DIR = $(PIPELINE_DIR)/model/pan_cancer

# Grouped file variables
DATA_FILES := \
  $(DATA_DIR)/c_data.h5ad \
  $(DATA_DIR)/filtered_ms_adata.h5ad

MODEL_FILES := \
  $(MODEL_DIR)/best_model.pt \
  $(MODEL_DIR)/vocab.json \
  $(MODEL_DIR)/args.json

help: ## Show this help message
	@echo "Nextflow Pipeline Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make run                 - Run the pipeline with default parameters"
	@echo "  make prereq              - Install the requirements"
	@echo "  make image               - Build the Docker image"
	@echo "  make clean              - Clean Nextflow work directories and results"
	@echo "  make test               - Run with test parameters"
	@echo ""
	@echo "Parameters (can be overridden):"
	@echo "  INPUT_VALUE=$(INPUT_VALUE)"
	@echo "  PROCESSING_MODE=$(PROCESSING_MODE)"
	@echo ""
	@echo "Examples:"
	@echo "  make run INPUT_VALUE=my_data PROCESSING_MODE=debug"
	@echo "  make test"

prereq: ## Install the requirements
	@docker version
	@nextflow info
	@pip show gdown
	@echo "prerequisites installed"

image: ## Build the Docker image
	@docker build --quiet -t bh2025scllmsp30:latest -f scgpt.Dockerfile .

run: image download-data download-model ## Run the Nextflow pipeline
	@cd $(PIPELINE_DIR) && nextflow run main.nf \
		--input_value $(INPUT_VALUE) \
		--processing_mode $(PROCESSING_MODE)

test: ## Run with test parameters
	$(MAKE) run INPUT_VALUE=test PROCESSING_MODE=test

clean: ## Clean Nextflow work directories and results
	rm -rf results
	rm -rf reports
	rm -rf $(PIPELINE_DIR)/.nextflow*
	rm -rf .nextflow*
	cd $(PIPELINE_DIR) && nextflow clean -f


clean-old: ## Clean work/metadata of runs older than the latest
	@RUN_ID=$$(cd $(PIPELINE_DIR) && nextflow log -q | tail -1); \
	if [ -n "$$RUN_ID" ]; then \
	  echo "Cleaning runs before: $$RUN_ID"; \
	  cd $(PIPELINE_DIR) && nextflow clean -f -before $$RUN_ID; \
	else \
	  echo "No Nextflow runs found"; \
	fi

# Data files - make will skip if they exist
$(DATA_DIR)/c_data.h5ad:
	@echo "Downloading c_data.h5ad"
	@mkdir -p $(DATA_DIR)
	@python3 -m gdown 1bV1SHKVZgkcL-RmmuN51_IIUJTSJbXOi --quiet -O $@

$(DATA_DIR)/filtered_ms_adata.h5ad:
	@echo "Downloading filtered_ms_adata.h5ad"
	@mkdir -p $(DATA_DIR)
	@python3 -m gdown 1casFhq4InuBNhJLMnGebzkRXM2UTTeQG --quiet -O $@

# Model files - make will skip if they exist
$(MODEL_DIR)/best_model.pt:
	@echo "Downloading best_model.pt"
	@mkdir -p $(MODEL_DIR)
	@python3 -m gdown 1PsOOAXioZ7twJZiIhvxg5c5HD18fAtYt --quiet -O $@

$(MODEL_DIR)/vocab.json:
	@echo "Downloading vocab.json"
	@mkdir -p $(MODEL_DIR)
	@python3 -m gdown 10D8_BtS3PORqawUwjWh5Dm_blDfRhm3r --quiet -O $@

$(MODEL_DIR)/args.json:
	@echo "Downloading args.json"
	@mkdir -p $(MODEL_DIR)
	@python3 -m gdown 1GTQXIwa4yzbRZlarGgHkmC9k8hD6x1hA --quiet -O $@

# Convenience targets that depend on the files
download-data: $(DATA_FILES) ## Download the data

download-model: $(MODEL_FILES) ## Download the model