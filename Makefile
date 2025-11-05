.PHONY: help run clean test

# Default parameters (can be overridden)
INPUT_VALUE ?= test2
PROCESSING_MODE ?= production

# Pipeline directory
PIPELINE_DIR = pipelines
MAIN_NF = $(PIPELINE_DIR)/main.nf

help: ## Show this help message
	@echo "Nextflow Pipeline Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make run                 - Run the pipeline with default parameters"
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

image: ## Build the Docker image
	docker build -t bh2025scllmsp30:latest -f scgpt.Dockerfile .

run: image ## Run the Nextflow pipeline
	cd $(PIPELINE_DIR) && nextflow run main.nf \
		--input_value $(INPUT_VALUE) \
		--processing_mode $(PROCESSING_MODE)

test: ## Run with test parameters
	$(MAKE) run INPUT_VALUE=test PROCESSING_MODE=test

clean: ## Clean Nextflow work directories and results
	cd $(PIPELINE_DIR) && nextflow clean -f
	rm -rf results
	rm -rf reports
	rm -rf $(PIPELINE_DIR)/.nextflow*
	rm -rf .nextflow*

clean-all: clean ## Clean everything including cache
	cd $(PIPELINE_DIR) && nextflow clean --all
	rm -rf .cache

