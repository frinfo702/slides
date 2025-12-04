.DEFAULT_GOAL := usage

MARPCMD ?= marp
PPTX_OUT_DIR ?= nishikawa-lab/pptx

ifeq ($(filter pptx,$(MAKECMDGOALS)),pptx)
  ifeq ($(origin FILE), undefined)
    FILE := $(word 2,$(MAKECMDGOALS))
  endif
  ifneq ($(strip $(FILE)),)
    $(eval $(FILE):;@:)
  endif
  ifeq ($(strip $(FILE)),)
    $(error Usage: make pptx path/to/slide.md or FILE=/path/to/slide.md)
  endif
endif

TARGET_NAME := $(basename $(notdir $(FILE)))
OUTPUT := $(PPTX_OUT_DIR)/$(TARGET_NAME).pptx

.PHONY: pptx
pptx:
	@mkdir -p $(PPTX_OUT_DIR)
	$(MARPCMD) --pptx "$(FILE)" --output "$(OUTPUT)"

.PHONY: usage
usage:
	@echo "Usage: make pptx path/to/slide.md"
