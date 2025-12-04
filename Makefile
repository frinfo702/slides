.DEFAULT_GOAL := usage

MARPCMD ?= marp
PPTX_OUT_DIR ?= nishikawa-lab/pptx
PDF_OUT_DIR ?= nishikawa-lab/pdf
DEFAULT_THEME ?= theme/academic.css
BUILD_TARGETS := pptx pdf

ifneq ($(filter $(BUILD_TARGETS),$(MAKECMDGOALS)),)
  ifeq ($(origin FILE), undefined)
    FILE := $(word 2,$(MAKECMDGOALS))
  endif
  ifeq ($(origin THEME), undefined)
    THEME := $(word 3,$(MAKECMDGOALS))
  endif
  ifneq ($(strip $(FILE)),)
    $(eval $(FILE):;@:)
  endif
  ifneq ($(strip $(THEME)),)
    $(eval $(THEME):;@:)
  endif
  ifeq ($(strip $(FILE)),)
    $(error Usage: make {pptx|pdf} path/to/slide.md [path/to/theme.css] or FILE=/path/to/slide.md [THEME=/path/to/theme.css])
  endif
endif

ifeq ($(strip $(THEME)),)
  THEME := $(DEFAULT_THEME)
endif

TARGET_NAME := $(basename $(notdir $(FILE)))
OUTPUT := $(PPTX_OUT_DIR)/$(TARGET_NAME).pptx
PDF_OUTPUT := $(PDF_OUT_DIR)/$(TARGET_NAME).pdf

.PHONY: pptx
pptx:
	@mkdir -p $(PPTX_OUT_DIR)
	$(MARPCMD) --pptx "$(FILE)" --theme "$(THEME)" --output "$(OUTPUT)"

.PHONY: pdf
pdf:
	@mkdir -p $(PDF_OUT_DIR)
	$(MARPCMD) --pdf "$(FILE)" --theme "$(THEME)" --output "$(PDF_OUTPUT)"

.PHONY: usage
usage:
	@echo "Usage: make {pptx|pdf} path/to/slide.md [path/to/theme.css]"
