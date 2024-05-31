COMPETITION := playground-series-s4e5

# Inputs
DIR_DATA := data/raw
DATA_TRAIN := $(DIR_DATA)/train.csv
DATA_TEST := $(DIR_DATA)/test.csv

.PHONY: all

all: $(DATA_TRAIN) $(DATA_TEST)

$(DIR_DATA):
	mkdir -p $@

# Downloading competition data
$(DATA_TRAIN) $(DATA_TEST): $(DIR_DATA)
	# Verifies if train.csv or test.csv are present
	@if [ ! -f $(DATA_TRAIN) ] || [ ! -f $(DATA_TEST) ]; then \
		pdm run python src/download_data.py \
			--competition_name $(COMPETITION) \
			--download_path $(DIR_DATA); \
	fi

# Config
YAML_FILE := config.yaml

include config.mk

config.mk:
	@echo "Generating configuration from YAML file..."
	@python parse_yaml.py $(YAML_FILE) > config.mk

train: config.mk
	# Feature Engineering
	pdm run python $(FEATURE_ENGINEERING_PATH)
	# Preprocessing
	pdm run python $(PREPROCESS_PATH)
	# Feature selection
	pdm run python $(FEATURE_SELECTION_PATH)
	# Model Path
	pdm run python src/model.py
	# Train the model
	pdm run python src/train.py
	rm config.mk

optimize: config.mk
	pdm run python src/optimize.py

submission: $(BUILD_DIR)/submission.csv config.mk
	kaggle competitions submit -c $(COMPETITION) -f $< -m "$(BUILD_DESCRIPTION)"

submissions:
	kaggle competitions submissions $(COMPETITION) > submissions.csv

clean:
	rm -Rf $(BUILD_DIR)
	rm config.mk