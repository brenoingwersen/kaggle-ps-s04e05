# kaggle-ps-s04e05

Repository template for a regression pipeline for Kaggle competitions. The project uses Makefile for task automation, a config.yaml file for configuration, and PDM for dependency management.

## Installation

1. Clone the repository

```sh
git clone https://github.com/your-username/kaggle-competition-project.git
cd kaggle-competition-project
```

2. Install dependencies with PDM
   pdm install

## Usage

1. Data preparation
   Download the competition data:

```sh
make all
```

This will create the necessary directories and download the data if it's not already present. 2. Training
To train the model:

```sh
make train
```

This will run the feature engineering, preprocessing, feature selection, and model training steps using the specified configuration. 3. Submit to the competition

```sh
make submission
```

4. Cleaning
   To clean up the build directory and configuration files:

```sh
make clean
```

## Configuration

- Available models: `XGBoost`, `LightGBM`, `CatBoost`.

Alter the experiment configurations through `config.py`.
