import argparse
import os
from src.utils.kaggle_api import download_competition_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition_name",
                        required=True,
                        dest="competition_name")
    parser.add_argument("--download_path",
                        required=True,
                        dest="download_path")
    args = parser.parse_args()
    download_competition_data(args.competition_name, args.download_path)
