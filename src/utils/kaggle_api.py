import os
from zipfile import ZipFile
import kaggle
import logging


def download_competition_data(competition_name, download_path="data/raw"):
    """
    Download competition data from Kaggle.
    """
    kaggle.api.competition_download_files(competition_name,
                                          path=download_path,
                                          quiet=False)
    # Unzipping
    zip_path = os.path.join(download_path, f"{competition_name}.zip")
    with ZipFile(zip_path) as zf:
        zf.extractall(download_path)


def submit_prediction(competition_name, file_path, message):
    """
    Submit prediction to Kaggle competition.
    """
    kaggle.api.competition_submit(file_path,
                                  message,
                                  competition_name)
