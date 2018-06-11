"""
Import data from dataset, and preprocess it.
"""
import imghdr
import os
import re

import numpy as np
import pandas as pd


DATA_VIR = "1.1"

DATA_DIR = os.path.abspath(__file__)        # ?/dataset.py
DATA_DIR = os.path.dirname(DATA_DIR)        # ?/
DATA_DIR = os.path.join(
    DATA_DIR,
    "dataset",
    "MURA-v" + DATA_VIR,
)                                           # ?/dataset/MURA-v1.1/

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")

# import labeled dataset into Dataframe
TRAIN_LABELED = pd.read_csv(
    os.path.join(DATA_DIR, "train_labeled_studies.csv"),
    names=["patient", "label"]
)

VALID_LABELED = pd.read_csv(
    os.path.join(DATA_DIR, "valid_labeled_studies.csv"),
    names=["patient", "label"]
)

# import image paths
TRAIN_PATH = pd.read_csv(
    os.path.join(DATA_DIR, "train_image_paths.csv"),
    names=["path"]
)

VALID_PATH = pd.read_csv(
    os.path.join(DATA_DIR, "valid_image_paths.csv"),
    names=["path"]
)

BPARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


def classify_bpart(data):
    """
    Divide TRAIN_LABELED into sub-sets based on the body parts in the image.
    Also add body part as a new feature of the dataset.
    :param data: dataset to process.
    :return:
    """
    for bpart in BPARTS:
        data.loc[data["path"].str.contains(bpart.upper()), "body_part"] = bpart


def complete_path(data, column):
    """
    Convert relative image path to absolute path so that the execution does not depend
    on working directory. Also clean up the patient name
    :param data: dataset to process.
    :param column: column to perform the operation.
    :return:
    """
    data[column] = np.where(
        data[column].str.startswith("MURA-v" + DATA_VIR),
        data[column].str.replace("MURA-v" + DATA_VIR, DATA_DIR),
        data[column]
    )


def extract_study(row):
    """
    Callback function to generate a column for unique patient-study combo.
    :param row: a row from processing table
    :return:
    """
    match = re.search("study\d+", row["path"])
    if match:
        study = match.group()
        return "{}-{}-{}".format(row["patient"], row["body_part"], study)
    else:
        raise ValueError("study not found in " + row["path"])


def get_patient(row):
    """
    Call back function to check if the image column is a valid path,
    and grab the parent directory if it is.
    :param row: a row from processing table
    :return:
    """
    try:
        img_type = imghdr.what(row["path"])
    except IsADirectoryError:
        img_type = None

    if img_type:
        return os.path.dirname(row["path"]) + "/"
    return row["patient"]


def build_dataframe(df_label, df_path):
    """
    Build datasets by combining image paths with labels, so that we have a dataframe
    where each row is an image and has the patient it belongs to, as well as the label
    :param df_label: labeled dataset.
    :param df_path: image paths.
    :return: training table, validation table
    """
    df_label = df_label.copy(deep=True)
    df_path = df_path.copy(deep=True)

    complete_path(df_path, "path")
    complete_path(df_label, "patient")

    # Apply a transformation over each row to save image directory as a new column
    df_path["patient"] = df_path.apply(get_patient, axis=1)

    # Merge two table on patient column
    result = df_path.merge(df_label, on="patient")

    classify_bpart(result)

    # change .../patient00001/... to patient00001
    result["patient"] = result["patient"].str.extract("(patient\d{5})")

    # Apply a transformation over each row to create a column for unique
    # patient-bpart-study combo
    result["study"] = result.apply(extract_study, axis=1)
    return result


def preprocess():
    """
    Preprocess datasets.
    :return: training set, validation set
    """
    df_train = build_dataframe(TRAIN_LABELED, TRAIN_PATH)
    df_valid = build_dataframe(VALID_LABELED, VALID_PATH)

    return df_train, df_valid
