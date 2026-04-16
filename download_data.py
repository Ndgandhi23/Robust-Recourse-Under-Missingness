"""
download_data.py
----------------
Downloads the Pima Indians Diabetes dataset and saves it locally
as 'data/diabetes.csv' relative to this script.

Run this once before using the pipeline:
    python download_data.py

Source: UCI Machine Learning Repository (via jbrownlee GitHub mirror)
Citation: Smith et al., Pima Indians Diabetes Database, 1988.
"""

import urllib.request
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

OUTPUT_FILE = os.path.join(_DIR, "data", "diabetes.csv")


def download():
    if os.path.exists(OUTPUT_FILE):
        print(f"'{OUTPUT_FILE}' already exists — skipping download.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Downloading Pima Indians Diabetes dataset...")
    print(f"Source: {URL}")

    try:
        urllib.request.urlretrieve(URL, OUTPUT_FILE)

        # Add column headers to the file
        with open(OUTPUT_FILE, "r") as f:
            data = f.read()

        header = ",".join(COLUMN_NAMES) + "\n"
        with open(OUTPUT_FILE, "w") as f:
            f.write(header + data)

        print(f"Saved to '{OUTPUT_FILE}'")
        print(f"Rows: 768, Columns: {len(COLUMN_NAMES)}")
        print(f"Columns: {', '.join(COLUMN_NAMES)}")

    except Exception as e:
        print(f"Download failed: {e}")
        print("You can manually download the dataset from:")
        print("  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print("and save it as 'data/diabetes.csv' in this directory.")


if __name__ == "__main__":
    download()