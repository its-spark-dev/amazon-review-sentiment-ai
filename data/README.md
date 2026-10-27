# Amazon Fine Food Reviews dataset

Obtain the dataset from the [Amazon Fine Food Reviews page on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), published by SNAP. Review the dataset description and usage terms on the source page before using it.

## Local download (when needed)

1. Open the source page and sign in to Kaggle if prompted.
2. Download the dataset manually and extract the archive locally.
3. Place the extracted `Reviews.csv` at `data/Reviews.csv`.

The initial Streamlit placeholder does not require any data. This repository does not download the dataset automatically, and no dataset is included in the scaffold.

## Storage and collaboration

- Keep downloaded archives, raw data, and processed datasets local. The root `.gitignore` excludes the contents of `data/` except this guide.
- Never commit Kaggle credentials, API tokens, or other secrets. No Kaggle API setup is required for the scaffold.
- Record future preprocessing, sentiment label definitions, splits, and data quality decisions in the project notebooks and report. Those decisions have not been finalized.
- Share the source link and reproducible processing instructions with contributors instead of committing the dataset.
