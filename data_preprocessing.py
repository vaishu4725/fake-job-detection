import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore")

RAW_PATH   = r"C:\Users\ELCOT\Documents\IV-Project\code\fake and real (1)\irrelevent\final_balanced_fake_job_postings.csv"
CLEAN_PATH = r"C:\Users\ELCOT\Documents\IV-Project\code\fake and real (1)\irrelevent\cleaned_jobs.csv"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess():
    print("====================================")
    print("Fake Job Dataset Preprocessing")
    print("====================================")

    print("\nLoading dataset...")
    df = pd.read_csv(RAW_PATH)
    print("Original Shape:", df.shape)

    print("\nRemoving rows with missing values...")
    df = df.dropna()
    print("Shape after dropna:", df.shape)

    print("\nRemoving duplicate rows...")
    df = df.drop_duplicates()
    print("Shape after dedup:", df.shape)

    print("\nCleaning text columns...")
    for col in ["title", "description", "company_profile", "requirements", "benefits"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Normalize label column → always 'label'
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    elif "label" not in df.columns:
        raise KeyError("Dataset must contain a 'fraudulent' or 'label' column.")

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    df.to_csv(CLEAN_PATH, index=False)
    print("\nCleaned dataset saved as:", CLEAN_PATH)
    print("Final Shape:", df.shape)

    return df


if __name__ == "__main__":
    preprocess()