import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_DATA_PATH    = r"C:\Users\ELCOT\Documents\IV-Project\code\fake and real (1)\irrelevent\cleaned_jobs.csv"
FEATURES_PATH      = r"C:\Users\ELCOT\Documents\IV-Project\code\fake and real (1)\irrelevent\features.npz"
VECTORIZER_PATH    = r"C:\Users\ELCOT\Documents\IV-Project\code\fake and real (1)\irrelevent\tfidf_vectorizer.pkl"
META_FEATURES_PATH = r"C:\Users\ELCOT\Documents\IV-Project\code\fake and real (1)\irrelevent\meta_features.npy"

SCAM_KEYWORDS = [
    "no investment", "quick earning", "earn from home",
    "easy money", "guaranteed income", "unlimited income",
    "be your own boss", "daily payout", "weekly payout",
    "risk free", "free registration", "mlm",
    "network marketing", "instant payment",
    "make money fast", "data entry work"
]


def build_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=5000,       # ✅ reduced from 15000
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=5,                # ✅ increased from 2
        max_df=0.90,             # ✅ reduced from 0.95
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
        norm="l2"
    )


def create_combined_text(df):
    text_columns = ["title", "company_profile", "description",
                    "requirements", "salary_range", "location", "industry"]
    text_columns = [c for c in text_columns if c in df.columns]
    df[text_columns] = df[text_columns].fillna("")
    return df[text_columns].astype(str).agg(" ".join, axis=1)


def extract_features(df):
    # Accept either 'label' or 'fraudulent'
    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    elif "label" not in df.columns:
        raise KeyError("DataFrame must contain a 'label' or 'fraudulent' column.")

    print("[INFO] Creating combined text...")
    combined_text = create_combined_text(df)

    print("[INFO] Training TF-IDF vectorizer...")
    vectorizer = build_tfidf_vectorizer()
    X_tfidf = vectorizer.fit_transform(combined_text)
    print("TF-IDF Shape:", X_tfidf.shape)

    possible_meta = [
        "has_scam_keywords", "has_salary", "has_company_desc",
        "has_phone_in_desc", "title_len", "desc_len"
    ]
    meta_cols = [c for c in possible_meta if c in df.columns]

    if meta_cols:
        X_meta = csr_matrix(df[meta_cols].fillna(0).astype(float).values)
        X = hstack([X_tfidf, X_meta])
    else:
        X = X_tfidf

    print("Final Feature Matrix Shape:", X.shape)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    save_npz(FEATURES_PATH, X)
    np.save(META_FEATURES_PATH, np.array(meta_cols))

    y = df["label"].values

    return X, y, vectorizer, meta_cols


def transform_single(text_dict, vectorizer, meta_cols):
    pattern = "|".join(SCAM_KEYWORDS)

    def clean(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+", " url ", text)
        text = re.sub(r"\b\d{10,}\b", " phone ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    combined = " ".join([
        clean(text_dict.get("title", "")),
        clean(text_dict.get("company_profile", "")),
        clean(text_dict.get("description", "")),
        clean(text_dict.get("requirements", "")),
        clean(text_dict.get("salary_range", "")),
        clean(text_dict.get("location", "")),
        clean(text_dict.get("industry", ""))
    ])

    X_tfidf = vectorizer.transform([combined])

    meta = {
        "has_scam_keywords" : int(bool(re.search(pattern, combined))),
        "has_salary"        : int(bool(text_dict.get("salary_range", "").strip())),
        "has_company_desc"  : int(bool(text_dict.get("company_profile", "").strip())),
        "has_phone_in_desc" : int(bool(re.search(r"\b\d{10}\b", combined))),
        "title_len"         : len(text_dict.get("title", "").split()),
        "desc_len"          : len(text_dict.get("description", "").split()),
    }

    if meta_cols:
        X_meta = csr_matrix([[meta.get(c, 0) for c in meta_cols]])
        X = hstack([X_tfidf, X_meta])
    else:
        X = X_tfidf

    return X


if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    X, y, vectorizer, meta_cols = extract_features(df)
    print("[DONE] Feature extraction complete.")