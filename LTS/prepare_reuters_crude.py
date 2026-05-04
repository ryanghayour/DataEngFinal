"""
prepare_reuters_crude.py

Builds two datasets from Reuters-21578 (ModApte split) for LTS experiments:
  - data_use_cases/data_reuters_crude.csv   : unlabeled pool for LTS active learning
  - data_use_cases/reuters_crude_validation.csv : labeled validation set (ground truth)

Target category: 'crude' (oil market news), treated as the positive class.
All other articles are treated as negatives.

Source splits used:
  - ModApte_train -> unlabeled pool  (LTS generates labels via Qwen)
  - ModApte_test  -> validation set  (Reuters ground-truth labels, no LLM needed)
"""

import ast
import re
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

TRAIN_PATH = "reuters/ModApte_train.csv"
TEST_PATH  = "reuters/ModApte_test.csv"
POOL_OUT   = "data_use_cases/data_reuters_crude.csv"
VAL_OUT    = "data_use_cases/reuters_crude_validation.csv"

TARGET_CATEGORY = "crude"
MIN_WORDS       = 10     # drop articles shorter than this after cleaning


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_topics(raw):
    """Parse the topics column stored as a Python list string."""
    try:
        return ast.literal_eval(raw) if isinstance(raw, str) else []
    except (ValueError, SyntaxError):
        return []


def clean_text(text):
    """
    Normalise Reuters wire article text:
      1. Replace newlines / tabs with spaces
      2. Remove the Reuters boilerplate signature ('Reuter' at end)
      3. Collapse multiple spaces into one
      4. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    # Reuters articles end with ' Reuter' or ' REUTER' (wire service tag)
    text = re.sub(r'\s+[Rr][Ee][Uu][Tt][Ee][Rr]\s*$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_title(title):
    """Normalise article headline."""
    if not isinstance(title, str):
        return ""
    return re.sub(r'\s+', ' ', title).strip()


def clean_id(raw_id):
    """Strip surrounding quotes that appear in the new_id column."""
    return str(raw_id).strip('"').strip("'").strip()


def word_count(text):
    return len(text.split())


# ── Load ─────────────────────────────────────────────────────────────────────

print("Loading Reuters ModApte splits...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
print(f"  Train rows : {len(train_raw):,}")
print(f"  Test  rows : {len(test_raw):,}")


# ── Parse topics ─────────────────────────────────────────────────────────────

train_raw["topics_list"] = train_raw["topics"].apply(parse_topics)
test_raw["topics_list"]  = test_raw["topics"].apply(parse_topics)


# ── Clean text & title ───────────────────────────────────────────────────────

for df in [train_raw, test_raw]:
    df["text"]  = df["text"].apply(clean_text)
    df["title"] = df["title"].apply(clean_title)


# ── Drop nulls / empty ────────────────────────────────────────────────────────

def drop_empty(df, name):
    before = len(df)
    df = df[df["text"].str.len() > 0]
    df = df[df["title"].str.len() > 0]
    df = df[df["text"].apply(word_count) >= MIN_WORDS]
    after = len(df)
    print(f"  {name}: dropped {before - after:,} rows (null/empty/too-short) -> {after:,} remaining")
    return df.reset_index(drop=True)

print("\nDropping empty / too-short articles...")
train_clean = drop_empty(train_raw, "train")
test_clean  = drop_empty(test_raw,  "test")


# ── Deduplicate on id ────────────────────────────────────────────────────────

train_clean["id"] = train_clean["new_id"].apply(clean_id)
before = len(train_clean)
train_clean = train_clean.drop_duplicates(subset="id")
print(f"\nDeduplication (train): {before - len(train_clean):,} duplicates removed")


# ── Build unlabeled pool (train split, no labels) ────────────────────────────

pool = train_clean[["id", "title", "text"]].copy()
pool = pool.reset_index(drop=True)

n_true_positives = train_clean["topics_list"].apply(
    lambda t: TARGET_CATEGORY in t
).sum()

print(f"\n--- Unlabeled pool ---")
print(f"  Total articles     : {len(pool):,}")
print(f"  True crude positives (hidden from LTS): {n_true_positives:,} "
      f"({n_true_positives / len(pool) * 100:.1f}%)")
print(f"  True negatives     : {len(pool) - n_true_positives:,}")
print(f"  Avg text length    : {pool['text'].apply(word_count).mean():.0f} words")
print(f"  Median text length : {pool['text'].apply(word_count).median():.0f} words")


# ── Build validation set (test split, ground-truth labels) ───────────────────

val = test_clean[["title", "text", "topics_list"]].copy()
val["label"] = val["topics_list"].apply(
    lambda t: 1 if TARGET_CATEGORY in t else 0
)
val = val.drop(columns=["topics_list"])
val = val.reset_index(drop=True)

n_pos = val["label"].sum()
n_neg = (val["label"] == 0).sum()

print(f"\n--- Validation set ---")
print(f"  Total articles     : {len(val):,}")
print(f"  Crude positives    : {n_pos:,} ({n_pos / len(val) * 100:.1f}%)")
print(f"  Negatives          : {n_neg:,} ({n_neg / len(val) * 100:.1f}%)")
print(f"  Avg text length    : {val['text'].apply(word_count).mean():.0f} words")


# ── Save ─────────────────────────────────────────────────────────────────────

pool.to_csv(POOL_OUT, index=False)
val.to_csv(VAL_OUT, index=False)

print(f"\nSaved:")
print(f"  {POOL_OUT}")
print(f"  {VAL_OUT}")
print("\nDone.")
