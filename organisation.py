import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "Data_v2_clean.csv")
IMG_DIRS = {
    "non_seg": os.path.join(BASE_DIR, "../mission_herbonaute_2000"),
    "seg": os.path.join(BASE_DIR, "../mission_herbonaute_2000_seg_black"),
}
OUT_DIR = os.path.join(BASE_DIR, "data", "herbonaute")

# ===============================
# Load & normalize CSV
# ===============================
df = pd.read_csv(CSV_PATH, sep=";")

df["label"] = df["epines"].astype(int).astype(str)
df["filename"] = df["code"].astype(str) + ".jpg"

# ===============================
# Train / Val split (reproductible)
# ===============================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ===============================
# Copy helper
# ===============================
def copy_split(df_split, src_root, split, tag):
    for _, row in df_split.iterrows():
        src = os.path.join(src_root, row["filename"])
        if not os.path.exists(src):
            continue

        dst = os.path.join(OUT_DIR, tag, split, row["label"])
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, os.path.join(dst, row["filename"]))

# ===============================
# Execute
# ===============================
for tag, src_root in IMG_DIRS.items():
    print(f"[INFO] Preparing {tag} dataset...")
    copy_split(train_df, src_root, "train", tag)
    copy_split(val_df, src_root, "val", tag)

print("Dataset preparation DONE !")
