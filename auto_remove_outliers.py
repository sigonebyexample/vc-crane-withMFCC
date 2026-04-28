import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist

INPUT_CSV = "features_table.csv"
DATASET_DIR = "dataset"
OUTLIER_PERCENT = 5.0
REMOVED_REPORT = "removed_files_report_fixed.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Total initial samples: {len(df)}")

class_col = 'class'
filename_col = 'filename'
feature_cols = [c for c in df.columns if c not in [class_col, filename_col]]

def delete_audio_file(class_name, filename):
    filepath = os.path.join(DATASET_DIR, class_name, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

removed_entries = []

for cls in df[class_col].unique():
    df_cls = df[df[class_col] == cls].copy()
    X = df_cls[feature_cols].values.astype(np.float64)

    mean_vec = np.mean(X, axis=0)
    std_vec = np.std(X, axis=0)
    std_vec[std_vec < 1e-8] = 1.0
    X_norm = (X - mean_vec) / std_vec

    centroid = np.mean(X_norm, axis=0)
    distances = cdist(X_norm, [centroid], metric='euclidean').flatten()
    df_cls['euclidean_dist'] = distances

    threshold = np.percentile(distances, 100 - OUTLIER_PERCENT)

    outliers = df_cls[df_cls['euclidean_dist'] > threshold]

    for _, row in outliers.iterrows():
        filename = row[filename_col]
        success = delete_audio_file(cls, filename)
        if success:
            print(f"Deleted: {cls}/{filename} (distance: {row['euclidean_dist']:.4f})")
        else:
            print(f"File not found: {cls}/{filename}")
        removed_entries.append({
            'class': cls,
            'filename': filename,
            'euclidean_dist': row['euclidean_dist']
        })

    print(f"Class '{cls}': {len(outliers)} samples ({OUTLIER_PERCENT:.1f}%) removed.")
    print(f"   Distance range: {distances.min():.2f} to {distances.max():.2f}")
    print(f"   Removal threshold: {threshold:.2f}")

if removed_entries:
    df_report = pd.DataFrame(removed_entries)
    df_report.to_csv(REMOVED_REPORT, index=False)
    print(f"\nRemoved files report saved to '{REMOVED_REPORT}'.")
else:
    print("\nNo files were removed.")

print("\nDataset cleaning completed successfully.")
print("=" * 60)
print("Note: You must run train_model.py again to rebuild the model with the cleaned data.")
