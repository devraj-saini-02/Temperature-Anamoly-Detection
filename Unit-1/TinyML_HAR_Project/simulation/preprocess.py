import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load feature names
features = pd.read_csv(
    "data/features.txt",
    sep="\s+",
    header=None,
    names=["index", "feature"]
)

feature_names = []
seen = {}

for name in features["feature"]:
    if name in seen:
        seen[name] += 1
        feature_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        feature_names.append(name)

# Load training data
X_train = pd.read_csv(
    "data/X_train.txt",
    sep="\s+",
    header=None,
    names=feature_names
)

y_train = pd.read_csv(
    "data/y_train.txt",
    header=None,
    names=["activity"]
)

# Load activity labels
activity_labels = pd.read_csv(
    "data/activity_labels.txt",
    sep="\s+",
    header=None,
    names=["id", "label"]
)

# Map activity numbers to names
y_train["activity"] = y_train["activity"].map(
    dict(zip(activity_labels.id, activity_labels.label))
)

# Keep only 3 activities (TinyML-friendly)
allowed = ["WALKING", "SITTING", "STANDING"]
mask = y_train["activity"].isin(allowed)

X_train = X_train[mask]
y_train = y_train[mask]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train["activity"])

print("Classes used:", encoder.classes_)
print("Sample input shape:", X_train.iloc[0].shape)
print("Sample label:", y_train.iloc[0].values[0])

# Save processed data for training
np.save("data/X_processed.npy", X_train.values)
np.save("data/y_processed.npy", y_encoded)

print("Preprocessing completed successfully!")