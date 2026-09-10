"""
Leakage check and honest evaluation for the pose-recognition project.

The nested cross-validation in the main notebook returns 100% accuracy.
This script explains why, and re-evaluates the model with a split that
respects the structure of the data.

Run:  python leakage_check.py
Outputs: figures/pose-recognition.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

DATA = "data/input_data.pkl"
FIGDIR = "figures"


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
d = pd.read_pickle(DATA)
X, y = d["X"], d["y"]
labels = np.unique(y)
y_enc = np.array([list(labels).index(v) for v in y])

print(f"{X.shape[0]} samples, {X.shape[1]} features "
      f"({X.shape[1] // 4} keypoints x 4)")
print("classes:", ", ".join(labels))


# ----------------------------------------------------------------------
# 1. is the dataset made of independent samples?
# ----------------------------------------------------------------------
print("\n--- structure of the data ---")

consecutive = np.linalg.norm(X[1:] - X[:-1], axis=1)
rng = np.random.default_rng(0)
i, j = rng.integers(0, len(X), 5000), rng.integers(0, len(X), 5000)
random_pairs = np.linalg.norm(X[i] - X[j], axis=1)

ratio = np.median(random_pairs) / np.median(consecutive)
print(f"median distance, consecutive frames : {np.median(consecutive):.4f}")
print(f"median distance, random pairs       : {np.median(random_pairs):.4f}")
print(f"ratio                               : {ratio:.0f}x")

label_changes = np.where(y[1:] != y[:-1])[0]
n_blocks = len(label_changes) + 1
print(f"\ncontiguous label blocks: {n_blocks} "
      f"({len(y) / n_blocks:.0f} frames each on average)")
print("-> each class is one continuous recording, not independent samples")


# ----------------------------------------------------------------------
# model (same architecture as the notebook)
# ----------------------------------------------------------------------
def create_model(input_dim, output_dim, n_units=128, n_layers=2,
                 learning_rate=1e-3, dropout_rate=0.1):
    m = Sequential()
    m.add(Input(shape=(input_dim,)))
    m.add(Dense(n_units, activation="relu", kernel_initializer="he_uniform"))
    m.add(Dropout(dropout_rate))
    for _ in range(n_layers - 1):
        m.add(Dense(n_units, activation="relu", kernel_initializer="he_uniform"))
        m.add(Dropout(dropout_rate))
    m.add(Dense(output_dim, activation="softmax", kernel_initializer="he_uniform"))
    m.compile(optimizer=Adam(learning_rate=learning_rate),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def fit_predict(train_idx, test_idx):
    """Scaler fitted on the training split only, as in the notebook."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])

    model = create_model(X_tr.shape[1], len(labels))
    model.fit(X_tr, y_enc[train_idx], epochs=35, batch_size=64, verbose=0)
    return np.argmax(model.predict(X_te, verbose=0), axis=-1)


# ----------------------------------------------------------------------
# 2. shuffled split — what the notebook does
# ----------------------------------------------------------------------
print("\n--- A) shuffled split over frames ---")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = [accuracy_score(y_enc[te], fit_predict(tr, te))
          for tr, te in skf.split(X, y_enc)]
acc_shuffled = float(np.mean(scores))
print(f"accuracy: {acc_shuffled:.4f} +/- {np.std(scores):.4f}")


# ----------------------------------------------------------------------
# 3. temporal split — last 30% of each recording held out
# ----------------------------------------------------------------------
print("\n--- B) temporal split ---")

train_idx, test_idx = [], []
for c in np.unique(y_enc):
    pos = np.where(y_enc == c)[0]          # already contiguous
    cut = int(len(pos) * 0.7)
    train_idx += list(pos[:cut])
    test_idx += list(pos[cut:])
train_idx, test_idx = np.array(train_idx), np.array(test_idx)

pred = fit_predict(train_idx, test_idx)
y_true = y_enc[test_idx]
acc_temporal = accuracy_score(y_true, pred)
print(f"accuracy: {acc_temporal:.4f}")
print()
print(classification_report(y_true, pred, target_names=labels, digits=3))

cm = confusion_matrix(y_true, pred)
errors = cm.sum() - np.trace(cm)

idx = {l: k for k, l in enumerate(labels)}
lr_pairs = [("left_bicep", "right_bicep"),
            ("left_tricep", "right_tricep"),
            ("left_shoulder", "right_shoulder")]
lr_errors = sum(cm[idx[a], idx[b]] + cm[idx[b], idx[a]] for a, b in lr_pairs)
print(f"errors: {errors} / {len(y_true)} | left-right confusions: {lr_errors}")


# ----------------------------------------------------------------------
# 4. figure
# ----------------------------------------------------------------------
os.makedirs(FIGDIR, exist_ok=True)
short = [l.replace("_", " ").replace("left", "L").replace("right", "R")
         for l in labels]

fig, ax = plt.subplots(1, 2, figsize=(16, 10), dpi=100,
                       gridspec_kw={"width_ratios": [1, 1.15]})
fig.patch.set_facecolor("white")

cm_norm = cm / cm.sum(axis=1, keepdims=True)
ax[0].imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)
ax[0].set_xticks(range(len(labels)))
ax[0].set_yticks(range(len(labels)))
ax[0].set_xticklabels(short, rotation=40, ha="right", fontsize=11)
ax[0].set_yticklabels(short, fontsize=11)
for a in range(len(labels)):
    for b in range(len(labels)):
        if cm[a, b]:
            ax[0].text(b, a, cm[a, b], ha="center", va="center", fontsize=11,
                       color="white" if cm_norm[a, b] > 0.55 else "#333")
ax[0].set_title(f"Confusion matrix \u2014 held-out frames  ({acc_temporal:.1%})",
                fontsize=14, pad=16)
ax[0].set_xlabel("predicted", fontsize=11)
ax[0].set_ylabel("true", fontsize=11)
ax[0].set_aspect("equal")

pc1 = PCA(n_components=1).fit_transform(X).ravel()
cmap = plt.get_cmap("tab10")
for k, l in enumerate(labels):
    mask = y == l
    ax[1].scatter(np.where(mask)[0], pc1[mask], s=5, color=cmap(k), label=short[k])
ax[1].set_title("Why that number means less than it looks", fontsize=14, pad=16)
ax[1].set_xlabel("sample index, in recording order", fontsize=11)
ax[1].set_ylabel("first principal component", fontsize=11)
ax[1].legend(markerscale=3, fontsize=10, frameon=False, ncol=7,
             loc="upper center", bbox_to_anchor=(0.5, -0.09))
ax[1].text(0.02, 0.97,
           "Each class is a single continuous recording.\n"
           f"Neighbouring frames sit ~{ratio:.0f}x closer together than\n"
           "random pairs, so a shuffled split puts near-duplicates\n"
           f"on both sides and accuracy goes to {acc_shuffled:.0%}.",
           transform=ax[1].transAxes, fontsize=11, color="#3A2E8F",
           va="top", linespacing=1.5)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=2.6)
out = os.path.join(FIGDIR, "pose-recognition.png")
plt.savefig(out, facecolor="white")
print(f"\nfigure written to {out}")
