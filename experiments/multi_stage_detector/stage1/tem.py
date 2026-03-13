import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ── Simulated TF-IDF counts (higher FN, lower TP vs BGE Large) ──────────────
# Scenario 1: PARAPHRASE = DUPLICATE (Positive Class)
s1 = {
    "NEW      ":  {"NEW": 9800,  "DUPE/PARAPHRASE": 2748},   # ↑ FP slightly
    "DUPE/\nPARAPHRASE": {"NEW": 3200,  "DUPE/PARAPHRASE":  958},   # ↑ FN massively, ↓ TP
}

# Scenario 2: PARAPHRASE = NEW (Negative Class)
s2 = {
    "NEW/\nPARAPHRASE": {"NEW/PARAPHRASE": 10900, "DUPE": 1648},   # ↑ FP
    "DUPE     ":         {"NEW/PARAPHRASE":  3600, "DUPE":  328},   # ↑ FN massively, ↓ TP
}

def make_matrix(data):
    rows = list(data.keys())
    cols = list(list(data.values())[0].keys())
    mat  = np.array([[data[r][c] for c in cols] for r in rows])
    return mat, rows, cols

def plot_cm(ax, mat, row_labels, col_labels, cmap, title):
    vmax = mat.max()
    im   = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    # Cell text
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v / vmax > 0.45 else "#444444"
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Prediction (verdict)", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth (llm_verdict)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    plt.colorbar(im, ax=ax, label="Count")

# ── Colormaps matching original style ────────────────────────────────────────
blue_cmap   = LinearSegmentedColormap.from_list("blue_cm",  ["#dce9f5", "#08306b"])
orange_cmap = LinearSegmentedColormap.from_list("orange_cm",["#fde8d8", "#7f1900"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("TF-IDF + Clustering",
             fontsize=14, fontweight="bold", y=1.02)

mat1, r1, c1 = make_matrix(s1)
plot_cm(ax1, mat1, r1, c1, blue_cmap,
        "Scenario 1: PARAPHRASE = DUPLICATE\n(Positive Class)")

mat2, r2, c2 = make_matrix(s2)
plot_cm(ax2, mat2, r2, c2, orange_cmap,
        "Scenario 2: PARAPHRASE = NEW\n(Negative Class)")

plt.tight_layout()
plt.savefig("tfidf_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Print metric comparison ───────────────────────────────────────────────────
def metrics(tn, fp, fn, tp):
    p  = tp / (tp + fp) if (tp + fp) else 0
    r  = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2*p*r / (p+r)  if (p + r)  else 0
    acc= (tp+tn)/(tp+tn+fp+fn)
    return p, r, f1, acc

print("\n" + "="*55)
print("  METRIC COMPARISON  —  BGE Large vs TF-IDF (Simulated)")
print("="*55)

bge_s1   = metrics(10811, 1737,  970, 1358)
tfidf_s1 = metrics( 9800, 2748, 3200,  958)
bge_s2   = metrics(11861,  687, 1513,  815)
tfidf_s2 = metrics(10900, 1648, 3600,  328)

for label, bge, tfidf in [("Scenario 1", bge_s1, tfidf_s1),
                           ("Scenario 2", bge_s2, tfidf_s2)]:
    print(f"\n  {label}")
    print(f"  {'Metric':<12} {'BGE Large':>10} {'TF-IDF':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for name, b, t in zip(["Precision","Recall","F1","Accuracy"],bge,tfidf):
        print(f"  {name:<12} {b*100:>9.1f}% {t*100:>9.1f}% {(b-t)*100:>+9.1f}%")
print("="*55)