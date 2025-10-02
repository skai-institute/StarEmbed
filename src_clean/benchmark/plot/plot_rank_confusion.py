# Trio v5:
# - Make the *matrix area* of the rank heatmap square (not the cells), matching confusions.
#   Achieved via ax.set_aspect('auto') + ax.set_box_aspect(1).
# - Keep inverse colormap for rank and reversed annotation colors.
# - Reduce whitespace between middle and right panels (very small gap).

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Resampling

DPI = 200
FIGSIZE = (6, 6)            # square figure for all three
TICK_FONTSIZE = 16
TITLE_FONTSIZE = 16
ANNOT_FONTSIZE = 13
CBAR_LABELSIZE = 12

LEFT = 0.18; RIGHT = 0.98; BOTTOM = 0.12; TOP = 0.88

GAP_12 = 30   # rank ↔ middle
GAP_23 = 2    # middle ↔ right (tighter, per request)

classes = ["EW", "EA", "RRab", "RRc", "RRd", "RS CVn", "LPV"]

models = [
    "Astromer 1",
    "Astromer 2",
    "Moirai-small",
    "Chronos-tiny",
    "Chronos-Bolt",
    "Random",
    "Hand-crafted",
]
classifiers = ["k-NN", "Linear", "RF", "MLP"]

f1 = np.array([
    [0.122, 0.072, 0.115, 0.134],
    [0.537, 0.521, 0.582, 0.388],
    [0.554, 0.579, 0.551, 0.592],
    [0.672, 0.617, 0.636, 0.645],
    [0.570, 0.580, 0.580, 0.612],
    [0.120, 0.076, 0.115, 0.096],
    [0.712, 0.714, 0.803, 0.735],
])

cm_chronos_mlp = np.array([
    [0.739, 0.052, 0.012, 0.075, 0.018, 0.102, 0.001],
    [0.031, 0.943, 0.001, 0.000, 0.002, 0.022, 0.001],
    [0.009, 0.002, 0.849, 0.057, 0.035, 0.047, 0.001],
    [0.093, 0.002, 0.069, 0.693, 0.101, 0.040, 0.001],
    [0.060, 0.000, 0.157, 0.353, 0.341, 0.072, 0.016],
    [0.101, 0.028, 0.028, 0.029, 0.028, 0.754, 0.033],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.100, 0.900],
])

cm_hf_rf = np.array([
    [0.960, 0.019, 0.002, 0.008, 0.000, 0.011, 0.000],
    [0.108, 0.891, 0.000, 0.000, 0.000, 0.001, 0.000],
    [0.053, 0.000, 0.908, 0.034, 0.003, 0.003, 0.000],
    [0.104, 0.001, 0.013, 0.873, 0.009, 0.000, 0.000],
    [0.120, 0.000, 0.020, 0.526, 0.309, 0.024, 0.000],
    [0.332, 0.011, 0.021, 0.004, 0.000, 0.625, 0.007],
    [0.057, 0.000, 0.000, 0.000, 0.000, 0.071, 0.871],
])

def to_rank(matrix: np.ndarray) -> np.ndarray:
    order = (-matrix).argsort(axis=0)
    return order.argsort(axis=0) + 1

def save_square(fig, png_path, pdf_path):
    fig.set_size_inches(*FIGSIZE)
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(pdf_path, dpi=DPI)
    plt.close(fig)

# ---- Rank heatmap: square matrix area ----
rank_mat = to_rank(f1)
fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(rank_mat, cmap="viridis_r", interpolation="nearest", vmin=1, vmax=rank_mat.max())
# Make the axes box square and allow rectangular data scaling (overall square matrix)
ax.set_aspect('auto')
try:
    ax.set_box_aspect(1)   # square axes box (Matplotlib ≥3.3)
except Exception:
    pass

ax.set_xticks(np.arange(len(classifiers)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(classifiers, fontsize=TICK_FONTSIZE)
ax.set_yticklabels(models, fontsize=TICK_FONTSIZE)

threshold = (rank_mat.min() + rank_mat.max()) / 2.0
for i in range(rank_mat.shape[0]):
    for j in range(rank_mat.shape[1]):
        r = rank_mat[i, j]
        color = "black" if r <= threshold else "white"  # reversed color logic
        ax.text(j, i, r, ha="center", va="center", color=color, fontsize=ANNOT_FONTSIZE)

ax.set_title("F1 Rank (1 = best)", fontsize=TITLE_FONTSIZE)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20, shrink=0.8)
cbar.set_label("Rank (1 = best)", fontsize=CBAR_LABELSIZE)
cbar.ax.tick_params(labelsize=CBAR_LABELSIZE)
fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
rank_png = "/mnt/data/_rank_square_box_v5.png"
rank_pdf = "/mnt/data/_rank_square_box_v5.pdf"
save_square(fig, rank_png, rank_pdf)

# ---- Middle CM (no colorbar) ----
fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(cm_chronos_mlp, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=TICK_FONTSIZE, rotation=45, ha="right")
ax.set_yticklabels(classes, fontsize=TICK_FONTSIZE)
for i in range(cm_chronos_mlp.shape[0]):
    for j in range(cm_chronos_mlp.shape[1]):
        v = cm_chronos_mlp[i, j]
        color = "white" if v < 0.45 else "black"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=ANNOT_FONTSIZE)
ax.set_title("Chronos‑tiny + MLP", fontsize=TITLE_FONTSIZE)
fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
cm1_png = "/mnt/data/_cm_mid_v5.png"
cm1_pdf = "/mnt/data/_cm_mid_v5.pdf"
save_square(fig, cm1_png, cm1_pdf)

# ---- Right CM (no y-labels, keep colorbar) ----
fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(cm_hf_rf, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
ax.set_xticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=TICK_FONTSIZE, rotation=45, ha="right")
ax.set_yticks([]); ax.set_yticklabels([])
for i in range(cm_hf_rf.shape[0]):
    for j in range(cm_hf_rf.shape[1]):
        v = cm_hf_rf[i, j]
        color = "white" if v < 0.45 else "black"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=ANNOT_FONTSIZE)
ax.set_title("Hand‑crafted + RF", fontsize=TITLE_FONTSIZE)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20, shrink=0.8)
cbar.set_label("Proportion", fontsize=CBAR_LABELSIZE)
cbar.ax.tick_params(labelsize=CBAR_LABELSIZE)
fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)
cm2_png = "/mnt/data/_cm_right_v5.png"
cm2_pdf = "/mnt/data/_cm_right_v5.pdf"
save_square(fig, cm2_png, cm2_pdf)

# ---- Stitch with tighter middle↔right spacing ----
im_rank = Image.open(rank_png)
im_mid  = Image.open(cm1_png)
im_right= Image.open(cm2_png)

h = max(im.height for im in [im_rank, im_mid, im_right])
def resize_to_height(img, target_h):
    if img.height == target_h:
        return img
    scale = target_h / img.height
    return img.resize((int(img.width * scale), target_h), Resampling.LANCZOS)

im_rank = resize_to_height(im_rank, h)
im_mid  = resize_to_height(im_mid,  h)
im_right= resize_to_height(im_right,h)

total_w = im_rank.width + GAP_12 + im_mid.width + GAP_23 + im_right.width
canvas = Image.new("RGB", (total_w, h), "white")

x = 0
canvas.paste(im_rank, (x, 0)); x += im_rank.width + GAP_12
canvas.paste(im_mid,  (x, 0)); x += im_mid.width + GAP_23
canvas.paste(im_right,(x, 0))

combined_png = "/projects/b1094/StarEmbed/src/output/combined_rank_and_confusions_row_v5.png"
combined_pdf = "/projects/b1094/StarEmbed/src/output/combined_rank_and_confusions_row_v5.pdf"
canvas.save(combined_png)
canvas.save(combined_pdf)

combined_png, combined_pdf
