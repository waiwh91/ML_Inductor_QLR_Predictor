import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
x = np.linspace(-5, 5, 400)
y_linear = x
y_sigmoid = 1 / (1 + np.exp(-x))
y_silu = x / (1 + np.exp(-x))

# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 10
})

TITLE_SIZE = 14
LABEL_SIZE = 17
XLABEL_SIZE = 12
SUBLABEL_SIZE = 11

# -----------------------------
# Figure
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))

# Curves
axs[0].plot(x, y_linear, linewidth=1.6)
axs[1].plot(x, y_sigmoid, linewidth=1.6)
axs[2].plot(x, y_silu, linewidth=1.6)

# Titles
axs[0].set_title("Linear", fontsize=TITLE_SIZE)
axs[1].set_title("Sigmoid", fontsize=TITLE_SIZE)
axs[2].set_title("SiLU", fontsize=TITLE_SIZE)

# X-axis labels
for ax in axs:
    ax.set_xlabel("Input", fontsize=XLABEL_SIZE, labelpad=12)

# Axis limits and grid
for ax in axs:
    ax.set_xlim(-5, 5)
    ax.grid(True, linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

axs[0].set_ylim(-5, 5)
axs[1].set_ylim(0, 1)
axs[2].set_ylim(-1, 5)

# Formula label box style
box_kw = dict(
    boxstyle="square,pad=0.18",
    facecolor="white",
    edgecolor="black",
    linewidth=0.9
)

# Formula labels
axs[0].text(
    0.045, 0.93, r"$y=x$",
    transform=axs[0].transAxes,
    ha="left", va="top",
    fontsize=LABEL_SIZE,
    bbox=box_kw
)

axs[1].text(
    0.045, 0.93, r"$y=\frac{1}{1+e^{-x}}$",
    transform=axs[1].transAxes,
    ha="left", va="top",
    fontsize=LABEL_SIZE,
    bbox=box_kw
)

axs[2].text(
    0.045, 0.93, r"$y=\frac{x}{1+e^{-x}}$",
    transform=axs[2].transAxes,
    ha="left", va="top",
    fontsize=LABEL_SIZE,
    bbox=box_kw
)

# Subfigure labels
for i, ax in enumerate(axs):
    ax.text(
        0.5, -0.5, f"({chr(97+i)})",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=SUBLABEL_SIZE
    )

plt.tight_layout()

# Save
plt.savefig("activation_functions.pdf", bbox_inches="tight")
plt.savefig("activation_functions.png", dpi=900, bbox_inches="tight")

plt.show()