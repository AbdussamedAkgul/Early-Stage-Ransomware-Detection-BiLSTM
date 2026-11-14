"""
Visualization & Diagram Generator for Ransomware Detection
Generates:
1. confusion_matrix.png (High-Res 300 DPI annotated heatmap)
2. roc_curve.png (High-Res 300 DPI ROC curve with AUC score)
3. training_history.png (Loss and Accuracy curves across epochs)
4. architecture_diagram.png (Professional schematic block diagram of dataflow & deep learning architecture)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Apply clean aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def plot_confusion_matrix(y_true, y_pred_prob, output_path: str, threshold: float = 0.5):
    """Generates high-resolution confusion matrix heatmap."""
    y_pred = (np.array(y_pred_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Custom color palette: clean blues
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
        square=True,
        linewidths=1.5,
        linecolor="#E2E8F0",
        annot_kws={"size": 16, "weight": "bold", "color": "#1E293B"}
    )
    
    # Calculate percentages
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)", ha="center", va="center", fontsize=11, color="#475569")

    ax.set_title("Confusion Matrix - Ransomware API Sequence Detection", fontsize=14, weight="bold", pad=15, color="#0F172A")
    ax.set_xlabel("Predicted Label", fontsize=12, weight="bold", labelpad=10, color="#1E293B")
    ax.set_ylabel("True Label", fontsize=12, weight="bold", labelpad=10, color="#1E293B")
    ax.set_xticklabels(["Benign (0)", "Malware / Ransomware (1)"], fontsize=11)
    ax.set_yticklabels(["Benign (0)", "Malware / Ransomware (1)"], fontsize=11, va="center")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved Confusion Matrix to {output_path}")


def plot_roc_curve(y_true, y_pred_prob, output_path: str):
    """Generates high-resolution ROC curve with AUC metric."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"BiLSTM Model (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance Baseline (AUC = 0.50)")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12, weight="bold", labelpad=10, color="#1E293B")
    ax.set_ylabel("True Positive Rate / Recall (TPR)", fontsize=12, weight="bold", labelpad=10, color="#1E293B")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, weight="bold", pad=15, color="#0F172A")
    
    ax.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved ROC Curve to {output_path}")


def plot_training_history(history_dict: dict, output_path: str):
    """Generates dual-panel Loss and Accuracy history over epochs."""
    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Loss subplot
    ax1.plot(epochs, history_dict.get("loss", []), "o-", color="#DC2626", lw=2, markersize=5, label="Train Loss")
    if "val_loss" in history_dict:
        ax1.plot(epochs, history_dict["val_loss"], "s--", color="#EA580C", lw=2, markersize=5, label="Validation Loss")
    ax1.set_title("Model Loss Progression", fontsize=13, weight="bold", color="#0F172A", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, weight="bold", color="#1E293B")
    ax1.set_ylabel("Binary Crossentropy Loss", fontsize=11, weight="bold", color="#1E293B")
    ax1.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    ax1.legend(loc="upper right", fontsize=10, frameon=True, facecolor="#F8FAFC")
    
    # Accuracy subplot
    ax2.plot(epochs, history_dict.get("accuracy", []), "o-", color="#16A34A", lw=2, markersize=5, label="Train Accuracy")
    if "val_accuracy" in history_dict:
        ax2.plot(epochs, history_dict["val_accuracy"], "s--", color="#0D9488", lw=2, markersize=5, label="Validation Accuracy")
    ax2.set_title("Model Accuracy Progression", fontsize=13, weight="bold", color="#0F172A", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, weight="bold", color="#1E293B")
    ax2.set_ylabel("Accuracy", fontsize=11, weight="bold", color="#1E293B")
    ax2.grid(True, linestyle="--", alpha=0.6, color="#CBD5E1")
    ax2.legend(loc="lower right", fontsize=10, frameon=True, facecolor="#F8FAFC")
    
    plt.suptitle("Deep Learning Model Training & Validation History", fontsize=15, weight="bold", y=1.02, color="#0F172A")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved Training History to {output_path}")


def draw_architecture_diagram(output_path: str):
    """
    Renders a high-resolution, modern schematic block diagram showing:
    1. Dynamic API sequence ingestion
    2. Tokenizer & Sequence Padding (maxlen=100)
    3. Neural Network Layers (Embedding -> SpatialDropout1D -> BiLSTM -> BatchNorm -> Dense -> Sigmoid)
    4. Early-Stage Ransomware Decision & Alerting Engine
    """
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Background canvas card
    canvas = patches.FancyBboxPatch(
        (0.3, 0.3), 14.4, 9.4,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        linewidth=1.5, edgecolor="#CBD5E1", facecolor="#F8FAFC"
    )
    ax.add_patch(canvas)

    # Title block
    ax.text(7.5, 9.2, "Windows API Call Sequence Deep Learning Architecture", ha="center", va="center",
            fontsize=17, weight="bold", color="#0F172A")
    ax.text(7.5, 8.8, "Early-Stage Ransomware Behavioral Detection Pipeline", ha="center", va="center",
            fontsize=12, color="#475569", style="italic")

    # Workflow blocks definitions: (x, y, w, h, title, subtitle, bg_color, border_color, text_color)
    blocks = [
        # Stage 1: Input & Ingestion
        (1.0, 7.0, 3.8, 1.2, "1. Dynamic API Call Stream", "Parent Process Calls (Cuckoo Sandbox)\n[NtCreateFile, RegSetValue, CryptEncrypt...]", "#EFF6FF", "#3B82F6", "#1E3A8A"),
        # Stage 2: Preprocessing
        (1.0, 5.0, 3.8, 1.3, "2. Tokenization & Padding", "Integer Mapping (0..306 -> 1..307)\nSequence Padding (maxlen=100, post-pad/trunc)", "#F0FDF4", "#22C55E", "#14532D"),
        # Stage 3: Embedding & Regularization
        (1.0, 2.8, 3.8, 1.4, "3. Embedding & Dropout", "Embedding(input_dim=310, output_dim=64)\nSpatialDropout1D(rate=0.40)\nOutput Shape: (Batch, 100, 64)", "#FAF5FF", "#A855F7", "#581C87"),
        
        # Stage 4: Bidirectional LSTM
        (5.6, 4.5, 4.0, 1.8, "4. Temporal Feature Extraction", "Bidirectional(LSTM(units=64))\nCaptures Forward & Backward Context\nConcatenated Output: 128 Hidden Features\nParams: 66,048", "#FEF3C7", "#F59E0B", "#78350F"),
        # Stage 5: Normalization & Dense
        (5.6, 2.0, 4.0, 1.6, "5. Normalization & Classifier", "BatchNormalization(128 features)\nDense(32, activation='relu')\nDropout / Feature Condensation\nParams: 4,640", "#FFF1F2", "#F43F5E", "#881337"),
        
        # Stage 6: Output & Decision Engine
        (10.4, 5.2, 3.8, 1.4, "6. Probability Output", "Dense(1, activation='sigmoid')\nBinary Crossentropy Loss\nRansomware Likelihood P ∈ [0.0, 1.0]", "#F1F5F9", "#64748B", "#0F172A"),
        # Stage 7: Early Warning Action
        (10.4, 2.5, 3.8, 1.8, "7. Early Warning & Mitigation", "P >= 0.50 Threshold Decision\nEarly Trigger @ First 20-50 Calls\nImmediate Process Kill & VSS Lock\nALERT: Ransomware Activity Detected!", "#FEF2F2", "#EF4444", "#991B1B"),
    ]

    for x, y, w, h, title, subtitle, bg, border, tc in blocks:
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.2",
            linewidth=2, edgecolor=border, facecolor=bg
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
                fontsize=11.5, weight="bold", color=tc)
        ax.text(x + w / 2, y + (h - 0.30) / 2, subtitle, ha="center", va="center",
                fontsize=9.2, color="#334155", linespacing=1.3)

    # Connective arrows
    arrows = [
        # Stage 1 -> Stage 2
        ((2.9, 7.0), (2.9, 6.3)),
        # Stage 2 -> Stage 3
        ((2.9, 5.0), (2.9, 4.2)),
        # Stage 3 -> Stage 4
        ((4.8, 3.5), (5.6, 5.2)),
        # Stage 4 -> Stage 5
        ((7.6, 4.5), (7.6, 3.6)),
        # Stage 5 -> Stage 6
        ((9.6, 2.8), (10.4, 5.7)),
        # Stage 6 -> Stage 7
        ((12.3, 5.2), (12.3, 4.3))
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#3B82F6",
                lw=2.2,
                mutation_scale=16,
                shrinkA=2,
                shrinkB=2
            )
        )

    # Footnote / Metrics summary bar at the bottom
    footnote_box = patches.FancyBboxPatch(
        (1.0, 0.6), 13.2, 0.9,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        linewidth=1, edgecolor="#93C5FD", facecolor="#DBEAFE"
    )
    ax.add_patch(footnote_box)
    ax.text(
        7.6, 1.05,
        "System Highlights: Stratified 80/20 Split • EarlyStopping(patience=5) • ReduceLROnPlateau • Test Metrics: Accuracy, Precision, Recall, F1, ROC-AUC",
        ha="center", va="center", fontsize=10, weight="bold", color="#1E40AF"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved Architecture Diagram to {output_path}")
