"""
Windows API Call Sequence Deep Learning Model Training Script
Topic: Windows API Çağrı Dizilerinin Zaman Serisi Derin Öğrenme Modeli ile Analizi ve Erken Aşama Fidye Yazılımı Tespiti

Pipeline:
1. Loads Kaggle 'ang3loliveira/malware-analysis-datasets-api-call-sequences' (or synthetic fallback).
2. Data Preprocessing: Tokenization & Sequence Padding (maxlen=100, post-padding & post-truncating).
3. Stratified Split (80% Train, 20% Test).
4. Model Architecture:
   Embedding(input_dim=310, output_dim=64) ->
   SpatialDropout1D(0.4) ->
   Bidirectional(LSTM(64)) ->
   BatchNormalization() ->
   Dense(32, activation='relu') ->
   Dense(1, activation='sigmoid')
5. Optimizer: Adam, Loss: binary_crossentropy
6. Callbacks: EarlyStopping(patience=5), ReduceLROnPlateau(factor=0.5, patience=2)
7. Outputs & Metrics:
   Accuracy, Precision, Recall, F1, ROC-AUC
   Plots: confusion_matrix.png, roc_curve.png, training_history.png, architecture_diagram.png
   Model artifact: models/ransomware_lstm_model.keras
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from data_loader import (
    load_raw_dataset,
    preprocess_sequences,
    get_stratified_train_test_split,
    save_api_mapping,
    MAX_LEN,
    VOCAB_SIZE
)
from visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    draw_architecture_diagram
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_ransomware_bilstm_model(vocab_size: int = VOCAB_SIZE, maxlen: int = MAX_LEN, lr: float = 1e-3) -> keras.Model:
    """
    Constructs the exact Bidirectional LSTM model specified:
    Embedding(input_dim=310, output_dim=64) ->
    SpatialDropout1D(0.4) ->
    Bidirectional(LSTM(64)) ->
    BatchNormalization() ->
    Dense(32, activation='relu') ->
    Dense(1, activation='sigmoid')
    """
    model = keras.Sequential([
        layers.Input(shape=(maxlen,), name="api_sequence_input"),
        layers.Embedding(input_dim=vocab_size, output_dim=64, name="api_embedding"),
        layers.SpatialDropout1D(0.4, name="spatial_dropout_1d"),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bidirectional_lstm"),
        layers.BatchNormalization(name="batch_norm"),
        layers.Dense(32, activation="relu", name="dense_dense32"),
        layers.Dense(1, activation="sigmoid", name="ransomware_probability_output")
    ], name="Ransomware_Early_Stage_BiLSTM")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM for Early Stage Ransomware Detection via API Sequences")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset CSV")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained model")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Directory to save plots and metrics")
    parser.add_argument("--sample-size", type=int, default=15000, help="Stratified sample size for CPU speed (0 for full dataset)")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    models_dir = os.path.join(script_dir, args.models_dir)
    outputs_dir = os.path.join(script_dir, args.outputs_dir)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 80)
    print("  EARLY-STAGE RANSOMWARE DETECTION VIA TIME-SERIES DEEP LEARNING (BiLSTM)  ")
    print("=" * 80)

    # 1. Save vocabulary mapping for predict.py
    api_map_path = os.path.join(models_dir, "api_mapping.json")
    save_api_mapping(api_map_path)

    # 2. Draw architecture diagram
    arch_diagram_path = os.path.join(outputs_dir, "architecture_diagram.png")
    print(f"\n[STEP 1/6] Generating Architecture Diagram: {arch_diagram_path}")
    draw_architecture_diagram(arch_diagram_path)

    # 3. Load & Preprocess Data
    sample_sz = None if args.sample_size <= 0 else args.sample_size
    print(f"\n[STEP 2/6] Loading Dataset (Target sample size: {'FULL' if sample_sz is None else sample_sz})...")
    df = load_raw_dataset(data_dir=data_dir, sample_size=sample_sz)

    print("\n[STEP 3/6] Preprocessing API Sequences (Tokenization + Sequence Padding maxlen=100)...")
    X, y = preprocess_sequences(df, maxlen=MAX_LEN)

    print("\n[STEP 4/6] Performing Stratified Train/Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y, test_size=0.20, random_state=42)

    # Calculate class weights to handle natural class imbalance
    unique_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weights = dict(zip(unique_classes.astype(int), weights))
    print(f"[DATA] Computed balanced class weights: {class_weights}")

    # 4. Build Model
    print("\n[STEP 5/6] Building BiLSTM Deep Learning Model...")
    model = build_ransomware_bilstm_model(vocab_size=VOCAB_SIZE, maxlen=MAX_LEN, lr=args.lr)
    model.summary()

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    print(f"\n[TRAINING] Starting Model Training (Epochs: {args.epochs}, Batch Size: {args.batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Save model
    model_save_path = os.path.join(models_dir, "ransomware_lstm_model.keras")
    model.save(model_save_path)
    print(f"\n[MODEL] Trained model successfully saved to: {model_save_path}")

    # 5. Evaluate on Test Set
    print("\n[STEP 6/6] Evaluating on Held-Out Test Set (20% Stratified Test)...")
    y_pred_probs = model.predict(X_test, batch_size=args.batch_size, verbose=0).ravel()
    y_pred = (y_pred_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_probs)

    print("\n" + "=" * 60)
    print("               FINAL TEST SET PERFORMANCE METRICS            ")
    print("=" * 60)
    print(f"  Accuracy  : {acc * 100:.2f}% ({acc:.4f})")
    print(f"  Precision : {prec * 100:.2f}% ({prec:.4f})")
    print(f"  Recall    : {rec * 100:.2f}% ({rec:.4f})")
    print(f"  F1-Score  : {f1 * 100:.2f}% ({f1:.4f})")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Ransomware (1)"]))

    # Save metrics JSON
    metrics_summary = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "total_test_samples": int(len(y_test)),
        "malware_test_count": int((y_test == 1).sum()),
        "benign_test_count": int((y_test == 0).sum())
    }
    metrics_path = os.path.join(outputs_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[METRICS] Saved metrics JSON to {metrics_path}")

    # Generate the 3 high-res plots
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    roc_path = os.path.join(outputs_dir, "roc_curve.png")
    hist_path = os.path.join(outputs_dir, "training_history.png")

    plot_confusion_matrix(y_test, y_pred_probs, cm_path)
    plot_roc_curve(y_test, y_pred_probs, roc_path)
    plot_training_history(history.history, hist_path)

    print("\n" + "=" * 80)
    print("  ALL TASKS SUCCESSFULLY COMPLETED!")
    print(f"  Trained Model: {model_save_path}")
    print(f"  Visualizations: {outputs_dir}/")
    print(f"    - {cm_path}")
    print(f"    - {roc_path}")
    print(f"    - {hist_path}")
    print(f"    - {arch_diagram_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
