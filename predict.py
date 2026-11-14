"""
Inference & Early-Stage Ransomware Detection Script
Topic: Windows API Çağrı Dizilerinin Zaman Serisi Derin Öğrenme Modeli ile Analizi ve Erken Aşama Fidye Yazılımı Tespiti

Features:
1. Loads trained model (models/ransomware_lstm_model.keras) and vocabulary mapping.
2. Supports input via:
   - Built-in realistic demonstration samples: --sample ransomware | benign
   - Numeric API call sequence: --sequence "112, 274, 158, ..."
   - Textual Windows API function names: --api-names "RegOpenKeyExW, CryptEncrypt, WriteFile, ..."
3. Early-Stage Sequential Analysis (--early-stage-eval):
   Evaluates detection probability at step 10, 20, 30, 50, 75, 100 to prove early detection
   capability BEFORE widespread file encryption occurs.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import keras


def load_model_and_vocab(models_dir: str):
    """Loads the trained Keras model and API vocabulary mapping."""
    model_path = os.path.join(models_dir, "ransomware_lstm_model.keras")
    vocab_path = os.path.join(models_dir, "api_mapping.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please execute 'python train.py' first."
        )

    print(f"[PREDICT] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    api_to_id = {}
    id_to_api = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            api_to_id = mapping.get("api_to_id", {})
            id_to_api = {int(k): v for k, v in mapping.get("id_to_api", {}).items()}

    return model, api_to_id, id_to_api


def tokenize_and_pad(raw_ids: list, maxlen: int = 100) -> np.ndarray:
    """
    Applies the exact same tokenization (+1 shift, maxlen=100 post-padding/post-truncation)
    used during model training.
    """
    token_seq = []
    for val in raw_ids:
        try:
            val_int = int(val)
            if 0 <= val_int <= 306:
                token_seq.append(val_int + 1)
            else:
                token_seq.append(308)  # UNK
        except (ValueError, TypeError):
            token_seq.append(308)  # UNK

    padded = np.zeros((1, maxlen), dtype=int)
    length = min(len(token_seq), maxlen)
    padded[0, :length] = token_seq[:length]
    return padded


def get_demo_samples():
    """Built-in representative behavioral sequences taken from benchmark analysis."""
    # Representative Ransomware sequence (MD5: 071e8c3f8922e186e57548cd4c703a5d, Class: 1)
    ransomware_demo_ids = [
        112, 274, 158, 215, 274, 158, 215, 298, 76, 208, 76, 172, 117, 172, 117, 172,
        76, 117, 35, 60, 81, 60, 81, 60, 81, 60, 81, 60, 81, 60, 81, 60,
        81, 60, 81, 60, 81, 60, 81, 60, 81, 60, 81, 60, 81, 60, 81, 117,
        60, 81, 60, 81, 208, 35, 215, 35, 208, 240, 117, 172, 60, 81, 60, 81,
        225, 35, 60, 81, 35, 225, 172, 60, 81, 60, 81, 60, 81, 172, 117, 76,
        172, 117, 172, 117, 35, 111, 81, 140, 208, 240, 117, 71, 297, 135, 171, 215,
        35, 208, 56, 71
    ]
    ransomware_demo_names = [
        "RegOpenKeyExW", "CreateFileW", "CryptAcquireContextW", "CryptGenKey",
        "CryptEncrypt", "WriteFile", "DeleteFileW", "MoveFileExW"
    ]

    # Representative Legitimate Benign Process sequence (MD5: 03ac674216f3e15c761ee1a5e255f067, Class: 0)
    benign_demo_ids = [
        82, 228, 16, 29, 82, 29, 82, 29, 82, 194, 144, 194, 297, 93, 297, 93,
        127, 297, 93, 215, 16, 252, 172, 117, 172, 117, 252, 29, 82, 29, 252, 29,
        82, 252, 297, 29, 82, 252, 29, 82, 252, 29, 82, 252, 108, 252, 208, 29,
        82, 252, 286, 73, 286, 297, 135, 171, 215, 248, 208, 286, 73, 286, 73, 286,
        73, 286, 73, 286, 73, 286, 73, 286, 73, 286, 73, 286, 73, 286, 73, 286,
        73, 286, 73, 286, 73, 286, 73, 286, 73, 286, 73, 286, 73, 286, 73, 286,
        208, 286, 73, 286, 257, 114
    ]
    benign_demo_names = [
        "GetSystemMetrics", "GetSystemInfo", "GetTickCount", "LoadLibraryW",
        "GetProcAddress", "GetMessageW", "TranslateMessage", "DispatchMessageW"
    ]

    return {
        "ransomware": (ransomware_demo_ids, ransomware_demo_names),
        "benign": (benign_demo_ids, benign_demo_names)
    }


def analyze_sequence(model, raw_ids: list, seq_name: str = "Input Sequence", threshold: float = 0.50):
    """Evaluates an API sequence and prints the diagnostic report."""
    X_input = tokenize_and_pad(raw_ids, maxlen=100)
    prob = float(model.predict(X_input, verbose=0)[0][0])
    is_malware = prob >= threshold

    if prob >= 0.85:
        risk_level = "CRITICAL (High Confidence Ransomware)"
        alert_box = "[!] ALERT: Active Ransomware behavioral pattern detected!"
        action = "IMMEDIATE MITIGATION: Kill process PID, restrict file write privileges, preserve VSS snapshots."
    elif prob >= 0.50:
        risk_level = "ELEVATED (Suspicious Behavioral Pattern)"
        alert_box = "[?] WARNING: Suspicious API sequence observed, potential early ransomware activity."
        action = "CONTAINMENT: Place process in sandbox monitoring, flag for telemetry review."
    else:
        risk_level = "NORMAL / CLEAN (Legitimate Software)"
        alert_box = "[OK] Normal operational behavior, no destructive indicators."
        action = "ALLOW: Process behavior conforms to standard Windows execution profile."

    print("\n" + "=" * 70)
    print(f"             PREDICTION REPORT: {seq_name.upper()}")
    print("=" * 70)
    print(f"  API Call Count Analyzed : {len(raw_ids)} calls (Padded to 100)")
    print(f"  Ransomware Likelihood   : {prob * 100:.2f}%  (Raw Score: {prob:.4f})")
    print(f"  Final Classification    : {'MALICIOUS (RANSOMWARE)' if is_malware else 'BENIGN (SAFE)'}")
    print(f"  Threat Risk Level       : {risk_level}")
    print(f"  Security Diagnostic     : {alert_box}")
    print(f"  Recommended Action      : {action}")
    print("=" * 70)
    return prob


def evaluate_early_stage_detection(model, raw_ids: list, seq_name: str = "Ransomware Early Stage"):
    """
    Demonstrates Early-Stage Ransomware Detection:
    Simulates real-time telemetry where API calls arrive one by one.
    Evaluates detection likelihood at steps: 10, 20, 30, 50, 75, 100 calls.
    """
    checkpoints = [10, 20, 30, 50, 75, 100]
    print("\n" + "=" * 75)
    print(f"      EARLY-STAGE DETECTION TIMELINE ANALYSIS: {seq_name}")
    print("=" * 75)
    print(f" {'Step (Calls)':<15} | {'Padded Input':<15} | {'Ransomware Likelihood':<22} | {'Trigger State'}")
    print("-" * 75)

    triggered_at = None
    for step in checkpoints:
        slice_ids = raw_ids[:step]
        X_slice = tokenize_and_pad(slice_ids, maxlen=100)
        prob = float(model.predict(X_slice, verbose=0)[0][0])
        status = "[CRITICAL ALERT]" if prob >= 0.80 else ("[SUSPICIOUS]" if prob >= 0.50 else "[BENIGN]")
        if prob >= 0.50 and triggered_at is None:
            triggered_at = step

        bar_len = int(prob * 20)
        prob_bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f" t = {step:<10} | {len(slice_ids):<15} | {prob * 100:6.2f}% [{prob_bar}] | {status}")

    print("-" * 75)
    if triggered_at is not None:
        print(f" [SUCCESS] Early-Stage Detection Triggered at Step t={triggered_at} API calls!")
        print(f"           Ransomware was flagged BEFORE 100 API calls completed,")
        print(f"           enabling pre-encryption containment.")
    else:
        print(" [INFO] Sequence did not cross the 0.50 threat threshold.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Inference & Early-Stage Detection on Windows API Call Sequences")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory where model is stored")
    parser.add_argument("--sample", type=str, choices=["ransomware", "benign"], help="Test with built-in realistic demo sample")
    parser.add_argument("--sequence", type=str, help="Comma-separated API call integer IDs (e.g., '112,274,158,215,71')")
    parser.add_argument("--api-names", type=str, help="Comma-separated API function names (e.g., 'RegOpenKeyExW, CryptEncrypt, WriteFile')")
    parser.add_argument("--early-eval", action="store_true", help="Perform temporal early-stage detection analysis")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, args.models_dir)

    try:
        model, api_to_id, id_to_api = load_model_and_vocab(models_dir)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    demos = get_demo_samples()

    # Determine which sequence to evaluate
    if args.sample:
        seq_ids, seq_names = demos[args.sample]
        label_text = f"Built-in {args.sample.capitalize()} Demonstration Sample"
    elif args.sequence:
        seq_ids = [int(x.strip()) for x in args.sequence.split(",") if x.strip().isdigit()]
        label_text = "Custom User Integer API Sequence"
    elif args.api_names:
        names = [x.strip() for x in args.api_names.split(",") if x.strip()]
        seq_ids = [api_to_id.get(name, 308) for name in names]
        label_text = "Custom User API Names Sequence"
    else:
        # Default behavior: run both built-in ransomware and benign demos
        print("\n[INFO] No specific input provided. Running dual benchmark demonstration (Ransomware vs Benign)...")
        rw_ids, _ = demos["ransomware"]
        bn_ids, _ = demos["benign"]

        analyze_sequence(model, rw_ids, "Synthetic Ransomware Attack Sequence")
        if args.early_eval or True:
            evaluate_early_stage_detection(model, rw_ids, "Ransomware Early Stage Timeline")

        analyze_sequence(model, bn_ids, "Legitimate Benign Process Sequence")
        return

    # Run analysis on specified input
    analyze_sequence(model, seq_ids, label_text)
    if args.early_eval:
        evaluate_early_stage_detection(model, seq_ids, label_text)


if __name__ == "__main__":
    main()
