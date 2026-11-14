"""
Model Architecture Definition for Ransomware API Sequence Detection.
Deep Learning Architecture:
Embedding(310, 64) -> SpatialDropout1D(0.4) -> Bidirectional(LSTM(64))
-> BatchNormalization() -> Dense(32, relu) -> Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics


def build_ransomware_detector(
    input_dim: int = 310,
    output_dim: int = 64,
    maxlen: int = 100,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Constructs and compiles the Bidirectional LSTM deep learning model for
    time-series Windows API sequence classification.
    """
    inputs = layers.Input(shape=(maxlen,), name="api_sequence_input", dtype="int32")
    
    # 1. Embedding Layer: maps token IDs (0..309) to 64-dimensional dense vectors
    x = layers.Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        mask_zero=False,
        name="api_embedding"
    )(inputs)
    
    # 2. Spatial Dropout: drops entire 1D feature maps to prevent co-adaptation of API embeddings
    x = layers.SpatialDropout1D(dropout_rate, name="spatial_dropout_1d")(x)
    
    # 3. Bidirectional LSTM: captures forward and backward temporal dependencies across API calls
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        name="bidirectional_lstm"
    )(x)
    
    # 4. Batch Normalization: stabilizes intermediate activations
    x = layers.BatchNormalization(name="batch_norm")(x)
    
    # 5. Dense Feature Layer: non-linear feature representation
    x = layers.Dense(dense_units, activation="relu", name="dense_feature")(x)
    
    # 6. Output Sigmoid Layer: binary classification probability (0: Benign, 1: Ransomware)
    outputs = layers.Dense(1, activation="sigmoid", name="ransomware_probability")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Ransomware_Early_Detection_BiLSTM")
    
    # Compile model with Adam optimizer and binary crossentropy loss
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="roc_auc")
        ]
    )
    
    return model


def get_default_callbacks(
    checkpoint_path: str = "models/ransomware_bilstm_model.keras",
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 2
) -> list:
    """
    Returns EarlyStopping and ReduceLROnPlateau callbacks.
    """
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-5,
        verbose=1,
        mode="min"
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )
    
    return [early_stopping, reduce_lr, model_checkpoint]
