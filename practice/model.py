"""
5가지 핵심 모델:
1. MLP: 기본 다층 퍼셉트론
2. CNN: 1D Convolutional Neural Network (시계열)
3. CNN+Attention: CNN with Multi-Head Attention
4. LSTM: Long Short-Term Memory
5. Transformer: Multi-Head Self-Attention + Feed-Forward
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp(input_dim, n_classes, dropout_rate=0.3):
    """
    MLP (Multi-Layer Perceptron)
    - 시계열이 아닌 평탄화된 피처에 적합
    - 빠른 학습 속도
    """
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ], name="MLP")


def build_cnn(window_size, n_features, n_classes, dropout_rate=0.3):
    """
    CNN (1D Convolutional Neural Network)
    - 시계열 데이터의 국소 패턴 추출
    - 급격한 변화 감지에 효과적
    - window_size가 작을 때 MaxPooling을 조건부 적용
    """
    inputs = layers.Input(shape=(window_size, n_features))

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    seq_len = window_size
    if seq_len >= 4:
        x = layers.MaxPool1D(pool_size=2)(x)
        seq_len = seq_len // 2
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if seq_len >= 4:
        x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="CNN")


def build_cnn_attention(window_size, n_features, n_classes, dropout_rate=0.3, num_heads=4):
    """
    CNN + Multi-Head Attention
    - CNN으로 국소 패턴 추출
    - Attention으로 중요한 시점에 집중
    - 3초 진단에 매우 효과적
    """
    inputs = layers.Input(shape=(window_size, n_features))

    # CNN 블록
    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Multi-Head Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=128 // num_heads,
        dropout=dropout_rate
    )(x, x)

    # Residual connection
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="CNN_Attention")


def build_lstm(window_size, n_features, n_classes, dropout_rate=0.3):
    """
    LSTM (Long Short-Term Memory)
    - 시계열 순차 패턴 학습
    - 장기 의존성 포착
    """
    return models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(256, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        layers.BatchNormalization(),
        layers.LSTM(128, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ], name="LSTM")


def build_transformer(window_size, n_features, n_classes, dropout_rate=0.3, num_heads=8, ff_dim=256):
    """
    Transformer (Multi-Head Self-Attention)
    - 전체 시계열에 대한 Self-Attention
    - 사고 시점과 관련 센서 간 관계 학습
    - 가장 정교한 모델
    """
    inputs = layers.Input(shape=(window_size, n_features))

    # Positional encoding (learnable)
    x = layers.Dense(n_features)(inputs)

    # Transformer block 1
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=n_features // num_heads,
        dropout=dropout_rate
    )(x, x)
    x1 = layers.Add()([x, attention_output])
    x1 = layers.LayerNormalization()(x1)

    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation="relu")(x1)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.Dense(n_features)(ff_output)
    x1 = layers.Add()([x1, ff_output])
    x1 = layers.LayerNormalization()(x1)

    # Transformer block 2
    attention_output2 = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=n_features // num_heads,
        dropout=dropout_rate
    )(x1, x1)
    x2 = layers.Add()([x1, attention_output2])
    x2 = layers.LayerNormalization()(x2)

    # Feed-forward network 2
    ff_output2 = layers.Dense(ff_dim, activation="relu")(x2)
    ff_output2 = layers.Dropout(dropout_rate)(ff_output2)
    ff_output2 = layers.Dense(n_features)(ff_output2)
    x2 = layers.Add()([x2, ff_output2])
    x2 = layers.LayerNormalization()(x2)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x2)

    # Classification head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Transformer")
