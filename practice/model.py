"""
5가지 핵심 모델:
1. MLP: 기본 다층 퍼셉트론
2. CNN: Multi-Scale 1D CNN (InceptionTime 기반)
3. CNN+Attention: CNN with Multi-Head Attention
4. TCN: Temporal Convolutional Network (LSTM 대체)
5. Transformer: Multi-Head Self-Attention + Feed-Forward

References:
  - TCN: "Accident severity assessment for NPP based on TCN"
         (J. Nuclear Sci. Tech., 2025)
  - CNN Multi-Scale: "InceptionTime: Finding AlexNet for TSC"
         (Data Mining & Knowledge Discovery, 2020)
  - Transformer: "Improving position encoding of transformers for
         multivariate TSC" (Data Mining & KD, 2023, ConvTran)
"""

import numpy as np
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
    CNN (Multi-Scale 1D Convolutional Neural Network)
    - InceptionTime 기반 다중 스케일 커널 (k=1, 2, 3)
    - 각 스케일에서 다른 시간 패턴 추출
    - window_size=3에서도 효율적 (MaxPooling 불필요)

    Ref: "InceptionTime: Finding AlexNet for TSC"
         (Data Mining & Knowledge Discovery, 2020)
    """
    inputs = layers.Input(shape=(window_size, n_features))

    # Multi-Scale Block: 3가지 커널 크기 병렬 적용
    branch_a = layers.Conv1D(64, kernel_size=1, padding="same", activation="relu")(inputs)
    branch_b = layers.Conv1D(64, kernel_size=2, padding="same", activation="relu")(inputs)
    branch_c = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)

    x = layers.Concatenate(axis=-1)([branch_a, branch_b, branch_c])  # (window, 192)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # 채널 통합
    x = layers.Conv1D(256, kernel_size=1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
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
    - 2단계 CNN (k=1 → k=3)으로 효율적 피처 추출
    - Attention으로 중요한 시점에 집중
    - 3초 진단에 매우 효과적
    """
    inputs = layers.Input(shape=(window_size, n_features))

    # CNN 블록: k=1(시점별 변환) → k=3(윈도우 패턴)
    x = layers.Conv1D(128, kernel_size=1, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
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


def _tcn_residual_block(x, n_filters, kernel_size, dilation_rate, dropout_rate):
    """
    TCN Residual Block.
    Causal Conv1D × 2 + BatchNorm + ReLU + Dropout + Residual shortcut.
    """
    # Branch 1: Causal convolutions
    conv_out = layers.Conv1D(
        n_filters, kernel_size, padding="causal",
        dilation_rate=dilation_rate, activation="relu",
    )(x)
    conv_out = layers.BatchNormalization()(conv_out)
    conv_out = layers.Dropout(dropout_rate)(conv_out)

    conv_out = layers.Conv1D(
        n_filters, kernel_size, padding="causal",
        dilation_rate=dilation_rate, activation="relu",
    )(conv_out)
    conv_out = layers.BatchNormalization()(conv_out)
    conv_out = layers.Dropout(dropout_rate)(conv_out)

    # Branch 2: Residual shortcut (1×1 conv if channel mismatch)
    if x.shape[-1] != n_filters:
        shortcut = layers.Conv1D(n_filters, kernel_size=1)(x)
    else:
        shortcut = x

    out = layers.Add()([conv_out, shortcut])
    out = layers.Activation("relu")(out)
    return out


def build_tcn(window_size, n_features, n_classes, dropout_rate=0.3):
    """
    TCN (Temporal Convolutional Network)
    - Dilated causal convolution으로 미래 정보 유출 방지
    - 병렬 처리 가능 (LSTM보다 빠른 추론)
    - Residual connection으로 안정적 gradient 전파
    - window_size=3에서도 효과적 (dilation=1,2)

    Ref: "Accident severity assessment for NPP based on TCN
          and Bayesian optimization"
         (J. Nuclear Sci. Tech., Vol.62(5), 2025)
    """
    inputs = layers.Input(shape=(window_size, n_features))

    # Residual Block 1: dilation=1 → 수용영역 = 3 (k=2, d=1 → 2스텝)
    x = _tcn_residual_block(inputs, n_filters=128, kernel_size=2,
                            dilation_rate=1, dropout_rate=dropout_rate)

    # Residual Block 2: dilation=2 → 수용영역 = 7 (k=2, d=2 → 4스텝 추가)
    x = _tcn_residual_block(x, n_filters=128, kernel_size=2,
                            dilation_rate=2, dropout_rate=dropout_rate)

    # Global pooling + Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="TCN")


def build_transformer(window_size, n_features, n_classes, dropout_rate=0.3,
                      num_heads=8, ff_dim=256):
    """
    Transformer (Multi-Head Self-Attention)
    - 입력을 8의 배수 차원(d_model)으로 투영하여 head 정렬
    - 1-block 구조 (window=3에 최적화)
    - Learnable positional embedding

    Ref: "Improving position encoding of transformers for
          multivariate TSC" (Data Mining & KD, 2023, ConvTran)
    """
    # d_model: n_features를 num_heads의 배수로 올림
    d_model = int(np.ceil(n_features / num_heads)) * num_heads  # 207 → 208

    inputs = layers.Input(shape=(window_size, n_features))

    # Input projection + Learnable positional embedding
    pos_embedding = layers.Embedding(input_dim=window_size, output_dim=d_model)
    positions = tf.range(start=0, limit=window_size, delta=1)
    x = layers.Dense(d_model)(inputs) + pos_embedding(positions)

    # Transformer Block (1블록, 3스텝에 충분)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,   # 208 / 8 = 26 (정확히 나누어떨어짐)
        dropout=dropout_rate
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.Dense(d_model)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)

    # Global pooling + Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="Transformer")
