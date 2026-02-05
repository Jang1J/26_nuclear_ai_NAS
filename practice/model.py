import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp(input_dim, n_classes):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])


def to_square_per_timestep(X):
    """
    (N, D) -> (N, S, S, 1) 로 reshape.
    D가 완전 제곱이 아니면 뒤를 0으로 패딩.
    """
    N, D = X.shape
    S = int(np.ceil(np.sqrt(D)))
    D2 = S * S
    if D2 != D:
        Xp = np.zeros((N, D2), dtype=X.dtype)
        Xp[:, :D] = X
    else:
        Xp = X
    Xsq = Xp.reshape(N, S, S, 1)
    return Xsq, S


def build_cnn_2d(input_shape, n_classes):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])


# =====================================================================
# 개선된 모델들
# =====================================================================

def build_mlp_v2(input_dim, n_classes, dropout_rate=0.3):
    """
    개선된 MLP: 4층 + BatchNormalization + Dropout.
    기존 MLP 대비 깊고 안정적인 학습 가능.
    """
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ])


def build_cnn1d(window_size, n_features, n_classes, dropout_rate=0.3):
    """
    1D CNN: 시계열 시간축을 따라 Conv1D 적용.
    Input shape: (window_size, n_features)
    """
    return models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2),
        layers.Dropout(dropout_rate),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2),
        layers.Dropout(dropout_rate),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ])


def build_lstm(window_size, n_features, n_classes, dropout_rate=0.3):
    """
    2-layer LSTM: 시계열 순차 패턴 학습.
    Input shape: (window_size, n_features)
    """
    return models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(128, return_sequences=True, dropout=dropout_rate),
        layers.LSTM(64, return_sequences=False, dropout=dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ])


def build_cnn1d_lstm(window_size, n_features, n_classes, dropout_rate=0.3):
    """
    CNN+LSTM 하이브리드: Conv1D로 국소 패턴 추출 → LSTM으로 시간 흐름 학습.
    원전 사고의 급격한 변화(CNN) + 점진적 추세(LSTM) 모두 포착.
    Input shape: (window_size, n_features)
    """
    return models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2),
        layers.Dropout(dropout_rate),
        layers.LSTM(64, return_sequences=False, dropout=dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(n_classes, activation="softmax"),
    ])