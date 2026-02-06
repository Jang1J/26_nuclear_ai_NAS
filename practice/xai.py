"""
XAI (Explainable AI) 분석 모듈.

원전 사고 진단 모델의 해석성 제공:
1. SHAP 분석: 글로벌/클래스별 피처 중요도
2. Attention 시각화: CNN+Attention 모델의 attention weight 분석

사용법:
    python -m practice.xai \
        --model_dir models \
        --data_folder data/data_new \
        --output_dir xai_results \
        --analysis_type all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from .dataloader import load_Xy_runs, create_sliding_windows_from_runs, LABELS, ID2LABEL
from .data_split import split_runs
from .preprocessing import PreprocessingPipeline

# ── 한글 폰트 설정 ──────────────────────────────────────
plt.rcParams["axes.unicode_minus"] = False
for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font_name
        break


# ═══════════════════════════════════════════════════════════
# 1. Attention 시각화
# ═══════════════════════════════════════════════════════════
class AttentionVisualizer:
    """CNN+Attention 모델의 attention weight 추출 및 시각화."""

    def __init__(self, model):
        self.model = model
        self.attention_layer = self._find_attention_layer()
        self.attention_model = self._build_attention_model()

    def _find_attention_layer(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.MultiHeadAttention):
                return layer
        raise ValueError("MultiHeadAttention 레이어를 찾을 수 없습니다.")

    def _build_attention_model(self):
        """
        Attention weight를 추출하는 중간 모델 구축.
        MultiHeadAttention의 return_attention_scores=True 활용.
        """
        inp = self.model.input

        # 모델의 레이어를 순서대로 실행하며 attention 레이어에서 분기
        x = inp
        attention_scores = None

        for layer in self.model.layers[1:]:  # Input 레이어 제외
            if layer == self.attention_layer:
                # Attention 레이어: attention scores도 반환
                attn_out, attn_scores = layer(x, x, return_attention_scores=True)
                attention_scores = attn_scores
                # Residual connection: 다음 레이어가 Add인지 확인
                x = attn_out  # attention output으로 진행 (후속 레이어에서 처리)
            else:
                # 일반 레이어: 직접 호출
                try:
                    if isinstance(layer, tf.keras.layers.Add):
                        # Add 레이어는 2개 입력 필요 → 스킵하고 모델 출력만 사용
                        break
                    x = layer(x)
                except Exception:
                    break

        if attention_scores is None:
            raise ValueError("Attention scores를 추출할 수 없습니다.")

        return tf.keras.Model(inputs=inp, outputs=attention_scores)

    def get_attention_weights(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Attention weights 추출 (직접 그래디언트 테이프 방식).

        대안: call_and_return_attention 방식으로 직접 추출.

        Args:
            X: (N, W, D)
            batch_size: 배치 크기

        Returns:
            attn_weights: (N, num_heads, W, W)
        """
        # 직접 forward pass로 attention weight 추출
        inp = self.model.input
        x = inp

        # CNN 블록 찾기 (Attention 전까지)
        cnn_layers = []
        attn_layer = None
        for layer in self.model.layers[1:]:
            if isinstance(layer, tf.keras.layers.MultiHeadAttention):
                attn_layer = layer
                break
            cnn_layers.append(layer)

        # CNN 블록을 통과시킨 후 Attention 호출
        @tf.function
        def extract_attention(batch_x):
            h = batch_x
            for layer in cnn_layers:
                h = layer(h, training=False)
            _, scores = attn_layer(h, h, return_attention_scores=True, training=False)
            return scores

        all_scores = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            scores = extract_attention(tf.constant(batch, dtype=tf.float32))
            all_scores.append(scores.numpy())

        return np.concatenate(all_scores, axis=0)

    def visualize_attention_heatmap(
        self, X: np.ndarray, y: np.ndarray, output_dir: Path
    ):
        """
        클래스별 평균 attention weight 히트맵.

        Output: attention_heatmap_all_classes.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        attn_weights = self.get_attention_weights(X)
        # attn_weights: (N, num_heads, W, W)

        num_heads = attn_weights.shape[1]
        W = attn_weights.shape[2]
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        fig, axes = plt.subplots(
            n_classes, num_heads,
            figsize=(3 * num_heads + 1, 2 * n_classes + 1),
            squeeze=False,
        )
        fig.suptitle("Attention Weights by Class & Head", fontsize=14, y=1.02)

        for i, cls_id in enumerate(unique_classes):
            mask = y == cls_id
            avg_attn = attn_weights[mask].mean(axis=0)  # (num_heads, W, W)

            for h in range(num_heads):
                ax = axes[i, h]
                im = ax.imshow(avg_attn[h], cmap="YlOrRd", vmin=0, aspect="auto")
                ax.set_xticks(range(W))
                ax.set_yticks(range(W))
                ax.set_xticklabels([f"t-{W - 1 - t}" for t in range(W)], fontsize=7)
                ax.set_yticklabels([f"t-{W - 1 - t}" for t in range(W)], fontsize=7)

                if h == 0:
                    ax.set_ylabel(LABELS[cls_id], fontsize=9, fontweight="bold")
                if i == 0:
                    ax.set_title(f"Head {h}", fontsize=9)

        fig.colorbar(im, ax=axes, shrink=0.6, label="Attention Score")
        fig.tight_layout()
        path = output_dir / "attention_heatmap_all_classes.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")

    def visualize_per_head_patterns(
        self, X: np.ndarray, y: np.ndarray, output_dir: Path
    ):
        """
        각 Head별 시점 집중도 분석.

        Output: per_head_analysis.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        attn_weights = self.get_attention_weights(X)

        num_heads = attn_weights.shape[1]
        W = attn_weights.shape[2]

        # 각 head별로 어떤 시점(query→key)에 집중하는지 분석
        # 마지막 query(최신 시점)가 어디에 주목하는지가 가장 중요
        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4), squeeze=False)
        fig.suptitle("Per-Head Attention Patterns (Last Query → All Keys)", fontsize=13)

        unique_classes = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

        time_labels = [f"t-{W - 1 - t}s" for t in range(W)]

        for h in range(num_heads):
            ax = axes[0, h]
            for ci, cls_id in enumerate(unique_classes):
                mask = y == cls_id
                # 마지막 query row의 attention
                head_attn = attn_weights[mask, h, -1, :]  # (N_cls, W)
                avg = head_attn.mean(axis=0)
                ax.plot(range(W), avg, marker="o", label=LABELS[cls_id],
                        color=colors[ci], linewidth=1.5, markersize=4)

            ax.set_title(f"Head {h}", fontsize=11)
            ax.set_xlabel("Key 시점")
            ax.set_ylabel("Attention Score")
            ax.set_xticks(range(W))
            ax.set_xticklabels(time_labels, fontsize=8)
            ax.legend(fontsize=6, ncol=2, loc="upper left")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = output_dir / "per_head_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")

    def attention_by_class(
        self, X: np.ndarray, y: np.ndarray, output_dir: Path
    ):
        """
        클래스별 attention 집중 시점 비교.

        Output: attention_by_class.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        attn_weights = self.get_attention_weights(X)

        W = attn_weights.shape[2]
        unique_classes = np.unique(y)

        # 모든 head 평균, 마지막 query의 attention
        avg_attn_per_class = []
        class_labels = []
        for cls_id in unique_classes:
            mask = y == cls_id
            # (N_cls, heads, W, W) → heads 평균 → 마지막 query → (N_cls, W) → 평균
            head_avg = attn_weights[mask].mean(axis=1)  # (N_cls, W, W)
            last_query = head_avg[:, -1, :]  # (N_cls, W)
            avg_attn_per_class.append(last_query.mean(axis=0))
            class_labels.append(LABELS[cls_id])

        avg_matrix = np.array(avg_attn_per_class)  # (n_classes, W)

        fig, ax = plt.subplots(figsize=(8, 6))
        time_labels = [f"t-{W - 1 - t}s" for t in range(W)]

        im = ax.imshow(avg_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(W))
        ax.set_xticklabels(time_labels, fontsize=10)
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(class_labels, fontsize=10)
        ax.set_xlabel("Key 시점 (Attention 대상)", fontsize=11)
        ax.set_ylabel("사고 유형", fontsize=11)
        ax.set_title("Class-wise Attention Pattern\n(All Heads Avg, Last Query)", fontsize=12)

        # 값 표시
        for i in range(len(class_labels)):
            for j in range(W):
                ax.text(j, i, f"{avg_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if avg_matrix[i, j] > avg_matrix.max() * 0.6 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Attention Score")
        fig.tight_layout()
        path = output_dir / "attention_by_class.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════
# 2. SHAP 분석
# ═══════════════════════════════════════════════════════════
class SHAPAnalyzer:
    """
    SHAP 기반 피처 중요도 분석.
    Keras 3 호환성을 위해 여러 fallback 전략 사용.
    """

    def __init__(self, model, background_data: np.ndarray):
        """
        Args:
            model: Keras 모델
            background_data: SHAP 배경 데이터 (N_bg, W, D) or (N_bg, D)
        """
        self.model = model
        self.background = background_data
        self.explainer = None
        self.method_used = None
        self._init_explainer()

    def _init_explainer(self):
        """SHAP Explainer 초기화 (호환성 순서대로 시도)."""
        import shap

        # 시도 1: GradientExplainer (Keras 3와 호환성 좋음)
        try:
            self.explainer = shap.GradientExplainer(self.model, self.background)
            self.method_used = "GradientExplainer"
            print(f"  [SHAP] {self.method_used} 초기화 성공")
            return
        except Exception as e:
            print(f"  [SHAP] GradientExplainer 실패: {e}")

        # 시도 2: DeepExplainer
        try:
            self.explainer = shap.DeepExplainer(self.model, self.background)
            self.method_used = "DeepExplainer"
            print(f"  [SHAP] {self.method_used} 초기화 성공")
            return
        except Exception as e:
            print(f"  [SHAP] DeepExplainer 실패: {e}")

        # 시도 3: KernelExplainer (가장 느리지만 확실함)
        try:
            def model_predict(x):
                return self.model.predict(x, verbose=0)

            # KernelExplainer용 배경 데이터는 2D여야 함
            bg = self.background
            if bg.ndim == 3:
                bg = bg.reshape(bg.shape[0], -1)

            self.explainer = shap.KernelExplainer(model_predict, bg[:50])
            self.method_used = "KernelExplainer"
            print(f"  [SHAP] {self.method_used} 초기화 성공 (느릴 수 있음)")
            return
        except Exception as e:
            print(f"  [SHAP] KernelExplainer 실패: {e}")

        # 최종 fallback: Permutation Importance
        self.method_used = "PermutationImportance"
        print(f"  [SHAP] 모든 SHAP 방법 실패. Permutation Importance 사용")

    def compute_shap_values(
        self, X: np.ndarray, n_samples: int = 500
    ) -> np.ndarray:
        """
        SHAP values 계산.

        Args:
            X: 설명할 데이터 (N, W, D)
            n_samples: 최대 설명 샘플 수

        Returns:
            shap_values: (n_classes, N, W, D) or (N, W, D)
        """
        X_explain = X[:n_samples]

        if self.method_used == "PermutationImportance":
            return self._permutation_importance(X_explain)

        try:
            shap_values = self.explainer.shap_values(X_explain)
            return shap_values
        except Exception as e:
            print(f"  [SHAP] shap_values 계산 실패: {e}")
            print(f"  [SHAP] Permutation Importance로 대체")
            return self._permutation_importance(X_explain)

    def _permutation_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Permutation importance (모델 무관 방법).

        피처를 하나씩 셔플하면서 예측 변화 측정.
        """
        baseline_pred = self.model.predict(X, verbose=0)
        baseline_class = np.argmax(baseline_pred, axis=1)
        n_classes = baseline_pred.shape[1]

        # 피처 차원 (마지막 축)
        n_features = X.shape[-1]
        importance = np.zeros((n_classes, *X.shape))

        for f_idx in range(n_features):
            X_perm = X.copy()
            # 피처 셔플
            perm_idx = np.random.permutation(len(X))
            if X.ndim == 3:
                X_perm[:, :, f_idx] = X[perm_idx, :, f_idx]
            else:
                X_perm[:, f_idx] = X[perm_idx, f_idx]

            perm_pred = self.model.predict(X_perm, verbose=0)
            # 각 클래스 확률 변화량 = importance
            diff = baseline_pred - perm_pred  # (N, n_classes)

            for c in range(n_classes):
                if X.ndim == 3:
                    importance[c, :, :, f_idx] = diff[:, c:c + 1]
                else:
                    importance[c, :, f_idx] = diff[:, c]

        return importance

    def explain_global(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        output_dir: Path,
        top_k: int = 20,
        n_samples: int = 500,
    ):
        """
        글로벌 피처 중요도 분석 + 시각화.

        Output: global_importance_top20.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [SHAP Global] {n_samples}개 샘플 분석 중...")

        shap_values = self.compute_shap_values(X, n_samples)
        n_explain = min(n_samples, len(X))

        # shap_values 형태 통일
        if isinstance(shap_values, list):
            # GradientExplainer: list of (N, W, D) 배열 (클래스별)
            sv = np.array(shap_values)  # (n_classes, N, W, D)
        else:
            sv = shap_values
            if sv.ndim == len(X[:n_explain].shape):
                sv = np.expand_dims(sv, 0)

        # 시간 축 + 배치 평균하여 글로벌 중요도 계산
        sv_abs = np.abs(sv)

        if sv_abs.ndim == 4:
            # (n_classes, N, W, D) → 시간축 평균 → (n_classes, N, D)
            sv_flat = sv_abs.mean(axis=2)
            global_importance = sv_flat.mean(axis=(0, 1))  # (D,)
        elif sv_abs.ndim == 3:
            sv_flat = sv_abs
            global_importance = sv_flat.mean(axis=(0, 1))  # (D,)
        else:
            sv_flat = sv_abs
            global_importance = sv_flat.mean(axis=0)  # (D,)

        # 피처 이름과 중요도 정렬
        n_feat = len(global_importance)
        names = feature_names[:n_feat] if len(feature_names) >= n_feat else \
            feature_names + [f"feat_{i}" for i in range(len(feature_names), n_feat)]

        sorted_idx = np.argsort(global_importance)[::-1]
        actual_top_k = min(top_k, n_feat)
        top_idx = sorted_idx[:actual_top_k]

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        top_names = [names[i] for i in top_idx]
        top_values = global_importance[top_idx]

        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, actual_top_k))
        bars = ax.barh(range(actual_top_k)[::-1], top_values, color=colors)
        ax.set_yticks(range(actual_top_k)[::-1])
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"Global Feature Importance (Top {actual_top_k})\n[Method: {self.method_used}]",
                      fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        path = output_dir / "global_importance_top20.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")

        # CSV 저장
        csv_path = output_dir / "global_importance_all.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("rank,feature,importance\n")
            for rank, idx in enumerate(sorted_idx):
                f.write(f"{rank + 1},{names[idx]},{global_importance[idx]:.6f}\n")
        print(f"  [Saved] {csv_path}")

        return global_importance, names

    def explain_per_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        output_dir: Path,
        top_k: int = 15,
        n_samples: int = 500,
    ):
        """
        클래스별 피처 중요도 분석.

        Output: per_class_importance.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [SHAP Per-Class] 분석 중...")

        shap_values = self.compute_shap_values(X, n_samples)
        n_explain = min(n_samples, len(X))
        unique_classes = np.unique(y[:n_explain])

        if isinstance(shap_values, list):
            sv = np.array(shap_values)
        else:
            sv = shap_values
            if sv.ndim == len(X[:n_explain].shape):
                sv = np.expand_dims(sv, 0)

        # 시간 축 평균
        sv_abs = np.abs(sv)
        if sv_abs.ndim == 4:
            sv_flat = sv_abs.mean(axis=2)  # (n_classes, N, D)
        else:
            sv_flat = sv_abs

        n_feat = sv_flat.shape[-1]
        names = feature_names[:n_feat] if len(feature_names) >= n_feat else \
            feature_names + [f"feat_{i}" for i in range(len(feature_names), n_feat)]

        n_cls = len(unique_classes)
        n_cols = min(3, n_cls)
        n_rows = (n_cls + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle("Per-Class Feature Importance (Top 15)", fontsize=14, y=1.02)

        actual_top_k = min(top_k, n_feat)

        for ci, cls_id in enumerate(unique_classes):
            row, col = ci // n_cols, ci % n_cols
            ax = axes[row, col]

            cls_mask = y[:n_explain] == cls_id

            if sv_flat.ndim == 3:
                # sv_flat shape: (n_classes, N, D) 또는 (N, W, D)
                if sv_flat.shape[1] == n_explain:
                    # (n_classes, N, D) — 첫 차원이 클래스
                    cls_idx = min(int(cls_id), sv_flat.shape[0] - 1)
                    cls_samples = sv_flat[cls_idx, cls_mask]
                else:
                    # (N, W, D) — GradientExplainer에서 단일 출력
                    # 시간 축 평균 후 cls_mask 적용
                    sv_2d = sv_flat.mean(axis=1)  # (N, D)
                    cls_samples = sv_2d[cls_mask]
                cls_importance = cls_samples.mean(axis=0) if len(cls_samples) > 0 else np.zeros(n_feat)
            elif sv_flat.ndim == 2:
                cls_samples = sv_flat[cls_mask]
                cls_importance = cls_samples.mean(axis=0) if len(cls_samples) > 0 else np.zeros(n_feat)
            else:
                cls_importance = np.zeros(n_feat)

            sorted_idx = np.argsort(cls_importance)[::-1][:actual_top_k]
            top_names_cls = [names[i] for i in sorted_idx]
            top_values = cls_importance[sorted_idx]

            colors = plt.cm.Blues(np.linspace(0.3, 0.9, actual_top_k))
            ax.barh(range(actual_top_k)[::-1], top_values, color=colors)
            ax.set_yticks(range(actual_top_k)[::-1])
            ax.set_yticklabels(top_names_cls, fontsize=8)
            ax.set_xlabel("Mean |SHAP|", fontsize=9)
            ax.set_title(f"{LABELS[cls_id]}", fontsize=11, fontweight="bold")
            ax.grid(True, axis="x", alpha=0.3)

        # 남은 subplot 숨기기
        for ci in range(n_cls, n_rows * n_cols):
            row, col = ci // n_cols, ci % n_cols
            axes[row, col].set_visible(False)

        fig.tight_layout()
        path = output_dir / "per_class_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")

    def explain_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        output_dir: Path,
        n_samples: int = 10,
    ):
        """
        개별 샘플 예측 설명.

        Output: force_plots/sample_N.png
        """
        force_dir = output_dir / "force_plots"
        force_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [SHAP Samples] {n_samples}개 개별 설명 생성 중...")

        shap_values = self.compute_shap_values(X, n_samples)
        predictions = self.model.predict(X[:n_samples], verbose=0)

        if isinstance(shap_values, list):
            sv = np.array(shap_values)
        else:
            sv = shap_values
            if sv.ndim == len(X[:n_samples].shape):
                sv = np.expand_dims(sv, 0)

        # 시간 축 평균
        if sv.ndim == 4:
            sv_flat = sv.mean(axis=2)  # (n_classes, N, D)
        else:
            sv_flat = sv

        n_feat = sv_flat.shape[-1]
        names = feature_names[:n_feat] if len(feature_names) >= n_feat else \
            feature_names + [f"feat_{i}" for i in range(len(feature_names), n_feat)]

        for s in range(min(n_samples, len(X))):
            pred_cls = int(np.argmax(predictions[s]))
            true_cls = int(y[s])
            pred_prob = predictions[s, pred_cls]

            if sv_flat.ndim == 3:
                if sv_flat.shape[1] == min(n_samples, len(X)):
                    # (n_classes, N, D)
                    cls_idx = min(pred_cls, sv_flat.shape[0] - 1)
                    sample_sv = sv_flat[cls_idx, s]
                else:
                    # (N, W, D) → 시간 축 평균
                    sample_sv = sv_flat[s].mean(axis=0)
            elif sv_flat.ndim == 2:
                sample_sv = sv_flat[s]
            else:
                continue

            # Top 기여 피처 (양/음 포함)
            top_k = min(15, n_feat)
            abs_sv = np.abs(sample_sv)
            top_idx = np.argsort(abs_sv)[::-1][:top_k]

            fig, ax = plt.subplots(figsize=(10, 6))
            top_names = [names[i] for i in top_idx]
            top_values = sample_sv[top_idx]
            colors = ["#ff4444" if v > 0 else "#4444ff" for v in top_values]

            ax.barh(range(top_k)[::-1], top_values, color=colors)
            ax.set_yticks(range(top_k)[::-1])
            ax.set_yticklabels(top_names, fontsize=9)
            ax.set_xlabel("SHAP value", fontsize=11)
            ax.axvline(x=0, color="black", linewidth=0.8)

            result_str = "✓" if pred_cls == true_cls else "✗"
            ax.set_title(
                f"Sample #{s}  |  "
                f"True: {LABELS[true_cls]}  |  "
                f"Pred: {LABELS[pred_cls]} ({pred_prob:.1%})  {result_str}",
                fontsize=11,
            )
            ax.grid(True, axis="x", alpha=0.3)

            # 범례
            from matplotlib.patches import Patch
            ax.legend(
                handles=[Patch(color="#ff4444", label="Positive"),
                         Patch(color="#4444ff", label="Negative")],
                loc="lower right", fontsize=9,
            )

            fig.tight_layout()
            path = force_dir / f"sample_{s}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  [Saved] {force_dir}/ ({min(n_samples, len(X))}개)")


# ═══════════════════════════════════════════════════════════
# 3. 통합 분석기
# ═══════════════════════════════════════════════════════════
class XAIAnalyzer:
    """XAI 분석 통합 오케스트레이터."""

    def __init__(self, model_dir: str, data_folder: str, seed: int = 0):
        self.model_dir = Path(model_dir)
        self.data_folder = data_folder
        self.seed = seed

        # 모델/파이프라인 로드
        self.config = self._load_config()
        self.model = self._load_model()
        self.pipeline = PreprocessingPipeline.load(str(self.model_dir))

        # 테스트 데이터 준비
        self.X_test, self.y_test, self.feature_names = self._prepare_data()

    def _load_config(self) -> dict:
        configs = list(self.model_dir.glob("*__config.json"))
        if not configs:
            raise FileNotFoundError(f"Config 파일이 없습니다: {self.model_dir}")

        # cnn_attention 우선 선택
        for c in configs:
            if "cnn_attention" in c.name:
                with open(c) as f:
                    config = json.load(f)
                print(f"[Config] {c.name}")
                return config

        with open(configs[0]) as f:
            config = json.load(f)
        print(f"[Config] {configs[0].name}")
        return config

    def _load_model(self) -> tf.keras.Model:
        model_type = self.config.get("model_type", "cnn_attention")
        models = list(self.model_dir.glob("*__model.keras"))
        if not models:
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.model_dir}")

        # 커스텀 객체 등록 (WarmupCosineSchedule 등)
        try:
            from .main import WarmupCosineSchedule
        except Exception:
            pass

        # config와 일치하는 모델 찾기
        for m in models:
            if model_type in m.name:
                try:
                    model = tf.keras.models.load_model(m)
                except Exception:
                    # compile 없이 로드 (커스텀 Loss/Optimizer 직렬화 문제 회피)
                    model = tf.keras.models.load_model(m, compile=False)
                print(f"[Model] {m.name}")
                return model

        try:
            model = tf.keras.models.load_model(models[0])
        except Exception:
            model = tf.keras.models.load_model(models[0], compile=False)
        print(f"[Model] {models[0].name}")
        return model

    def _prepare_data(self):
        """테스트 데이터 로드 및 전처리."""
        print(f"\n[Data] {self.data_folder}")

        X_runs, y_runs, run_names, feature_names = load_Xy_runs(
            self.data_folder, include_time=False, verbose=False
        )

        # 동일한 분할 사용
        (train_X, train_y), _, (test_X, test_y) = split_runs(
            X_runs, y_runs, run_names,
            test_ratio=0.2,
            seed=self.config.get("seed", self.seed),
            use_val=False,
        )

        # 전처리 적용 (런 단위)
        feat = self.pipeline.feature_transformer
        feat_names_out = self.pipeline.feature_names_out

        test_X_feat, test_y_out, _ = feat.transform_runs(test_X, test_y)
        test_X_scaled = [self.pipeline.scaler.transform(X) for X in test_X_feat]

        # 윈도우 생성
        W = self.config.get("window_size", 3)
        stride = self.config.get("stride", 1)

        X_test, y_test = create_sliding_windows_from_runs(
            test_X_scaled, test_y_out, W, stride
        )

        print(f"  Test: {X_test.shape}, classes: {np.unique(y_test)}")
        return X_test, y_test, feat_names_out

    def run_attention_analysis(self, output_dir: Path):
        """Attention 분석 실행."""
        model_type = self.config.get("model_type", "")
        if model_type != "cnn_attention":
            print(f"\n[Skip] Attention 분석은 cnn_attention 모델에서만 가능 (현재: {model_type})")
            return

        print("\n" + "=" * 60)
        print("  ATTENTION WEIGHT 분석")
        print("=" * 60)

        attn_dir = output_dir / "attention"
        try:
            vis = AttentionVisualizer(self.model)
            vis.visualize_attention_heatmap(self.X_test, self.y_test, attn_dir)
            vis.visualize_per_head_patterns(self.X_test, self.y_test, attn_dir)
            vis.attention_by_class(self.X_test, self.y_test, attn_dir)
        except Exception as e:
            print(f"  [Error] Attention 분석 실패: {e}")
            import traceback
            traceback.print_exc()

    def run_shap_analysis(
        self, output_dir: Path, n_background: int = 100, n_explain: int = 500
    ):
        """SHAP 분석 실행."""
        print("\n" + "=" * 60)
        print("  SHAP 피처 중요도 분석")
        print("=" * 60)

        shap_dir = output_dir / "shap"
        n_bg = min(n_background, len(self.X_test))
        bg_data = self.X_test[:n_bg]

        try:
            analyzer = SHAPAnalyzer(self.model, bg_data)
            analyzer.explain_global(
                self.X_test, self.y_test, self.feature_names,
                shap_dir, top_k=20, n_samples=n_explain
            )
            analyzer.explain_per_class(
                self.X_test, self.y_test, self.feature_names,
                shap_dir, top_k=15, n_samples=n_explain
            )
            analyzer.explain_samples(
                self.X_test, self.y_test, self.feature_names,
                shap_dir, n_samples=10
            )
        except Exception as e:
            print(f"  [Error] SHAP 분석 실패: {e}")
            import traceback
            traceback.print_exc()

    def run_all(self, output_dir: str, n_background: int = 100, n_explain: int = 500):
        """전체 XAI 분석 실행."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print("\n" + "═" * 60)
        print("  NAS 원전 사고 진단 모델 - XAI 분석")
        print("═" * 60)
        print(f"  모델: {self.config.get('model_type')}")
        print(f"  피처: {self.config.get('feature_method')}")
        print(f"  윈도우: {self.config.get('window_size')}초")
        print(f"  테스트 샘플: {len(self.X_test)}")

        self.run_attention_analysis(out)
        self.run_shap_analysis(out, n_background, n_explain)

        print("\n" + "═" * 60)
        print(f"  [완료] 모든 XAI 결과 → {out}")
        print("═" * 60)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="XAI 분석 - 원전 사고 진단 모델")
    p.add_argument("--model_dir", type=str, default="models",
                    help="학습된 모델 디렉토리")
    p.add_argument("--data_folder", type=str, required=True,
                    help="테스트 데이터 폴더")
    p.add_argument("--output_dir", type=str, default="xai_results",
                    help="XAI 결과 저장 디렉토리")
    p.add_argument("--analysis_type", type=str, default="all",
                    choices=["all", "shap", "attention"],
                    help="분석 유형")
    p.add_argument("--n_background", type=int, default=100,
                    help="SHAP 배경 샘플 수")
    p.add_argument("--n_explain", type=int, default=500,
                    help="SHAP 설명 샘플 수")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    analyzer = XAIAnalyzer(
        model_dir=args.model_dir,
        data_folder=args.data_folder,
        seed=args.seed,
    )

    out = Path(args.output_dir)
    if args.analysis_type == "all":
        analyzer.run_all(str(out), args.n_background, args.n_explain)
    elif args.analysis_type == "shap":
        analyzer.run_shap_analysis(out, args.n_background, args.n_explain)
    elif args.analysis_type == "attention":
        analyzer.run_attention_analysis(out)


if __name__ == "__main__":
    main()
