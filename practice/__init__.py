"""
원전 사고 진단 AI 시스템 (practice 모듈)

모듈 구성:
    - main: 학습/평가 파이프라인
    - dataloader: 런 단위 데이터 로딩
    - preprocessing: 전처리 파이프라인 (학습=추론 일치)
    - model: 5가지 모델 아키텍처 (MLP, CNN, CNN+Attention, LSTM, Transformer)
    - feature_method: 피처 엔지니어링 (all, physics, diff, stats, selection, change)
    - data_split: 런 단위 데이터 분할 (누수 방지)
    - augmentation: 시계열 데이터 증강 (Jitter, Scaling, Mixup, MinorityOversampling)
    - xai: XAI 분석 (SHAP + Attention 시각화)
    - utils_plot: 시각화 유틸리티
"""
