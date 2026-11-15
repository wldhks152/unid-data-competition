"""처음부터 학습하는 시퀀스 모델 (사전학습 가중치·외부 데이터 없음, 엄격한 인과성).

기존 LightGBM/CatBoost 파이프라인(src/, train.py, test.py)은 건드리지 않는 별도 모듈이다.
  seq/prep.py    시퀀스 입력 만들기 (현재·과거 행만 사용), fold 분할(train.py와 동일), 스케일러
  seq/nets.py    인과 GRU · 인과 TCN (PyTorch)
  seq/rocket.py  인과 MiniRocket 스타일 랜덤 합성곱 피처 + 선형 분류기
  seq/synth.py   동작 확인용 합성 데이터
"""
