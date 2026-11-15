"""학습 — 최종 model.joblib을 처음부터 다시 만든다 (운영진은 실행하지 않고 읽기만 함).

    python train.py --data_dir <battery_data_light 경로> [--out model.joblib] [--work_dir runs/stack_trv]

순서 (README '실행 방법'과 같은 명령)
  1. stack.py      lgb(LightGBM)·dino(DINOv2 + CatBoost) 멤버를 Train 실험 단위 5-fold + Train 전체로 학습.
                   Validation은 학습에 넣지 않고 홀드아웃 평가에만 사용
  2. train_seq.py  인과 GRU 멤버를 같은 fold로 학습
  3. tools/make_final.py  세 멤버를 0.4 : 0.4 : 0.2로 묶고 재확인 후처리(src/late.py, late)를 고정해 model.joblib 저장
피처 함수는 model.py에 모아 두었다. GPU가 있으면 GPU로 학습한다.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="battery_data_light 폴더")
    ap.add_argument("--work_dir", default="runs/stack_trv", help="멤버 모델·OOF 저장 폴더")
    ap.add_argument("--out", default="model.joblib", help="최종 가중치 파일 (test.py 기본 경로)")
    a = ap.parse_args()
    py, w = sys.executable, a.work_dir
    run([py, "stack.py", "--data_dir", a.data_dir, "--splits", "train", "--holdout_valid",
         "--members", "lgb", "dino", "--out_dir", w])
    run([py, "train_seq.py", "--data_dir", a.data_dir, "--model", "gru", "--out_dir", w,
         "--folds_json", os.path.join(w, "folds.json"), "--holdout_valid"])
    run([py, "tools/make_final.py",
         "--members", f"lgb={w}/lgb.joblib", f"dino={w}/dino.joblib", f"seq_gru={w}/seq_gru.joblib",
         "--weights", "0.4", "0.4", "0.2", "--post", "late", "--post_module", "src.late", "--out", a.out])
    print(f"\n완료: {a.out}  (추론: python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>)")


if __name__ == "__main__":
    main()
