# coding: utf-8
"""
[0단계] 베이스라인(Baseline) 실험 — Fashion-MNIST
- 목적: 6장 고급 기법(He/Xavier, Adam, BatchNorm, Weight Decay, Dropout) 없이
        기본 MLP + SGD만으로 학습했을 때의 기준 성능을 확인한다.
- 규칙 준수:
  * 6층 이하: hidden_size_list=[100, 100] (은닉 2개 → 총 3층)
  * train(48,000) / val(12,000)만 사용, test 데이터 미사용
  * 시드(42), epoch(20), batch_size(128) 고정
"""

import sys
from pathlib import Path

# 같은 폴더의 common_exp 모듈을 불러오기 위해 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_exp import (
    RESULTS_DIR,
    load_fashion_train_val,
    plot_accuracy_curves,
    run_experiment,
    save_results_csv,
)


def main():
    # 1. Fashion-MNIST train/val 분할 로드 (test 데이터는 사용하지 않음)
    (x_train, t_train), (x_val, t_val) = load_fashion_train_val(val_rate=0.20, seed=42)
    print(f"train: {x_train.shape}, val: {x_val.shape}")

    # 2. 6장 기법 없는 베이스라인 3가지 설정 정의
    configs = [
        {
            "name": "Baseline_Sigmoid_SGD_lr0.01",
            "activation": "sigmoid",
            "optimizer_param": {"lr": 0.01},
        },
        {
            "name": "Baseline_ReLU_SGD_lr0.01",
            "activation": "relu",
            "optimizer_param": {"lr": 0.01},
        },
        {
            "name": "Baseline_ReLU_SGD_lr0.1",
            "activation": "relu",
            "optimizer_param": {"lr": 0.1},
        },
    ]

    results = []
    for cfg in configs:
        print(f"\n▶ 실행 중: {cfg['name']}")
        res = run_experiment(
            name=cfg["name"],
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            hidden_size_list=[100, 100],   # 은닉 2층 → 3층 신경망
            activation=cfg["activation"],
            weight_init_std=0.01,          # 6장 초기화 기법 미적용 (단순 표준편차 0.01)
            weight_decay_lambda=0.0,       # 가중치 감소 미적용
            use_dropout=False,             # 드롭아웃 미적용
            use_batchnorm=False,           # 배치 정규화 미적용
            optimizer="sgd",               # 기본 SGD만 사용
            optimizer_param=cfg["optimizer_param"],
            epochs=20,
            mini_batch_size=128,
            seed=42,
            verbose=False,
        )
        results.append(res)
        print(
            f"  완료 ({res['elapsed_sec']:.1f}s) | "
            f"train acc: {res['train_acc']:.4f} | "
            f"val acc: {res['val_acc']:.4f} | "
            f"best val acc: {res['best_val_acc']:.4f}"
        )

    # 3. 결과 CSV 및 그래프 저장
    csv_path = RESULTS_DIR / "minseoHan_00_baseline.csv"
    png_path = RESULTS_DIR / "minseoHan_00_baseline.png"
    save_results_csv(results, csv_path)
    plot_accuracy_curves(
        results,
        title="Stage 0 Baseline (3-layer [100, 100], std=0.01, SGD, 20 epochs)",
        png_path=png_path,
    )

    # 4. 개발일지.md 기록용 요약 출력
    print("\n=== 요약 표 ===")
    for r in results:
        lr = r["optimizer_param"]["lr"]
        print(
            f"| 2026-11-26 | minseoHan | {r['name']} | "
            f"3층[100,100], {r['activation']}, std=0.01, SGD(lr={lr}), seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | "
            f"best val={r['best_val_acc']:.4f} |"
        )
    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
