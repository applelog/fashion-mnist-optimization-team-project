# coding: utf-8
"""
[1A단계] 층 수(2~6층) 비교 실험 — Fashion-MNIST
- 목적: 은닉층 뉴런 수를 100으로 고정한 채 신경망 깊이(2~6층)만 바꿨을 때
        학습 정확도와 검증 정확도(일반화 성능)가 어떻게 달라지는지 확인한다.
- 규칙 준수:
  * 6층 이하: hidden_size_list 길이 1~5 (총 층 수 2~6층)
  * train(48,000) / val(12,000)만 사용, test 데이터 미사용
  * 시드(42), epoch(20), batch_size(128), ReLU + He + Adam(lr=0.001) 고정
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


def count_params(network):
    """네트워크 내 모든 학습 가능 매개변수(W, b 등)의 총 원소 수를 계산한다."""
    return sum(v.size for v in network.params.values())


def main():
    # 1. Fashion-MNIST train/val 분할 로드 (test 데이터는 사용하지 않음)
    (x_train, t_train), (x_val, t_val) = load_fashion_train_val(val_rate=0.20, seed=42)
    print(f"train: {x_train.shape}, val: {x_val.shape}")

    # 2. 비교할 층 수 설정 (은닉층 뉴런 수 100 고정, 총 2층~6층)
    configs = [
        ("L2_100", [100]),
        ("L3_100x2", [100, 100]),
        ("L4_100x3", [100, 100, 100]),
        ("L5_100x4", [100, 100, 100, 100]),
        ("L6_100x5", [100, 100, 100, 100, 100]),
    ]

    results = []
    for name, hidden_list in configs:
        print(f"\n▶ 실행 중: {name} (총 {len(hidden_list) + 1}층, hidden={hidden_list})")
        res = run_experiment(
            name=name,
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            hidden_size_list=hidden_list,
            activation="relu",
            weight_init_std="he",          # 6.2 ReLU 권장 He 초깃값
            weight_decay_lambda=0.0,       # 층 수 순수 비교를 위해 정규화 OFF
            use_dropout=False,
            use_batchnorm=False,
            optimizer="adam",
            optimizer_param={"lr": 0.001},
            epochs=20,
            mini_batch_size=128,
            seed=42,
            verbose=False,
        )
        res["num_params"] = count_params(res["network"])
        results.append(res)
        gap = res["train_acc"] - res["val_acc"]
        print(
            f"  완료 ({res['elapsed_sec']:.1f}s) | params: {res['num_params']:,} | "
            f"train acc: {res['train_acc']:.4f} | val acc: {res['val_acc']:.4f} | "
            f"best val: {res['best_val_acc']:.4f} | gap: {gap:+.4f}"
        )

    # 3. 결과 CSV 및 그래프 저장
    csv_path = RESULTS_DIR / "minseoHan_01a_layers.csv"
    png_path = RESULTS_DIR / "minseoHan_01a_layers.png"
    save_results_csv(results, csv_path)
    plot_accuracy_curves(
        results,
        title="Stage 1A Layer Depth Comparison (2~6 layers, width=100, ReLU+He+Adam)",
        png_path=png_path,
    )

    # 4. 개발일지.md 기록용 요약 출력
    print("\n=== 개발일지.md 기록용 표 ===")
    for r in results:
        gap = r["train_acc"] - r["val_acc"]
        print(
            f"| 2026-11-26 | minseoHan | {r['name']} | "
            f"{r['total_layers']}층{r['hidden_size_list']}, relu, He, Adam(lr=0.001), seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | "
            f"params={r['num_params']:,}, best={r['best_val_acc']:.4f}, gap={gap:+.4f} |"
        )
    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
