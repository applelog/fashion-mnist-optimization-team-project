# coding: utf-8
"""
[3A단계] 배치 정규화(Batch Normalization) 유무 비교 실험 — Fashion-MNIST
- 목적: 1B단계 상위 2개 구조([512, 256], [512, 256, 128])에 대해
        BatchNorm 적용 전/후의 학습 속도 및 검증 정확도 변화를 비교한다 (교재 6.3절).
- 규칙 준수:
  * 6층 이하: 3층 [512, 256], 4층 [512, 256, 128]
  * train(48,000) / val(12,000)만 사용, test 데이터 미사용
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
    # 1. Fashion-MNIST train/val 분할 로드
    (x_train, t_train), (x_val, t_val) = load_fashion_train_val(val_rate=0.20, seed=42)
    print(f"train: {x_train.shape}, val: {x_val.shape}")

    # 2. 두 구조에서 BatchNorm Off vs On 비교
    configs = [
        ("BN_Off_512_256", [512, 256], False),
        ("BN_On_512_256", [512, 256], True),
        ("BN_Off_512_256_128", [512, 256, 128], False),
        ("BN_On_512_256_128", [512, 256, 128], True),
    ]

    results = []
    for name, hidden_list, use_bn in configs:
        print(f"\n▶ 실행 중: {name} (hidden={hidden_list}, use_batchnorm={use_bn})")
        res = run_experiment(
            name=name,
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            hidden_size_list=hidden_list,
            activation="relu",
            weight_init_std="he",
            weight_decay_lambda=0.0,
            use_dropout=False,
            use_batchnorm=use_bn,
            optimizer="adam",
            optimizer_param={"lr": 0.001},
            epochs=20,
            mini_batch_size=128,
            seed=42,
            verbose=False,
        )
        results.append(res)
        gap = res["train_acc"] - res["val_acc"]
        print(
            f"  완료 ({res['elapsed_sec']:.1f}s) | "
            f"train acc: {res['train_acc']:.4f} | val acc: {res['val_acc']:.4f} | "
            f"best val: {res['best_val_acc']:.4f} | gap: {gap:+.4f}"
        )

    # 3. 결과 CSV 및 그래프 저장
    csv_path = RESULTS_DIR / "minseoHan_03a_batchnorm.csv"
    png_path = RESULTS_DIR / "minseoHan_03a_batchnorm.png"
    save_results_csv(results, csv_path)
    plot_accuracy_curves(
        results,
        title="Stage 3A BatchNorm Effect (ReLU+He+Adam, 20 epochs)",
        png_path=png_path,
    )

    # 4. 개발일지.md 기록용 요약 출력
    print("\n=== 개발일지.md 기록용 표 ===")
    for r in results:
        gap = r["train_acc"] - r["val_acc"]
        bn_str = "BN=True" if r["use_batchnorm"] else "BN=False"
        print(
            f"| 2026-11-26 | minseoHan | {r['name']} | "
            f"{r['total_layers']}층{r['hidden_size_list']}, relu, He, Adam(lr=0.001), {bn_str}, seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | "
            f"best={r['best_val_acc']:.4f}, gap={gap:+.4f} |"
        )
    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
