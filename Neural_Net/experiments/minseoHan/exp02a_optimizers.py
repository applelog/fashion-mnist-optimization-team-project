# coding: utf-8
"""
[2A단계] 옵티마이저(매개변수 갱신 방법) 4종 비교 실험 — Fashion-MNIST
- 목적: 1단계에서 선정한 [512, 256, 128] 구조(ReLU + He)를 고정하고,
        교재 6.1절에서 배운 SGD / Momentum / AdaGrad / Adam의 수렴 속도와 검증 성능을 비교한다.
- 규칙 준수:
  * 6층 이하: [512, 256, 128] (총 4층)
  * 허용 옵티마이저 4종만 사용 (Nesterov, RMSprop 제외)
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

    # 2. 교재 6.1절 4가지 옵티마이저 설정 (다른 조건은 모두 동일하게 고정)
    configs = [
        ("Opt_SGD_lr0.01", "sgd", {"lr": 0.01}),
        ("Opt_Momentum_lr0.01", "momentum", {"lr": 0.01, "momentum": 0.9}),
        ("Opt_AdaGrad_lr0.01", "adagrad", {"lr": 0.01}),
        ("Opt_Adam_lr0.001", "adam", {"lr": 0.001}),
    ]

    results = []
    for name, opt_name, opt_param in configs:
        print(f"\n▶ 실행 중: {name} (optimizer={opt_name}, param={opt_param})")
        res = run_experiment(
            name=name,
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            hidden_size_list=[512, 256, 128],
            activation="relu",
            weight_init_std="he",
            weight_decay_lambda=0.0,
            use_dropout=False,
            use_batchnorm=False,
            optimizer=opt_name,
            optimizer_param=opt_param,
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
    csv_path = RESULTS_DIR / "minseoHan_02a_optimizers.csv"
    png_path = RESULTS_DIR / "minseoHan_02a_optimizers.png"
    save_results_csv(results, csv_path)
    plot_accuracy_curves(
        results,
        title="Stage 2A Optimizer Comparison ([512, 256, 128], ReLU+He, 20 epochs)",
        png_path=png_path,
    )

    # 4. 개발일지.md 기록용 요약 출력
    print("\n=== 개발일지.md 기록용 표 ===")
    for r in results:
        gap = r["train_acc"] - r["val_acc"]
        lr = r["optimizer_param"]["lr"]
        print(
            f"| 2026-11-26 | minseoHan | {r['name']} | "
            f"4층[512,256,128], relu, He, {r['optimizer']}(lr={lr}), seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | "
            f"best={r['best_val_acc']:.4f}, gap={gap:+.4f} |"
        )
    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
