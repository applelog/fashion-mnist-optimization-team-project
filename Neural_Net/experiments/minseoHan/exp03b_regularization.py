# coding: utf-8
"""
[3B단계] 오버피팅 방지 기법(가중치 감소 L2 & 드롭아웃) 비교 실험 — Fashion-MNIST
- 목적: BatchNorm이 적용된 [512, 256, 128] 모델에서 발생하는 오버피팅(train-val 격차)을
        가중치 감소(weight_decay_lambda)와 드롭아웃(dropout_ration)으로 얼마나 억제하고
        검증 정확도(val acc)를 높일 수 있는지 비교한다 (교재 6.4절).
- 규칙 준수:
  * 6층 이하: [512, 256, 128] (총 4층)
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

    # 2. 가중치 감소(L2) 및 드롭아웃 조합 6가지 비교
    configs = [
        ("Reg_None", 0.0, False, 0.0),
        ("Reg_WD_1e-5", 1e-5, False, 0.0),
        ("Reg_WD_1e-4", 1e-4, False, 0.0),
        ("Reg_DO_0.2", 0.0, True, 0.2),
        ("Reg_DO_0.3", 0.0, True, 0.3),
        ("Reg_WD1e-5_DO0.2", 1e-5, True, 0.2),
    ]

    results = []
    for name, wd, use_do, do_ratio in configs:
        print(f"\n▶ 실행 중: {name} (wd={wd}, dropout={use_do}, ratio={do_ratio})")
        res = run_experiment(
            name=name,
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            hidden_size_list=[512, 256, 128],
            activation="relu",
            weight_init_std="he",
            weight_decay_lambda=wd,
            use_dropout=use_do,
            dropout_ratio=do_ratio,
            use_batchnorm=True,            # 3A단계 BatchNorm 적용 기반 위에서 정규화 비교
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
    csv_path = RESULTS_DIR / "minseoHan_03b_regularization.csv"
    png_path = RESULTS_DIR / "minseoHan_03b_regularization.png"
    save_results_csv(results, csv_path)
    plot_accuracy_curves(
        results,
        title="Stage 3B Regularization ([512, 256, 128], BN+ReLU+He+Adam)",
        png_path=png_path,
    )

    # 4. 개발일지.md 기록용 요약 출력
    print("\n=== 개발일지.md 기록용 표 ===")
    for r in results:
        gap = r["train_acc"] - r["val_acc"]
        do_str = f"DO={r['dropout_ratio']}" if r["use_dropout"] else "DO=False"
        print(
            f"| 2026-11-26 | minseoHan | {r['name']} | "
            f"4층[512,256,128], relu, He, BN, Adam(lr=0.001), WD={r['weight_decay_lambda']}, {do_str}, seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | "
            f"best={r['best_val_acc']:.4f}, gap={gap:+.4f} |"
        )
    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
