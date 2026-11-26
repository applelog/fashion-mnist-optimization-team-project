# coding: utf-8
"""
[5단계] 최종 모델 확정, x_test 딱 1회 최종 평가 및 .pkl 저장 — Fashion-MNIST
- 목적: 4B단계(exp04b)에서 선정된 Best-1 설정(R2_T01)으로 최대 30 epoch까지 학습하며
        검증 정확도(val acc)가 가장 높은 최적 epoch을 확정한 뒤,
        확정된 최종 모델 1개에 대해 test 데이터(10,000장)로 딱 한 번 최종 평가하고
        Competition 제출 형식과 동일한 MultiLayerNetExtend 객체(.pkl)를 저장한다.
- 규칙 준수:
  * 6층 이하: 3층 [512, 256] (hidden_size_list 길이 = 2)
  * 교재 4·5·6장 기법만 사용: ReLU + He + BatchNorm + AdaGrad + L2(weight_decay) + Dropout
  * x_test는 모든 모델·epoch 결정이 끝난 마지막 단계에서만 딱 1회 사용
"""

import csv
import pickle
import sys
from pathlib import Path

# 같은 폴더의 common_exp 모듈을 불러오기 위해 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common_exp import (
    NN,
    RESULTS_DIR,
    load_fashion_train_val,
    run_experiment,
)
from data import mnist_reader


def strip_inference_caches(network):
    """저장 용량 절감(100MB 초과 방지)을 위해 순전파/역전파 임시 캐시만 비운다.

    추론에 필요한 가중치(W, b), 배치 정규화 통계량(gamma, beta, running_mean, running_var),
    드롭아웃 비율(dropout_ratio)은 그대로 보존된다.
    """
    cache_attrs = (
        "x",
        "original_x_shape",
        "dW",
        "db",
        "input_shape",
        "batch_size",
        "xc",
        "xn",
        "std",
        "dgamma",
        "dbeta",
        "mask",
        "out",
    )
    for layer in network.layers.values():
        for attr in cache_attrs:
            if hasattr(layer, attr):
                setattr(layer, attr, None)

    if hasattr(network, "last_layer"):
        for attr in ("loss", "y", "t"):
            if hasattr(network.last_layer, attr):
                setattr(network.last_layer, attr, None)


def main():
    # 1. Fashion-MNIST train(48,000) / val(12,000) 분할 로드 (이 시점에서는 test 미사용)
    (x_train, t_train), (x_val, t_val) = load_fashion_train_val(val_rate=0.20, seed=42)
    print(f"train: {x_train.shape}, val: {x_val.shape}")

    # 2. 4B단계(Round 2) Best-1 하이퍼파라미터 설정
    best_cfg = {
        "hidden_size_list": [512, 256],
        "activation": "relu",
        "weight_init_std": "he",
        "use_batchnorm": True,
        "use_dropout": True,
        "dropout_ratio": 0.2674,
        "weight_decay_lambda": 4.672e-05,
        "optimizer": "adagrad",
        "optimizer_param": {"lr": 0.014164},
        "mini_batch_size": 128,
        "seed": 42,
    }

    # 3. Step A: 30 epoch 학습을 통해 검증 정확도 기준 최적 epoch 탐색
    print("\n▶ [Step A] 30 epoch 탐색으로 최적 epoch 확정 중 (val 기준)...")
    res_30 = run_experiment(
        name="Final_R2_T01_30ep",
        x_train=x_train,
        t_train=t_train,
        x_val=x_val,
        t_val=t_val,
        epochs=30,
        verbose=False,
        **best_cfg,
    )

    # val_acc_history[0]은 학습 시작 전, [1:]은 1..30 epoch 완료 후 정확도
    val_per_epoch = res_30["val_acc_history"][1:]
    best_epoch = int(np.argmax(val_per_epoch)) + 1
    best_val_at_30 = float(val_per_epoch[best_epoch - 1])
    print(
        f"  30 epoch 완료 ({res_30['elapsed_sec']:.1f}s) | "
        f"30ep val={res_30['val_acc']:.4f} | "
        f"최적 epoch={best_epoch} (val={best_val_at_30:.4f})"
    )

    # 최적 epoch이 30이 아니면, 동일 시드로 딱 best_epoch만큼만 재학습해 최종 모델 확정
    if best_epoch == 30:
        final_res = res_30
    else:
        print(f"\n▶ [Step A-2] 확정된 최적 epoch={best_epoch}으로 최종 모델 학습 중...")
        final_res = run_experiment(
            name=f"Final_R2_T01_{best_epoch}ep",
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            epochs=best_epoch,
            verbose=False,
            **best_cfg,
        )
        print(
            f"  최종 학습 완료 ({final_res['elapsed_sec']:.1f}s) | "
            f"train acc={final_res['train_acc']:.4f} | val acc={final_res['val_acc']:.4f}"
        )

    final_network = final_res["network"]

    # 4. Step B: 최종 확정된 모델 1개에 대해 test 데이터(10,000장)로 딱 1회 최종 평가
    print("\n▶ [Step B] test 데이터(t10k, 10,000장) 최종 1회 평가 중...")
    data_dir = NN / "data" / "fashion"
    x_test, t_test = mnist_reader.load_mnist(str(data_dir), kind="t10k")
    x_test = x_test.astype(np.float32) / 255.0
    final_test_acc = float(final_network.accuracy(x_test, t_test))
    print(f"  ★ 최종 Test Accuracy (10,000장): {final_test_acc:.4f} ({final_test_acc * 100:.2f}%)")

    # 5. Step C: 임시 캐시를 비운 순수 MultiLayerNetExtend 객체를 .pkl 파일로 저장
    strip_inference_caches(final_network)
    exp_dir = Path(__file__).resolve().parent
    pkl_path = exp_dir / "minseoHan_final_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(final_network, f)
    pkl_size_mb = pkl_path.stat().st_size / (1024 * 1024)

    # 최종 수치 CSV 저장
    csv_path = RESULTS_DIR / "minseoHan_05_final_model.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "name",
            "total_layers",
            "hidden_size_list",
            "activation",
            "weight_init_std",
            "optimizer",
            "lr",
            "use_batchnorm",
            "weight_decay_lambda",
            "use_dropout",
            "dropout_ratio",
            "best_epoch",
            "mini_batch_size",
            "seed",
            "train_acc",
            "val_acc",
            "test_acc",
            "pkl_path",
            "pkl_size_mb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "name": final_res["name"],
                "total_layers": final_res["total_layers"],
                "hidden_size_list": str(final_res["hidden_size_list"]),
                "activation": final_res["activation"],
                "weight_init_std": final_res["weight_init_std"],
                "optimizer": final_res["optimizer"],
                "lr": best_cfg["optimizer_param"]["lr"],
                "use_batchnorm": final_res["use_batchnorm"],
                "weight_decay_lambda": final_res["weight_decay_lambda"],
                "use_dropout": final_res["use_dropout"],
                "dropout_ratio": final_res["dropout_ratio"],
                "best_epoch": best_epoch,
                "mini_batch_size": best_cfg["mini_batch_size"],
                "seed": best_cfg["seed"],
                "train_acc": f"{final_res['train_acc']:.4f}",
                "val_acc": f"{final_res['val_acc']:.4f}",
                "test_acc": f"{final_test_acc:.4f}",
                "pkl_path": str(pkl_path),
                "pkl_size_mb": f"{pkl_size_mb:.2f}",
            }
        )

    # 최종 학습 곡선 그래프 저장 (30 epoch 기준 추이 + best_epoch 수직선 + final test_acc 표시)
    png_path = RESULTS_DIR / "minseoHan_05_final_model.png"
    epochs_axis = np.arange(len(res_30["val_acc_history"]))
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        epochs_axis,
        res_30["val_acc_history"],
        linewidth=2.2,
        label=f"val acc (best@{best_epoch}: {final_res['val_acc']:.4f})",
    )
    plt.plot(
        epochs_axis,
        res_30["train_acc_history"],
        linestyle="--",
        linewidth=1.6,
        alpha=0.8,
        label=f"train acc (@{best_epoch}: {final_res['train_acc']:.4f})",
    )
    plt.axvline(
        best_epoch,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"selected epoch = {best_epoch}",
    )
    plt.axhline(
        final_test_acc,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"final test acc (1-shot): {final_test_acc:.4f}",
    )
    plt.title("Stage 5 Final Model ([512, 256], BN+ReLU+He+AdaGrad+L2+Dropout)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.75, 1.0)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # 6. 개발일지.md 기록용 요약 출력
    print("\n=== 개발일지.md 기록용 최종 행 ===")
    print(
        f"| 2026-11-26 | minseoHan | {final_res['name']} | "
        f"3층[512, 256], relu, He, BN=True, adagrad(lr=0.014164), wd=4.672e-05, do=0.2674, seed=42 | "
        f"{best_epoch} | {final_res['train_acc']:.4f} | {final_res['val_acc']:.4f} | "
        f"최종 1회 test acc={final_test_acc:.4f} |"
    )
    print(f"\nPKL 저장: {pkl_path} ({pkl_size_mb:.2f} MB)")
    print(f"CSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
