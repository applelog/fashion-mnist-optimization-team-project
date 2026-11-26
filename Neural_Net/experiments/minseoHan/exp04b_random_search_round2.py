# coding: utf-8
"""
[4B단계] 2차 좁혀진 범위 Randomized Search — Fashion-MNIST
- 목적: 1차 탐색(exp04a)의 Top-5 결과를 바탕으로 유망한 구간으로 범위를 좁혀
        최적의 하이퍼파라미터 조합을 정밀 탐색한다 (교재 6.5.3절).
  * 1차 Top-5 분석 기반 고정 설정:
    - 구조: 3층 [512, 256] (Top-5 중 5/5)
    - 활성화 함수: ReLU, 초깃값: He, 배치 정규화: True (Top-5 중 4/5), 드롭아웃: True
  * 좁혀진 탐색 구간 (16회 시행, 15 epoch):
    - AdaGrad (12회): lr = 10 ** uniform(-2.45, -1.70)  [0.00355 ~ 0.01995]
    - Adam    (4회):  lr = 10 ** uniform(-3.15, -2.55)  [0.00071 ~ 0.00282]
    - weight_decay_lambda = 10 ** uniform(-6.50, -4.10) [3.16e-07 ~ 7.94e-05]
    - dropout_ratio       = uniform(0.08, 0.32)
- 규칙 준수:
  * 6층 이하: 3층 [512, 256]
  * train(48,000) / val(12,000)만 사용, test 데이터 미사용
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

# 같은 폴더의 common_exp 모듈을 불러오기 위해 경로 추가 (Windows spawn 호환)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common_exp import (
    RESULTS_DIR,
    load_fashion_train_val,
    run_experiment,
    save_results_csv,
)

# 워커 프로세스마다 데이터를 1회만 로드하기 위한 전역 변수
_WORKER_DATA = None


def _init_worker():
    """각 병렬 워커 프로세스 시작 시 train/val 데이터를 1회 로드한다."""
    global _WORKER_DATA
    _WORKER_DATA = load_fashion_train_val(val_rate=0.20, seed=42)


def _run_single_trial(trial_cfg):
    """단일 2차 Randomized Search 시행을 실행하고 결과를 반환한다."""
    (x_train, t_train), (x_val, t_val) = _WORKER_DATA
    res = run_experiment(
        name=trial_cfg["name"],
        x_train=x_train,
        t_train=t_train,
        x_val=x_val,
        t_val=t_val,
        hidden_size_list=[512, 256],
        activation="relu",
        weight_init_std="he",
        weight_decay_lambda=trial_cfg["weight_decay_lambda"],
        use_dropout=True,
        dropout_ratio=trial_cfg["dropout_ratio"],
        use_batchnorm=True,
        optimizer=trial_cfg["optimizer"],
        optimizer_param={"lr": trial_cfg["lr"]},
        epochs=15,
        mini_batch_size=128,
        seed=42,
        verbose=False,
    )
    # IPC 직렬화 부담을 줄이기 위해 network 객체 제거
    res.pop("network", None)
    return res


def sample_round2_trials(adagrad_trials=12, adam_trials=4, sample_seed=2027):
    """1차 Top-5에서 도출한 좁은 범위 내에서 2차 하이퍼파라미터 조합 16개를 추출한다."""
    np.random.seed(sample_seed)
    trials = []
    total = adagrad_trials + adam_trials

    for i in range(1, total + 1):
        if i <= adagrad_trials:
            opt = "adagrad"
            lr = float(10 ** np.random.uniform(-2.45, -1.70))
        else:
            opt = "adam"
            lr = float(10 ** np.random.uniform(-3.15, -2.55))

        weight_decay = float(10 ** np.random.uniform(-6.50, -4.10))
        dropout_ratio = float(np.random.uniform(0.08, 0.32))

        trials.append(
            {
                "name": f"R2_T{i:02d}",
                "optimizer": opt,
                "lr": lr,
                "weight_decay_lambda": weight_decay,
                "dropout_ratio": dropout_ratio,
            }
        )
    return trials


def plot_round2_grid(sorted_results, title, png_path):
    """Best-1 ~ Best-16 학습/검증 정확도 곡선을 4x4 격자로 저장한다."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    num_plots = min(16, len(sorted_results))
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(13, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx in range(rows * cols):
        ax = axes_flat[idx]
        if idx < num_plots:
            r = sorted_results[idx]
            x = np.arange(len(r["val_acc_history"]))
            ax.plot(x, r["val_acc_history"], linewidth=1.8, label="val")
            ax.plot(x, r["train_acc_history"], linestyle="--", linewidth=1.2, label="train")
            opt_short = "Adm" if r["optimizer"] == "adam" else "Ada"
            lr_val = r["optimizer_param"]["lr"]
            ax.set_title(
                f"Best-{idx + 1}: {r['val_acc']:.4f}\n"
                f"{opt_short} lr={lr_val:.1e} do={r['dropout_ratio']:.2f}",
                fontsize=8,
            )
            ax.set_ylim(0.75, 1.00)
            ax.grid(True, linestyle=":", alpha=0.4)
            if idx == 0:
                ax.legend(loc="lower right", fontsize=7)
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.supxlabel("Epoch")
    fig.supylabel("Accuracy")
    plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.96])
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    # 1. 2차 좁혀진 범위 후보 16개 생성
    trials = sample_round2_trials(adagrad_trials=12, adam_trials=4, sample_seed=2027)
    print(f"총 {len(trials)}개 조합에 대해 2차 Randomized Search 시작 (4 병렬 워커, 15 epochs)")

    # 2. 병렬 실행
    results = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init_worker) as executor:
        future_map = {executor.submit(_run_single_trial, t): t for t in trials}
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            lr_val = res["optimizer_param"]["lr"]
            print(
                f"  [{len(results):02d}/16] {res['name']} | "
                f"val={res['val_acc']:.4f} (best={res['best_val_acc']:.4f}, train={res['train_acc']:.4f}) | "
                f"{res['optimizer']}(lr={lr_val:.5f}) | wd={res['weight_decay_lambda']:.2e} | "
                f"do={res['dropout_ratio']:.3f} ({res['elapsed_sec']:.1f}s)"
            )

    # 3. 최종 검증 정확도(val_acc) 기준 내림차순 정렬
    sorted_results = sorted(
        results,
        key=lambda r: (r["val_acc"], r["best_val_acc"]),
        reverse=True,
    )

    # 4. CSV 및 4x4 그래프 저장
    csv_path = RESULTS_DIR / "minseoHan_04b_random_search_r2.csv"
    png_path = RESULTS_DIR / "minseoHan_04b_random_search_r2.png"
    save_results_csv(sorted_results, csv_path)
    plot_round2_grid(
        sorted_results,
        title="Stage 4B Randomized Search Round 2 (Best-1 ~ Best-16, [512, 256]+BN, 15 epochs)",
        png_path=png_path,
    )

    # 5. Best-1 ~ Best-10 요약 출력
    print("\n=== Round 2 Best-1 ~ Best-10 ===")
    for rank, r in enumerate(sorted_results[:10], start=1):
        lr_val = r["optimizer_param"]["lr"]
        gap = r["train_acc"] - r["val_acc"]
        print(
            f"Best-{rank:02d} ({r['name']}) | val={r['val_acc']:.4f} | best={r['best_val_acc']:.4f} | "
            f"train={r['train_acc']:.4f} (gap={gap:+.4f}) | {r['optimizer']}(lr={lr_val:.5f}) | "
            f"wd={r['weight_decay_lambda']:.2e} | do={r['dropout_ratio']:.3f}"
        )

    # 6. 5단계 최종 모델 학습 후보(Best-1) 출력
    best = sorted_results[0]
    print("\n=== Best-1 최종 추천 설정 (5단계 후보) ===")
    print(
        f"name={best['name']}, hidden=[512, 256], relu, He, BN=True, "
        f"optimizer={best['optimizer']}, lr={best['optimizer_param']['lr']:.6f}, "
        f"weight_decay={best['weight_decay_lambda']:.3e}, dropout_ratio={best['dropout_ratio']:.4f}"
    )

    # 7. 개발일지.md 기록용 Top-5 표 출력
    print("\n=== 개발일지.md 기록용 Top-5 표 ===")
    for rank, r in enumerate(sorted_results[:5], start=1):
        lr_val = r["optimizer_param"]["lr"]
        print(
            f"| 2026-11-26 | minseoHan | R2_Best{rank}({r['name']}) | "
            f"3층[512, 256], relu, He, BN=True, {r['optimizer']}(lr={lr_val:.5f}), "
            f"wd={r['weight_decay_lambda']:.2e}, do={r['dropout_ratio']:.3f}, seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | best={r['best_val_acc']:.4f} |"
        )

    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
