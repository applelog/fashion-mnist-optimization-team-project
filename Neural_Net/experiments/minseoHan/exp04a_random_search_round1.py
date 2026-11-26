# coding: utf-8
"""
[4A단계] 1차 넓은 범위 Randomized Search — Fashion-MNIST
- 목적: 교재 6.5절(Randomized Grid Search) 절차에 따라
        구조([512, 256] / [512, 256, 128]), 옵티마이저(adam / adagrad),
        배치 정규화(True / False), 학습률(lr), 가중치 감소(weight_decay),
        드롭아웃 비율(dropout_ratio)을 넓은 범위에서 무작위로 20회 샘플링하여
        유망한 하이퍼파라미터 구간(Top-5)을 도출한다.
- 규칙 준수:
  * 6층 이하: 3층 [512, 256] / 4층 [512, 256, 128]
  * train(48,000) / val(12,000)만 사용, test 데이터 미사용
  * 효율적 탐색을 위해 1차 탐색 에폭은 10 epoch으로 설정 (교재 6.5.2 2단계)
"""

from collections import Counter
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

# 워커 프로세스마다 데이터를 한 번만 로드해 재사용하기 위한 전역 변수
_WORKER_DATA = None


def _init_worker():
    """각 병렬 워커 프로세스 시작 시 train/val 데이터를 1회 로드한다."""
    global _WORKER_DATA
    _WORKER_DATA = load_fashion_train_val(val_rate=0.20, seed=42)


def _run_single_trial(trial_cfg):
    """단일 Randomized Search 시행을 실행하고 (네트워크 객체 제외) 결과를 반환한다."""
    (x_train, t_train), (x_val, t_val) = _WORKER_DATA
    res = run_experiment(
        name=trial_cfg["name"],
        x_train=x_train,
        t_train=t_train,
        x_val=x_val,
        t_val=t_val,
        hidden_size_list=trial_cfg["hidden_size_list"],
        activation="relu",
        weight_init_std="he",
        weight_decay_lambda=trial_cfg["weight_decay_lambda"],
        use_dropout=True,
        dropout_ratio=trial_cfg["dropout_ratio"],
        use_batchnorm=trial_cfg["use_batchnorm"],
        optimizer=trial_cfg["optimizer"],
        optimizer_param={"lr": trial_cfg["lr"]},
        epochs=10,
        mini_batch_size=128,
        seed=42,
        verbose=False,
    )
    # 프로세스 간 직렬화 비용을 줄이기 위해 학습된 network 객체는 제거
    res.pop("network", None)
    return res


def sample_round1_trials(num_trials=20, sample_seed=2026):
    """1차 넓은 범위에서 하이퍼파라미터 조합 num_trials개를 무작위 추출한다."""
    np.random.seed(sample_seed)
    hidden_candidates = [[512, 256], [512, 256, 128]]
    opt_candidates = ["adam", "adagrad"]
    bn_candidates = [True, False]

    trials = []
    for i in range(1, num_trials + 1):
        hidden = hidden_candidates[int(np.random.randint(0, len(hidden_candidates)))]
        opt = opt_candidates[int(np.random.randint(0, len(opt_candidates)))]
        use_bn = bn_candidates[int(np.random.randint(0, len(bn_candidates)))]

        # 옵티마이저별 적정 로그 스케일 범위에서 학습률(lr) 샘플링
        if opt == "adam":
            lr = float(10 ** np.random.uniform(-3.7, -2.5))
        else:
            lr = float(10 ** np.random.uniform(-2.7, -1.5))

        # 교재 6.5.2절 권장 범위: 가중치 감소 계수 1e-8 ~ 1e-4
        weight_decay = float(10 ** np.random.uniform(-8, -4))
        # 드롭아웃 비율 0.05 ~ 0.45
        dropout_ratio = float(np.random.uniform(0.05, 0.45))

        trials.append(
            {
                "name": f"R1_T{i:02d}",
                "hidden_size_list": hidden,
                "optimizer": opt,
                "lr": lr,
                "use_batchnorm": use_bn,
                "weight_decay_lambda": weight_decay,
                "dropout_ratio": dropout_ratio,
            }
        )
    return trials


def plot_random_search_grid(sorted_results, title, png_path):
    """교재 그림 6-24 스타일로 Best-1 ~ Best-20 학습/검증 곡선을 4x5 격자로 저장한다."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    num_plots = min(20, len(sorted_results))
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx in range(rows * cols):
        ax = axes_flat[idx]
        if idx < num_plots:
            r = sorted_results[idx]
            x = np.arange(len(r["val_acc_history"]))
            ax.plot(x, r["val_acc_history"], linewidth=1.8, label="val")
            ax.plot(x, r["train_acc_history"], linestyle="--", linewidth=1.2, label="train")
            opt_short = "Adm" if r["optimizer"] == "adam" else "Ada"
            bn_short = "BN" if r["use_batchnorm"] else "NoBN"
            lr_val = r["optimizer_param"]["lr"]
            ax.set_title(
                f"Best-{idx + 1}: {r['val_acc']:.4f}\n"
                f"{opt_short} lr={lr_val:.1e} {bn_short}",
                fontsize=8,
            )
            ax.set_ylim(0.70, 1.00)
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
    # 1. 1차 Randomized Search 후보 20개 생성
    trials = sample_round1_trials(num_trials=20, sample_seed=2026)
    print(f"총 {len(trials)}개 조합에 대해 1차 Randomized Search 시작 (4 병렬 워커, 10 epochs)")

    # 2. 병렬 실행
    results = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init_worker) as executor:
        future_map = {executor.submit(_run_single_trial, t): t for t in trials}
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            lr_val = res["optimizer_param"]["lr"]
            print(
                f"  [{len(results):02d}/20] {res['name']} | "
                f"val={res['val_acc']:.4f} (best={res['best_val_acc']:.4f}, train={res['train_acc']:.4f}) | "
                f"{res['hidden_size_list']} | {res['optimizer']}(lr={lr_val:.2e}) | "
                f"BN={res['use_batchnorm']} | wd={res['weight_decay_lambda']:.2e} | "
                f"do={res['dropout_ratio']:.3f} ({res['elapsed_sec']:.1f}s)"
            )

    # 3. 최종 검증 정확도(val_acc) 기준 내림차순 정렬
    sorted_results = sorted(
        results,
        key=lambda r: (r["val_acc"], r["best_val_acc"]),
        reverse=True,
    )

    # 4. CSV 및 4x5 그래프 저장
    csv_path = RESULTS_DIR / "minseoHan_04a_random_search_r1.csv"
    png_path = RESULTS_DIR / "minseoHan_04a_random_search_r1.png"
    save_results_csv(sorted_results, csv_path)
    plot_random_search_grid(
        sorted_results,
        title="Stage 4A Randomized Search Round 1 (Best-1 ~ Best-20, 10 epochs)",
        png_path=png_path,
    )

    # 5. Best-1 ~ Best-10 요약 출력
    print("\n=== Round 1 Best-1 ~ Best-10 ===")
    for rank, r in enumerate(sorted_results[:10], start=1):
        lr_val = r["optimizer_param"]["lr"]
        print(
            f"Best-{rank:02d} ({r['name']}) | val={r['val_acc']:.4f} | best={r['best_val_acc']:.4f} | "
            f"train={r['train_acc']:.4f} | {r['hidden_size_list']} | {r['optimizer']}(lr={lr_val:.5f}) | "
            f"BN={r['use_batchnorm']} | wd={r['weight_decay_lambda']:.2e} | do={r['dropout_ratio']:.3f}"
        )

    # 6. Top-5 기반 2차 탐색 좁힌 범위 요약 (교재 6.5.3 방식)
    top5 = sorted_results[:5]
    top5_wd = [r["weight_decay_lambda"] for r in top5]
    top5_do = [r["dropout_ratio"] for r in top5]
    top5_hidden = Counter(str(r["hidden_size_list"]) for r in top5)
    top5_opt = Counter(r["optimizer"] for r in top5)
    top5_bn = Counter(r["use_batchnorm"] for r in top5)

    print("\n=== Top-5 기반 2차 탐색 범위 제안 ===")
    print(f"- hidden_size_list 분포: {dict(top5_hidden)}")
    print(f"- optimizer 분포:        {dict(top5_opt)}")
    print(f"- use_batchnorm 분포:    {dict(top5_bn)}")
    for opt_name in top5_opt:
        lrs = [r["optimizer_param"]["lr"] for r in top5 if r["optimizer"] == opt_name]
        print(f"- {opt_name} lr 범위:      {min(lrs):.5f} ~ {max(lrs):.5f} (log10: {np.log10(min(lrs)):.2f} ~ {np.log10(max(lrs)):.2f})")
    print(
        f"- weight_decay 범위:     {min(top5_wd):.2e} ~ {max(top5_wd):.2e} "
        f"(log10: {np.log10(min(top5_wd)):.2f} ~ {np.log10(max(top5_wd)):.2f})"
    )
    print(f"- dropout_ratio 범위:    {min(top5_do):.3f} ~ {max(top5_do):.3f}")

    # 7. 개발일지.md 기록용 Top-5 출력
    print("\n=== 개발일지.md 기록용 Top-5 표 ===")
    for rank, r in enumerate(top5, start=1):
        lr_val = r["optimizer_param"]["lr"]
        print(
            f"| 2026-11-26 | minseoHan | R1_Best{rank}({r['name']}) | "
            f"{r['total_layers']}층{r['hidden_size_list']}, relu, He, BN={r['use_batchnorm']}, "
            f"{r['optimizer']}(lr={lr_val:.5f}), wd={r['weight_decay_lambda']:.2e}, do={r['dropout_ratio']:.3f}, seed=42 | "
            f"{r['epochs']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | best={r['best_val_acc']:.4f} |"
        )

    print(f"\nCSV 저장: {csv_path}")
    print(f"PNG 저장: {png_path}")


if __name__ == "__main__":
    main()
