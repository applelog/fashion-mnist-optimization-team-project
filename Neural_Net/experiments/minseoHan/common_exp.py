# coding: utf-8
"""
minseoHan 실험 공통 유틸리티 모듈

규칙 준수 사항:
1. 교재 4·5·6장 기법만 사용 (SGD, Momentum, AdaGrad, Adam만 허용)
2. 6층 이하 신경망만 사용 (hidden_size_list 길이 <= 5)
3. 학습·검증에는 train 데이터(60,000장)만 분할하여 사용하고,
   test 데이터(t10k, 10,000장)는 최종 평가 전까지 절대 로드/사용하지 않음
"""

import csv
import sys
import time
from pathlib import Path

# 저장소 루트와 Neural_Net 폴더를 모듈 검색 경로 맨 앞에 추가
REPO = Path(__file__).resolve().parents[3]
NN = REPO / "Neural_Net"
if str(NN) not in sys.path:
    sys.path.insert(0, str(NN))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")  # GUI 창 없이 파일로만 그래프를 저장하기 위한 백엔드 설정
import matplotlib.pyplot as plt
import numpy as np

from data import mnist_reader
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from common.util import shuffle_dataset

# 팀플 규칙상 허용된 옵티마이저와 최대 은닉층 수 상수
ALLOWED_OPTIMIZERS = ("sgd", "momentum", "adagrad", "adam")
MAX_HIDDEN_LAYERS = 5
RESULTS_DIR = NN / "results"


def load_fashion_train_val(val_rate=0.20, seed=42):
    """Fashion-MNIST train 데이터를 불러와 train/val로 분할한다.

    주의: test 데이터(t10k)는 실험 및 하이퍼파라미터 튜닝 중 유출을 막기 위해
    이 함수에서 아예 읽지 않는다.

    Parameters
    ----------
    val_rate : float
        검증 데이터(validation) 비율 (기본 0.20 → train 48,000 / val 12,000)
    seed : int
        데이터 셔플 시드 고정값 (재현성 확보용)

    Returns
    -------
    (x_train, t_train), (x_val, t_val)
    """
    data_dir = NN / "data" / "fashion"
    # 훈련 데이터(60,000장)만 로드
    x_all, t_all = mnist_reader.load_mnist(str(data_dir), kind="train")

    # 3장 전처리: 0~255 픽셀값을 0.0~1.0 실수로 정규화
    x_all = x_all.astype(np.float32) / 255.0

    # 시드를 고정한 뒤 데이터를 뒤섞어 클래스 편향을 방지 (6.5.1 검증 데이터 분할)
    np.random.seed(seed)
    x_all, t_all = shuffle_dataset(x_all, t_all)

    # 앞부분 val_rate 만큼을 검증 데이터, 나머지를 훈련 데이터로 분할
    val_num = int(x_all.shape[0] * val_rate)
    x_val, t_val = x_all[:val_num], t_all[:val_num]
    x_train, t_train = x_all[val_num:], t_all[val_num:]

    return (x_train, t_train), (x_val, t_val)


def run_experiment(
    name,
    x_train,
    t_train,
    x_val,
    t_val,
    hidden_size_list,
    activation="relu",
    weight_init_std="he",
    weight_decay_lambda=0.0,
    use_dropout=False,
    dropout_ratio=0.5,
    use_batchnorm=False,
    optimizer="adam",
    optimizer_param=None,
    epochs=20,
    mini_batch_size=128,
    seed=42,
    verbose=False,
):
    """주어진 설정으로 MultiLayerNetExtend 모델을 학습하고 검증 성능을 반환한다."""
    # 규칙 1: 6층 이하 확인 (은닉층 수 + 출력층 1 <= 6 → 은닉층 최대 5개)
    if not (1 <= len(hidden_size_list) <= MAX_HIDDEN_LAYERS):
        raise ValueError(
            f"hidden_size_list 길이는 1~{MAX_HIDDEN_LAYERS}이어야 합니다 (현재: {len(hidden_size_list)})."
        )

    # 규칙 2: 교재 6.1에서 다룬 4가지 옵티마이저만 허용
    opt_key = optimizer.lower()
    if opt_key not in ALLOWED_OPTIMIZERS:
        raise ValueError(
            f"허용되지 않은 옵티마이저 '{optimizer}'. 허용 목록: {ALLOWED_OPTIMIZERS}"
        )

    # Trainer 내부에서 current_iter % iter_per_epoch == 0 으로 에폭을 체크하므로
    # train 크기가 mini_batch_size로 나누어떨어져야 에폭별 정확도가 정확히 기록됨
    if x_train.shape[0] % mini_batch_size != 0:
        raise ValueError(
            f"x_train 크기({x_train.shape[0]})가 mini_batch_size({mini_batch_size})로 나누어떨어져야 합니다."
        )

    if optimizer_param is None:
        optimizer_param = {"lr": 0.001 if opt_key == "adam" else 0.01}

    # 가중치 초기화 및 미니배치 샘플링 재현을 위해 시드 고정
    np.random.seed(seed)

    # 교재 MultiLayerNetExtend 생성 (인자명 오타 'dropout_ration' 그대로 전달)
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=hidden_size_list,
        output_size=10,
        activation=activation,
        weight_init_std=weight_init_std,
        weight_decay_lambda=weight_decay_lambda,
        use_dropout=use_dropout,
        dropout_ration=dropout_ratio,
        use_batchnorm=use_batchnorm,
    )

    # Trainer의 test 자리에는 반드시 검증 데이터(x_val, t_val)를 전달
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        optimizer=opt_key,
        optimizer_param=optimizer_param,
        verbose=verbose,
    )

    start_time = time.time()
    trainer.train()
    elapsed_sec = time.time() - start_time

    # Trainer는 각 에폭 시작 시점(iter 0, iter_per_epoch, ...)의 정확도만 기록하므로
    # 마지막 학습 스텝이 끝난 직후의 최종 train/val 정확도를 별도로 계산해 붙임
    final_train_acc = float(network.accuracy(x_train, t_train))
    final_val_acc = float(network.accuracy(x_val, t_val))

    train_acc_history = [float(v) for v in trainer.train_acc_list] + [final_train_acc]
    val_acc_history = [float(v) for v in trainer.test_acc_list] + [final_val_acc]
    best_val_acc = max(val_acc_history)

    return {
        "name": name,
        "hidden_size_list": list(hidden_size_list),
        "total_layers": len(hidden_size_list) + 1,
        "activation": activation,
        "weight_init_std": weight_init_std,
        "weight_decay_lambda": weight_decay_lambda,
        "use_dropout": use_dropout,
        "dropout_ratio": dropout_ratio if use_dropout else 0.0,
        "use_batchnorm": use_batchnorm,
        "optimizer": opt_key,
        "optimizer_param": dict(optimizer_param),
        "epochs": epochs,
        "mini_batch_size": mini_batch_size,
        "seed": seed,
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "elapsed_sec": elapsed_sec,
        "network": network,
    }


def save_results_csv(results, csv_path):
    """실험 결과 리스트를 CSV 파일로 저장한다."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

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
        "epochs",
        "mini_batch_size",
        "seed",
        "train_acc",
        "val_acc",
        "best_val_acc",
        "elapsed_sec",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "name": r["name"],
                    "total_layers": r["total_layers"],
                    "hidden_size_list": str(r["hidden_size_list"]),
                    "activation": r["activation"],
                    "weight_init_std": r["weight_init_std"],
                    "optimizer": r["optimizer"],
                    "lr": r["optimizer_param"].get("lr", ""),
                    "use_batchnorm": r["use_batchnorm"],
                    "weight_decay_lambda": r["weight_decay_lambda"],
                    "use_dropout": r["use_dropout"],
                    "dropout_ratio": r["dropout_ratio"],
                    "epochs": r["epochs"],
                    "mini_batch_size": r["mini_batch_size"],
                    "seed": r["seed"],
                    "train_acc": f"{r['train_acc']:.4f}",
                    "val_acc": f"{r['val_acc']:.4f}",
                    "best_val_acc": f"{r['best_val_acc']:.4f}",
                    "elapsed_sec": f"{r['elapsed_sec']:.2f}",
                }
            )


def plot_accuracy_curves(results, title, png_path):
    """에폭별 검증 정확도(실선)와 훈련 정확도(점선) 그래프를 저장한다."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, r in enumerate(results):
        color = colors[idx % len(colors)]
        epochs_axis = np.arange(len(r["val_acc_history"]))
        plt.plot(
            epochs_axis,
            r["val_acc_history"],
            color=color,
            linewidth=2.0,
            label=f"{r['name']} val ({r['val_acc']:.4f})",
        )
        plt.plot(
            epochs_axis,
            r["train_acc_history"],
            color=color,
            linestyle="--",
            alpha=0.65,
            label=f"{r['name']} train ({r['train_acc']:.4f})",
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
