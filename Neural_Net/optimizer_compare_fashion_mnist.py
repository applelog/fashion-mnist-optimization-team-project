# coding: utf-8
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 상위 폴더(신경망/)를 모듈 검색 경로에 추가 → common, dataset 패키지 사용 가능
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import mnist_reader
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD, Momentum, AdaGrad, Adam

# ── 0. Fashion-MNIST 데이터 로드 ──────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent / "data" / "fashion"

x_train, t_train = mnist_reader.load_mnist(str(DATA_PATH), kind='train')
x_test,  t_test  = mnist_reader.load_mnist(str(DATA_PATH), kind='t10k')

# 정규화 (0–255 → 0.0–1.0)
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32)  / 255.0

# 레이블을 one-hot 벡터로 변환
def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot

t_train = to_one_hot(t_train)
t_test  = to_one_hot(t_test)

print(f"훈련 데이터: {x_train.shape}, 레이블: {t_train.shape}")
print(f"테스트 데이터: {x_test.shape},  레이블: {t_test.shape}")

# ── 1. 실험 설정 ──────────────────────────────────────────────────────────────
train_size    = x_train.shape[0]
batch_size    = 128
max_iterations = 2000      # 그래프를 보며 적절한 에포크 수 조정 가능

optimizers = {
    "SGD":      SGD(lr=0.01),
    "Momentum": Momentum(lr=0.01),
    "AdaGrad":  AdaGrad(lr=0.01),
    "Adam":     Adam(lr=0.001),
}

networks   = {}
train_loss = {}

for key in optimizers:
    networks[key]   = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10
    )
    train_loss[key] = []

# ── 2. 학습 ───────────────────────────────────────────────────────────────────
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch    = x_train[batch_mask]
    t_batch    = t_train[batch_mask]

    for key in optimizers:
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print(f"========== iteration: {i} ==========")
        for key in optimizers:
            print(f"  {key}: {networks[key].loss(x_batch, t_batch):.4f}")

# ── 3. 그래프 출력 ────────────────────────────────────────────────────────────
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)

for key in optimizers:
    plt.plot(
        x,
        smooth_curve(train_loss[key]),
        marker=markers[key],
        markevery=100,
        label=key,
    )

plt.title("Optimizer Comparison — Fashion-MNIST")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

save_path = Path(__file__).resolve().parent / "optimizer_compare_fashion_mnist.png"
plt.savefig(save_path, dpi=150)
print(f"\n그래프 저장: {save_path}")
plt.show()
