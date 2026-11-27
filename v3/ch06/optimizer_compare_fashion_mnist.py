# coding: utf-8
# [과제 4번] 교재 ch06/optimizer_compare_mnist.py 를 Fashion-MNIST 용으로 수정
# 실행: v3/ch06 폴더에서  python optimizer_compare_fashion_mnist.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # v3 폴더를 import 경로에 추가

import numpy as np
import matplotlib.pyplot as plt
from data import mnist_reader
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD, Momentum, AdaGrad, Adam


# ===== 실험 옵션 (여기만 수정) =====
batch_size = 128
max_iterations = 2000          # 참고: 1 epoch = 60000 / batch_size ≈ 469 iterations
hidden_size_list = [100, 100, 100, 100]   # 은닉층 구조 (팀플 안내: 6층 이하)
weight_decay_lambda = 0        # 가중치 감소 (0이면 사용 안 함)
learning_rates = {'SGD': 0.01, 'Momentum': 0.01, 'AdaGrad': 0.01, 'Adam': 0.001}  # 교재 기본값
print_interval = 100           # loss 출력 간격 + 그래프 마커 간격
loss_ylim = (0, 1)             # 그래프 y축 범위
# ====================================


# 0. Fashion-MNIST 데이터 읽기==========
# README 힌트 방식: '../data/fashion/' (실행 위치와 상관없이 이 파일 기준 경로 사용)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fashion')
x_train, t_train = mnist_reader.load_mnist(data_dir, kind='train')
x_test, t_test = mnist_reader.load_mnist(data_dir, kind='t10k')  # 테스트 데이터는 학습에 사용하지 않음

# mnist_reader는 0~255 값을 반환 -> 원본 load_mnist(normalize=True)와 같게 0~1로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

train_size = x_train.shape[0]


# 1. 실험용 설정==========
optimizers = {}
optimizers['SGD'] = SGD(lr=learning_rates['SGD'])
optimizers['Momentum'] = Momentum(lr=learning_rates['Momentum'])
optimizers['AdaGrad'] = AdaGrad(lr=learning_rates['AdaGrad'])
optimizers['Adam'] = Adam(lr=learning_rates['Adam'])

# 옵티마이저마다 같은 구조의 신경망을 하나씩 생성
networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=hidden_size_list,
        output_size=10, weight_decay_lambda=weight_decay_lambda)
    train_loss[key] = []


# 2. 훈련 시작==========
for i in range(max_iterations):
    # 미니배치 무작위 추출
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 같은 미니배치로 각 옵티마이저의 신경망을 갱신하고 loss 기록
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % print_interval == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=print_interval, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(*loss_ylim)
plt.legend()
plt.show()
