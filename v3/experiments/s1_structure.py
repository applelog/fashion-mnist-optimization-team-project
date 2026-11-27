# coding: utf-8
"""
[구조 결정] 층 수 → 층별 뉴런 수          담당: 팀장
(안내문: "모델(구조) 결정을 위한 실험 필요", "6층 이하" / 수업 슬라이드 "모델 결정: 층 수, 층별 노드 수")

  1) STEP = 'depth' : 은닉층 1~5개 (= 2~6층), 뉴런 100 고정
  2) STEP = 'width' : 1) 에서 정한 층 수로 뉴런 64 / 128 / 256 / 512 + 점점 줄어드는 모양

한 옵티마이저에 유리한 구조를 고르지 않도록 4개 옵티마이저(교재 기본 학습률 = 과제 4번 설정)로 각각 학습하고,
'4개 평균 val 정확도'가 가장 높은 구조를 고른다 → team_common.py 의 HIDDEN 에 적는다.
실행: v3/experiments 폴더에서  python s1_structure.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # v3 폴더 (common, data)

import time
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
import team_common as S


# ===== 설정 (팀장) =====
STEP = 'depth'                      # 'depth' → 'width'

OPTIMIZERS = {'sgd': 0.01, 'momentum': 0.01, 'adagrad': 0.01, 'adam': 0.001}   # 교재 기본 학습률 (과제 4번과 같음)

DEPTH_CANDIDATES = [[100] * n for n in range(1, 6)]          # 은닉층 1~5개 = 2~6층

WIDTH_LAYERS = 4                    # 1) depth 결과로 정한 은닉층 수
WIDTH_CANDIDATES = [[w] * WIDTH_LAYERS for w in (64, 128, 256, 512)] \
                   + [[512 // 2 ** i for i in range(WIDTH_LAYERS)]]    # 마지막: 512 → 256 → 128 … 점점 줄어드는 모양

SEED = 0
QUICK = False                       # True: 데이터 1,000장·1 epoch 로 코드만 확인 (결과는 quick_ 파일)
# =======================


# 0. 데이터 (train / val 만, test 는 읽지 않음) ==========
x_train, t_train, x_val, t_val = S.load_train_val(QUICK)
candidates = DEPTH_CANDIDATES if STEP == 'depth' else WIDTH_CANDIDATES
for hidden in candidates:
    S.check_layers(hidden)
epochs = 1 if QUICK else S.EPOCHS_MANUAL


def __train(hidden, optimizer, lr):
    """구조(hidden) 하나를 옵티마이저 하나로 학습. Trainer 의 test 자리에 검증 데이터를 넣는다."""
    np.random.seed(SEED)            # 가중치 초깃값·미니배치 순서 고정
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=hidden, output_size=10,
                                  activation=S.ACTIVATION, weight_init_std=S.WEIGHT_INIT)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epochs, mini_batch_size=S.BATCH_SIZE,
                      optimizer=optimizer, optimizer_param={'lr': lr}, verbose=False)
    start = time.time()
    trainer.train()
    sec = time.time() - start

    # Trainer 는 epoch 시작 때만 기록하므로, 학습이 끝난 직후 정확도를 따로 계산 (val 12,000장 전체)
    val_acc = network.accuracy(x_val, t_val)
    train_acc = network.accuracy(x_train, t_train)
    return trainer.test_acc_list + [val_acc], trainer.train_acc_list + [train_acc], val_acc, train_acc, sec


# 1. 구조 × 옵티마이저 전부 학습 ==========
rows = []
total, k = len(candidates) * len(OPTIMIZERS), 0
for hidden in candidates:
    for optimizer, lr in OPTIMIZERS.items():
        k += 1
        val_list, train_list, val_acc, train_acc, sec = __train(hidden, optimizer, lr)
        print("[%d/%d] %d층 %s | %s lr:%g | val acc:%.4f, train acc:%.4f (%.0fs)"
              % (k, total, len(hidden) + 1, hidden, optimizer, lr, val_acc, train_acc, sec))
        rows.append(S.result_row(optimizer, 'structure_' + STEP, lr, 0, SEED, epochs, train_acc, val_acc, sec,
                                 val_list, train_list, hidden))
S.save_csv('structure_%s.csv' % STEP, rows, QUICK)


# 2. 결과 정리: 구조별 4개 옵티마이저 평균 ==========
print("=========== Structure Result (%s) ===========" % STEP)
mean_val = []
for hidden in candidates:
    same = [r for r in rows if r['hidden'] == str(hidden)]
    mean_val.append(np.mean([r['val_acc'] for r in same]))
    print("%d층 %-26s | 평균 val acc:%.4f | %s" % (len(hidden) + 1, hidden, mean_val[-1],
          '  '.join('%s %.4f' % (r['optimizer'], r['val_acc']) for r in same)))
best = int(np.argmax(mean_val))
print("→ 4개 옵티마이저 평균 val 정확도가 가장 높은 구조: %s (%d층)" % (candidates[best], len(candidates[best]) + 1))
print("  차이가 0.5%p 안쪽이면 더 작은(빠른) 구조를 골라도 합리적이다. 정한 값은 team_common.py 의 HIDDEN 에 적는다.")


# 3. 그래프: 구조별 val 정확도 (옵티마이저별 선 + 평균) ==========
markers = {'sgd': 'o', 'momentum': 'x', 'adagrad': 's', 'adam': 'D'}
x = np.arange(len(candidates))
plt.figure(figsize=(9, 4))
for optimizer in OPTIMIZERS:
    y = [r['val_acc'] for r in rows if r['optimizer'] == optimizer]
    plt.plot(x, y, marker=markers.get(optimizer, 'o'), label=optimizer)
plt.plot(x, mean_val, 'k-', linewidth=3, alpha=0.4, label='mean')
plt.xticks(x, ['%d layers\n%s' % (len(h) + 1, h) for h in candidates], fontsize=7)
plt.ylabel("val acc")
plt.title("structure: %s (%d epochs)" % (STEP, epochs))
plt.legend()
plt.grid(alpha=0.3)
S.save_fig('structure_%s.png' % STEP, QUICK)
plt.show()
