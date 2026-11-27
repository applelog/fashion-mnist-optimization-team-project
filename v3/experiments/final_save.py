# coding: utf-8
"""
[최종] 최선의 모델 하나를 학습해서 network_Team13.pkl 로 저장          담당: 팀장
(안내문: 팀별로 가장 좋은 성능의 MultiLayerNet / MultiLayerNetExtend 객체 하나를 pkl 로 저장해 제출)

순서
  1) 비교 실험에서 val 평균이 가장 높았던 설정을 아래 '설정'에 적는다 (test 점수로 고르지 않는다)
  2) 그 설정으로 학습 (Trainer 의 test 자리에 검증 데이터)
  3) 성능평가: test(t10k) 정확도는 여기서 한 번만 잰다
  4) 교수님 입력(0~255 원본)에 맞춤: (x/255)·W1 = x·(W1/255) 이므로 W1 을 255 로 나눈다 (결과는 수학적으로 같음)
  5) 용량 줄이기: 층마다 남은 마지막 입력(1만 장)을 1장짜리로 바꾼다 (교수님 예시 pkl 이 49MB 인 이유)
  6) ../Competition/network_Team13.pkl 저장 → Competition 폴더에서  python check_Team13.py  로 확인
실행: v3/experiments 폴더에서  python final_save.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # v3 폴더 (common, data)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from data import mnist_reader
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
import team_common as S


# ===== 설정: 실험으로 정한 최선의 값 (팀장) =====
TEAM_NO = 13

HIDDEN = S.HIDDEN                   # 구조 (s1_structure.py 결과)
ACTIVATION = 'relu'
WEIGHT_INIT = 'he'                  # ② 가중치 초깃값
OPTIMIZER = 'sgd'                   # ① 옵티마이저 비교(s2_compare.py) 결과
LR = 0.01
OPTIMIZER_PARAM = {}                # 예: Momentum {'momentum': 0.9}, Adam {'beta1': 0.9, 'beta2': 0.999}
WEIGHT_DECAY = 0                    # ④ 가중치 감소
USE_BATCHNORM = False               # ③ 배치 정규화
USE_DROPOUT = False                 # ④ 드롭아웃
DROPOUT_RATIO = 0.2
BATCH_SIZE = S.BATCH_SIZE
EPOCHS = 20                         # confirm 곡선에서 val 이 가장 높았던 epoch 근처 (교재 Trainer 는 마지막 epoch 가중치를 남김)
SEED = 0
USE_VAL_FOR_TRAIN = False           # True: 설정 확정 후 train+val 60,000장 전부로 학습 (이때 val 점수는 볼 수 없음)

QUICK = False                       # True: 데이터 1,000장·1 epoch 로 코드만 확인 (network_Team13_quick.pkl 로 저장)
# ===============================================


# 0. 데이터 (학습은 train / val 만) ==========
S.check_layers(HIDDEN)
x_train, t_train, x_val, t_val = S.load_train_val(QUICK)
if USE_VAL_FOR_TRAIN:
    x_train = np.concatenate([x_train, x_val])
    t_train = np.concatenate([t_train, t_val])
epochs = 1 if QUICK else EPOCHS


# 1. 학습 ==========
np.random.seed(SEED)
network = MultiLayerNetExtend(input_size=784, hidden_size_list=HIDDEN, output_size=10,
                              activation=ACTIVATION, weight_init_std=WEIGHT_INIT,
                              weight_decay_lambda=WEIGHT_DECAY, use_dropout=USE_DROPOUT,
                              dropout_ration=DROPOUT_RATIO, use_batchnorm=USE_BATCHNORM)
optimizer_param = {'lr': LR}
optimizer_param.update(OPTIMIZER_PARAM)
trainer = Trainer(network, x_train, t_train, x_val, t_val,
                  epochs=epochs, mini_batch_size=BATCH_SIZE,
                  optimizer=OPTIMIZER, optimizer_param=optimizer_param, verbose=False)
print("학습 중... (%d층 %s, %s lr %g, %d epoch)" % (len(HIDDEN) + 1, HIDDEN, OPTIMIZER, LR, epochs))
trainer.train()

train_acc = network.accuracy(x_train, t_train)
val_acc = None if USE_VAL_FOR_TRAIN else network.accuracy(x_val, t_val)
print("train acc: %.4f" % train_acc + ("" if val_acc is None else "   val acc: %.4f" % val_acc))


# 2. 성능평가: test 는 여기서 한 번만 ==========
x_test, t_test = mnist_reader.load_mnist(S.DATA_DIR, kind='t10k')   # 0~255 원본 (교수님 test_data.pkl 과 같은 형식)
if QUICK:
    x_test, t_test = x_test[:200], t_test[:200]
x_test_01 = x_test / 255.0          # float64 로 나눠야 아래 변환 전후 비교가 반올림 오차 없이 정확함
score_before = network.predict(x_test_01)
test_acc = network.accuracy(x_test_01, t_test)


# 3. 교수님 입력(0~255)에 맞게 변환 ==========
network.params['W1'] /= 255.0       # 제자리 나눗셈 → 같은 배열을 쓰는 Affine1 층도 함께 바뀐다
assert network.layers['Affine1'].W is network.params['W1']
score_after = network.predict(x_test)
rel_err = np.abs(score_before - score_after).max() / np.abs(score_before).max()
assert rel_err < 1e-6 and (score_before.argmax(axis=1) == score_after.argmax(axis=1)).all(), \
    "변환 전후 결과가 다릅니다 (상대 오차 %.1e)" % rel_err
test_acc_raw = network.accuracy(x_test, t_test)
print("test acc: %.4f (0~1 입력)  →  변환 후 %.4f (0~255 입력, 교수님 방식)" % (test_acc, test_acc_raw))


# 4. 용량 줄이기 ==========
network.predict(x_test[:1])         # 층마다 남아 있는 마지막 입력을 1장짜리로 바꾼다


# 5. 저장 (네트워크 객체 그대로) ==========
pkl_name = 'network_Team%d%s.pkl' % (TEAM_NO, '_quick' if QUICK else '')
pkl_path = os.path.join(S.HERE, '..', 'Competition', pkl_name)
with open(pkl_path, 'wb') as f:
    pickle.dump(network, f, protocol=4)   # protocol 4: 파이썬 3.4 이상이면 어디서나 열림
pkl_mb = os.path.getsize(pkl_path) / 1e6
print("저장: %s (%.1f MB)" % (os.path.abspath(pkl_path), pkl_mb))
print("확인: Competition 폴더에서  python check_Team13.py" + (" " + pkl_name if QUICK else ""))

S.save_csv('final.csv', [{
    'team_no': TEAM_NO, 'hidden': str(HIDDEN), 'activation': ACTIVATION, 'weight_init': WEIGHT_INIT,
    'optimizer': OPTIMIZER, 'lr': LR, 'optimizer_param': str(OPTIMIZER_PARAM), 'weight_decay': WEIGHT_DECAY,
    'use_batchnorm': USE_BATCHNORM, 'use_dropout': USE_DROPOUT, 'dropout_ratio': DROPOUT_RATIO,
    'batch_size': BATCH_SIZE, 'epochs': epochs, 'seed': SEED, 'use_val_for_train': USE_VAL_FOR_TRAIN,
    'train_acc': round(float(train_acc), 4), 'val_acc': '' if val_acc is None else round(float(val_acc), 4),
    'test_acc': round(float(test_acc_raw), 4), 'pkl': pkl_name, 'pkl_mb': round(pkl_mb, 2),
}], QUICK)


# 6. 학습 곡선 ==========
if not USE_VAL_FOR_TRAIN:
    S.plot_curves({'final': (trainer.test_acc_list + [val_acc], trainer.train_acc_list + [train_acc])},
                  'final model (%s)' % OPTIMIZER, 'final.png', QUICK)
    plt.show()
