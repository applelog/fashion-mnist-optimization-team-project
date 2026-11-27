# coding: utf-8
"""
[옵티마이저 트랙] SGD  (교재 6.1.2)          담당: __________

목표: SGD 로 낼 수 있는 가장 좋은 val 정확도와 그 설정(학습률·가중치 감소)을 찾는다.
      4개 트랙(SGD·Momentum·AdaGrad·Adam)이 같은 조건(team_common.py)·같은 탐색 횟수로 찾으므로
      마지막에 s2_compare.py 로 '각자 최선끼리' 공정하게 비교할 수 있다.

순서 (안내문: "하이퍼파라미터는 수동 탐색 후 ... Randomized Search 수행 필수", 교재 6.5)
  1) STEP = 'manual'  : MANUAL_LR 의 학습률 몇 개를 직접 비교 → 어느 범위가 좋은지 확인
  2) STEP = 'random'  : 그 범위에서 학습률·가중치 감소를 무작위로 뽑아 비교 (교재 hyperparameter_optimization.py)
                        → 출력된 '상위 5개 범위'로 LR_RANGE, WD_RANGE 를 좁히고 ROUND = 2 로 한 번 더
  3) STEP = 'confirm' : 찾은 최고값(BEST_LR, BEST_WD)을 seed 3개로 다시 학습 → 평균 ± 표준편차

규칙: 이 파일의 '설정'만 고친다 (team_common.py, common/ 은 고치지 않음).
      판단은 검증(val) 정확도로만 한다. test 데이터는 쓰지 않는다.
실행: v3/experiments 폴더에서  python s2_sgd.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # v3 폴더 (common, data)

import time
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
import team_common as S


# ===== 설정 (이 파일 담당자만 수정) =====
OPTIMIZER = 'sgd'                   # 이 파일의 옵티마이저 (바꾸지 않음)
OPTIMIZER_PARAM = {}                # 학습률 외 고유 값 (SGD 는 없음)

STEP = 'manual'                     # 'manual' → 'random' → 'confirm' 순서로 바꿔 가며 실행

# 1) 수동 탐색: 학습률 후보 (교재 기본값 0.01 주변)
MANUAL_LR = [0.1, 0.03, 0.01, 0.003, 0.001]

# 2) 무작위 탐색: lr = 10 ** uniform(LR_RANGE), weight decay = 10 ** uniform(WD_RANGE)  (교재 6.5.3)
ROUND = 1                           # 범위를 좁혀 다시 할 때 2 (이전 라운드 결과 파일은 남는다)
LR_RANGE = (-3, 0)                  # 10^-3 ~ 10^0  ← 1) 결과를 보고 고친다
WD_RANGE = (-8, -4)                 # 10^-8 ~ 10^-4 (교재 범위)
SAMPLE_SEED = 100                   # 무작위로 뽑는 값의 seed (같은 seed = 같은 후보 → 재현 가능)

# 3) 확인: 2) 에서 찾은 최고값
BEST_LR = 0.01
BEST_WD = 0

QUICK = False                       # True: 데이터 1,000장·1 epoch 로 코드만 확인 (결과는 quick_ 파일)
# =========================================


# 0. 데이터 (train / val 만, test 는 읽지 않음) ==========
x_train, t_train, x_val, t_val = S.load_train_val(QUICK)
S.check_layers(S.HIDDEN)


def __train(lr, weight_decay, epochs, seed=0):
    """교재 hyperparameter_optimization.py 의 __train 과 같은 구조.
    Trainer 의 test 자리에 검증 데이터를 넣는다 (test 데이터는 쓰지 않음)."""
    np.random.seed(seed)            # 가중치 초깃값·미니배치 순서 고정 → 같은 seed 면 같은 결과
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=S.HIDDEN, output_size=10,
                                  activation=S.ACTIVATION, weight_init_std=S.WEIGHT_INIT,
                                  weight_decay_lambda=weight_decay)
    optimizer_param = {'lr': lr}
    optimizer_param.update(OPTIMIZER_PARAM)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epochs, mini_batch_size=S.BATCH_SIZE,
                      optimizer=OPTIMIZER, optimizer_param=optimizer_param, verbose=False)
    start = time.time()
    trainer.train()
    sec = time.time() - start

    # Trainer 는 epoch 시작 때만 기록하므로, 학습이 끝난 직후 정확도를 따로 계산 (val 12,000장 전체)
    val_acc = network.accuracy(x_val, t_val)
    train_acc = network.accuracy(x_train, t_train)
    val_acc_list = trainer.test_acc_list + [val_acc]
    train_acc_list = trainer.train_acc_list + [train_acc]
    return val_acc_list, train_acc_list, val_acc, train_acc, sec


def __row(step, lr, weight_decay, seed, epochs, result, round_no=''):
    """결과 한 줄 (모든 트랙이 같은 열 → s2_compare.py 가 읽는다)"""
    val_acc_list, train_acc_list, val_acc, train_acc, sec = result
    return S.result_row(OPTIMIZER, step, lr, weight_decay, seed, epochs, train_acc, val_acc, sec,
                        val_acc_list, train_acc_list, S.HIDDEN, OPTIMIZER_PARAM, round_no)


if STEP == 'manual':
    # 1) 수동 탐색 ==========
    epochs = 1 if QUICK else S.EPOCHS_MANUAL
    rows, curves = [], {}
    for lr in MANUAL_LR:
        result = __train(lr, 0, epochs)
        print("lr:%g | val acc:%.4f, train acc:%.4f (%.0fs)" % (lr, result[2], result[3], result[4]))
        rows.append(__row('manual', lr, 0, 0, epochs, result))
        curves['lr=%g' % lr] = (result[0], result[1])
    S.save_csv('%s_manual.csv' % OPTIMIZER, rows, QUICK)
    best = max(rows, key=lambda r: r['val_acc'])
    print("→ 가장 좋은 학습률: %g (val acc %.4f). 이 주변으로 LR_RANGE 를 정하고 STEP = 'random'"
          % (best['lr'], best['val_acc']))
    S.plot_curves(curves, '%s manual search' % OPTIMIZER, '%s_manual.png' % OPTIMIZER, QUICK)

elif STEP == 'random':
    # 2) 무작위 탐색 (교재 hyperparameter_optimization.py) ==========
    optimization_trial = 2 if QUICK else S.RANDOM_TRIAL
    epochs = 1 if QUICK else S.EPOCHS_RANDOM

    # 탐색할 값을 먼저 모두 뽑아 둔다 (학습할 때 seed 를 고정하므로, 값 뽑기와 학습의 난수를 분리)
    np.random.seed(SAMPLE_SEED + ROUND)
    samples = []
    for _ in range(optimization_trial):
        # 탐색한 하이퍼파라미터의 범위 지정 (교재와 같은 방식) ===============
        weight_decay = 10 ** np.random.uniform(WD_RANGE[0], WD_RANGE[1])
        lr = 10 ** np.random.uniform(LR_RANGE[0], LR_RANGE[1])
        samples.append((lr, weight_decay))

    rows = []
    for i, (lr, weight_decay) in enumerate(samples):
        result = __train(lr, weight_decay, epochs)
        print("[%d/%d] val acc:%.4f | lr:%.2e, weight decay:%.2e (%.0fs)"
              % (i + 1, len(samples), result[2], lr, weight_decay, result[4]))
        rows.append(__row('random', lr, weight_decay, 0, epochs, result, ROUND))
    S.save_csv('%s_random_r%d.csv' % (OPTIMIZER, ROUND), rows, QUICK)
    S.print_top(rows)
    S.plot_random(rows, '%s random search round %d' % (OPTIMIZER, ROUND), '%s_random_r%d' % (OPTIMIZER, ROUND), QUICK)

elif STEP == 'confirm':
    # 3) 확인: 같은 설정을 seed 3개로 → 평균 ± 표준편차 ==========
    epochs = 1 if QUICK else S.EPOCHS_CONFIRM
    rows, curves = [], {}
    for seed in S.SEEDS:
        result = __train(BEST_LR, BEST_WD, epochs, seed)
        print("seed %d | val acc:%.4f, train acc:%.4f (%.0fs)" % (seed, result[2], result[3], result[4]))
        rows.append(__row('confirm', BEST_LR, BEST_WD, seed, epochs, result))
        curves['seed %d' % seed] = (result[0], result[1])
    S.save_csv('%s_confirm.csv' % OPTIMIZER, rows, QUICK)
    vals = [r['val_acc'] for r in rows]
    print("=========== %s 최종: lr %g, weight decay %g ===========" % (OPTIMIZER, BEST_LR, BEST_WD))
    print("val acc %.4f ± %.4f (seed %d개)  → 이 줄과 그래프를 PR 에 붙인다"
          % (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0, len(vals)))
    S.plot_curves(curves, '%s confirm (lr=%g, wd=%g)' % (OPTIMIZER, BEST_LR, BEST_WD),
                  '%s_confirm.png' % OPTIMIZER, QUICK)

else:
    raise ValueError("STEP 은 'manual', 'random', 'confirm' 중 하나")

plt.show()
