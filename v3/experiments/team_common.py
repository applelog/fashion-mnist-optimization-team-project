# coding: utf-8
"""
팀 공통 조건 + 공통 함수  (팀장만 수정)

s1_structure.py, s2_sgd.py, s2_momentum.py, s2_adagrad.py, s2_adam.py, s2_compare.py, final_save.py 가
모두 이 파일을 불러 쓴다 →  모든 실험이 같은 조건(같은 데이터 분할·구조·epoch·시도 횟수)으로 돌아가서
결과를 서로 공정하게 비교할 수 있다.

  - 데이터 분할: 교재 ch06/hyperparameter_optimization.py 방식 (shuffle_dataset 후 20%를 검증 데이터로)
  - test 데이터(t10k)는 여기서 읽지 않는다 → 성능평가는 final_save.py 에서 한 번만
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # v3 폴더 (common, data)

import csv
import numpy as np
import matplotlib.pyplot as plt
from data import mnist_reader
from common.util import shuffle_dataset


# ===== 공통 조건 (팀장만 수정) =====
HIDDEN = [100, 100, 100, 100]   # 은닉층 뉴런 수 → s1_structure.py 결과로 팀장이 정함 (지금은 과제 4번 구조)
ACTIVATION = 'relu'             # 활성화 함수 (교재 3장·5장)
WEIGHT_INIT = 'he'              # 가중치 초깃값 (교재 6.2: ReLU → He)
BATCH_SIZE = 128                # 미니배치 크기 (과제 4번과 같음)

VALIDATION_RATE = 0.20          # train 60,000장 중 20% = 12,000장을 검증 데이터로 (교재와 같은 비율)
SPLIT_SEED = 0                  # 누가 돌려도 같은 train/val 분할이 되도록 고정 (바꾸지 않음)

EPOCHS_MANUAL = 10              # 수동 탐색·구조 실험 epoch
EPOCHS_RANDOM = 5               # 무작위 탐색 epoch (교재 6.5.3: 한 번 학습을 짧게 해서 많이 시도)
RANDOM_TRIAL = 30               # 무작위 탐색 1라운드 시도 횟수 (모든 트랙 같게 = 같은 탐색 예산)
SEEDS = [0, 1, 2]               # 확인 단계: seed 3개 평균 ± 표준편차로 비교 (한 번의 우연을 막음)
EPOCHS_CONFIRM = 20             # 확인 단계 epoch
# ==================================

MAX_HIDDEN_LAYERS = 5           # 안내문 "6층 이하" → 은닉층 최대 5개 + 출력층 1개

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data', 'fashion')
RESULT_DIR = os.path.join(HERE, '..', 'results')


# 1. 데이터 ==========
def load_train_val(quick=False):
    """Fashion-MNIST train 60,000장을 읽어 train 48,000 / val 12,000 으로 나눈다 (test 는 읽지 않음).
    quick=True 이면 코드 확인용으로 train 1,000 / val 200 장만 쓴다 (교재 예제처럼 데이터를 줄임)."""
    x, t = mnist_reader.load_mnist(DATA_DIR, kind='train')
    x = x.astype(np.float32) / 255.0      # 0~255 → 0~1 (교재 load_mnist(normalize=True) 와 같음)

    np.random.seed(SPLIT_SEED)            # 분할 고정
    x, t = shuffle_dataset(x, t)
    validation_num = int(x.shape[0] * VALIDATION_RATE)
    x_val, t_val = x[:validation_num], t[:validation_num]
    x_train, t_train = x[validation_num:], t[validation_num:]

    if quick:
        x_train, t_train = x_train[:1000], t_train[:1000]
        x_val, t_val = x_val[:200], t_val[:200]
    return x_train, t_train, x_val, t_val


def check_layers(hidden_size_list):
    """안내문 조건: 6층 이하 (은닉층 + 출력층)"""
    if len(hidden_size_list) > MAX_HIDDEN_LAYERS:
        raise ValueError("6층 초과: 은닉층 %d개 + 출력층 1개 = %d층 (은닉층은 최대 %d개)"
                         % (len(hidden_size_list), len(hidden_size_list) + 1, MAX_HIDDEN_LAYERS))


# 2. 결과 기록 ==========
def result_row(optimizer, step, lr, weight_decay, seed, epochs, train_acc, val_acc, sec,
               val_curve, train_curve, hidden, optimizer_param=None, round_no=''):
    """실험 1회 결과를 csv 한 줄(dict)로 만든다. 모든 스크립트가 같은 열을 쓰게 하기 위함."""
    return {
        'optimizer': optimizer, 'step': step, 'round': round_no,
        'lr': lr, 'weight_decay': weight_decay,
        'optimizer_param': str(optimizer_param or {}), 'hidden': str(hidden),
        'seed': seed, 'epochs': epochs,
        'train_acc': round(float(train_acc), 4), 'val_acc': round(float(val_acc), 4),
        'gap': round(float(train_acc - val_acc), 4),   # train − val: 클수록 오버피팅
        'sec': round(sec, 1),
        'val_curve': ' '.join('%.4f' % v for v in val_curve),
        'train_curve': ' '.join('%.4f' % v for v in train_curve),
    }


def _result_path(filename, quick):
    os.makedirs(RESULT_DIR, exist_ok=True)
    if quick:
        filename = 'quick_' + filename   # 코드 확인용 결과는 따로 (git 에 안 올라감)
    return os.path.join(RESULT_DIR, filename)


def save_csv(filename, rows, quick=False):
    path = _result_path(filename, quick)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:   # utf-8-sig: 엑셀에서 열어도 안 깨짐
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("결과 저장:", path)


def read_csv(filename):
    """save_csv 로 저장한 결과를 다시 읽는다 (숫자 열은 float 로)."""
    path = os.path.join(RESULT_DIR, filename)
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ('lr', 'weight_decay', 'train_acc', 'val_acc', 'gap', 'sec'):
            r[key] = float(r[key])
    return rows


def curve(text):
    """csv 에 공백으로 저장한 정확도 곡선을 숫자 리스트로"""
    return [float(v) for v in text.split()]


# 3. 결과 정리·그래프 ==========
def save_fig(filename, quick=False):
    path = _result_path(filename, quick)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("그래프 저장:", path)


def plot_curves(curves, title, filename, quick=False):
    """curves: {이름: (val 곡선, train 곡선)}.  실선 = val, 점선 = train (교재 overfit 그래프 형식)"""
    plt.figure()
    for label, (val, train) in curves.items():
        x = np.arange(len(val))
        line, = plt.plot(x, val, label=str(label))
        plt.plot(x, train, '--', color=line.get_color())
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.title(title + "  (solid: val, dashed: train)")
    plt.legend(loc='lower right')
    save_fig(filename, quick)


def print_top(rows, k=5):
    """무작위 탐색 결과에서 val 정확도 상위 k개와 그 범위를 출력 → 다음 라운드 범위를 정하는 근거 (교재 6.5.3)"""
    top = sorted(rows, key=lambda r: r['val_acc'], reverse=True)[:k]
    print("=========== Hyper-Parameter Optimization Result ===========")
    for i, r in enumerate(top):
        print("Best-%d (val acc:%.4f) | lr:%.2e, weight decay:%.2e" % (i + 1, r['val_acc'], r['lr'], r['weight_decay']))
    lrs = [r['lr'] for r in top]
    wds = [r['weight_decay'] for r in top]
    print("상위 %d개 범위 → 다음 라운드 설정 예:  LR_RANGE = (%.2f, %.2f)   WD_RANGE = (%.2f, %.2f)"
          % (len(top), np.log10(min(lrs)), np.log10(max(lrs)), np.log10(min(wds)), np.log10(max(wds))))
    print("가장 좋은 값 → BEST_LR = %.3g, BEST_WD = %.3g" % (top[0]['lr'], top[0]['weight_decay']))


def plot_random(rows, title, filename, quick=False, graph_draw_num=20, col_num=5):
    """교재 hyperparameter_optimization.py 의 그래프 (상위 20개: 실선 val, 점선 train)
    + 학습률·가중치 감소별 val 정확도 산점도 (어느 범위가 좋은지 보기)"""
    top = sorted(rows, key=lambda r: r['val_acc'], reverse=True)[:graph_draw_num]
    row_num = int(np.ceil(graph_draw_num / col_num))
    plt.figure(figsize=(10, 8))
    for i, r in enumerate(top):
        plt.subplot(row_num, col_num, i + 1)
        plt.title("Best-" + str(i + 1))
        plt.ylim(0.0, 1.0)
        if i % 5: plt.yticks([])
        plt.xticks([])
        val, train = curve(r['val_curve']), curve(r['train_curve'])
        x = np.arange(len(val))
        plt.plot(x, val)
        plt.plot(x, train, "--")
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    save_fig(filename + '_top20.png', quick)

    plt.figure(figsize=(10, 4))
    for j, key in enumerate(['lr', 'weight_decay']):
        plt.subplot(1, 2, j + 1)
        plt.scatter([r[key] for r in rows], [r['val_acc'] for r in rows])
        plt.xscale('log')
        plt.xlabel(key)
        plt.ylabel("val acc")
        plt.grid(alpha=0.3)
    plt.suptitle(title)
    save_fig(filename + '_scatter.png', quick)
