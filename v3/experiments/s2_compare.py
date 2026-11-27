# coding: utf-8
"""
[옵티마이저 비교] 4개 트랙의 '각자 최선' 결과를 같은 기준으로 비교          담당: 팀장

각 트랙의 3) confirm 결과(results/<옵티마이저>_confirm.csv, seed 3개)를 읽어
val 정확도 평균 ± 표준편차로 순위를 매기고 판정한다.
  ★ 최고  : val 평균이 가장 높음
  비슷함  : 최고와의 차이 < max(0.5%p, 두 표준편차의 합) → 우연 범위 안. 더 단순하거나 빠른 쪽을 골라도 합리적
  낮음    : 그보다 차이가 큼
실행: v3/experiments 폴더에서  python s2_compare.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # v3 폴더 (common, data)

import numpy as np
import matplotlib.pyplot as plt
import team_common as S


# ===== 설정 (팀장) =====
OPTIMIZERS = ['sgd', 'momentum', 'adagrad', 'adam']
MIN_DIFF = 0.005                    # 0.5%p 보다 작은 차이는 '비슷함' (val 12,000장의 오차가 약 ±0.3%p)
QUICK = False                       # True: quick_ 결과 파일로 코드만 확인
# =======================


# 1. 각 트랙의 confirm 결과 읽기 ==========
summary = []
for optimizer in OPTIMIZERS:
    filename = ('quick_' if QUICK else '') + '%s_confirm.csv' % optimizer
    if not os.path.exists(os.path.join(S.RESULT_DIR, filename)):
        print("(아직 없음) %s → 담당자가 STEP = 'confirm' 까지 실행하고 PR 해야 함" % filename)
        continue
    rows = S.read_csv(filename)
    vals = np.array([r['val_acc'] for r in rows])
    summary.append({
        'optimizer': optimizer,
        'val_mean': vals.mean(),
        'val_std': vals.std(ddof=1) if len(vals) > 1 else 0.0,
        'train_mean': np.mean([r['train_acc'] for r in rows]),
        'gap_mean': np.mean([r['gap'] for r in rows]),
        'sec_mean': np.mean([r['sec'] for r in rows]),
        'lr': rows[0]['lr'], 'weight_decay': rows[0]['weight_decay'],
        'optimizer_param': rows[0]['optimizer_param'], 'hidden': rows[0]['hidden'],
        'seeds': len(rows), 'epochs': rows[0]['epochs'],
        'val_curve': np.mean([S.curve(r['val_curve']) for r in rows], axis=0),
    })
if not summary:
    sys.exit("비교할 결과가 없습니다.")

# 공정한 비교인지 확인: 구조·epoch 가 모두 같아야 한다
if len(set((s['hidden'], s['epochs']) for s in summary)) > 1:
    print("⚠ 트랙마다 구조나 epoch 가 다릅니다 → 공정한 비교가 아님. team_common.py 조건을 맞춰 다시 confirm")


# 2. 순위와 판정 ==========
summary.sort(key=lambda s: s['val_mean'], reverse=True)
best = summary[0]
for s in summary:
    if s is best:
        s['verdict'] = '★ 최고'
    else:
        diff = best['val_mean'] - s['val_mean']
        s['verdict'] = '비슷함' if diff < max(MIN_DIFF, best['val_std'] + s['val_std']) else '낮음'

print("=========== Optimizer Comparison (각자 최선끼리, seed %d개 평균) ===========" % best['seeds'])
print("| 순위 | 옵티마이저 | val 정확도 | train | gap | 시간(s) | 학습률 | 가중치 감소 | 판정 |")
print("|---|---|---|---|---|---|---|---|---|")
for i, s in enumerate(summary):
    print("| %d | %s | %.4f ± %.4f | %.4f | %.4f | %.0f | %.3g | %.2e | %s |"
          % (i + 1, s['optimizer'], s['val_mean'], s['val_std'], s['train_mean'], s['gap_mean'],
             s['sec_mean'], s['lr'], s['weight_decay'], s['verdict']))
print("→ 제안: %s (lr %.3g, weight decay %.2e). '비슷함'이 있으면 더 빠르거나 gap 이 작은 쪽도 근거와 함께 고려"
      % (best['optimizer'], best['lr'], best['weight_decay']))

S.save_csv('optimizer_compare.csv',
           [{'rank': i + 1, 'optimizer': s['optimizer'], 'verdict': s['verdict'],
             'val_mean': round(s['val_mean'], 4), 'val_std': round(s['val_std'], 4),
             'train_mean': round(s['train_mean'], 4), 'gap_mean': round(s['gap_mean'], 4),
             'sec_mean': round(s['sec_mean'], 1), 'lr': s['lr'], 'weight_decay': s['weight_decay'],
             'optimizer_param': s['optimizer_param'], 'hidden': s['hidden'], 'seeds': s['seeds']}
            for i, s in enumerate(summary)], QUICK)


# 3. 그래프: 막대(평균 ± 표준편차) + 평균 학습 곡선 ==========
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
names = [s['optimizer'] for s in summary]
plt.bar(names, [s['val_mean'] for s in summary], yerr=[s['val_std'] for s in summary], capsize=5)
low = min(s['val_mean'] for s in summary)
plt.ylim(max(0.0, low - 0.05), 1.0)
plt.ylabel("val acc (mean ± std)")
plt.title("best of each optimizer")

plt.subplot(1, 2, 2)
for s in summary:
    plt.plot(np.arange(len(s['val_curve'])), s['val_curve'], label=s['optimizer'])
plt.xlabel("epochs")
plt.ylabel("val acc (mean of seeds)")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("learning curve")
S.save_fig('optimizer_compare.png', QUICK)
plt.show()
