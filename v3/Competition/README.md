# Competition 폴더 (13조)

교수님 안내: **`Neural_net\Competition` 폴더에서 실행되도록 코드 작성.**
교수님 코드는 `sys.path.append(os.pardir)` 로 **부모 폴더(`v3/`)의 `common`** 을 불러옵니다.
그래서 이 저장소의 `v3/` 폴더가 교수님 PC의 `Neural_net/` 역할입니다 (`v3/common` + `v3/Competition`).

## 들어갈 파일
| 파일 | 누가 | git |
|---|---|---|
| `Competition_Code_NeuralNet.py` | 교수님 배포 (accuracy 계산 코드) | ❌ 올리지 않음 (연락처 포함) → 게시판에서 받아 여기에 |
| `test_data.pkl` | 교수님 배포 (예시 데이터. 대회 당일엔 새 데이터) | ❌ 게시판에서 받아 여기에 |
| `fashion_mnist_test_sample_selection.py` | 교수님 배포 (test_data.pkl 만든 방법, 참고용) | ❌ |
| `network_Team13.pkl` | **우리 팀** 최종 모델 (`experiments/final_save.py` 가 만듦) | ❌ 이메일로 제출 |
| `check_Team13.py` | 우리 팀 확인용 (교수님 코드와 같은 방식, 파일명만 Team13) | ✅ |

## 교수님 데이터에서 확인한 것 (모의 Competition 파일)
- `test_data.pkl` = `{"x_test": (N, 784) uint8, "t_test": (N,) uint8}` → **정규화하지 않은 0~255 값**
- 그래서 우리 모델은 0~1 로 학습한 뒤, 저장 직전에 첫 층 가중치를 255로 나눠 **0~255 입력에서 같은 결과**가 나오게 한다
  (`(x/255)·W1 = x·(W1/255)`, `final_save.py` 가 변환 전후 결과가 같은지 확인)
- 교수님 예시 pkl 은 49MB → 층마다 마지막 입력을 들고 있어서. `final_save.py` 는 저장 전에 비워서 작게 만든다

## 확인 방법
```bash
cd v3/Competition
python check_Team13.py          # network_Team13.pkl → accuracy 출력
```

## 제출 전 체크
- [ ] `python check_Team13.py` 가 에러 없이 accuracy 출력 (0~255 입력 그대로)
- [ ] 파일명 `network_Team13.pkl`, 객체는 `MultiLayerNetExtend` (교재 `common` 클래스만)
- [ ] `common/` 을 고치지 않았음 (교수님 PC 는 원본 `common` 사용)
- [ ] 은닉층 5개 이하 (6층 이하)
- [ ] test 데이터로 학습·선택하지 않음 (설정은 val 로만 고름)
- [ ] 파일 크기 확인 (이메일 첨부 가능한 크기)
- [ ] **15주 수업 전 이메일 제출, 메일 제목 = 조 이름(13조)** (주소는 교수님 코드 주석 참고)
