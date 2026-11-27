# coding: utf-8
"""
교수님 Competition_Code_NeuralNet.py 와 같은 방식으로 우리 pkl 을 확인한다 (파일명만 Team13).
교수님 PC(Neural_net/common + Neural_net/Competition)와 같은 구조: v3/common + v3/Competition (여기서 실행)

실행: v3/Competition 폴더에서
    python check_Team13.py                         ← network_Team13.pkl
    python check_Team13.py network_Team13_quick.pkl  ← 코드 확인용 pkl
(test_data.pkl 은 게시판에서 받아 이 폴더에 둔다)
"""
import os, sys
sys.path.append(os.pardir)  # 부모 디렉터리(v3)의 common 을 쓰기 위해 (교수님 코드와 같음)
import pickle

pkl_name = sys.argv[1] if len(sys.argv) > 1 else 'network_Team13.pkl'

# test_data.pkl 파일 불러오기 (교수님 코드와 같음)
with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)

x_test = data["x_test"]
t_test = data["t_test"]
print("x_test:", x_test.shape, x_test.dtype, "값 범위 %d ~ %d" % (x_test.min(), x_test.max()))
print("t_test:", t_test.shape)

# 우리 팀 network 파일 불러오기
with open(pkl_name, 'rb') as f:
    network = pickle.load(f)
print("모델: %s, 은닉층 %s → %d층, 파일 크기 %.1f MB"
      % (type(network).__name__, network.hidden_size_list, len(network.hidden_size_list) + 1,
         os.path.getsize(pkl_name) / 1e6))

# accuracy 출력 (교수님 코드와 같음)
accuracy = network.accuracy(x_test, t_test)
print("accuracy:", accuracy)
