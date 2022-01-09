# [PyTorch] DQN(Deep Q-Network)을 이용하여 Tetris 학습

- 발표 영상 링크: [YouTube](https://youtu.be/RSsgcNyvkzc)

<br>

|Epoch  `84`|Epoch `1185`|Epoch `2042`|Epoch `3651`|
|-|-|-|-|
|총 제거된 행 `1`|총 제거된 행 `13`|총 제거된 행 `42`|총 제거된 행 `312`|
|최종 점수 `26`|최종 점수 `201`|최종 점수 `781`|최종 점수 `7515`|
|![image](Demo/1.gif)|![image](Demo/2.gif)|![image](Demo/3.gif)|![image](Demo/4.gif)|

<br>

## Details
### ✔ 테트리스에서의 State, Reward, Action 정의
![image](https://user-images.githubusercontent.com/42428487/148688331-8ffa4186-c359-427d-8129-deb72f90219c.png)

### ✔ DQN 알고리즘 (Q-function)
<img src="https://user-images.githubusercontent.com/42428487/148688790-87299f9f-0766-48a4-93b4-ecad71ded425.png" width="600">


<br>




## RUN CODE
### Requirements
```
imageio==2.13.5
importlib-metadata==4.8.2
matplotlib==3.5.0
numpy==1.18.1
opencv-python==4.5.4.60
Pillow==8.4.0
tensorboard==1.15.0
tensorboardX==2.4.1
tensorflow==1.15.5
torch==1.10.0
torchvision==0.11.1
tqdm==4.62.3
```

<br>

### Installing Packages

```shell
$ pip install -r requirements.txt
```

### Train & Save model
```shell
$ python3 train.py
```

### Test & Save the results in gif format
```shell
$ python3 test.py
```


<br>

---
###### 소스코드 참고: https://github.com/uvipen/Tetris-deep-Q-learning-pytorch
