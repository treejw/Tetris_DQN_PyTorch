# [PyTorch] Tetris with DQN(Deep Q-Network)

- Presentation video: [YouTube](https://youtu.be/RSsgcNyvkzc)

<br>

## Results

<p align="center">

|Epoch|  `84`| `1185`| `2042`| `3651`|
|:-:|:-:|:-:|:-:|:-:|
|**total number of rows removed**| 1| 13 | 42 | 312 |
|**final score**| 26 | 201 | 781 | 7515 |
||![image](Demo/1.gif)|![image](Demo/2.gif)|![image](Demo/3.gif)|![image](Demo/4.gif)|

  
</p>
<br>

## Details
### ✔ Definition of State, Reward, Action in Tetris
![image](https://user-images.githubusercontent.com/42428487/148688331-8ffa4186-c359-427d-8129-deb72f90219c.png)

### ✔ DQN algorithm (Q-function)
<img src="https://user-images.githubusercontent.com/42428487/148688790-87299f9f-0766-48a4-93b4-ecad71ded425.png" width="600">


<br>




## How to run codes
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
