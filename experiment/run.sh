#!/bin/bash

cd /root/github

source /root/anaconda3/etc/profile.d/conda.sh
conda acitvate venv

# 여가부 토론
python3 experiment/run/main.py 5 0 1 A B
python3 experiment/run/main.py 5 1 1 A B
python3 experiment/run/main.py 5 2 1 A B

# 퀴어 퍼레이드
python3 experiment/run/main.py 5 0 2 C D
python3 experiment/run/main.py 5 1 2 C D
python3 experiment/run/main.py 5 2 2 C D

# 모병제 찬반
python3 experiment/run/main.py 5 0 3 E F
python3 experiment/run/main.py 5 1 3 E F
python3 experiment/run/main.py 5 2 3 E F

# 난민수용정책
python3 experiment/run/main.py 5 0 4 G H
python3 experiment/run/main.py 5 1 4 G H
python3 experiment/run/main.py 5 2 4 G H