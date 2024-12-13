# -*- coding: utf-8 -*-

# hmmfunc.py에서 MonoPhoneHMM 클래스를 임포트
from hmmfunc import MonoPhoneHMM

import numpy as np

import os

if __name__ == "__main__":

    # HMM 프로토타입
    hmmproto = '../model_output/hmm_3_state_1mix/hmmproto'

    # 평균 및 표준편차 파일 경로
    mean_file = '../features/train/global_mean.npy'
    std_file = '../features/train/global_std.npy'

    # 출력 디렉토리
    out_dir = os.path.dirname(hmmproto)

    # 출력 디렉토리가 없으면 생성
    os.makedirs(out_dir, exist_ok=True)

    # 평균 및 표준편차 읽기
    mean = np.load(mean_file)
    std = np.load(std_file)
    var = std ** 2

    # ########## float32에서 float64로 변환 ##########
    mean = mean.astype(np.float64)
    var = var.astype(np.float64)
    # ########## 변경 완료 ##########

    # MonoPhoneHMM 클래스를 호출
    hmm = MonoPhoneHMM()

    # HMM 프로토타입을 읽어들임
    hmm.load_hmm(hmmproto)

    # 플랫 스타트 초기화를 실행
    hmm.flat_init(mean, var)

    # HMM 프로토타입을 저장
    hmm.save_hmm(os.path.join(out_dir, '0.hmm'))
