# -*- coding: utf-8 -*-
from hmmfunc import MonoPhoneHMM
import numpy as np
import os

def load_feature_list(bin_dir, max_utts=None):
    """특징량 .bin 파일을 발화 ID로 매핑"""
    feat_list = {}
    files = [f for f in os.listdir(bin_dir) if f.endswith('.bin')]
    if max_utts:
        files = files[:max_utts]  # 발화 수 제한 적용
    for bin_file in files:
        utt = bin_file.split('.')[0]
        feat_list[utt] = os.path.join(bin_dir, bin_file)
    return feat_list

def load_label_list(label_file, max_utts=None):
    """발화 ID와 라벨 리스트를 매핑"""
    label_list = {}
    with open(label_file, mode='r') as f:
        lines = f.readlines()
        if max_utts:
            lines = lines[:max_utts]  # 발화 수 제한 적용
        for line in lines:
            utt = line.split()[0]
            lab = list(map(int, line.split()[1:]))
            label_list[utt] = lab
    return label_list

def compute_gaussian_probability(x, mean, var):
    """
    다차원 가우시안 확률 밀도 함수 계산.

    Args:
        x (np.ndarray): 입력 데이터 벡터.
        mean (np.ndarray): 가우시안의 평균 벡터.
        var (np.ndarray): 가우시안의 분산 벡터.

    Returns:
        float: 주어진 데이터 벡터의 확률.
    """
    dim = len(mean)
    cov_det = np.prod(var)  # 공분산 행렬의 행렬식
    cov_inv = 1.0 / var  # 공분산 행렬의 역행렬

    diff = x - mean
    exponent = -0.5 * np.sum(diff * diff * cov_inv)
    coefficient = 1.0 / np.sqrt((2 * np.pi) ** dim * cov_det)

    return coefficient * np.exp(exponent)

def calculate_validation_loss(hmm, feat_list, label_list):
    """검증 손실 계산"""
    total_log_likelihood = 0
    total_frames = 0

    for utt, feat_path in feat_list.items():
        features = np.fromfile(feat_path, dtype=np.float32)
        features = features.reshape(-1, hmm.num_dims)

        if utt not in label_list:
            continue
        labels = label_list[utt]

        log_likelihood = 0
        for frame in features:
            frame_likelihood = 0
            for label in labels:
                # 상태 정보 가져오기
                states = hmm.get_state(label)
                for state in states:
                    for mixture in state['mixtures']:
                        mean = np.array(mixture['mean'])
                        var = np.array(mixture['var'])
                        weight = mixture['weight']
                        
                        # 가우시안 확률 계산
                        prob = compute_gaussian_probability(frame, mean, var)
                        frame_likelihood += weight * prob

            # 로그 우도 계산
            log_likelihood += np.log(frame_likelihood + 1e-10)

        total_log_likelihood += log_likelihood
        total_frames += len(features)

    if total_frames > 0:
        return total_log_likelihood / total_frames
    else:
        return float('inf')  # 데이터가 없는 경우


if __name__ == "__main__":
    # 경로 설정
    base_hmm = '../model_output/hmm_3_state_1mix/0.hmm'
    train_bin_dir = '../features/train/'
    val_bin_dir = '../features/validation/'
    train_label_file = '../data/label/train/text_int'
    val_label_file = '../data/label/val/text_int'
    work_dir = './exp'
    num_iter = 5
    mixup_time = 2
    num_utters_train = 74063
    num_utters_val = 21151

    # 데이터 로드
    train_feat_list = load_feature_list(train_bin_dir, max_utts=num_utters_train)
    val_feat_list = load_feature_list(val_bin_dir, max_utts=num_utters_val)
    train_label_list = load_label_list(train_label_file, max_utts=num_utters_train)
    val_label_list = load_label_list(val_label_file, max_utts=num_utters_val)

    hmm = MonoPhoneHMM()
    hmm.load_hmm(base_hmm)

    num_states = hmm.num_states
    num_mixture = hmm.num_mixture

    # 출력 디렉토리 설정
    model_name = f'model_{num_states}state_{num_mixture}mix'
    out_dir = os.path.join(work_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    for m in range(mixup_time + 1):
        if m > 0:
            hmm.mixup()
            num_mixture *= 2
            model_name = f"model_{num_states}state_{num_mixture}mix"
            out_dir = os.path.join(work_dir, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_hmm = os.path.join(out_dir, "0.hmm")
            hmm.save_hmm(out_hmm)
            print(f"Mixture increased: {num_mixture // 2} -> {num_mixture}")
            print(f"Saved model: {out_hmm}")

        for iter in range(num_iter):
            print(f"{iter + 1}-th iteration for {num_mixture}-mixture model")
            hmm.train(train_feat_list, train_label_list)
            val_loss = calculate_validation_loss(hmm, val_feat_list, val_label_list)
            print(f"Validation loss after {iter + 1}-th iteration: {val_loss:.4f}")

            out_hmm = os.path.join(out_dir, f"{iter + 1}.hmm")
            hmm.save_hmm(out_hmm)
            print(f"Saved model: {out_hmm}")
