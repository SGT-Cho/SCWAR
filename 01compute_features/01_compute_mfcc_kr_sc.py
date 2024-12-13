# -*- coding: utf-8 -*-

# 
# MFCC 특징을 계산합니다.
# 
import wave

import numpy as np

import os
import sys

class FeatureExtractor():
    '''특징량(FBANK, MFCC)을 추출하는 클래스'''
    def __init__(self, 
                 sample_frequency=16000, 
                 frame_length=25, 
                 frame_shift=10, 
                 num_mel_bins=23, 
                 num_ceps=13, 
                 lifter_coef=22, 
                 low_frequency=20, 
                 high_frequency=8000, 
                 dither=1.0):
        self.sample_freq = sample_frequency
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lifter_coef = lifter_coef
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.dither_coef = dither

        # FFT 포인트 수 = 창폭 이상의 2제곱
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # 멜 필터뱅크를 생성
        self.mel_filter_bank = self.MakeMelFilterBank()

        # 이산 코사인 변환(DCT) 기저 행렬을 생성
        self.dct_matrix = self.MakeDCTMatrix()

        # 리프터(lifter) 생성
        self.lifter = self.MakeLifter()


    def Herz2Mel(self, herz):
        ''' 주파수를 헤르츠에서 Mel로 변환 '''
        return (1127.0 * np.log(1.0 + herz / 700))


    def MakeMelFilterBank(self):
        ''' Mel 필터 뱅크 생성 '''
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        mel_points = np.linspace(mel_low_freq, 
                                 mel_high_freq, 
                                 self.num_mel_bins+2)

        dim_spectrum = int(self.fft_size / 2) + 1
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            left_mel = mel_points[m]
            center_mel = mel_points[m+1]
            right_mel = mel_points[m+2]
            for n in range(dim_spectrum):
                freq = 1.0 * n * self.sample_freq/2 / dim_spectrum
                mel = self.Herz2Mel(freq)
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel-mel) / (right_mel-center_mel)
                    mel_filter_bank[m][n] = weight
         
        return mel_filter_bank

    
    def ExtractWindow(self, waveform, start_index, num_samples):
        '''
        1프레임 분량의 파형 데이터를 추출하여 전처리하고,
        로그 파워값을 계산
        '''
        window = waveform[start_index:start_index + self.frame_size].copy()
        if self.dither_coef > 0:
            window = window \
                     + np.random.rand(self.frame_size) \
                     * (2*self.dither_coef) - self.dither_coef

        window = window - np.mean(window)
        power = np.sum(window ** 2)
        if power < 1E-10:
            power = 1E-10
        log_power = np.log(power)
        window = np.convolve(window,np.array([1.0, -0.97]), mode='same')
        window[0] -= 0.97*window[0]
        window *= np.hamming(self.frame_size)

        return window, log_power


    def ComputeFBANK(self, waveform):
        '''로그 Mel 필터뱅크특징(FBANK) 계산'''
        num_samples = np.size(waveform)
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        log_power = np.zeros(num_frames)

        for frame in range(num_frames):
            start_index = frame * self.frame_shift
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)
            spectrum = np.fft.fft(window, n=self.fft_size)
            spectrum = spectrum[:int(self.fft_size/2) + 1]
            spectrum = np.abs(spectrum) ** 2
            fbank = np.dot(spectrum, self.mel_filter_bank.T)
            fbank[fbank<0.1] = 0.1
            fbank_features[frame] = np.log(fbank)
            log_power[frame] = log_pow

        return fbank_features, log_power


    def MakeDCTMatrix(self):
        ''' 이산 코사인 변환(DCT)의 기저 행렬 작성 '''
        N = self.num_mel_bins
        dct_matrix = np.zeros((self.num_ceps,self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2/N) \
                    * np.cos(((2.0*np.arange(N)+1)*k*np.pi) / (2*N))

        return dct_matrix


    def MakeLifter(self):
        ''' 리프터 계산 '''
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter


    def ComputeMFCC(self, waveform):
        ''' MFCC 계산 '''
        fbank, log_power = self.ComputeFBANK(waveform)
        mfcc = np.dot(fbank, self.dct_matrix.T)
        mfcc *= self.lifter
        mfcc[:,0] = log_power
        return mfcc

def extract_features(data_dir, output_dir, feat_extractor):
    """
    Extract MFCC features for all .wav files in the specified directory.
    
    Args:
        data_dir (str): Directory containing .wav files (train, validation, or test).
        output_dir (str): Output directory for the extracted features.
        feat_extractor (FeatureExtractor): Feature extraction class instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_scp = os.path.join(output_dir, 'feats.scp')

    with open(feat_scp, 'w') as feat_file:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    utterance_id = os.path.splitext(file)[0]  # 파일 이름에서 확장자 제거
                    try:
                        with wave.open(wav_path, 'rb') as wav:
                            # 샘플링 주파수 확인
                            if wav.getframerate() != feat_extractor.sample_freq:
                                raise ValueError("Sampling rate mismatch!")

                            # 1채널 (모노) 확인
                            if wav.getnchannels() != 1:
                                raise ValueError("Only mono files are supported!")

                            num_samples = wav.getnframes()
                            waveform = np.frombuffer(wav.readframes(num_samples), dtype=np.int16)

                            # MFCC 추출
                            mfcc = feat_extractor.ComputeMFCC(waveform)
                            num_frames, num_dims = mfcc.shape

                            # 저장 경로 생성
                            out_file = os.path.join(output_dir, f"{utterance_id}.bin")
                            mfcc.astype(np.float32).tofile(out_file)
                            feat_file.write(f"{utterance_id} {out_file} {num_frames} {num_dims}\n")
                    except Exception as e:
                        print(f"Error processing {wav_path}: {e}")

# 데이터 경로 및 출력 경로 설정
base_data_dir = "D:/Dataset/scwar/data"
output_base_dir = "D:/Dataset/scwar/features"

# 데이터셋 디렉토리
dataset_folders = {
    "train": os.path.join(base_data_dir, "train"),
    "validation": os.path.join(base_data_dir, "validation"),
    "test": os.path.join(base_data_dir, "test"),
}

# 출력 디렉토리
output_folders = {
    "train": os.path.join(output_base_dir, "train"),
    "validation": os.path.join(output_base_dir, "validation"),
    "test": os.path.join(output_base_dir, "test"),
}


# 각 데이터셋에 대해 특징 추출

# 메인 함수
if __name__ == "__main__":


    sample_frequency = 16000
    frame_length = 25
    frame_shift = 10
    low_frequency = 20
    high_frequency = sample_frequency / 2
    num_mel_bins = 23
    num_ceps = 13
    dither = 1.0

    np.random.seed(seed=0)

    feat_extractor = FeatureExtractor(
        sample_frequency=sample_frequency, 
        frame_length=frame_length, 
        frame_shift=frame_shift, 
        num_mel_bins=num_mel_bins, 
        num_ceps=num_ceps,
        low_frequency=low_frequency, 
        high_frequency=high_frequency, 
        dither=dither
    )
#데이터를 다 합친다음 train val test로 나눈 거에서 각각 특징을 추출한다.
for split in ["train", "validation", "test"]:
    print(f"Extracting features for {split} dataset...")
    extract_features(dataset_folders[split], output_folders[split], feat_extractor)
    print(f"Features for {split} dataset saved to {output_folders[split]}.")

   