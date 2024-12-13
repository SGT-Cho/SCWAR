import os
import numpy as np

# bin 파일이 저장된 디렉토리
feature_dir = 'D:/Dataset/scwar/features/test'
#feature_dir = '../features/test'
# 출력될 feats.scp 파일 경로
output_scp_path = 'D:/Dataset/scwar/01compute_features/mfcc/feats.scp'
#output_scp_path = '../01compute_features/mfcc/feats.scp'
# feats.scp 생성
with open(output_scp_path, 'w') as scp_file:
    for file in os.listdir(feature_dir):
        if file.endswith('.bin'):  # bin 파일만 처리
            utterance_id = os.path.splitext(file)[0]  # 파일명에서 확장자 제거
            file_path = os.path.join(feature_dir, file)  # 전체 경로 생성
            
            # bin 파일에서 차원 추출
            try:
                data = np.fromfile(file_path, dtype=np.float32)
                num_frames = len(data)  # 총 데이터 개수
                num_dims = 13  # HMM에서 정의된 차원으로 설정 (필요에 따라 수정)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            
            # feats.scp에 기록
            scp_file.write(f"{utterance_id} {file_path} 0 {num_dims}\n")

print(f"feats.scp generated at {output_scp_path}")
