import os

# 테스트 데이터 경로와 출력 파일 설정
test_folder = '../data/test/'  # 테스트 파일들이 저장된 폴더 경로
output_label_file = '../data/label/test/test_label.txt'

# 테스트 파일 리스트 가져오기
test_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.wav')])

# 레이블 파일 생성
with open(output_label_file, 'w') as label_file:
    for file_name in test_files:
        # 파일명에서 라벨 추출 (예: backward_001.wav -> backward)
        label = file_name.split('_')[0]
        # 레이블 파일에 작성 (예: backward_001 -> backward)
        label_file.write(f"{file_name.split('.')[0]} {label}\n")

print(f"{len(test_files)}개의 레이블 파일이 {output_label_file}에 저장되었습니다.")