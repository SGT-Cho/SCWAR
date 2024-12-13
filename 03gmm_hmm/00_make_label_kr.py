# -*- coding: utf-8 -*-
import os

def phone_to_int(label_str, label_int, phone_list, insert_sil=False):
    ''' 
    라벨 파일의 음소를 숫자로 변환합니다.
    label_str: 입력 라벨 파일 경로 (text_phone)
    label_int: 출력 숫자 라벨 파일 경로
    phone_list: 음소 리스트
    insert_sil: True일 경우, 시작과 끝에 포즈를 삽입합니다.
    '''
    with open(label_str, mode='r') as f_in, open(label_int, mode='w') as f_out:
        for line in f_in:
            text = line.split()
            f_out.write('%s' % text[0])  # 발화 ID
            
            if insert_sil:
                f_out.write(' 0')  # 포즈 삽입

            for u in text[1:]:
                if u not in phone_list:
                    ####################
                    print(f"Unknown phone: {u}")  # 수정: 오류 대신 경고 메시지 출력
                    ####################
                    continue  # 수정: 알 수 없는 음소를 건너뜀
                f_out.write(f" {phone_list.index(u)}")
            
            if insert_sil:
                f_out.write(' 0')  # 포즈 삽입
            
            f_out.write('\n')

if __name__ == "__main__":
    # 음소 리스트 파일 (cmu39.txt)
    ####################
    phone_file = './cmu39.txt'  # 음소 리스트 파일 경로
    ####################
    silence_phone = 'pau'
    insert_sil = True

    # text_phone 파일 경로
    ####################
    label_train_str = '../data/label/text_phone'  #경로 지정
    ####################

    # 출력 디렉토리 및 파일
    ####################
    out_train_dir = '../data/label/train'  # 기존 출력 경로 유지
    ####################
    label_int = os.path.join(out_train_dir, 'text_int')
    
    os.makedirs(out_train_dir, exist_ok=True)

    # 음소 리스트 생성
    phone_list = [silence_phone]
    with open(phone_file, mode='r') as f:
        ####################
        phone_list.extend([line.strip() for line in f if line.strip()])  # 수정: 빈 줄 제거
        ####################

    # 음소와 숫자의 대응 관계 저장
    out_phone_list = os.path.join(out_train_dir, 'phone_list')
    with open(out_phone_list, 'w') as f:
        for i, phone in enumerate(phone_list):
            f.write(f"{phone} {i}\n")

    # text_phone 파일을 숫자 라벨로 변환
    ####################
    phone_to_int(label_train_str, label_int, phone_list, insert_sil)  # 수정: 함수 호출 위치 정리
    ####################
    print(f"Numeric labels saved to: {label_int}")
