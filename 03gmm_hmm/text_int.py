# -*- coding: utf-8 -*-

# os 모듈을 임포트
import os

def phone_to_int(label_str, label_int, phone_list, insert_sil=False):
    ''' 
    음소 리스트를 사용하여 라벨 파일을 숫자로 변환합니다.
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
                    print(f"Unknown phone: {u}")
                    continue
                f_out.write(f" {phone_list.index(u)}")
            
            if insert_sil:
                f_out.write(' 0')  # 포즈 삽입
            
            f_out.write('\n')


if __name__ == "__main__":
    # 음소 리스트 파일 (cmu39.txt)
    phone_file = os.path.realpath('D:/Dataset/scwar/03gmm_hmm/cmu39.txt')
    #phone_file ='./cmu39.txt'
    silence_phone = 'pau'
    insert_sil = True

    # Validation 데이터의 text_phone 파일 경로
    label_val_str = os.path.realpath('D:/Dataset/scwar/data/label/val/text_phone')
    #label_val_str = '../data/label/val/text_phone'

    # 출력 디렉토리 및 파일
    out_val_dir = os.path.realpath('D:/Dataset/scwar/data/label/val')
    #out_val_dir = '../data/label/val'
    label_val_int = os.path.join(out_val_dir, 'text_int')
    
    os.makedirs(out_val_dir, exist_ok=True)

    # 음소 리스트 생성
    phone_list = [silence_phone]
    with open(phone_file, mode='r') as f:
        phone_list.extend([line.strip() for line in f if line.strip()])

    # text_phone 파일을 숫자 라벨로 변환
    phone_to_int(label_val_str, label_val_int, phone_list, insert_sil)
    print(f"Validation numeric labels saved to: {label_val_int}")
