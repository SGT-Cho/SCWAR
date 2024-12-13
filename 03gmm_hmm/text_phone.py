import os

def load_word_to_phoneme_map(dic_file):
    """
    sc35.dic 파일에서 단어-음소 매핑을 로드합니다.
    Args:
        dic_file (str): Dictionary file path.
    Returns:
        dict: 단어와 음소 표현의 매핑.
    """
    word_to_phoneme = {}
    with open(dic_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            phonemes = parts[1:]
            word_to_phoneme[word] = phonemes
    return word_to_phoneme

def create_text_phone(data_dir, dic_file, output_file):
    """
    74063개의 파일에 대해 text_phone 파일을 생성합니다.
    Args:
        data_dir (str): 데이터 파일이 저장된 디렉토리.
        dic_file (str): 단어-음소 매핑 파일 (sc35.dic).
        output_file (str): 생성된 text_phone 파일 경로.
    """
    # 단어-음소 매핑 로드
    word_to_phoneme = load_word_to_phoneme_map(dic_file)
    
    with open(output_file, 'w') as f_out:
        for subdir, _, files in os.walk(data_dir):
            for file_name in files:
                if file_name.endswith('.wav'):
                    # 파일 이름에서 발화 ID 추출
                    utterance_id = os.path.splitext(file_name)[0]
                    # 파일 이름에서 단어 추출 (예: backward_1 -> backward)
                    word = file_name.split('_')[0].lower()
                    
                    if word in word_to_phoneme:
                        phonemes = word_to_phoneme[word]
                        # 발화 ID와 음소 표현을 기록
                        f_out.write(f"{utterance_id} {' '.join(phonemes)}\n")
                    else:
                        print(f"Word not found in dictionary: {word}")
                        continue

if __name__ == "__main__":
    # 데이터 파일 경로
    data_dir = r'D:/Dataset/scwar/data/validation'
    #data_dir = '../data/validation'
    # sc35.dic 파일 경로
    dic_file = r'D:/Dataset/scwar/03gmm_hmm/sc35.dic'
    #dic_file = './sc35.dic'
    # 출력될 text_phone 파일 경로
    output_file = os.path.realpath('D:/Dataset/scwar/data/label/val/text_phone')
    #output_file = '../data/label/val/text_phone'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    create_text_phone(data_dir, dic_file, output_file)
    print(f"text_phone file created at: {output_file}")
