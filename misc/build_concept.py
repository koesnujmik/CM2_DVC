# coding:utf-8
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import numpy as np
import pdb

# 만약 NLTK 데이터가 없다면 다운로드 (최초 한 번)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# 파일 경로 (학습 데이터셋 사용)
file_path_list = ["data/yc2/captiondata/train.json"]

# 개념 집합과 관련 파일 저장 경로
output_vocab_path = './data/concept_set_youcook2.json'
output_labels_path = './data/concept_labels_youcook2.json'

# 단어 집합 사이즈(youcook2는 600)
Nc = 600

# 제거할 특수문자 리스트 (기본 제공)
mark = [',', ':', '!', '_', ';', '-', '.', '?', '/', '"', '\\n', '\\']

def preprocess_sentence(sentence):
    """문장 내 특수문자를 제거하고 소문자로 변환 후 토크나이즈."""
    for m in mark:
        sentence = sentence.replace(m, " ")
    # 여러 공백을 하나의 공백으로 변환
    sentence = ' '.join(sentence.split())
    sentence = sentence.strip().lower()
    tokens = word_tokenize(sentence)
    return tokens

def build_concept_set(file_path_list, Nc):
    """
    여러 JSON 파일의 캡션 데이터를 읽어 전체 단어 빈도수를 계산하고,
    빈도수가 높은 순으로 상위 Nc개의 단어로 개념 집합 E를 구성합니다.
    """
    count_vocab = {}
    # 학습 데이터셋의 모든 캡션을 순회
    for file_path in file_path_list:
        data = json.load(open(file_path, encoding='utf-8'))
        for video_id, info in data.items():
            captions = info.get("sentences", [])
            for sentence in captions:
                tokens = preprocess_sentence(sentence)
                for token in tokens:
                    if token == "":
                        continue
                    count_vocab[token] = count_vocab.get(token, 0) + 1
    # 빈도수를 기준으로 정렬한 후 상위 Nc개의 단어 선택
    sorted_vocab = sorted(count_vocab.items(), key=lambda x: x[1], reverse=True)
    concept_set = [word for word, freq in sorted_vocab[:Nc]]
    return concept_set, count_vocab

def build_concept_label_for_video(captions, concept_set):
    """
    주어진 캡션 리스트(비디오의 모든 캡션)에 대해, 
    concept_set에 해당하는 단어의 등장 여부를 멀티핫 벡터(Y_c)로 반환.
    각 단어가 한 번이라도 등장하면 1, 아니면 0.
    """
    # 모든 캡션을 하나의 문자열로 합침
    combined_text = " ".join(captions)
    tokens = set(preprocess_sentence(combined_text))
    # concept_set의 각 단어가 tokens에 있으면 1, 아니면 0
    label = [1 if word in tokens else 0 for word in concept_set]
    return label

if __name__ == '__main__':
    concept_set, count_vocab = build_concept_set(file_path_list, Nc)
    print("Concept set E (vocabulary):", concept_set)
    print("Total number of concepts:", len(concept_set))

    # 각 비디오에 대한 concept label Y 구성 (비디오 수준 멀티핫 레이블)
    data = json.load(open(file_path_list[0], encoding='utf-8'))
    video_concept_labels = {}  # video_id -> Y_c (list or np.array)
    for video_id, info in data.items():
        captions = info.get("sentences", [])
        Y_c = build_concept_label_for_video(captions, concept_set)
        video_concept_labels[video_id] = Y_c

    # 결과 저장
    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'concept_set': concept_set, 'word_counts': count_vocab}, f, ensure_ascii=False, indent=4)
    with open(output_labels_path, 'w', encoding='utf-8') as f:
        json.dump(video_concept_labels, f, ensure_ascii=False, indent=4)

    print("Saved concept set to:", output_vocab_path)
    print("Saved concept labels to:", output_labels_path)