# coding: utf-8
"""
MNIST 데이터셋 다운로드 및 로딩 모듈

이 모듈은 MNIST 손글씨 이미지 데이터셋을 자동으로 다운로드하고,
NumPy 배열로 변환하여 저장 및 로드하는 기능을 제공합니다.

MNIST 데이터셋:
- 훈련 데이터: 60,000개 이미지 (28x28 픽그레이, 흑백)
- 테스트 데이터: 10,000개 이미지 (28x28 픽셀, 흑백)
- 레이블: 0~9 숫자 (10 클래스)

Usage
-----
>>> from dataset.mnist import load_mnist
>>> (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
"""
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


# MNIST 데이터셋 다운로드 URL
url_base = 'http://yann.lecun.com/exdb/mnist/'

# 파일명 매핑
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',   # 훈련 이미지
    'train_label': 'train-labels-idx1-ubyte.gz',  # 훈련 레이블
    'test_img': 't10k-images-idx3-ubyte.gz',      # 테스트 이미지
    'test_label': 't10k-labels-idx1-ubyte.gz'     # 테스트 레이블
}

# 데이터셋 디렉토리 및 저장 파일 경로
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + os.sep + "mnist.pkl"

# 데이터셋 설정
train_num = 60000   # 훈련 데이터 수
test_num = 10000    # 테스트 데이터 수
img_dim = (1, 28, 28)  # 이미지 차원 (채널, 높이, 너비)
img_size = 784        # 이미지 평탄화 크기 (1 * 28 * 28)


def _download(file_name):
    """
    개별 파일 다운로드
    
    이미 다운로드된 파일은 다시 다운로드하지 않음.
    
    Parameters
    ----------
    file_name : 다운로드할 파일명
    """
    file_path = dataset_dir + os.sep + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    

def download_mnist():
    """
    MNIST 데이터셋의 모든 파일 다운로드
    """
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    """
    레이블 데이터 로드 및 파싱
    
    Parameters
    ----------
    file_name : 레이블 파일명
    
    Returns
    -------
    numpy array: 레이블 배열 (shape: (60000,) 또는 (10000,))
    """
    file_path = dataset_dir + os.sep + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        # 8바이트 헤더 건너뛰고 레이블 데이터 읽기
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    """
    이미지 데이터 로드 및 파싱
    
    Parameters
    ----------
    file_name : 이미지 파일명
    
    Returns
    -------
    numpy array: 이미지 배열 (shape: (N, 784))
    """
    file_path = dataset_dir + os.sep + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
        # 16바이트 헤더 건너뛰고 이미지 데이터 읽기
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 각 이미지를 784차원 벡터로 reshape
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    """
    모든 데이터를 NumPy 배열로 변환 및 딕셔너리로 묶음
    
    Returns
    -------
    dict: 훈련/테스트 이미지와 레이블을 담은 딕셔너리
    """
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    """
    MNIST 데이터셋 초기화: 다운로드 -> 변환 -> pickle 저장
    
    처음 실행 시 MNIST 데이터셋을 다운로드하고
    pickle 파일로 저장하여 향후 빠른 로딩을 가능하게 함.
    """
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    """
    정수 레이블을 원-핫 인코딩(One-Hot Encoding) 배열로 변환
    
    Parameters
    ----------
    X : 정수 레이블 배열
    
    Returns
    -------
    numpy array: 원-핫 인코딩 배열 (shape: (N, 10))
    """
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    MNIST 데이터셋 읽기
    
    pickle 파일에서 데이터를 로드하고, 필요시 정규화 및 변환을 수행.
    
    Parameters
    ----------
    normalize : 이미지 픽셀 값을 0.0~1.0 사이로 정규화할지 여부
    one_hot_label : 레이블을 원-핫 배열로 변환할지 여부
        True: [0,0,1,0,0,0,0,0,0,0] 형태
        False: 정수 인덱스 (예: 2)
    flatten : 입력 이미지를 1차원 배열로 펼칠지 여부
        True: 784차원 벡터
        False: (1, 28, 28) 형태
    
    Returns
    -------
    ((훈련 이미지, 훈련 레이블), (테스트 이미지, 테스트 레이블))
    """
    # pickle 파일이 없으면 초기화 (다운로드 및 변환)
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # 픽셀 값 정규화 (0.0~1.0)
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    # 원-핫 인코딩 적용
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    # 평탄화 해제 (이미지 형태 유지)
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
