# coding: utf-8
"""
유틸리티 함수 모듈

신경망 학습에 필요한 다양한 유틸리티 함수를 제공합니다:
- smooth_curve: 손실 함수 그래프 매끄럽게 하기
- shuffle_dataset: 데이터셋 셔플링
- im2col / col2im: 합성곱 연산을 위한 텐서 변환
"""
import numpy as np


def smooth_curve(x):
    """
    손실 함수(loss curve)의 그래프를 매끄럽게 하는 함수
    
    Kaiser 윈도우를 사용한 컨볼루션(convolution)으로
    학습 과정에서의 손실 값 변동을 완화한다.
    
    Parameters
    ----------
    x : 매끄럽게 할 배열 (손실 값 시계열)
    
    Returns
    -------
    numpy array: 매끄럽게 된 배열
    
    Reference
    ---------
    http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11  # 윈도우 길이 (11개 포인트로 평균)
    # 양쪽 끝을 대칭으로 확장 (패딩)
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # Kaiser 윈도우 생성
    w = np.kaiser(window_len, 2)
    # 컨볼루션으로 매끄러운 값 계산
    y = np.convolve(w/w.sum(), s, mode='valid')
    # 양쪽 끝 부분 제거
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """
    데이터셋을 무작위로 섞는 함수
    
    학습 시 데이터 순서에 의한 편향을 방지하기 위해
    훈련 데이터와 정답 레이블을 동일한 순서로 섞는다.
    
    Parameters
    ----------
    x : 훈련 데이터 (이미지 등 입력 데이터)
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])  # 무작위 순서 생성
    # 2차원 (완전연결망): x[permutation, :]
    # 4차원 (합성곱망): x[permutation, :, :, :]
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    """
    합성곱 연산 후 출력 크기를 계산하는 함수
    
    Parameters
    ----------
    input_size : 입력 크기 (높이 또는 너비)
    filter_size : 필터 크기
    stride : 스트라이드 (스킬 크기)
    pad : 패딩 (주변에 추가할 패딩 크기)
    
    Returns
    -------
    float: 출력 크기
    """
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    다수의 이미지를 입력받아 2차원 배열로 변환 (평탄화)
    
    합성곱 연산을 완전연결층의 행렬 곱으로 변환하여
    계산 효율을 높이는 핵심 함수입니다.
    
    이미지에서 필터 크기의 지역 영역(local receptive field)을
    추출하여 2차원 행렬로 재배열합니다.
    
    Parameters
    ----------
    input_data : 4차원 배열 (이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열 (N*out_h*out_w, C*filter_h*filter_w)
    
    Example
    -------
    입력: (N, C, H, W) 형태의 이미지 배치
    출력: (N * out_h * out_w, C * filter_h * filter_w) 형태의 행렬
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # 패딩 적용 (기본값: 0으로 패딩)
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # 출력 배열 초기화
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 필터 크기의 지역 영역을 추출하여 col 배열에 저장
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 6차원 배열을 2차원 배열로 재배열 (전치 후 평탄화)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    im2col의 역연산: 2차원 배열을 이미지 형태로 복원
    
    합성곱 연산의 backward pass에서 사용되며,
    im2col로 평탄화된 배열을 원래 이미지 형태로 되돌린다.
    
    Parameters
    ----------
    col : 2차원 배열 (im2col로 변환된 데이터)
    input_shape : 원래 이미지 데이터의 형상 (예: (10, 1, 28, 28))
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들 (원래 shape로 복원)
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 2차원 배열을 6차원으로 재배열 후 전치
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 출력 배열 초기화 (패딩 포함 크기)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    # 각 위치에서 값을 누적 (패딩 영역 포함)
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 패딩 영역 제거
    return img[:, :, pad:H + pad, pad:W + pad]
