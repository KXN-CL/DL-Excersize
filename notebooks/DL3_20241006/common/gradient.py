# coding: utf-8
"""
수치 미분(Numerical Differentiation) 모듈

이 모듈은 수치 미분을 사용하여 함수의 기울기를 계산합니다.
신경망의 가중치 기울기를 근사하는 데 사용되며,
자동 미분(Automatic Differentiation)과 비교하여 정확도는 높지만 계산 비용이 큽니다.
"""
import numpy as np


def _numerical_gradient_1d(f, x):
    """
    1차원 배열에 대한 수치 미분 계산 (내부 사용 함수)
    
    중심 차분(Central Difference) 방법을 사용하여
    각 요소에서의 편미분 값을 계산합니다.
    
    수식: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    
    Parameters
    ----------
    f : 미분할 함수 (numpy array를 입력받아 스칼라를 반환)
    x : 입력 배열
    
    Returns
    -------
    numpy array: 각 요소에서의 기울기
    """
    h = 1e-4  # 미분 간격 (0.0001)
    grad = np.zeros_like(x)  # 기울기를 저장할 배열 초기화
    
    for idx in range(x.size):
        tmp_val = x[idx]  # 현재 값 백업
        
        # x[idx] = tmp_val + h
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h) 계산
        
        # x[idx] = tmp_val - h
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h) 계산
        
        # 중심 차분으로 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 원래 값 복원
        
    return grad


def numerical_gradient_2d(f, X):
    """
    2차원 배열에 대한 수치 미분 계산
    
    Parameters
    ----------
    f : 미분할 함수
    X : 입력 배열 (1차원 또는 2차원)
    
    Returns
    -------
    numpy array: 기울기
    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    """
    임의 차원 배열에 대한 수치 미분 계산
    
    np.nditer를 사용하여 다차원 배열의 모든 요소에 대해
    효율적으로 수치 미분을 수행합니다.
    
    수식: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    
    Parameters
    ----------
    f : 미분할 함수 (numpy array를 입력받아 스칼라를 반환)
    x : 입력 배열 (임의 차원)
    
    Returns
    -------
    numpy array: 각 요소에서의 기울기 (x와 동일한 shape)
    """
    h = 1e-4  # 미분 간격 (0.0001)
    grad = np.zeros_like(x)  # 기울기를 저장할 배열 초기화
    
    # 다차원 배열을 반복하면서 모든 요소에 접근
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index  # 현재 인덱스
        tmp_val = x[idx]  # 현재 값 백업
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        
        # 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 원래 값 복원
        it.iternext()   
        
    return grad
