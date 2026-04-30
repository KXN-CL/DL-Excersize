# coding: utf-8
"""
활성화 함수(Activation Function) 및 손실 함수(Loss Function) 모듈

이 모듈은 신경망에서 사용하는 주요 활성화 함수와 손실 함수를 정의합니다.
- 활성화 함수: step_function, sigmoid, relu, softmax
- 손실 함수: mean_squared_error, cross_entropy_error, softmax_loss
"""
import numpy as np


def identity_function(x):
    """
    항등 함수: 입력을 그대로 반환
    
    일반적으로 출력층에서 사용되며, regression 문제에서
    출력 값에 제약이 없을 때 활용한다.
    
    Parameters
    ----------
    x : 입력 데이터
    
    Returns
    -------
    입력 값을 그대로 반환
    """
    return x


def step_function(x):
    """
    단계 함수(Step Function) 활성화 함수
    
    입력 값이 0보다 크면 1, 아니면 0을 반환하는 이산적인 함수.
    신경망의 초기 단계에서 사용되었으나, 미분 불가능 지점이 있어
    현재는 주로 학습에 사용되지 않는다.
    
    Parameters
    ----------
    x : 입력 데이터
    
    Returns
    -------
    numpy array: x > 0이면 1, 아니면 0
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """
    시그모이드(Sigmoid) 활성화 함수
    
    입력 값을 0과 1 사이로 압축하는 S자 곡선 함수.
    신경망의 이진 분류 출력층에서 널리 사용된다.
    
    수식: σ(x) = 1 / (1 + exp(-x))
    
    Parameters
    ----------
    x : 입력 데이터
    
    Returns
    -------
    numpy array: 시그모이드 함수를 적용한 값 (0~1 사이)
    """
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    """
    시그모이드 함수의 미분(기울기) 계산
    
    시그모이드 함수의 미분은 시그모이드 출력 자체를 이용해
    효율적으로 계산할 수 있음: σ'(x) = σ(x) * (1 - σ(x))
    
    Parameters
    ----------
    x : 입력 데이터
    
    Returns
    -------
    numpy array: 시그모이드 함수의 미분 값
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    """
    ReLU(Rectified Linear Unit) 활성화 함수
    
    입력이 양수이면 그대로 반환하고, 음수이면 0을 반환.
    현재 가장 널리 사용되는 활성화 함수로, 계산이 간단하고
    기울기 소실 문제를 완화한다.
    
    수식: f(x) = max(0, x)
    
    Parameters
    ----------
    x : 입력 데이터
    
    Returns
    -------
    numpy array: x 중 양수 부분은 그대로, 음수 부분은 0
    """
    return np.maximum(0, x)


def relu_grad(x):
    """
    ReLU 함수의 미분(기울기) 계산
    
    x가 0 이상이면 1, 0 미만이면 0을 반환.
    
    Parameters
    ----------
    x : 입력 데이터
    
    Returns
    -------
    numpy array: x >= 0이면 1, 아니면 0
    """
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    
    

def softmax(x):
    """
    Softmax 활성화 함수
    
    입력 벡터의 각 요소를 확률로 변환. 모든 출력의 합이 1이 됨.
    다중 분류 문제의 출력층에서 널리 사용된다.
    
    수식: y_k = exp(x_k) / sum(exp(x_i))
    
    Parameters
    ----------
    x : 입력 데이터 (1차원 또는 2차원 배열)
    
    Returns
    -------
    numpy array: 확률 분포 (모든 요소의 합이 1)
    """
    if x.ndim == 2:
        # 2차원 배열인 경우: 각 열(샘플)별로 softmax 계산
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 방지를 위해 최대값 뺌
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    # 1차원 배열인 경우: 표준 softmax 계산
    x = x - np.max(x) # 오버플로 대책: 최대값을 빼서 지수 함수의 발산 방지
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    """
    평균 제곱 오차(Mean Squared Error) 손실 함수
    
    regression(회귀) 문제에서 주로 사용되며,
    예측값과 정답 값의 차이의 제곱을 평균한다.
    
    수식: L = 0.5 * sum((y - t)^2)
    
    Parameters
    ----------
    y : 예측값
    t : 정답 레이블
    
    Returns
    -------
    float: 평균 제곱 오차
    """
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    """
    교차 엔트로피 오차(Cross Entropy Error) 손실 함수
    
    분류(classification) 문제에서 주로 사용되며,
    예측 확률 분포와 정답 분포 간의 차이를 측정.
    로그 확률의 음의 합을 계산하여, 정답 레이블의 확률이 낮을수록 큰 손실 값을 가짐.
    
    수식: L = -sum(log(y[t_i])) / batch_size
    
    Parameters
    ----------
    y : 신경망의 출력 (softmax 등 확률 분포)
    t : 정답 레이블 (원-핫 인코딩 또는 정수 인덱스)
    
    Returns
    -------
    float: 교차 엔트로피 오차
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # 로그 계산 시 음수 로그 방지를 위해 작은 값(1e-7) 추가
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    """
    Softmax 함수와 교차 엔트로피 오차를 결합한 손실 함수
    
    완전연결 신경망의 출력층에서 널리 사용되며,
    softmax 출력과 교차 엔트로피 오차를 한 번에 계산.
    
    Parameters
    ----------
    X : 신경망의 미출력(logit)
    t : 정답 레이블
    
    Returns
    -------
    float: softmax + 교차 엔트로피 손실 값
    """
    y = softmax(X)
    return cross_entropy_error(y, t)
