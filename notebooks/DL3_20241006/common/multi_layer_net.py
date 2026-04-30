# coding: utf-8
"""
완전연결 다층 신경망(Multi-Layer Neural Network) 클래스 모듈

이 모듈은 완전연결(Fully Connected) 다층 신경망을 구현합니다.
MNIST 손글씨 분류 등을 학습하고 예측하는 데 사용됩니다.

주요 기능:
- 가중치 초기화 (He, Xavier 초기화 지원)
- 순전파(Predict) 및 손실 계산
- 오차역전파법(Backpropagation)을 통한 기울기 계산
- 가중치 감소(L2 Regularization) 지원
"""
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """
    완전연결 다층 신경망 클래스
    
    여러 개의 완전연결층(Affine)과 활성화 함수 계층을 순서대로 연결한
    다층 퍼셉트론(MLP)을 구현합니다.
    
    Parameters
    ----------
    input_size : 입력 크기 (MNIST의 경우 784 = 28*28)
    hidden_sizeList : 각 은닉층의 뉴런 수를 담은 리스트 (예: [100, 100, 100])
    output_size : 출력 크기 (MNIST의 경우 10 - 숫자 0~9)
    activation : 활성화 함수 - 'relu' 또는 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정
        - 'relu' 또는 'he': He 초깃값 (ReLU 활성화 함수용)
        - 'sigmoid' 또는 'xavier': Xavier 초깃값 (Sigmoid용)
    weight_decay_lambda : 가중치 감소(L2 norm regularization) 강도
    """
    def __init__(self, input_size, hidden_sizeList, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizeList = hidden_sizeList
        self.hidden_layer_num = len(hidden_sizeList)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}  # 모든 가중치와 편향 저장

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층 생성: OrderedDict를 사용하여 계층 순서 보장
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        
        # 은닉층 생성: Affine -> Activation 함수 순으로 연결
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        # 출력층: Affine 계층 (활성화 함수 없음)
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        # 마지막 계층: 손실 함수 (Softmax + Cross Entropy)
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        가중치 초기화
        
        He 초기화: ReLU 사용 시 권장 (std = sqrt(2/n_in))
        Xavier 초기화: Sigmoid 사용 시 권장 (std = sqrt(1/n_in))
        
        Parameters
        ----------
        weight_init_std : 가중치 표준편차 또는 초기화 방법 ('relu', 'he', 'sigmoid', 'xavier')
        """
        # 모든 층의 크기 목록: 입력 -> 은닉층들 -> 출력
        all_size_list = [self.input_size] + self.hidden_sizeList + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # ReHe 초기화: ReLU 활성화 함수 사용 시
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLU용 권장 초깃값
            # Xavier 초기화: Sigmoid 활성화 함수 사용 시
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # Sigmoid용 권장 초깃값
            # 가중치 초기화: 정규 분포에서 샘플링
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            # 편향 초기화: 0으로 초기화
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        """
        예측(Prediction): 순전파 수행
        
        Parameters
        ----------
        x : 입력 데이터
        
        Returns
        -------
        신경망의 출력 (softmax 전 logit 값)
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        손실 함수 계산
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블 
        
        Returns
        -------
        손실 함수 값 (가중치 감소 포함)
        """
        y = self.predict(x)

        # 가중치 감소(L2 regularization) 계산
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # 손실 함수 값 + 가중치 감소 항
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        """
        정확도(Accuracy) 계산
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        분류 정확도 (0~1 사이 값)
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 가장 확률이 높은 인덱스
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        기울기 계산: 수치 미분(Numerical Differentiation) 사용
        
        계산 비용이 크지만 정확도가 높음.
        주로 구현 검증에 사용.
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        각 층의 기울기 딕셔너리
            grads['W1'], grads['W2'], ... : 각 층의 가중치 기울기
            grads['b1'], grads['b2'], ... : 각 층의 편향 기울기
        """
        loss_W = lambda W: self.loss(x, t)  # 가중치 W에 대한 손실 함수

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """
        기울기 계산: 오차역전파법(Backpropagation) 사용
        
        수치 미분보다 훨씬 효율적으로 기울기를 계산.
        실제 학습에서 사용하는 방식.
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        Returns
        -------
        각 층의 기울기 딕셔너리
            grads['W1'], grads['W2'], ... : 각 층의 가중치 기울기
            grads['b1'], grads['b2'], ... : 각 층의 편향 기울기
        """
        # forward: 손실 계산 (내부에서 각 계층의 출력을 저장)
        self.loss(x, t)

        # backward: 마지막 계층에서 시작하여 거꾸로 기울기 전파
        dout = 1  # 초기 기울기
        dout = self.last_layer.backward(dout)  # 마지막 계층 (SoftmaxWithLoss)

        # 저장된 계층을 역순으로 반복
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 기울기 저장: 가중치 감소(L2 penalty) 추가
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
