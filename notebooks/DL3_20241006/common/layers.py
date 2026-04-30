# coding: utf-8
"""
신경망 계층(Neural Network Layer) 모듈

이 모듈은 신경망을 구성하는 다양한 계층(Layer)을 정의합니다.
각 계층은 forward(순전파)와 backward(역전파) 메서드를 구현하여,
자동 미분(Automatic Differentiation)을 통한 기울기 계산을 지원합니다.

주요 계층:
- Relu: ReLU 활성화 함수 계층
- Sigmoid: 시그모이드 활성화 함수 계층
- Affine: 완전연결층(Fully Connected Layer)
- SoftmaxWithLoss: Softmax 출력층 + 교차 엔트로피 손실 함수
- Dropout: 드롭아웃 정규화 계층
- BatchNormalization: 배치 정규화 계층
- Convolution: 합성곱 계층
- Pooling: 풀링 계층
"""
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    """
    ReLU(Rectified Linear Unit) 활성화 계층
    
    forward 시 입력이 0 이하인 부분을 0으로 마스킹하고,
    backward 시 마스킹된 부분의 기울기를 0으로 설정한다.
    """
    def __init__(self):
        # 입력이 0 이하인 부분을 마스킹하기 위한 배열
        self.mask = None

    def forward(self, x):
        """
        순전파: 입력이 0 이하인 부분을 0으로 변경
        
        Parameters
        ----------
        x : 입력 데이터
        
        Returns
        -------
        x 중 양수 부분만 유지 (음수는 0)
        """
        # 입력이 0 이하인 부분을 마스킹 배열로 표시
        self.mask = (x <= 0)
        out = x.copy()
        # 마스킹된 부분(0 이하)을 0으로 설정
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        역전파: 마스킹된 부분의 기울기를 0으로 설정
        
        Parameters
        ----------
        dout : 손실 함수의 미분 값 (위에서 내려오는 기울기)
        
        Returns
        -------
        dout 중 양수 입력에 해당하는 부분만 전달
        """
        # 마스킹된 부분(0 이하)의 기울기를 0으로 설정
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    """
    시그모이드(Sigmoid) 활성화 계층
    
    forward 시 시그모이드 함수를 적용하고,
    backward 시 시그모이드 함수의 미분을 이용한 기울기를 계산한다.
    """
    def __init__(self):
        # forward 시 출력 값을 저장 (backward 계산에 사용)
        self.out = None

    def forward(self, x):
        """
        순전파: 시그모이드 함수 적용
        
        Parameters
        ----------
        x : 입력 데이터
        
        Returns
        -------
        시그모이드 함수를 적용한 출력 (0~1 사이 값)
        """
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        """
        역전파: 시그모이드 함수의 미분 값을 이용한 기울기 계산
        
        시그모이드 함수의 미분: σ'(x) = σ(x) * (1 - σ(x))
        
        Parameters
        ----------
        dout : 손실 함수의 미분 값 (위에서 내려오는 기울기)
        
        Returns
        -------
        전 계층으로 전달될 기울기
        """
        # 시그모이드 미분 공식 활용: dx = dout * σ(x) * (1 - σ(x))
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    """
    완전연결층(Fully Connected / Dense Layer) 계층 (Affine Transform)
    
    신경망의 모든 뉴런이 이전 층의 모든 뉴런과 연결되는 계층.
    선형 변환 y = xW + b 를 수행하며,
    이미지 데이터 등 4차원 입력도 처리할 수 있도록 텐서 대응.
    
    Parameters
    ----------
    W : 가중치 행렬 (C_in, C_out)
    b : 편향 벡터 (C_out,)
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None  # forward 시 입력 데이터 저장
        self.original_x_shape = None  # 입력 데이터의 원래 shape 저장 (텐서 대응용)
        # 가중치와 편향 매개변수의 미분 (역전파 시 기울기 저장)
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        순전파: 선형 변환 y = xW + b 수행
        
        텐서 입력(배치 데이터)을 2차원 행렬로 reshaped 후
        행렬 곱셈과 편향 더하기를 수행.
        
        Parameters
        ----------
        x : 입력 데이터 (배치 차원 포함)
        
        Returns
        -------
        출력 데이터
        """
        # 텐서 대응: 배치 차원은 유지하고 나머지는 평탄화
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """
        역전파: 가중치와 편향의 기울기를 계산하고 입력 기울기를 반환
        
        - dW = x^T * dout (가중치 기울기)
        - db = sum(dout) (편향 기울기)
        - dx = dout * W^T (입력 기울기)
        
        Parameters
        ----------
        dout : 손실 함수의 미분 값 (위에서 내려오는 기울기)
        
        Returns
        -------
        dx : 전 계층으로 전달될 기울기 (원래 shape로 복원)
        """
        # 입력 기울기 계산: dx = dout * W^T
        dx = np.dot(dout, self.W.T)
        # 가중치 기울기 계산: dW = x^T * dout
        self.dW = np.dot(self.x.T, dout)
        # 편향 기울기 계산: db = sum(dout over batch)
        self.db = np.sum(dout, axis=0)
        
        # 입력 데이터 모양 변경 (원래 shape로 복원 - 텐서 대응)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    """
    Softmax 출력층 + 교차 엔트로피 손실 함수 계층
    
    forward 시 softmax 출력과 교차 엔트로피 손실 값을 계산하고,
    backward 시 손실 함수의 미분 값을 계산하여 전 계층으로 전달.
    """
    def __init__(self):
        self.loss = None  # 손실 함수 값
        self.y = None     # softmax 출력 (예측 확률 분포)
        self.t = None     # 정답 레이블 (원-핫 인코딩 형태)
        
    def forward(self, x, t):
        """
        순전파: softmax 출력과 교차 엔트로피 손실 계산
        
        Parameters
        ----------
        x : 신경망의 미출력(logit)
        t : 정답 레이블 (원-핫 인코딩 또는 정수 인덱스)
        
        Returns
        -------
        손실 함수 값
        """
        self.t = t
        self.y = softmax(x)  # softmax 확률 분포 계산
        self.loss = cross_entropy_error(self.y, self.t)  # 교차 엔트로피 손실 계산
        
        return self.loss

    def backward(self, dout=1):
        """
        역전파: softmax + 교차 엔트로피 손실 함수의 미분 계산
        
        softmax + cross-entropy 손실의 미분은 간단히 (y - t) / batch_size 로 표현됨.
        
        Parameters
        ----------
        dout : 손실 함수의 미분 값 (기본값 1)
        
        Returns
        -------
        dx : 전 계층으로 전달될 기울기
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:  # 정답 레이블이 정수 인덱스일 때
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    Dropout 정규화 계층
    
    학습 시 무작위로 일부 뉴런을 비활성화하여 과적합을 방지.
    논문: http://arxiv.org/abs/1207.0580
    
    Parameters
    ----------
    dropout_ratio : 드롭아웃 비율 (기본값 0.5 = 50% 뉴런 비활성화)
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None  # 비활성화 마스킹 배열

    def forward(self, x, train_flg=True):
        """
        순전파: 학습 시 드롭아웃 적용, 테스트 시 비율 조정
        
        Parameters
        ----------
        x : 입력 데이터
        train_flg : 학습 모드 여부 (True: 학습, False: 테스트)
        
        Returns
        -------
        드롭아웃 적용 출력
        """
        if train_flg:
            # 학습 시: dropout_ratio 비율로 무작위 뉴런 비활성화 (0 또는 1 마스킹)
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 테스트 시: 모든 뉴런 사용 but 출력에 dropout_ratio 보정
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        """
        역전파: 마스킹된 부분의 기울기만 전달
        
        Parameters
        ----------
        dout : 위에서 내려오는 기울기
        
        Returns
        -------
        마스킹 적용 기울기
        """
        return dout * self.mask


class BatchNormalization:
    """
    배치 정규화(Batch Normalization) 계층
    
    각 층의 출력을 평균 0, 분산 1로 정규화하여 학습 안정성과 수렴 속도 향상.
    논문: http://arxiv.org/abs/1502.03167
    
    Parameters
    ----------
    gamma : 스케일 파라미터 (학습 가능한 매개변수)
    beta : 이동 파라미터 (학습 가능한 매개변수)
    momentum : 이동 평균/분산 업데이트 모멘텀 (기본값 0.9)
    running_mean : 테스트 시 사용할 이동 평균
    running_var : 테스트 시 사용할 이동 분산
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma  # 스케일 파라미터
        self.beta = beta    # 이동 파라미터
        self.momentum = momentum
        self.input_shape = None  # 합성곱 계층은 4차원, 완전연결 계층은 2차원  
        
        # 시험(테스트) 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None  # 중심화된 입력 (x - mu)
        self.std = None  # 표준편차
        self.dgamma = None  # gamma의 기울기
        self.dbeta = None   # beta의 기울기

    def forward(self, x, train_flg=True):
        """
        순전파: 배치 정규화 수행
        
        학습 시: 배치 데이터의 평균과 분산을 사용하여 정규화
        테스트 시: 학습 중 누적한 이동 평균과 분산을 사용하여 정규화
        
        Parameters
        ----------
        x : 입력 데이터
        train_flg : 학습 모드 여부
        
        Returns
        -------
        배치 정규화 적용 출력
        """
        self.input_shape = x.shape
        if x.ndim != 2:
            # 4차원 입력(이미지)을 2차원으로 reshaped
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        # 처음 forward 시 running_mean, running_var 초기화
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            # 학습 시: 배치 통계 사용
            mu = x.mean(axis=0)  # 배치 평균
            xc = x - mu          # 중심화된 입력
            var = np.mean(xc**2, axis=0)  # 분산
            std = np.sqrt(var + 10e-7)     # 표준편차 (안정성을 위해 작은 값 추가)
            xn = xc / std          # 정규화
            
            # backward에서 사용할 중간 데이터 저장
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # 이동 평균/분산 업데이트 (학습 중 누적)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            # 테스트 시: 이동 평균/분산 사용
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        # 스케일과 이동 적용: out = gamma * xn + beta
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        """
        역전파: 배치 정규화의 기울기 계산
        
        Parameters
        ----------
        dout : 위에서 내려오는 기울기
        
        Returns
        -------
        전 계층으로 전달될 기울기
        """
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # beta 기울기
        dbeta = dout.sum(axis=0)
        # gamma 기울기
        dgamma = np.sum(self.xn * dout, axis=0)
        # 정규화된 입력에 대한 기울기
        dxn = self.gamma * dout
        # 중심화된 입력에 대한 기울기
        dxc = dxn / self.std
        # 표준편차에 대한 기울기
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        # 분산에 대한 기울기
        dvar = 0.5 * dstd / self.std
        # 중심화된 입력에 대한 추가 기울기
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        # 평균에 대한 기울기
        dmu = np.sum(dxc, axis=0)
        # 최종 입력 기울기
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    """
    합성곱(Convolution) 계층
    
    이미지 데이터에 국소 연결 패턴을 학습하는 합성곱 연산 수행.
    im2col을 사용하여 행렬 곱셈으로 효율적으로 구현.
    
    Parameters
    ----------
    W : 합성곱 필터 가중치 (FN, C, FH, FW)
        FN: 필터 수, C: 채널 수, FH: 필터 높이, FW: 필터 너비
    b : 편향 벡터 (FN,)
    stride : 스트라이드 (스킬 크기, 기본값 1)
    pad : 패딩 (기본값 0)
    """
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터 (backward 시 사용)
        self.x = None   
        self.col = None  # im2col로 변환된 입력
        self.col_W = None  # im2col로 변환된 필터
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        순전파: 합성곱 연산 수행
        
        Parameters
        ----------
        x : 입력 데이터 (배치, 채널, 높이, 너비)
        
        Returns
        -------
        합성곱 연산 출력
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 출력 크기 계산
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # im2col로 입력을 행렬로 변환
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 필터를 행렬로 변환
        col_W = self.W.reshape(FN, -1).T

        # 행렬 곱셈으로 합성곱 연산 수행
        out = np.dot(col, col_W) + self.b
        # 출력 shape 복원 (배치, 높이, 너비, 필터수) -> (배치, 필터수, 높이, 너비)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # backward에서 사용할 데이터 저장
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        역전파: 합성곱 계층의 기울기 계산
        
        Parameters
        ----------
        dout : 위에서 내려오는 기울기
        
        Returns
        -------
        전 계층으로 전달될 기울기 (입력 데이터 shape)
        """
        FN, C, FH, FW = self.W.shape
        # dout shape 변경 (배치, 높이, 너비, 필터수)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # 편향 기울기 계산
        self.db = np.sum(dout, axis=0)
        # 가중치 기울기 계산: dW = col^T * dout
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 입력 기울기 계산
        dcol = np.dot(dout, self.col_W.T)
        # col2im로 원래 이미지 shape로 복원
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    """
    풀링(Pooling) 계층 (Max Pooling)
    
    합성곱 출력의 공간적 차원을 축소하고 불변성 확보.
    최대값(Max)을 선택하여 주요 특징만 유지.
    
    Parameters
    ----------
    pool_h : 풀링 높이
    pool_w : 풀링 너비
    stride : 스트라이드 (기본값 1)
    pad : 패딩 (기본값 0)
    """
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None  # forward 시 입력 데이터 저장
        self.arg_max = None  # 최대값 인덱스 (backward 시 사용)

    def forward(self, x):
        """
        순전파: 최대 풀링 수행
        
        Parameters
        ----------
        x : 입력 데이터 (배치, 채널, 높이, 너비)
        
        Returns
        -------
        풀링 연산 출력
        """
        N, C, H, W = x.shape
        # 출력 크기 계산
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # im2col로 지역 영역 추출
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 각 지역 영역 내에서 최대값 찾기
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최대값 인덱스와 최대값 저장
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        역전파: 최대값 인덱스를 사용하여 기울기 전파
        
        최대값이 있던 위치에만 기울기를 전달하고 나머지는 0으로 설정.
        
        Parameters
        ----------
        dout : 위에서 내려오는 기울기
        
        Returns
        -------
        전 계층으로 전달될 기울기
        """
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        # 최대값 인덱스에만 기울기 전달
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        # col2im으로 원래 shape로 복원
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
