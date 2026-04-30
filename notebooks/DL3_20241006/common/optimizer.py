# coding: utf-8
"""
최적화(Optimization) 알고리즘 모듈

이 모듈은 신경망 가중치를 업데이트하는 다양한 최적화 알고리즘을 구현합니다.
경사 하강법(Gradient Descent)의 변형들로, 학습 속도와 안정성을 개선합니다.

주요 최적화 알고리즘:
- SGD: 확률적 경사 하강법
- Momentum: 모멘텀 추가
- Nesterov: Nesterov 가속 경사
- AdaGrad: 적응적 학습률
- RMSprop: RMSprop
- Adam: Adam (Momentum + AdaGrad 결합)
"""
import numpy as np


class SGD:
    """
    확률적 경사 하강법 (Stochastic Gradient Descent)
    
    가장 기본적인 최적화 알고리즘으로,
    가중치를 기울기 방향으로 일정 학습률로 업데이트.
    
    수식: params -= lr * grads
    """

    def __init__(self, lr=0.01):
        self.lr = lr  # 학습률 (learning rate)
        
    def update(self, params, grads):
        """
        가중치 업데이트
        
        Parameters
        ----------
        params : 매개변수 딕셔너리 {key: value}
        grads : 기울기 딕셔너리 {key: gradient}
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key] 


class Momentum:
    """
    모멘텀 SGD (Momentum Stochastic Gradient Descent)
    
    물리의 관성처럼 이전 업데이트 방향을 유지하여
    진동을 줄이고 더 빠르게 수렴하도록 함.
    
    수식:
        v = momentum * v - lr * grads
        params += v
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Momentum_(optimization)
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr  # 학습률
        self.momentum = momentum  # 모멘텀 계수 (기본값 0.9)
        self.v = None  # 속도 벡터 (관성)
        
    def update(self, params, grads):
        """
        가중치 업데이트 (모멘텀 적용)
        
        Parameters
        ----------
        params : 매개변수 딕셔너리
        grads : 기울기 딕셔너리
        """
        # 속도 벡터 초기화
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                 
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            # 모멘텀 업데이트: 이전 속도에 모멘텀 계수 곱하고 기울기 방향 힘 적용
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            # 속도만큼 가중치 업데이트
            params[key] += self.v[key]


class Nesterov:
    """
    Nesterov 가속 경사 (Nesterov's Accelerated Gradient)
    
    기존 모멘텀 방법을 개선한 알고리즘으로,
    실제 다음 위치에서의 기울기를 추정하여 더 정확한 업데이트 수행.
    
    수식:
        v = momentum * v - lr * grads
        params += momentum * momentum * v - lr * (1 + momentum) * grads
    
    Reference
    ---------
    http://arxiv.org/abs/1212.0901
    """
    # NAG는 모멘텀에서 한 단계 발전한 방법이다.

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        """
        가중치 업데이트 (Nesterov 알고리즘)
        
        Parameters
        ----------
        params : 매개변수 딕셔너리
        grads : 기울기 딕셔너리
        """
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 모멘텀 업데이트
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            # Nesterov 보정 적용
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    """
    AdaGrad (Adaptive Gradient)
    
    매개변수마다 적응적 학습률을 제공하며,
    자주 업데이트되는 매개변수의 학습률을 점차 감소시킴.
    희소 데이터(sparse data)에 효과적.
    
    수식:
        h += grads^2
        params -= lr * grads / (sqrt(h) + epsilon)
    
    Reference
    ---------
    http://www.jmlr.org/papers/v15/duchi11a.html
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None  # 기울기의 제곱의 누적
        
    def update(self, params, grads):
        """
        가중치 업데이트 (AdaGrad 알고리즘)
        
        Parameters
        ----------
        params : 매개변수 딕셔너리
        grads : 기울기 딕셔너리
        """
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 기울기의 제곱 누적
            self.h[key] += grads[key] * grads[key]
            # 적응적 학습률 적용 (자주 업데이트된 가중치는 학습률 감소)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation)
    
    AdaGrad의 학습률 과도 감소 문제를 개선한 알고리즘.
    기울기의 제곱에 지수 가중 평균을 적용하여
    오래된 정보를 서서히 잊어버림.
    
    수식:
        h = decay_rate * h + (1 - decay_rate) * grads^2
        params -= lr * grads / (sqrt(h) + epsilon)
    
    Reference
    ---------
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate  # 기울기 제곱의 가중 평균 감쇠 계수
        self.h = None
        
    def update(self, params, grads):
        """
        가중치 업데이트 (RMSprop 알고리즘)
        
        Parameters
        ----------
        params : 매개변수 딕셔너리
        grads : 기울기 딕셔너리
        """
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            # 기울기 제곱의 지수 가중 평균 업데이트
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            # 적응적 학습률 적용
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """
    Adam (Adaptive Moment Estimation)
    
    Momentum (첫 번째 모멘트)와 AdaGrad/RMSprop (두 번째 모멘트)를 결합한
    최적화 알고리즘. 현재 가장 널리 사용되는 최적화 방법.
    
    수식:
        m = beta1 * m + (1 - beta1) * grads  (기울기의 지수 가중 평균)
        v = beta2 * v + (1 - beta2) * grads^2 (기울기 제곱의 지수 가중 평균)
        m_hat = m / (1 - beta1^t)  (바이어스 보정)
        v_hat = v / (1 - beta2^t)  (바이어스 보정)
        params -= lr * m_hat / (sqrt(v_hat) + epsilon)
    
    Reference
    ---------
    http://arxiv.org/abs/1412.6980v8
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1  # 첫 번째 모멘트 지수 가중 평균 계수
        self.beta2 = beta2  # 두 번째 모멘트 지수 가중 평균 계수
        self.iter = 0  # 반복 횟수 (바이어스 보정용)
        self.m = None  # 첫 번째 모멘트 (기울기 평균)
        self.v = None  # 두 번째 모멘트 (기울기 제곱 평균)
        
    def update(self, params, grads):
        """
        가중치 업데이트 (Adam 알고리즘)
        
        Parameters
        ----------
        params : 매개변수 딕셔너리
        grads : 기울기 딕셔너리
        """
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        # 바이어스 보정 적용 학습률
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            # 첫 번째 모멘트 업데이트 (기울기 평균)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # 두 번째 모멘트 업데이트 (기울기 제곱 평균)
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            # 가중치 업데이트
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
