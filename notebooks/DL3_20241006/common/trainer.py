# coding: utf-8
"""
신경망 학습(Training) 모듈

이 모듈은 신경망 학습을 수행하는 Trainer 클래스를 제공합니다.
미니배치 경사 하강법, 학습률 조정, 평가 등을 한 번에 관리합니다.

주요 기능:
- 미니배치 샘플링 및 학습 단계 진행
- 학습/테스트 정확도 및 손실 추적
- 다양한 최적화 알고리즘 지원 (SGD, Momentum, Adam 등)
"""
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.optimizer import *


class Trainer:
    """
    신경망 학습을 대신 수행하는 클래스
    
    미니배치 기반 경사 하강법 학습을 관리하고,
    학습 과정에서의 손실 값과 정확도를 추적합니다.
    
    Parameters
    ----------
    network : 학습할 신경망 객체 (MultiLayerNet 등)
    x_train : 훈련 입력 데이터
    t_train : 훈련 정답 레이블
    x_test : 테스트 입력 데이터
    t_test : 테스트 정답 레이블
    epochs : 학습 에폭 수 (기본값 20)
    mini_batch_size : 미니배치 크기 (기본값 100)
    optimizer : 최적화 알고리즘 (기본값 'SGD')
    optimizer_param : 최적화 알고리즘 파라미터 (예: {'lr': 0.01})
    evaluate_sample_num_per_epoch : 에폭당 평가 샘플 수 (None이면 전체 사용)
    verbose : 출력 정보 표시 여부 (기본값 True)
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # 최적화 알고리즘 선택 및 인스턴스 생성
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        # 학습 크기 계산
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)  # 최대 반복 횟수
        self.current_iter = 0  # 현재 반복 횟수
        self.current_epoch = 0  # 현재 에폭 수
        
        # 학습 기록 저장소
        self.train_loss_list = []  # 에폭별 훈련 손실
        self.train_acc_list = []   # 에폭별 훈련 정확도
        self.test_acc_list = []    # 에폭별 테스트 정확도

    def train_step(self):
        """
        한 단계(한 미니배치) 학습 수행
        
        1. 미니배치 샘플링
        2. 오차역전파법으로 기울기 계산
        3. 최적화 알고리즘으로 가중치 업데이트
        4. 손실 값 기록
        5. 에폭 끝에서 훈련/테스트 정확도 평가
        """
        # 훈련 데이터에서 무작위 미니배치 샘플링
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        # 오차역전파법으로 기울기 계산
        grads = self.network.gradient(x_batch, t_batch)
        # 최적화 알고리즘으로 가중치 업데이트
        self.optimizer.update(self.network.params, grads)
        
        # 손실 값 계산 및 기록
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        # 에폭 끝에서 정확도 평가
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            # 평가 샘플 수 제한이 있으면 적용
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            # 훈련 정확도 계산
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            # 테스트 정확도 계산
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        """
        전체 학습 수행
        
        max_iter만큼 train_step()을 반복하여 신경망을 학습.
        학습 완료 후 최종 테스트 정확도를 출력.
        """
        for i in range(self.max_iter):
            self.train_step()

        # 최종 테스트 정확도 계산
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
