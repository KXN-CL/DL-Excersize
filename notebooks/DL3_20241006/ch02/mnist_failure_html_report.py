# coding: utf-8
"""
MNIST 실패 사례 HTML 리포트 생성 스크립트

이 스크립트는 MNIST 테스트에서 잘못 분류된 이미지들을
인터랙티브한 HTML 리포트로 시각화합니다.

주요 기능:
1. 모든 실패 사례를 이미지 + 메타데이터와 함께 표시
2. 클래스별 탭 필터링
3. 확신도별 정렬
4. 검색 기능 (글로벌 인덱스, 실제 레이블, 예측 레이블)
5. 혼동 행렬 인터랙티브 차트
6. 개별 실패 사례 상세 모달
"""

import sys
import os
import time
import pickle
import numpy as np
import json
import base64
from io import BytesIO
from collections import defaultdict

# PATH 설정
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

# ============================================================
# 설정
# ============================================================
OUTPUT_DIR = os.path.join(current_dir, "failure_analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 데이터 로드 및 분석 함수
# ============================================================
def get_data():
    """MNIST 테스트 데이터를 로드합니다."""
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test

def init_network():
    """사전 학습된 가중치를 로드합니다."""
    weight_path = os.path.join(current_dir, "sample_weight.pkl")
    with open(weight_path, "rb") as f:
        network = pickle.load(f)
    return network

def predict_batch(network, x_batch):
    """배치 단위로 예측을 수행합니다."""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = np.dot(x_batch, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

def image_to_base64(img_array):
    """numpy 배열을 base64 인코딩된 PNG 이미지로 변환합니다."""
    from PIL import Image
    img = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img, mode='L')
    
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def collect_all_data():
    """모든 데이터 수집 및 분석을 수행합니다."""
    print("데이터 로드 중...")
    x_test, t_test = get_data()
    
    print("신경망 초기화 중...")
    network = init_network()
    
    total = len(x_test)
    batch_size = 1000
    num_batches = (total + batch_size - 1) // batch_size
    
    all_predictions = []
    all_confidences = []
    all_second_confidences = []
    all_probabilities = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_x = x_test[start:end]
        y_batch = predict_batch(network, batch_x)
        
        preds = np.argmax(y_batch, axis=1)
        confs = np.max(y_batch, axis=1)
        
        second_confs = []
        for i in range(len(preds)):
            confs_copy = y_batch[i].copy()
            confs_copy[preds[i]] = 0
            second_confs.append(np.max(confs_copy))
        
        all_predictions.append(preds)
        all_confidences.append(confs)
        all_second_confidences.append(np.array(second_confs))
        all_probabilities.append(y_batch)
    
    all_predictions = np.concatenate(all_predictions)
    all_confidences = np.concatenate(all_confidences)
    all_second_confidences = np.concatenate(all_second_confidences)
    all_probabilities = np.concatenate(all_probabilities)
    
    correct_mask = all_predictions == t_test
    failure_indices = np.where(~correct_mask)[0]
    
    print(f"총 {len(failure_indices)}개의 실패 사례 발견")
    
    failures = []
    for idx in failure_indices:
        failures.append({
            "global_index": int(idx),
            "true_label": int(t_test[idx]),
            "predicted_label": int(all_predictions[idx]),
            "confidence": float(all_confidences[idx]),
            "second_confidence": float(all_second_confidences[idx]),
            "probabilities": all_probabilities[idx].tolist(),
            "image_base64": image_to_base64(x_test[idx])
        })
    
    accuracy = np.mean(all_predictions == t_test)
    
    class_stats = {}
    for digit in range(10):
        mask = t_test == digit
        total_class = np.sum(mask)
        correct_class = np.sum((all_predictions == t_test) & mask)
        fail_count = total_class - correct_class
        class_stats[str(digit)] = {
            "total": int(total_class),
            "correct": int(correct_class),
            "failures": int(fail_count),
            "accuracy": float(correct_class / total_class) if total_class > 0 else 1.0
        }
    
    confusion = np.zeros((10, 10), dtype=int)
    for i in range(total):
        confusion[t_test[i], all_predictions[i]] += 1
    
    stats = {
        "total": int(total),
        "accuracy": float(accuracy),
        "correct": int(accuracy * total),
        "failures": int(len(failure_indices)),
        "class_stats": class_stats,
        "confusion_matrix": confusion.tolist()
    }
    
    return failures, stats

# ============================================================
# HTML 리포트 생성
# ============================================================
def generate_html_report(failures, stats):
    """인터랙티브 HTML 리포트를 생성합니다."""
    
    by_class = defaultdict(list)
    for f in failures:
        by_class[f["true_label"]].append(f)
    
    confusion = stats["confusion_matrix"]
    
    # 탭 버튼 HTML 생성
    tab_buttons = '<button class="tab-btn active" data-class="all">전체 ({} )</button>'.format(len(failures))
    for d in range(10):
        count = len(by_class.get(d, []))
        tab_buttons += '<button class="tab-btn" data-class="{}">클래스 {} ({} )</button>'.format(d, d, count)
    
    # 열 헤더 HTML
    th_headers = ""
    for d in range(10):
        th_headers += '<th>{}</th>'.format(d)
    
    # 혼동 행렬 tbody HTML 생성
    confusion_table_body = ""
    for d in range(10):
        row_cells = ""
        for j in range(10):
            cls = ' class="highlight"' if (confusion[d][j] > 0 and d != j) else ""
            row_cells += '<td{}>{}</td>'.format(cls, confusion[d][j])
        confusion_table_body += '<tr><th>{}</th>{}</tr>\n'.format(d, row_cells)
    
    # 클래스별 통계 HTML 생성
    class_stats_html = ""
    for d in range(10):
        cs = stats["class_stats"][str(d)]
        acc_pct = cs["accuracy"] * 100
        class_stats_html += '''
            <div class="class-stat-card">
                <span class="digit">{}</span>
                <div class="detail">
                    <span>정확도: {:.1f}%</span>
                    <span>실패: {}</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" style="width: {:.1f}%"></div>
                </div>
            </div>'''.format(d, acc_pct, cs["failures"], acc_pct)
    
    # JSON 데이터
    json_data = json.dumps(failures, ensure_ascii=False)
    
    # HTML 파일에 직접 쓰기
    html_path = os.path.join(OUTPUT_DIR, "failure_report.html")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST 실패 사례 분석 리포트</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 30px;
            text-align: center;
            border-bottom: 3px solid #e94560;
        }
        
        .header h1 {
            font-size: 2.5em;
            color: #e94560;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            color: #a0a0a0;
        }
        
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .stat-card {
            background: #16213e;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid #0f3460;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(233, 69, 96, 0.2);
        }
        
        .stat-card .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #e94560;
        }
        
        .stat-card .label {
            font-size: 0.9em;
            color: #a0a0a0;
            margin-top: 5px;
        }
        
        .controls {
            padding: 20px 30px;
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .tab-container {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        
        .tab-btn {
            background: #16213e;
            border: 1px solid #0f3460;
            color: #e0e0e0;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        .tab-btn:hover {
            background: #0f3460;
        }
        
        .tab-btn.active {
            background: #e94560;
            border-color: #e94560;
            color: white;
        }
        
        .sort-btn {
            background: #16213e;
            border: 1px solid #0f3460;
            color: #e0e0e0;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        
        .sort-btn:hover {
            background: #0f3460;
        }
        
        .sort-btn.active {
            background: #533483;
            border-color: #533483;
            color: white;
        }
        
        .search-box {
            background: #16213e;
            border: 1px solid #0f3460;
            color: #e0e0e0;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 1em;
            width: 200px;
        }
        
        .search-box::placeholder {
            color: #666;
        }
        
        .content {
            padding: 20px 30px 50px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .section-title {
            font-size: 1.5em;
            color: #e94560;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #0f3460;
        }
        
        .failures-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .failure-card {
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #0f3460;
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }
        
        .failure-card:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3);
        }
        
        .failure-card img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: contain;
            background: #000;
        }
        
        .failure-card .info {
            padding: 12px;
        }
        
        .failure-card .index {
            font-size: 0.75em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .failure-card .labels {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .failure-card .labels .true-label {
            color: #4ecca3;
        }
        
        .failure-card .labels .arrow {
            color: #e94560;
            margin: 0 5px;
        }
        
        .failure-card .labels .pred-label {
            color: #e94560;
        }
        
        .failure-card .confidence {
            font-size: 0.85em;
            color: #a0a0a0;
        }
        
        .confidence-bar {
            height: 4px;
            background: #0f3460;
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s;
        }
        
        .confidence-low { background: #e94560; }
        .confidence-medium { background: #ffa500; }
        .confidence-high { background: #4ecca3; }
        
        .confusion-container {
            background: #16213e;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            border: 1px solid #0f3460;
            overflow-x: auto;
        }
        
        .confusion-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .confusion-table th, .confusion-table td {
            padding: 10px;
            text-align: center;
            border: 1px solid #0f3460;
            min-width: 40px;
        }
        
        .confusion-table th {
            background: #0f3460;
            color: #e94560;
        }
        
        .confusion-table td {
            background: #1a1a2e;
        }
        
        .confusion-table .highlight {
            background: rgba(233, 69, 96, 0.3);
            font-weight: bold;
            color: #e94560;
        }
        
        .class-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 40px;
        }
        
        .class-stat-card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        
        .class-stat-card .digit {
            font-size: 2em;
            font-weight: bold;
            color: #e94560;
            display: inline-block;
            width: 50px;
            text-align: center;
        }
        
        .class-stat-card .detail {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .class-stat-card .bar {
            height: 8px;
            background: #0f3460;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .class-stat-card .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ecca3, #e94560);
            border-radius: 4px;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: #16213e;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 90%;
            border: 2px solid #e94560;
            position: relative;
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .modal-close {
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 2em;
            color: #e94560;
            cursor: pointer;
            background: none;
            border: none;
        }
        
        .modal-image {
            width: 200px;
            height: 200px;
            object-fit: contain;
            background: #000;
            border-radius: 10px;
            margin: 20px auto;
            display: block;
        }
        
        .modal-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .modal-detail-item {
            background: #0f3460;
            padding: 15px;
            border-radius: 10px;
        }
        
        .modal-detail-item .label {
            font-size: 0.8em;
            color: #a0a0a0;
        }
        
        .modal-detail-item .value {
            font-size: 1.3em;
            font-weight: bold;
            color: #e94560;
        }
        
        .probability-bar-container {
            margin-top: 20px;
        }
        
        .probability-bar {
            display: flex;
            height: 30px;
            margin-bottom: 5px;
        }
        
        .probability-segment {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7em;
            color: white;
            transition: width 0.3s;
        }
        
        .probability-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: #a0a0a0;
        }
        
        .no-results {
            text-align: center;
            padding: 50px;
            color: #666;
            font-size: 1.2em;
        }
        
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.75em;
            font-weight: bold;
        }
        
        .badge-danger { background: #e94560; color: white; }
        .badge-warning { background: #ffa500; color: white; }
        .badge-success { background: #4ecca3; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>MNIST 실패 사례 분석 리포트</h1>
        <div class="subtitle">신경망이 잘못 분류한 이미지들의 상세 분석</div>
    </div>
    
    <div class="stats-container">
        <div class="stat-card">
            <div class="value">[TOTAL_IMGS]</div>
            <div class="label">총 테스트 이미지</div>
        </div>
        <div class="stat-card">
            <div class="value">[ACC_PCT]</div>
            <div class="label">전체 정확도</div>
        </div>
        <div class="stat-card">
            <div class="value">[CORRECT_CNT]</div>
            <div class="label">정답 개수</div>
        </div>
        <div class="stat-card">
            <div class="value" style="color: #e94560;">[FAIL_CNT]</div>
            <div class="label">실패 사례</div>
        </div>
        <div class="stat-card">
            <div class="value">[FAIL_RATE]</div>
            <div class="label">실패율</div>
        </div>
    </div>
    
    <div class="controls">
        <div class="tab-container" id="classTabs">
            [TAB_BUTTONS]
        </div>
        <div>
            <button class="sort-btn active" data-sort="index">인덱스순</button>
            <button class="sort-btn" data-sort="confidence_asc">확신도 낮은순</button>
            <button class="sort-btn" data-sort="confidence_desc">확신도 높은순</button>
        </div>
        <input type="text" class="search-box" id="searchBox" placeholder="검색 (인덱스, 레이블)...">
    </div>
    
    <div class="content">
        <h2 class="section-title">클래스별 정확도</h2>
        <div class="class-stats-grid">
            [CLASS_STATS_HTML]
        </div>
        
        <h2 class="section-title">혼동 행렬 (전체 카운트)</h2>
        <div class="confusion-container">
            <table class="confusion-table" id="confusionTable">
                <thead>
                    <tr>
                        <th>실제 \\ 예측</th>
                        [TH_HEADERS]
                    </tr>
                </thead>
                <tbody>
[CONFUSION_BODY]                </tbody>
            </table>
        </div>
        
        <h2 class="section-title">실패 사례</h2>
        <div class="failures-grid" id="failuresGrid"></div>
        <div class="no-results" id="noResults" style="display: none;">
            조건에 맞는 실패 사례가 없습니다.
        </div>
    </div>
    
    <div class="modal" id="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <h2 style="color: #e94560; margin-bottom: 10px;">실패 사례 상세</h2>
            <img class="modal-image" id="modalImage" src="" alt="Failure Image">
            <div class="modal-details">
                <div class="modal-detail-item">
                    <div class="label">글로벌 인덱스</div>
                    <div class="value" id="modalIndex">-</div>
                </div>
                <div class="modal-detail-item">
                    <div class="label">실제 레이블</div>
                    <div class="value" id="modalTrue" style="color: #4ecca3;">-</div>
                </div>
                <div class="modal-detail-item">
                    <div class="label">예측 레이블</div>
                    <div class="value" id="modalPred" style="color: #e94560;">-</div>
                </div>
                <div class="modal-detail-item">
                    <div class="label">확신도</div>
                    <div class="value" id="modalConf">-</div>
                </div>
            </div>
            <div class="probability-bar-container">
                <h3 style="margin-top: 20px; color: #e94560;">예측 확률 분포</h3>
                <div class="probability-bar" id="modalProbBar"></div>
                <div class="probability-labels" id="modalProbLabels"></div>
            </div>
        </div>
    </div>
    
    <script>
        const failures = [JSON_DATA];
        
        let currentFilter = 'all';
        let currentSort = 'index';
        let searchTerm = '';
        
        function getConfidenceClass(conf) {
            if (conf < 0.5) return 'confidence-low';
            if (conf < 0.8) return 'confidence-medium';
            return 'confidence-high';
        }
        
        function getConfidenceBadge(conf) {
            if (conf < 0.5) return '<span class="badge badge-danger">낮음</span>';
            if (conf < 0.8) return '<span class="badge badge-warning">중간</span>';
            return '<span class="badge badge-success">높음</span>';
        }
        
        function getFilteredFailures() {
            let filtered = [...failures];
            
            if (currentFilter !== 'all') {
                filtered = filtered.filter(f => f.true_label === parseInt(currentFilter));
            }
            
            if (searchTerm) {
                const term = parseInt(searchTerm);
                if (!isNaN(term)) {
                    filtered = filtered.filter(f => 
                        f.global_index === term || 
                        f.true_label === term || 
                        f.predicted_label === term
                    );
                }
            }
            
            switch (currentSort) {
                case 'index':
                    filtered.sort((a, b) => a.global_index - b.global_index);
                    break;
                case 'confidence_asc':
                    filtered.sort((a, b) => a.confidence - b.confidence);
                    break;
                case 'confidence_desc':
                    filtered.sort((a, b) => b.confidence - a.confidence);
                    break;
            }
            
            return filtered;
        }
        
        function renderGrid() {
            const grid = document.getElementById('failuresGrid');
            const noResults = document.getElementById('noResults');
            const filtered = getFilteredFailures();
            
            if (filtered.length === 0) {
                grid.innerHTML = '';
                noResults.style.display = 'block';
                return;
            }
            
            noResults.style.display = 'none';
            
            grid.innerHTML = filtered.map(function(f) {
                const confClass = getConfidenceClass(f.confidence);
                const badge = getConfidenceBadge(f.confidence);
                const card = document.createElement('div');
                card.className = 'failure-card';
                card.onclick = function() { openModal(f); };
                
                card.innerHTML = '<img src="' + f.image_base64 + '" alt="Digit ' + f.true_label + '">' +
                    '<div class="info">' +
                        '<div class="index">#' + f.global_index + ' ' + badge + '</div>' +
                        '<div class="labels">' +
                            '<span class="true-label">' + f.true_label + '</span>' +
                            '<span class="arrow">→</span>' +
                            '<span class="pred-label">' + f.predicted_label + '</span>' +
                        '</div>' +
                        '<div class="confidence">확신도: ' + f.confidence.toFixed(4) + '</div>' +
                        '<div class="confidence-bar">' +
                            '<div class="confidence-fill ' + confClass + '" style="width: ' + (f.confidence * 100) + '%"></div>' +
                        '</div>' +
                    '</div>';
                
                return card;
            }).reduce(function(arr, card) {
                arr.appendChild(card);
                return arr;
            }, grid);
        }
        
        document.querySelectorAll('#classTabs .tab-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                document.querySelectorAll('#classTabs .tab-btn').forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentFilter = btn.dataset.class;
                renderGrid();
            });
        });
        
        document.querySelectorAll('.sort-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.sort-btn').forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentSort = btn.dataset.sort;
                renderGrid();
            });
        });
        
        document.getElementById('searchBox').addEventListener('input', function(e) {
            searchTerm = e.target.value;
            renderGrid();
        });
        
        function openModal(data) {
            document.getElementById('modalImage').src = data.image_base64;
            document.getElementById('modalIndex').textContent = data.global_index;
            document.getElementById('modalTrue').textContent = data.true_label;
            document.getElementById('modalPred').textContent = data.predicted_label;
            document.getElementById('modalConf').textContent = data.confidence.toFixed(4);
            
            const probBar = document.getElementById('modalProbBar');
            const probLabels = document.getElementById('modalProbLabels');
            
            const colors = ['#e94560', '#4ecca3', '#533483', '#ffa500', '#0f3460',
                          '#e94560', '#4ecca3', '#533483', '#ffa500', '#0f3460'];
            
            probBar.innerHTML = data.probabilities.map(function(p, i) {
                const width = Math.max(p * 100, 2);
                const bgColor = i === data.predicted_label ? '#e94560' :
                               i === data.true_label ? '#4ecca3' : colors[i];
                const label = p > 0.05 ? p.toFixed(2) : '';
                return '<div class="probability-segment" style="width: ' + width + '%; background: ' + bgColor + '">' + label + '</div>';
            }).join('');
            
            probLabels.innerHTML = data.probabilities.map(function(p, i) { return '<span>' + i + '</span>'; }).join('');
            
            document.getElementById('modal').classList.add('active');
        }
        
        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }
        
        document.getElementById('modal').addEventListener('click', function(e) {
            if (e.target === document.getElementById('modal')) {
                closeModal();
            }
        });
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
        
        renderGrid();
    </script>
</body>
</html>''')
    
    # Placeholders 교체
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    total_imgs = "{:,}".format(stats['total'])
    acc_pct = "{:.2f}%".format(stats['accuracy'] * 100)
    correct_cnt = "{:,}".format(stats['correct'])
    fail_cnt = "{:,}".format(stats['failures'])
    fail_rate = "{:.2f}%".format(stats['failures'] / stats['total'] * 100)
    
    html = html.replace('[TOTAL_IMGS]', total_imgs)
    html = html.replace('[ACC_PCT]', acc_pct)
    html = html.replace('[CORRECT_CNT]', correct_cnt)
    html = html.replace('[FAIL_CNT]', fail_cnt)
    html = html.replace('[FAIL_RATE]', fail_rate)
    html = html.replace('[TAB_BUTTONS]', tab_buttons)
    html = html.replace('[TH_HEADERS]', th_headers)
    html = html.replace('[CONFUSION_BODY]', confusion_table_body)
    html = html.replace('[CLASS_STATS_HTML]', class_stats_html)
    html = html.replace('[JSON_DATA]', json_data)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return html_path

# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("  MNIST 실패 사례 HTML 리포트 생성")
    print("=" * 60)
    print()
    
    print("[1/2] 데이터 분석 중...")
    failures, stats = collect_all_data()
    
    print(f"  - 총 {stats['total']}개 이미지 분석")
    print(f"  - 정확도: {stats['accuracy']*100:.2f}%")
    print(f"  - 실패 사례: {stats['failures']}개")
    
    print("[2/2] HTML 리포트 생성 중...")
    html_path = generate_html_report(failures, stats)
    
    print()
    print("=" * 60)
    print(f"[완료] HTML 리포트가 생성되었습니다:")
    print(f"  {html_path}")
    print()
    print("브라우저에서 이 파일을 열어 인터랙티브하게 확인하세요.")
    print("=" * 60)

if __name__ == "__main__":
    main()
