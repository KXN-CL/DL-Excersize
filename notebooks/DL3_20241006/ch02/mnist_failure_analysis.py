# coding: utf-8
"""
MNIST 테스트 실패 사례 수집 및 가시화 자동화 스크립트

이 스크립트는 사전 학습된 MNIST 신경망을 사용하여 테스트 데이터에서
잘못 분류된 이미지(실패 사례)를 자동으로 수집하고, 인덱싱된 시각화 리포트를 생성합니다.

주요 기능:
1. 테스트 데이터 전체에 대한 추론 수행
2. 잘못 분류된 이미지 수집 (실패 사례)
3. 실패 사례에 대한 인덱싱 (클래스별, 확신도별)
4. 시각화 리포트 생성 (HTML + 이미지 파일)
5. 통계 분석 (클래스별 실패율, 확신도 분포 등)
"""

import sys
import os
import time
import pickle
import numpy as np
from collections import defaultdict
import json

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
# 1. 데이터 로드 및 신경망 초기화
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

def predict(network, x):
    """단일 이미지에 대해 예측을 수행합니다."""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

def predict_batch(network, x_batch):
    """배치 단위로 예측을 수행합니다 (성능 최적화)."""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = np.dot(x_batch, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

# ============================================================
# 2. 실패 사례 수집
# ============================================================
def collect_failure_cases(network, x_test, t_test, batch_size=1000):
    """
    테스트 데이터에서 잘못 분류된 이미지를 수집합니다.
    
    Parameters:
        network: 초기화된 신경망
        x_test: 테스트 이미지
        t_test: 테스트 레이블
        batch_size: 배치 크기
    
    Returns:
        failures: 실패 사례 목록
        stats: 통계 정보
    """
    total = len(x_test)
    print(f"총 {total}개 이미지 분석 시작...")
    print("-" * 60)
    
    start_time = time.time()
    
    # 실패 사례 저장
    failure_indices = []
    failure_images = []
    failure_labels = []
    failure_predictions = []
    failure_confidences = []
    failure_second_confidences = []
    
    # 배치 처리
    num_batches = (total + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_x = x_test[start:end]
        batch_t = t_test[start:end]
        
        # 배치 예측
        y_batch = predict_batch(network, batch_x)
        
        # 예측값 계산
        predictions = np.argmax(y_batch, axis=1)
        confidences = np.max(y_batch, axis=1)
        
        # 두 번째로 높은 확신도 계산
        second_confidences = []
        for i in range(len(predictions)):
            confs = y_batch[i]
            confs[predictions[i]] = 0
            second_confidences.append(np.max(confs))
        second_confidences = np.array(second_confidences)
        
        # 실패 사례 찾기
        incorrect_mask = predictions != batch_t
        batch_failures = np.where(incorrect_mask)[0]
        
        for idx in batch_failures:
            local_idx = idx
            global_idx = start + idx
            failure_indices.append(global_idx)
            failure_images.append(batch_x[idx])
            failure_labels.append(batch_t[idx])
            failure_predictions.append(predictions[idx])
            failure_confidences.append(confidences[idx])
            failure_second_confidences.append(second_confidences[idx])
        
        # 진행률 출력
        if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            processed = min((batch_idx + 1) * batch_size, total)
            print(f"\r진행률: {processed}/{total} ({processed/total*100:.1f}%) | "
                  f"현재 실패: {len(failure_indices)}개 | 경과: {elapsed:.1f}초", end='', flush=True)
    
    total_time = time.time() - start_time
    
    # 결과를 numpy array로 변환
    failure_indices = np.array(failure_indices)
    failure_images = np.array(failure_images)
    failure_labels = np.array(failure_labels)
    failure_predictions = np.array(failure_predictions)
    failure_confidences = np.array(failure_confidences)
    failure_second_confidences = np.array(failure_second_confidences)
    
    # 전체 정확도 계산
    all_predictions = []
    all_confidences = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_x = x_test[start:end]
        batch_t = t_test[start:end]
        y_batch = predict_batch(network, batch_x)
        preds = np.argmax(y_batch, axis=1)
        all_predictions.append(preds)
        all_confidences.append(np.max(y_batch, axis=1))
    
    all_predictions = np.concatenate(all_predictions)
    accuracy = np.mean(all_predictions == t_test)
    
    print(f"\n{'=' * 60}")
    print(f"분석 완료!")
    print(f"총 이미지: {total}")
    print(f"정확도: {accuracy*100:.2f}% ({int(accuracy*total)}/{total})")
    print(f"실패 사례: {len(failure_indices)}개 ({len(failure_indices)/total*100:.2f}%)")
    print(f"처리 시간: {total_time:.2f}초")
    print("-" * 60)
    
    stats = {
        "total": total,
        "accuracy": float(accuracy),
        "correct": int(accuracy * total),
        "failures": len(failure_indices),
        "processing_time": total_time
    }
    
    failures = {
        "indices": failure_indices,
        "images": failure_images,
        "labels": failure_labels,
        "predictions": failure_predictions,
        "confidences": failure_confidences,
        "second_confidences": failure_second_confidences
    }
    
    return failures, stats

# ============================================================
# 3. 인덱싱 및 통계 분석
# ============================================================
def analyze_failures(failures):
    """
    실패 사례에 대한 상세 분석을 수행합니다.
    
    Returns:
        analysis: 분석 결과 딕셔너리
    """
    labels = failures["labels"]
    predictions = failures["predictions"]
    confidences = failures["confidences"]
    
    analysis = {}
    
    # 1. 클래스별 실패 개수
    class_failure_count = defaultdict(list)
    for i in range(len(labels)):
        class_failure_count[int(labels[i])].append(i)
    analysis["class_failure_count"] = dict(class_failure_count)
    
    # 2. 클래스별 오분류 패턴 (어떤 숫자로 잘못 분류되었는지)
    confusion_patterns = defaultdict(lambda: defaultdict(int))
    for i in range(len(labels)):
        true_label = int(labels[i])
        pred_label = int(predictions[i])
        confusion_patterns[true_label][pred_label] += 1
    analysis["confusion_patterns"] = confusion_patterns
    
    # 3. 확신도별 실패 분포
    analysis["confidence_stats"] = {
        "mean": float(np.mean(confidences)),
        "median": float(np.median(confidences)),
        "std": float(np.std(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences))
    }
    
    # 4. 가장 낮은 확신도를 가진 실패 상위 10개
    lowest_conf_indices = np.argsort(confidences)[:10]
    analysis["lowest_confidence_failures"] = []
    for idx in lowest_conf_indices:
        analysis["lowest_confidence_failures"].append({
            "global_index": int(failures["indices"][idx]),
            "true_label": int(labels[idx]),
            "predicted_label": int(predictions[idx]),
            "confidence": float(confidences[idx])
        })
    
    # 5. 가장 높은 확신도로 잘못 예측한 사례 상위 10개
    highest_conf_indices = np.argsort(confidences)[-10:][::-1]
    analysis["highest_confidence_failures"] = []
    for idx in highest_conf_indices:
        analysis["highest_confidence_failures"].append({
            "global_index": int(failures["indices"][idx]),
            "true_label": int(labels[idx]),
            "predicted_label": int(predictions[idx]),
            "confidence": float(confidences[idx])
        })
    
    # 6. 클래스별 MNIST 데이터 분포 (참값)
    from dataset.mnist import load_mnist
    (_, t_train), (_, t_test) = load_mnist(normalize=False, flatten=False, one_hot_label=False)
    class_distribution = {}
    for digit in range(10):
        class_distribution[str(digit)] = {
            "train": int(np.sum(t_train == digit)),
            "test": int(np.sum(t_test == digit))
        }
    analysis["class_distribution"] = class_distribution
    
    return analysis

# ============================================================
# 4. 시각화 리포트 생성
# ============================================================
def create_visualization_report(failures, stats, analysis):
    """
    실패 사례에 대한 시각화 리포트를 생성합니다.
    
    생성되는 파일:
    - failure_summary.png: 전체 요약 이미지
    - failure_grid_{digit}.png: 클래스별 실패 그리드
    - confusion_matrix.png: 혼동 행렬
    - confidence_distribution.png: 확신도 분포
    - failure_report.json: JSON 통계 리포트
    """
    import matplotlib
    matplotlib.use('Agg')  # 백엔드 설정 (디스플레이 없음)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    labels = failures["labels"]
    predictions = failures["predictions"]
    confidences = failures["confidences"]
    images = failures["images"]
    indices = failures["indices"]
    
    # ----------------------------------------------------------
    # 1. 전체 요약 이미지
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1-1. 클래스별 실패 막대차트
    ax1 = fig.add_subplot(gs[0, :2])
    digits = range(10)
    failure_counts = [analysis["class_failure_count"].get(d, []) for d in digits]
    counts_only = [len(v) for v in failure_counts]
    total_per_class = [analysis["class_distribution"][str(d)]["test"] for d in digits]
    failure_rates = [(counts_per_class / total_per_class * 100) if total_per_class > 0 else 0 
                     for counts_per_class, total_per_class in zip(counts_only, total_per_class)]
    
    bars = ax1.bar(digits, counts_only, color='steelblue', label='Failure Count')
    ax1.set_ylabel('Failure Count', fontsize=12)
    ax1.set_title('Failure Count by True Label Digit', fontsize=14)
    ax1.set_xticks(digits)
    
    # 막대 위에 개수 표시
    for bar, count, rate in zip(bars, counts_only, failure_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({rate:.1f}%)', ha='center', va='bottom', fontsize=8)
    
    # 1-2. 확신도 분포 히스토그램
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(confidences, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution of Failures', fontsize=14)
    ax2.axvline(x=np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
    ax2.legend()
    
    # 1-3. 혼동 행렬
    ax3 = fig.add_subplot(gs[1:, :])
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(labels)):
        true_label = int(labels[i])
        pred_label = int(predictions[i])
        confusion_matrix[true_label][pred_label] += 1
    
    # 전체 테스트 데이터 기준 정규화
    test_per_class = np.array([analysis["class_distribution"][str(d)]["test"] for d in range(10)])
    confusion_normalized = confusion_matrix.astype(float) / test_per_class.reshape(-1, 1) * 100
    
    im = ax3.imshow(confusion_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=30)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_title('Confusion Matrix (Failure Rate %)', fontsize=14)
    ax3.set_xticks(range(10))
    ax3.set_yticks(range(10))
    
    # 셀에 값 표시
    for i in range(10):
        for j in range(10):
            if i == j:
                ax3.text(j, i, str(confusion_matrix[i][j]),
                        ha="center", va="center", color="black", fontsize=9)
            elif confusion_matrix[i][j] > 0:
                ax3.text(j, i, str(confusion_matrix[i][j]),
                        ha="center", va="center", color="white", fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'MNIST Failure Analysis Report\nAccuracy: {stats["accuracy"]*100:.2f}% | '
                 f'Total Failures: {stats["failures"]}', fontsize=16, fontweight='bold', y=0.98)
    
    summary_path = os.path.join(OUTPUT_DIR, "failure_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"요약 이미지 저장: {summary_path}")
    
    # ----------------------------------------------------------
    # 2. 클래스별 실패 그리드 이미지
    # ----------------------------------------------------------
    for digit in range(10):
        digit_failures = analysis["class_failure_count"].get(digit, [])
        if len(digit_failures) == 0:
            continue
        
        # 최대 20개까지 표시
        num_show = min(len(digit_failures), 20)
        show_indices = digit_failures[:num_show]
        
        n_cols = 5
        n_rows = (num_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, ax in enumerate(axes.flat):
            if i >= num_show:
                ax.axis('off')
                continue
            
            local_idx = show_indices[i]
            img = images[local_idx].reshape(28, 28)
            true_label = int(labels[local_idx])
            pred_label = int(predictions[local_idx])
            confidence = confidences[local_idx]
            global_idx = int(indices[local_idx])
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"#{global_idx}\nTrue: {true_label} → Pred: {pred_label}\nConf: {confidence:.3f}",
                        fontsize=9, color='red' if pred_label != true_label else 'green')
            ax.axis('off')
        
        fig.suptitle(f'Digit {digit} Failures ({len(digit_failures)} total)', fontsize=14, fontweight='bold')
        grid_path = os.path.join(OUTPUT_DIR, f"failure_grid_{digit}.png")
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"클래스 {digit} 그리드 저장: {grid_path}")
    
    # ----------------------------------------------------------
    # 3. 확신도 분포 상세 차트
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 3-1. 전체 확신도 분포
    axes[0].hist(confidences, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Confidence Distribution of All Failures', fontsize=13)
    axes[0].axvline(x=0.5, color='red', linestyle='--', label='50%')
    axes[0].axvline(x=np.mean(confidences), color='green', linestyle='--', 
                    label=f'Mean: {np.mean(confidences):.3f}')
    axes[0].legend()
    
    # 3-2. 클래스별 확신도 박스플롯
    digit_confidences = defaultdict(list)
    for i in range(len(labels)):
        digit_confidences[int(labels[i])].append(confidences[i])
    
    box_data = [digit_confidences[d] for d in range(10) if d in digit_confidences]
    box_labels = [str(d) for d in range(10) if d in digit_confidences]
    
    if box_data:
        bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(box_data)))):
            patch.set_facecolor(color)
        axes[1].set_xlabel('True Label Digit', fontsize=12)
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title('Confidence by True Label Digit', fontsize=13)
    
    # 3-3. 상위 20개 가장 낮은 확신도 실패
    lowest_20_idx = np.argsort(confidences)[:20]
    digits_low = [int(labels[i]) for i in lowest_20_idx]
    confs_low = [confidences[i] for i in lowest_20_idx]
    preds_low = [int(predictions[i]) for i in lowest_20_idx]
    
    colors_low = ['red' if d != p else 'green' for d, p in zip(digits_low, preds_low)]
    axes[2].bar(range(20), confs_low, color=colors_low, edgecolor='black')
    axes[2].set_xlabel('Failure Rank (lowest confidence)', fontsize=12)
    axes[2].set_ylabel('Confidence', fontsize=12)
    axes[2].set_title('Top 20 Lowest Confidence Failures', fontsize=13)
    axes[2].set_xticks(range(20))
    axes[2].set_xticklabels([f"#{int(indices[lowest_20_idx[i]])}" for i in range(20)], rotation=45, ha='right', fontsize=7)
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    conf_path = os.path.join(OUTPUT_DIR, "confidence_analysis.png")
    plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"확신도 분석 저장: {conf_path}")
    
    # ----------------------------------------------------------
    # 4. 대표 실패 사례 20개 시각화 (가장 확신이 낮은 것들)
    # ----------------------------------------------------------
    num_show = min(20, len(labels))
    lowest_20_global = np.argsort(confidences)[:num_show]
    
    n_cols = 5
    n_rows = (num_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ax in enumerate(axes.flat):
        if i >= num_show:
            ax.axis('off')
            continue
        
        local_idx = lowest_20_global[i]
        img = images[local_idx].reshape(28, 28)
        true_label = int(labels[local_idx])
        pred_label = int(predictions[local_idx])
        confidence = confidences[local_idx]
        global_idx = int(indices[local_idx])
        
        ax.imshow(img, cmap='gray')
        
        is_correct = pred_label == true_label
        title_color = 'green' if is_correct else 'red'
        ax.set_title(f"Global #\n{global_idx}\nTrue: {true_label}\nPred: {pred_label}\nConf: {confidence:.4f}",
                    fontsize=8, color=title_color)
        ax.axis('off')
    
    fig.suptitle(f'Representative Failures (Lowest Confidence) - Total: {len(labels)} failures',
                fontsize=14, fontweight='bold')
    rep_path = os.path.join(OUTPUT_DIR, "representative_failures.png")
    plt.savefig(rep_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"대표 실패 사례 저장: {rep_path}")

# ============================================================
# 5. JSON 리포트 생성
# ============================================================
def create_json_report(stats, analysis, failures, output_path):
    """통계 결과를 JSON 파일로 저장합니다."""
    
    # JSON 직렬화를 위해 타입 변환
    report = {
        "summary": {
            "total_images": stats["total"],
            "accuracy": stats["accuracy"],
            "correct_predictions": stats["correct"],
            "failure_count": stats["failures"],
            "failure_rate": stats["failures"] / stats["total"],
            "processing_time_seconds": stats["processing_time"]
        },
        "confidence_stats": analysis["confidence_stats"],
        "class_failure_distribution": {},
        "confusion_patterns": {},
        "lowest_confidence_failures": analysis["lowest_confidence_failures"],
        "highest_confidence_failures": analysis["highest_confidence_failures"],
        "class_distribution": analysis["class_distribution"]
    }
    
    # 클래스별 실패 분포
    for digit in range(10):
        digit_failures = analysis["class_failure_count"].get(digit, [])
        report["class_failure_distribution"][str(digit)] = {
            "failure_count": len(digit_failures),
            "failure_rate": len(digit_failures) / analysis["class_distribution"][str(digit)]["test"] 
                           if analysis["class_distribution"][str(digit)]["test"] > 0 else 0,
            "total_test_samples": analysis["class_distribution"][str(digit)]["test"]
        }
    
    # 혼동 패턴 변환
    for true_label, patterns in analysis["confusion_patterns"].items():
        report["confusion_patterns"][str(true_label)] = {
            str(k): v for k, v in patterns.items()
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"JSON 리포트 저장: {output_path}")

# ============================================================
# 6. 텍스트 리포트 생성
# ============================================================
def create_text_report(stats, analysis):
    """텍스트 형식의 리포트를 생성합니다."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("         MNIST 테스트 실패 사례 분석 리포트")
    lines.append("=" * 70)
    lines.append("")
    lines.append("【종합 통계】")
    lines.append(f"  총 테스트 이미지 수:    {stats['total']}")
    lines.append(f"  정확도:                 {stats['accuracy']*100:.2f}%")
    lines.append(f"  정답 개수:              {stats['correct']}")
    lines.append(f"  실패 개수:              {stats['failures']}")
    lines.append(f"  실패율:                 {stats['failures']/stats['total']*100:.2f}%")
    lines.append(f"  처리 시간:              {stats['processing_time']:.2f}초")
    lines.append("")
    
    lines.append("【확신도 통계】")
    conf_stats = analysis["confidence_stats"]
    lines.append(f"  평균 확신도:            {conf_stats['mean']:.4f}")
    lines.append(f"  중앙값 확신도:          {conf_stats['median']:.4f}")
    lines.append(f"  표준편차:               {conf_stats['std']:.4f}")
    lines.append(f"  최소 확신도:            {conf_stats['min']:.4f}")
    lines.append(f"  최대 확신도:            {conf_stats['max']:.4f}")
    lines.append("")
    
    lines.append("【클래스별 실패 분포】")
    lines.append(f"  {'Digit':<8} {'실패수':<8} {'테스트수':<8} {'실패율':<10}")
    lines.append("  " + "-" * 34)
    for digit in range(10):
        digit_str = str(digit)
        dist = analysis["class_distribution"][digit_str]
        test_count = dist["test"]
        failure_count = len(analysis["class_failure_count"].get(digit, []))
        failure_rate = failure_count / test_count * 100 if test_count > 0 else 0
        lines.append(f"  {digit_str:<8} {failure_count:<8} {test_count:<8} {failure_rate:<10.2f}%")
    lines.append("")
    
    lines.append("【혼동 패턴 (어떤 숫자로 잘못 분류되었는지)】")
    for true_label in sorted(analysis["confusion_patterns"].keys()):
        patterns = analysis["confusion_patterns"][true_label]
        lines.append(f"  실제 {true_label} → ", )
        pattern_str = ", ".join([f"{pred}({cnt})" for pred, cnt in sorted(patterns.items(), key=lambda x: -x[1])])
        lines.append(f"    {pattern_str}")
    lines.append("")
    
    lines.append("【가장 확신이 낮은 실패 사례 TOP 10】")
    lines.append(f"  {'Global#':<10} {'실제':<6} {'예측':<6} {'확신도':<10}")
    lines.append("  " + "-" * 32)
    for i, failure in enumerate(analysis["lowest_confidence_failures"]):
        lines.append(f"  {failure['global_index']:<10} {failure['true_label']:<6} "
                    f"{failure['predicted_label']:<6} {failure['confidence']:<10.4f}")
    lines.append("")
    
    lines.append("【가장 확신이 높은 실패 사례 TOP 10 (가장 위험한 실패)】")
    lines.append(f"  {'Global#':<10} {'실제':<6} {'예측':<6} {'확신도':<10}")
    lines.append("  " + "-" * 32)
    for i, failure in enumerate(analysis["highest_confidence_failures"]):
        lines.append(f"  {failure['global_index']:<10} {failure['true_label']:<6} "
                    f"{failure['predicted_label']:<6} {failure['confidence']:<10.4f}")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("생성된 파일:")
    lines.append(f"  - failure_summary.png (종합 요약)")
    lines.append(f"  - failure_grid_0.png ~ failure_grid_9.png (클래스별)")
    lines.append(f"  - confidence_analysis.png (확신도 분석)")
    lines.append(f"  - representative_failures.png (대표 실패 20개)")
    lines.append(f"  - failure_report.json (JSON 리포트)")
    lines.append("=" * 70)
    
    report_text = "\n".join(lines)
    
    # 파일 저장
    text_path = os.path.join(OUTPUT_DIR, "failure_report.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 콘솔 출력
    print("\n" + report_text)
    
    return text_path

# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("  MNIST 테스트 실패 사례 수집 및 분석 자동화")
    print("=" * 60)
    print()
    
    # 1. 데이터 로드
    print("[1/5] 데이터 로드 중...")
    x_test, t_test = get_data()
    print(f"  테스트 데이터: {len(x_test)}개 이미지")
    
    # 2. 신경망 초기화
    print("[2/5] 신경망 초기화 중...")
    network = init_network()
    print(f"  가중치 로드 완료")
    for key in ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']:
        print(f"    {key}: {network[key].shape}")
    
    # 3. 실패 사례 수집
    print("[3/5] 실패 사례 수집 중...")
    failures, stats = collect_failure_cases(network, x_test, t_test)
    
    if stats["failures"] == 0:
        print("\n경고: 실패 사례가 없습니다! (100% 정확도)")
        return
    
    # 4. 분석
    print("[4/5] 실패 사례 분석 중...")
    analysis = analyze_failures(failures)
    
    # 5. 리포트 생성
    print("[5/5] 리포트 생성 중...")
    create_visualization_report(failures, stats, analysis)
    
    json_path = os.path.join(OUTPUT_DIR, "failure_report.json")
    create_json_report(stats, analysis, failures, json_path)
    
    text_path = create_text_report(stats, analysis)
    
    print(f"\n\n[완료] 모든 리포트가 다음 폴더에 저장되었습니다:")
    print(f"  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
