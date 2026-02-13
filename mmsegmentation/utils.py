import random
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
import os
from PIL import  Image
from torch.utils.data import Dataset
import json
import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

def seed_all(seed):
    #  Set fixed seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True









def sample_action(actions_mean, action_std, attack_pixels):
    #  Function to sample actions
    if not isinstance(action_std, torch.Tensor):
        cov_mat = torch.eye(attack_pixels) * action_std**2
    else:
        cov_mat = torch.diag_embed(action_std.pow(2))
    distribution = dist.MultivariateNormal(actions_mean, cov_mat)
    if attack_pixels == 1:
        actions = distribution.sample()
    else:
        actions = distribution.sample_n(attack_pixels)
    

    actions_logprob = distribution.log_prob(actions)


    return actions, actions_logprob







def early_stopping(metric_value: float, patience_counter: int, 
                  min_improvement: float = 10.0, max_patience: int = 5) -> Tuple[int, bool]:
    """
    조기 종료(early stopping) 조건을 확인하는 함수입니다.
    
    Args:
        metric_value (float): 현재 에폭의 성능 지표 값 (예: loss, accuracy)
        patience_counter (int): 성능이 개선되지 않은 연속된 에폭 수
        min_improvement (float): 성능 개선으로 간주할 최소 임계값 (기본값: 10.0)
        max_patience (int): 조기 종료 전 허용할 최대 연속 실패 횟수 (기본값: 5)
                           0으로 설정하면 성능 개선 여부와 관계없이 즉시 종료
        
    Returns:
        Tuple[int, bool]: 
            - 업데이트된 patience_counter
            - 조기 종료 여부 (True면 학습 중단)
    """
    should_stop = False
    
    # max_patience가 0이면 즉시 종료
    if max_patience == 0:
        should_stop = True
        return patience_counter, should_stop
    
    # 성능 개선이 충분하지 않은 경우
    if metric_value <= min_improvement:
        patience_counter += 1
        
        # 허용된 최대 실패 횟수를 초과한 경우
        if patience_counter >= max_patience:
            should_stop = True
    else:
        # 성능이 개선된 경우 카운터 초기화
        patience_counter = 0
    
    return patience_counter, should_stop







def update(np_input1, np_input2, np_indices, all_same_shape):
    """
    이미지를 하나씩 처리하여 크기가 다른 이미지들도 처리할 수 있도록 수정된 update 함수입니다.
    all_same_shape이 True인 경우 GPU 배치 처리를 사용합니다.
    
    Args:
        np_input1: NumPy 배열 또는 NumPy 배열 리스트
        np_input2: NumPy 배열 또는 NumPy 배열 리스트
        np_indices: NumPy 배열 또는 NumPy 배열 리스트 (값이 0이면 np_input1의 값을 선택)
        all_same_shape: 모든 입력이 같은 크기인지 여부
    
    Returns:
        NumPy 배열 또는 NumPy 배열 리스트: 업데이트된 결과
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 입력이 리스트인지 확인
    is_list_input = isinstance(np_input1, list)
    
    if not is_list_input:
        return np.where(np_indices == 0, np_input1, np_input2)
    
    if all_same_shape and is_list_input:
        # GPU 배치 처리
        # 배치 텐서로 변환
        input1_tensor = torch.from_numpy(np.stack(np_input1)).to(device)
        input2_tensor = torch.from_numpy(np.stack(np_input2)).to(device)
        
        # indices를 적절한 shape으로 변환 (더 효율적인 방식)
        if np_indices[0].ndim == 0:  # 스칼라인 경우
            indices_tensor = torch.tensor(np_indices, device=device).view(-1, *([1] * (input1_tensor.ndim - 1)))
        else:
            # 모든 인덱스를 한 번에 스택
            indices_tensor = torch.from_numpy(np.stack([
                np.broadcast_to(idx, input1_tensor.shape[1:])
                for idx in np_indices
            ])).to(device)
        
        # 배치로 한번에 처리
        result_tensor = torch.where(indices_tensor == 0, input1_tensor, input2_tensor)
        
        # NumPy 배열 리스트로 변환하여 반환
        return list(result_tensor.cpu().numpy())
    
    else:
        # 기존 코드: 개별 처리
        results = []
        for i in range(len(np_input1)):
            # 각 이미지에 대해 개별적으로 처리
            input1 = np_input1[i]
            input2 = np_input2[i]
            idx = np_indices[i]
            
            # idx를 input1과 동일한 shape로 브로드캐스팅
            if idx.ndim == 0:  # 스칼라인 경우
                idx = np.full_like(input1, idx)
            elif idx.ndim < input1.ndim:
                # 필요한 차원만큼 확장
                for _ in range(input1.ndim - idx.ndim):
                    idx = np.expand_dims(idx, -1)
                idx = np.broadcast_to(idx, input1.shape)
            
            # 조건에 따라 input1 또는 input2 선택
            result = np.where(idx == 0, input1, input2)
            results.append(result)
        
        return results


#Visualization

# Cityscapes 19개 클래스에 대한 컬러 팔레트 (각 행이 [R, G, B])
cityscapes_palette = np.array([
    [128, 64,128],   # 0: road
    [244, 35,232],   # 1: sidewalk
    [ 70, 70, 70],   # 2: building
    [102,102,156],   # 3: wall
    [190,153,153],   # 4: fence
    [153,153,153],   # 5: pole
    [250,170, 30],   # 6: traffic light
    [220,220,  0],   # 7: traffic sign
    [107,142, 35],   # 8: vegetation
    [152,251,152],   # 9: terrain
    [ 70,130,180],   # 10: sky
    [220, 20, 60],   # 11: person
    [255,  0,  0],   # 12: rider
    [ 0,  0,142],    # 13: car
    [ 0,  0, 70],    # 14: truck
    [ 0, 60,100],    # 15: bus
    [ 0, 80,100],    # 16: train
    [ 0,  0,230],    # 17: motorcycle
    [119, 11, 32]    # 18: bicycle
])


def overlay_mask_on_image(image, mask, alpha=0.3):
    """
    원본 이미지 위에 segmentation mask를 alpha blending 방식으로 오버레이하여 반환합니다.
    """
    # 이미지가 torch.Tensor면 numpy array로 변환
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    # 두 이미지의 dtype이 다를 수 있으므로 float으로 변환 후 blending
    overlay = image_np.astype(np.float32) * (1 - alpha) + mask.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay    

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPy 배열을 리스트로 변환
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()  # 텐서를 리스트로 변환
        elif isinstance(obj, torch.device):
            return str(obj)  # device 객체를 문자열로 변환
        return super().default(obj)

def save_experiment_results(results, config, sweep_config=None, timestamp=None, save_dir="."):
    """
    config와 sweep_config, 그리고 실험 결과를 하나의 텍스트 파일에 저장합니다.
    
    Args:
        results (dict): 실험 결과를 담은 딕셔너리 (예: {"accuracy": 0.85, "loss": 0.35, ...})
        config (dict): 설정 정보 (예: config.py에서 불러온 config 딕셔너리)
        sweep_config (dict, optional): 스윕 설정 정보 (예: config.py에서 불러온 sweep_config 딕셔너리)
        save_dir (str): 파일을 저장할 디렉토리 경로 (기본값: 현재 디렉토리)
        
    파일명은 "experiment_results.txt"로 고정되어 있습니다.
    """
    # 날짜와 시간을 포함한 파일 이름 생성
    if timestamp == None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"experiment_results_{timestamp}.txt"
    file_path = os.path.join(save_dir, file_name)
    
    lines = []
    lines.append("Experiment Results")
    lines.append("=" * 40)
    lines.append("Timestamp: " + str(datetime.datetime.now()))
    lines.append("\n[Configuration]")
    lines.append(json.dumps(config, indent=4, ensure_ascii=False, cls=CustomJSONEncoder))
    
    if sweep_config is not None:
        lines.append("\n[Sweep Configuration]")
        lines.append(json.dumps(sweep_config, indent=4, ensure_ascii=False, cls=CustomJSONEncoder))
    
    # Experimental Results 섹션 (각 항목은 한 줄에 key: value 형식으로 출력)
    lines.append("\n[Experimental Results]")
    
    # Extract query labels if available
    query_labels = results.get("Query Labels", None)
    
    for key, value in results.items():
        if key == "Query Labels":
            # Skip Query Labels in output - only used internally
            continue
        if ("Per-category IoU" in key or "Per-category Accuracy" in key) and isinstance(value, list) and len(value) > 0:
            # Per-category IoU/Accuracy의 경우 쿼리별로 줄바꿈하여 가독성 개선
            lines.append(f"{key}:")
            for i, query_result in enumerate(value):
                # Use actual query labels from results
                query_label = f"{query_labels[i]}query" if query_labels and i < len(query_labels) else f"checkpoint{i}"
                
                if isinstance(query_result, list):
                    # NaN 값들을 "NaN"으로 변환하여 보기 좋게 처리
                    formatted_values = []
                    for val in query_result:
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            formatted_values.append("NaN")
                        else:
                            formatted_values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
                    lines.append(f"  {query_label}: [{', '.join(formatted_values)}]")
                else:
                    lines.append(f"  {query_label}: {json.dumps(query_result, ensure_ascii=False, cls=CustomJSONEncoder)}")
        elif key == "Per Query Averages":
             lines.append(f"{key}:")
             lines.append(json.dumps(value, indent=4, ensure_ascii=False, cls=CustomJSONEncoder))
        else:
            # 다른 결과들은 기존 방식으로 저장
            line = f"{key}: " + json.dumps(value, ensure_ascii=False, cls=CustomJSONEncoder)
            lines.append(line)
    
    os.makedirs(save_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Results saved to {file_path}")

def convert_to_train_id(label_array):
    """
    Cityscapes의 원본 레이블을 학습에 사용되는 레이블로 변환합니다.
    """
    mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    return np.vectorize(lambda x: mapping.get(x, 255))(label_array)