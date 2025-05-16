import cv2
import numpy as np
import os

# 이미지 경로
img_path = 'C:/Users/IIPL02/Desktop/inpainted_1/I26_23_04.png'
mask_save_path = 'input/test_mask.png'  # 저장 경로

# 이미지 불러오기
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

h, w = img.shape[:2]

# 중앙 사각형 마스크 생성 (예: 정사각형 100x100)
mask = np.zeros((h, w), dtype=np.uint8)
box_size = min(h, w) // 3  # 이미지 크기 기준 박스 크기 설정
start_y = (h - box_size) // 2
start_x = (w - box_size) // 2
mask[start_y:start_y+box_size, start_x:start_x+box_size] = 255  # 흰색 마스킹

# 저장 폴더 없으면 생성
os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)

# 마스크 저장
cv2.imwrite(mask_save_path, mask)

print(f"✅ 마스크 저장 완료: {mask_save_path}")
