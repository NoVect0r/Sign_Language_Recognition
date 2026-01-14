import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 폴더 경로 설정
dataset_dir = 'dataset_raw'  # 저장된 이미지 폴더
digits_csv_path = 'digits.csv'
hangul_csv_path = 'hangul.csv'

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 숫자 / 자모 분류 기준
digits_labels = {str(i) for i in range(1, 11)}
jaum_labels = {'ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ', 'conversion_model_1', 'space', 'back_space'}
moum_labels = {'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ',
               'ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅡ','ㅢ','ㅣ'} # ㅠ 제외
hangul_labels = jaum_labels | moum_labels

# CSV 파일 준비
digits_csv = open(digits_csv_path, mode='w', newline='', encoding='cp949')
hangul_csv = open(hangul_csv_path, mode='w', newline='', encoding='cp949')
digits_writer = csv.writer(digits_csv)
hangul_writer = csv.writer(hangul_csv)

# 헤더 작성
header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
digits_writer.writerow(header)
hangul_writer.writerow(header)

# 인식 성공률 기록용
success_rate_dict = {}

# 각 레이블 폴더 순회
for label_name in sorted(os.listdir(dataset_dir)):
    label_path = os.path.join(dataset_dir, label_name)
    if not os.path.isdir(label_path):
        continue

    success_count = 0
    count = 0

    for img_name in sorted(os.listdir(label_path)):
        img_path = os.path.join(label_path, img_name)

        # 한글 경로 이미지 로딩 처리
        try:
            with open(img_path, 'rb') as f:
                img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[!] 파일 열기 실패: {img_path} — {e}")
            continue

        if img is None:
            print(f"[!] 이미지 로드 실패: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            x_vals, y_vals = [], []
            for lm in hand.landmark:
                x_vals.append(lm.x)
                y_vals.append(lm.y)
            row = x_vals + y_vals + [label_name]

            if label_name in digits_labels:
                digits_writer.writerow(row)
            elif label_name in hangul_labels:
                hangul_writer.writerow(row)

            print(f"[✓] 좌표 저장 완료: {label_name}/{img_name}")
            success_count += 1
        else:
            print(f"[!] 손 인식 실패: {label_name}/{img_name} (스킵)")

        count += 1

    success_rate = (success_count / count) * 100 if count > 0 else 0
    success_rate_dict[label_name] = success_rate

# CSV 파일 닫기
digits_csv.close()
hangul_csv.close()

# 인식 성공률 요약 출력
print("\n📊 레이블별 인식 성공률:")
for label, rate in success_rate_dict.items():
    print(f"- {label}: {rate:.1f}%")

print(f"\n✅ CSV 파일 생성 완료:\n - {digits_csv_path}\n - {hangul_csv_path}")
