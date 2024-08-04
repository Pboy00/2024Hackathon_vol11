from djitellopy import Tello #Pythonでドローンを操作
import cv2 #画像処理・物体検出
import numpy as np 
import json #JSONファイルに変換
import socket
import threading
import time #データ送信のための遅延用

# YOLOのパラメータ
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# YOLOのファイルパス
YOLO_CONFIG = 'yolov3.cfg'
YOLO_WEIGHTS = 'yolov3.weights'
YOLO_NAMES = 'coco.names'

# UDPソケットの設定
udp_ip = "192.168.186.195"  # マイコンのIPアドレス
object_data_port = 8889  # 物体データを送信するポート番号

# ソケットの作成
sock = None
stop_event = threading.Event()

def udp_begin():
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1)
    print("UDP socket created")

# ラベルの読み込み
with open(YOLO_NAMES, 'r') as f:
    LABELS = f.read().strip().split('\n')

# カラーの設定
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# YOLOモデルの読み込み
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Telloドローンのインスタンスを作成
tello = Tello()

# Telloドローンに接続
tello.connect()

# ストリーミング開始
tello.streamon()

# フレームの中心を取得
def get_frame_center(frame):
    height, width, _ = frame.shape
    return width // 2, height // 2

# フレーム内の物体を検出
def detect_objects(frame):
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    return boxes, confidences, class_ids, idxs

# 認識した物体データをマイコンに送信
def send_object_data(boxes, confidences, class_ids, idxs, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            data = {
                "class": labels[class_ids[i]],
                "confidence": float(confidences[i]),
                "xy": {
                    "x": int(boxes[i][0]),
                    "y": int(boxes[i][1]),
                    "width": int(boxes[i][2]),
                    "height": int(boxes[i][3])
                }
            }
            message = json.dumps(data).encode('utf-8')
            try:
                sock.sendto(message, (udp_ip, object_data_port))
                print(f"Data sent successfully to {udp_ip}:{object_data_port} - {message}")
            except Exception as e:
                print(f"Failed to send data: {e}")
            time.sleep(0.1)  # 物体データ送信の間に短い遅延を追加
            

# 特定の物体を追尾
def track_object(tello, target_box, frame_center, frame):
    (x, y, w, h) = target_box
    target_center_x = x + w // 2
    target_center_y = y + h // 2
    diff_x = target_center_x - frame_center[0]
    diff_y = target_center_y - frame_center[1]
    area = w * h
    frame_area = frame.shape[0] * frame.shape[1]
    area_ratio = area / frame_area

    # X軸の制御（左右移動）
    if abs(diff_x) > 20:
        if diff_x > 0:
            tello.move_right(20)
        else:
            tello.move_left(20)
        time.sleep(0.5)  # 移動の間に短い遅延を追加

    # Y軸の制御（上下移動）
    if abs(diff_y) > 20:
        if diff_y > 0:
            tello.move_down(20)
        else:
            tello.move_up(20)
        time.sleep(0.5)  # 移動の間に短い遅延を追加

    # Z軸の制御（前進/後退）
    if area_ratio < 0.1:
        tello.move_forward(20)
    elif area_ratio > 0.2:
        tello.move_back(20)
    time.sleep(0.5)  # 移動の間に短い遅延を追加

    # Yawの制御（旋回）
    if diff_x > 100:
        tello.rotate_clockwise(30)
    elif diff_x < -100:
        tello.rotate_counter_clockwise(30)
    time.sleep(0.5)  # 旋回の間に短い遅延を追加

# キーボード入力を監視するスレッド
def keyboard_listener():
    global stop_event
    while not stop_event.is_set():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

def check_battery():
    battery_level = tello.get_battery()
    print(f"Battery level: {battery_level}%")
    if battery_level < 20:
        print("Battery low! Landing...")
        tello.land()
        return False
    return True

# UDPの初期化
udp_begin()

# 離陸
tello.takeoff()
time.sleep(2)  # 離陸後に短い遅延を追加

# キーボード入力スレッドの開始
keyboard_thread = threading.Thread(target=keyboard_listener)
keyboard_thread.start()

try:
    while not stop_event.is_set():

        if not check_battery():
            break
        
        # フレームを取得
        frame = tello.get_frame_read().frame

        # フレームの中心を取得
        frame_center = get_frame_center(frame)

        # 物体を検出
        boxes, confidences, class_ids, idxs = detect_objects(frame)

        #データの送信
        send_object_data(boxes, confidences, class_ids, idxs, LABELS)

        # 認識した物体データをマイコンに送信
        target_box = None
        if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[class_ids[i]] == 'person':  # 追尾する物体を指定
                    target_box = boxes[i]
                    print("now tracking")
                    break

        # 追尾
        if target_box is not None:
            track_object(tello, target_box, frame_center, frame)

except KeyboardInterrupt:
    print("緊急停止シグナルを受信。ストリーミングを停止します。")

finally:
    stop_event.set()
    keyboard_thread.join()
    tello.streamoff()
    tello.land()
    tello.end()
    if sock:
        sock.close()
    print("処理が完了しました。")
