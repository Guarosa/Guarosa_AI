'''
[Eye Blink Detector]

* 양안의 깜빡임을 감지

'''

'''
[Posture Corrector]

* 양 어깨의 keypoints를 받아 (y 거리)/(x 거리)를 자세 구분 기준으로 삼음

    기준 < 0.20         : 정상(normal)
    0.70 > 기준 >= 0.20 : 잘못된 자세(warning)
    (그 외에는 이상치(abnormal))

* 기준치에 도달한 뒤 30 frame이 유지된 경우에만 해당 자세로 인정
  (경계 값에서 불필요하게 빈번한 신호발송 방지 위함)

'''

import cv2, dlib
import mediapipe as mp
import numpy as np
from imutils import face_utils
from keras.models import load_model
from datetime import datetime
import MySQLdb
import os

################### image resize ratio ##################
fx, fy = 1, 1



######################## base #######################
def resource_path(relative_path):
  try:
    base_path = sys._MEIPASS
  except Exception:
    base_path = os.path.abspath(".")

  return os.path.join(base_path, relative_path)

def time_interval(datetime_now):
  time_str = str(datetime_now)
  second = float(time_str.split(' ')[1].split(':')[2])

  return second

user_id = int(input("사번을 입력하세요:"))
db = MySQLdb.connect('database_information')
cursor = db.cursor()
temp = datetime.now()



################## eye blink detector ##################
IMG_SIZE = (34, 26)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
model = load_model('models/eyeblink_detector.h5')
status = 'start'

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect




################# posture corrector ################
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def to_dict(landmark_idx, results):
  landmark = str(results.pose_landmarks.landmark[landmark_idx])
  dic = dict(x=float(landmark[landmark.find("x")+3:landmark.find("y")-1]), 
            y=float(landmark[landmark.find("y")+3:landmark.find("z")-1]), 
            z=float(landmark[landmark.find("z")+3:landmark.find("v")-1])
            )
  return dic

status_p = 'start'
status_c = 'start'
duration_temp = 0
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)





######################## execution ############################
cap = cv2.VideoCapture(0)

while cap.isOpened():
  success, image = cap.read()

  if not success:
    break

  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = pose.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  image = cv2.resize(image, dsize=(0, 0), fx=fx, fy=fy)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  try:
    face = detector(gray)[0]

    # for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

    # visualize
    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    if (pred_l <= 0.1 and pred_r <= 0.1):
      if (status == 'open'):
        # 눈깜빡했을 때 실행
        now = datetime.now()

        if (now.second - temp.second) < 0:
          insert_time = now.second - temp.second + 60
        else:
          insert_time = now.second - temp.second

        temp = now
        aa = str(insert_time)
        now_date = str(now.month)+'/'+str(now.day)+' '+str(now.hour)+':'+str(now.minute)
        cursor.execute("INSERT INTO eye_tracking (user_id,now_date,time_interval) VALUES(%s,%s,%s)", [user_id,now_date,insert_time])
        db.commit()
        print(insert_time, type(insert_time))
        status = 'close'

    elif (pred_l >= 0.5 and pred_r >= 0.5):
      status = 'open'
  except:
    print("EYE ERROR")


  mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
  if (duration_temp < 30): print(duration_temp) 

  try:
    # eye blink drawing
    cv2.rectangle(image, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=1)
    cv2.rectangle(image, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=1)

    cv2.putText(image, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, fx*0.7, (255,255,255), 2)
    cv2.putText(image, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, fx*0.7, (255,255,255), 2)

    # posture ratio calculation
    abs_diff_shoulder_x = abs(to_dict(11, results)['x']-to_dict(12, results)['x'])
    abs_diff_shoulder_y = abs(to_dict(11, results)['y']-to_dict(12, results)['y'])
    shoulder_angle = abs_diff_shoulder_y/abs_diff_shoulder_x
    font = cv2.FONT_HERSHEY_SIMPLEX
    angle = str(shoulder_angle)

    # posture recognition
    if shoulder_angle >= 0.20 and shoulder_angle < 0.70:
      cv2.putText(image, 'WARNING', (30, 50), font, 1, (0,0,155), 2, cv2.LINE_AA)
      cv2.putText(image, angle, (30, 30), font, 1, (0,0,155), 2, cv2.LINE_AA)
      if (status_p != 'warning'): duration_temp = 0
      status_p = 'warning'
      duration_temp += 1
      if duration_temp > 30: 
          if status_c != 'warning':
            print('warning')            # 이를 db에 저장해 신호 발송
          status_c = 'warning'
              
    elif shoulder_angle < 0.20:
      cv2.putText(image, 'normal', (30, 50), font, 1, (0,155,0), 2, cv2.LINE_AA)
      cv2.putText(image, angle, (30, 30), font, 1, (0,155,0), 2, cv2.LINE_AA)
      if (status_p != 'normal'): duration_temp = 0
      status_p = 'normal'
      duration_temp += 1
      if duration_temp > 30: 
        if status_c != 'normal':
          print('normal')            # 이를 db에 저장해 신호 발송
        status_c = 'normal'

    else: 
      cv2.putText(image, 'abnormal', (30, 50), font, 1, (155,155,155), 2, cv2.LINE_AA)
      cv2.putText(image, angle, (30, 30), font, 1, (155,155,155), 2, cv2.LINE_AA)
      status_p = 'abnormal'
      status_c = 'abnormal'
  except:
    print("POSTURE ERROR")

  cv2.imshow('Guarosa Project', image)
  if cv2.waitKey(5) & 0xFF == 27:
    cursor.close()
    db.close()
    pose.close()
    cap.release()
    break


