'''
[ Description of Speech-Emotion Recognition Model ]

    date: 12/7/2020
    produced by Project Guarosa

1. Data Input
- mic를 통해 3초 가량의 voice를 녹음함(output.wav)
- 이를 모델에 투입하고 결과를 산출해 내는 과정이 무한루프를 통해 반복됨.

2. Prediction
- 감정의 category는 총 8개('angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')로 구분됨.
- 음성 데이터를 모델에 투입하여 softmax 함수를 통해 최종 결과 값 산출해 냄.
- 즉, 전체 확률을 1로 두고 각각의 category에 대해 예측되는 확률을 출력함. 
- 이는 아래 # Prediction 섹션에서 변수 preds에 기록됨.
- preds의 index는 순서대로 'angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'와 상응함.

3. Moving Agerage
- 출력의 변동을 줄여 안정적으로 감정을 인식하고자 최근 5개의 음성데이터에 대해 이동 평균을 냄.
- 이는 아래 # Moving Average 섹션에서 변수 mean_stack에 기록됨.

4. Print Output
-  상위 3개의 감정결과와 이동평균 감정결과가 # Print Output 섹션을 통해 출력됨.
'''

import os
import keras
import librosa
import glob 
import numpy as np
import pandas as pd
import sys
import keyboard
import pyaudio
import wave
import datetime
import MySQLdb
from joblib import load
from keras.optimizers import RMSprop
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

f = open('ascii.txt','r')
print(f.read())
f.close()

# DB setting
# db = MySQLdb.connect(db="unknown", host="amazonaws.com",port=unknown,  user="unknown", passwd="unknown!",
# charset='utf8')
# cursor = db.cursor()

# Process audio file
MIC_DEVICE_ID = 1
CHUNK = 1024
# FORMAT = pyaudio.paInt16
FORMAT = pyaudio.paInt24
CHANNELS = 1
# CHANNELS = 2
RATE = 22050
# RATE = 44100
# SAMPLE_SIZE = 2 # FORMAT의 바이트 수

# Functions
def record(record_seconds):
    p = pyaudio.PyAudio()
    stream = p.open(input_device_index = MIC_DEVICE_ID,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    sys.stdout.write('='*40 + ' Start to record the audio' + ' // ' + 'count = {} '.format(count) + '='*40 + '\n')
    print()
    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

        progress(len(frames), int(RATE / CHUNK * record_seconds), status='Recording')
    print('='*50, 'Recording is finished', '='*47)
    print()
    
    SAMPLE_SIZE = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames, SAMPLE_SIZE

def save_wav(target, frames):
    wf = wave.open(resource_path(target), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPLE_SIZE)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

    if isinstance(target, str) :
        wf.close()

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) 
    
    return result

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def progress(count, total, status=''):
    bar_len = 80
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


# Setting model version
version = '002'

# Loading json and creating model
json_file = open(resource_path('saved_models_{0}/model_{0}.json'.format(version)), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Loading weights into new model
loaded_model.load_weights(resource_path('saved_models_{0}/model_hdf5_{0}.h5'.format(version)))
print('='*46, 'Loaded {}_model from disk'.format(version), '='*46)
print()

# Loading scaler
scaler = load('saved_models_{0}/scaler_{0}.save'.format(version))

# Loading categories
categories=np.array(['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad',
        'surprise'])

# Initial argument
count = 0
stack ='init'

while True:
    if keyboard.is_pressed('ESC'):
        sys.exit()
    
    count += 1

    RECORD_SECONDS = 3
    frames, SAMPLE_SIZE = record(RECORD_SECONDS)
    now = datetime.datetime.now()
    WAVE_OUTPUT_FILENAME = "output.wav"
    save_wav(WAVE_OUTPUT_FILENAME, frames)

    # Loading audio files and extracting features
    data, sample_rate = librosa.load('output.wav', duration=3.0, offset=0.2)
    ex_data = extract_features(data)

    # Prediction
    ex_data = np.expand_dims(ex_data, axis=0)
    ex_data = scaler.transform(ex_data)
    df = pd.DataFrame(data=ex_data)
    df = df.stack().to_frame().T
    ex_dim= np.expand_dims(df, axis=2)
    preds = loaded_model.predict(ex_dim, batch_size=32)
    sorted_pred = np.argsort(preds)

    # Moving Agerage
    if stack=='init':
        stack = preds
    else:
        if not stack.shape[0]==5:
            stack = np.vstack((stack, preds))
        else:
            stack = stack[1:5]
            stack = np.vstack((stack, preds))
    print(stack)
    mean_stack = stack.mean(axis=0)
    avg_pred = np.argmax(mean_stack, axis=0)

    # Print Output
    print('='*120)
    print()
    print('Rank 1 : ',' '*int(sorted_pred[:,-1])*10, categories[sorted_pred[:,-1]].tolist()[0].upper(),'(', preds[0][sorted_pred[:,-1]][0], ')')
    print('Rank 2 : ',' '*int(sorted_pred[:,-2])*10, categories[sorted_pred[:,-2]].tolist()[0].upper(),'(', preds[0][sorted_pred[:,-2]][0], ')')
    print('Rank 3 : ',' '*int(sorted_pred[:,-3])*10, categories[sorted_pred[:,-3]].tolist()[0].upper(),'(', preds[0][sorted_pred[:,-3]][0], ')')
    print()
    print('MoAvg5 : ',' '*int(avg_pred)*10,  categories[[avg_pred]].tolist()[0].upper(),'(', mean_stack[avg_pred], ')')
    print()
    print('='*120)
    print()
    print()



