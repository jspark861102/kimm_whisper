import sounddevice as sd
import numpy as np
import whisper
import queue
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int16
# from gtts import gTTS
import pyttsx3
import pygame
import io
import os

# Pygame 초기화
pygame.init()

# 1. Whisper 모델 로드
model = whisper.load_model("large")  # 모델 크기: tiny, base, small, medium, large

# 2. 설정
SAMPLE_RATE = 16000       # 샘플링 레이트
BLOCK_SIZE = 1024         # 블록 크기
PROCESSING_INTERVAL = 1   # 처리 간격 (초)
LANGUAGE = "ko"           # 언어 설정
THRESHOLD = 0.005         # 음량 임계값
# WORD_LIST = ["왼쪽", "오른쪽", "앞으로", "뒤로", "위로", "아래로"]  # 허용된 단어 리스트
WORD_LIST = {"교시": 1, "위로": 2, "모바일": 3, "팔": 4, "그리퍼": 5}
# WORD_LIST = {"teach": 16, "home": 1, "mobile": 3, "arm": 32, "finger": 99, "calibration": 41}


# ROS 노드 설정
PUB_TOPIC = "/recognized_word"

# 오디오 데이터를 처리할 큐 생성
audio_queue = queue.Queue()
buffer = np.zeros(0, dtype=np.float32)  # 버퍼를 전역적으로 관리

# 마이크 입력 콜백 함수
def audio_callback(indata, frames, time, status):
    if status:
        rospy.logwarn(f"Status: {status}")
    audio_queue.put(indata.copy())  # 오디오 데이터를 큐에 추가

# RMS 계산 함수
def calculate_rms(audio_data):
    return np.sqrt(np.mean(np.square(audio_data)))

# 가장 유사한 단어 찾기
def find_closest_word(transcribed_text, word_list):
    from difflib import get_close_matches
    matches = get_close_matches(transcribed_text, word_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

# 동기 TTS 처리
# def speak_text_gtts(text):
#     # gTTS로 음성 생성
#     tts = gTTS(text=text, lang="ko")
#     mp3_fp = io.BytesIO()  # 메모리 스트림 생성
#     tts.write_to_fp(mp3_fp)
#     mp3_fp.seek(0)  # 스트림 시작 위치로 이동

#     # Pygame으로 재생
#     pygame.mixer.init()
#     pygame.mixer.music.load(mp3_fp, "mp3")
#     pygame.mixer.music.play()

#     # 재생이 끝날 때까지 대기
#     while pygame.mixer.music.get_busy():
#         continue

def speak_text_offline(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # 음성 속도
    engine.setProperty('voice', 'ko-KR')  # 한국어 음성 (운영 체제에 따라 다름)
    engine.say(text)
    engine.runAndWait()

def speak_text_espeak(text):
    os.system(f'espeak -v ko "{text}"')


# 음성을 텍스트로 변환
def process_audio_stream(pub):
    global buffer
    rospy.loginfo(f"Streaming and transcribing every {PROCESSING_INTERVAL} seconds...")
    required_samples = SAMPLE_RATE * PROCESSING_INTERVAL

    try:
        while not rospy.is_shutdown():
            while not audio_queue.empty():
                audio_data = audio_queue.get()
                rms = calculate_rms(audio_data)

                if rms >= THRESHOLD:
                    buffer = np.append(buffer, audio_data)
                    rospy.loginfo(f"RMS passed threshold: {rms:.4f}")
                #else:
                #    rospy.loginfo(f"Skipped block due to low volume: {rms:.4f}")

            if len(buffer) >= required_samples or len(buffer) > SAMPLE_RATE // 2:
                rospy.loginfo("Transcribing audio...")
                audio_input = buffer[:required_samples]
                buffer = buffer[required_samples:]

                # Whisper로 텍스트 변환
                result = model.transcribe(audio_input, language=LANGUAGE, fp16=False)
                transcribed_text = result['text'].strip().lower()
                rospy.loginfo(f"Recognized Text: {transcribed_text}")

                # 첫 번째 단어만 추출
                first_word = transcribed_text.split()[0] if transcribed_text else None
                rospy.loginfo(f"First Word: {first_word}")

                # 단어 매칭
                closest_word = find_closest_word(first_word, WORD_LIST) if first_word else None
                if closest_word:
                    rospy.loginfo(f"Matched Word: {closest_word}")
                    command_value = WORD_LIST[closest_word]

                    # 퍼블리시
                    pub.publish(command_value)
                    # pub.publish(closest_word)

                    # TTS로 출력
                    # speak_text_gtts(f"{closest_word}")
                    # speak_text_offline(f"{closest_word}")
                    speak_text_espeak(f"{closest_word}")

                else:
                    rospy.loginfo("No matching word found.")
    except KeyboardInterrupt:
        rospy.loginfo("\nStopped streaming.")
        return

# ROS 노드 초기화 및 메인 함수
def main():
    rospy.init_node("speech_recognition_node", anonymous=True)
    # pub = rospy.Publisher(PUB_TOPIC, String, queue_size=10)
    pub = rospy.Publisher(PUB_TOPIC, Int16, queue_size=10)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, dtype="float32", callback=audio_callback):
            process_audio_stream(pub)
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
