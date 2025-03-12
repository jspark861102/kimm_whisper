import argparse
import io
import os
import speech_recognition as sr
import whisper
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep, time
from difflib import get_close_matches
import rospy
from std_msgs.msg import Int16

feedback = 0

def save_audio_recording(wav_data: io.BytesIO, folder: str = "recordings"):
    """
    녹음된 음성 데이터를 WAV 파일로 저장하는 함수.
    파일은 'recordings' 폴더에 타임스탬프를 이름으로 생성합니다.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = int(time())
    filename = os.path.join(folder, f"voice_{timestamp}.wav")
    with open(filename, 'wb') as f:
        f.write(wav_data.getvalue())
    rospy.loginfo("Saved voice recording to: %s", filename)

def main():
    PUB_TOPIC = "/recognized_word"

    global feedback    
    def callback(msg):
        global feedback
        feedback = msg.data        
        # rospy.loginfo("Received feedback: %s", feedback)
    
    rospy.init_node("speech_recognition_node", anonymous=True)
    pub = rospy.Publisher(PUB_TOPIC, Int16, queue_size=10)
    rospy.Subscriber("whisper_feedback", Int16, callback)
    
    WORD_LIST = {"teach": 97, "교시": 97, "教師": 97,
                 "home": 1, "homepose": 1, "홈포즈": 1, "준비자세": 1, "준비": 1,
                 "mobile": 98, "모바일": 98,
                 "arm": 32, "manipulator": 32, "매니퓰레이터": 32,
                 "finger": 99, "effector": 99,
                 "calibration": 41, "캘리브레이션": 41, "칼리브레이션": 41, 
                 "forward": 10, "앞으로": 10, 
                 "backward": 11, "뒤로": 11,
                 "left": 12, "왼쪽으로": 12,
                 "right": 13, "write": 13, "오른쪽으로": 13,
                 "upward": 14, "위쪽으로": 14, "위로": 14,
                 "down": 15, "downward": 15, "아래로": 15, "アレロー": 15, "arrero": 15, "arero": 15, "ábrero" :15,
                 "recover":30, "리커버": 30, "복구":30, "bokku":30,
                 "병렬":61, "parallel":61,"병열":61,"패러럴":61,"패럴럴":61,
                 "회전":62, "spin":62}
    
    OMIT_WORD_LIST = ["teach", "교시","教師", "mobile","모바일","finger","calibration","캘리브레이션","칼리브레이션", "병렬","병열","패러럴","패럴럴","parallel","spin","회전"]  


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="turbo", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])    
    parser.add_argument("--energy_threshold", default=3000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    args = parser.parse_args()    

    model = args.model    
    audio_model = whisper.load_model(model)

    temp_file = NamedTemporaryFile().name
    
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        if not data_queue.empty():  # 큐에 데이터가 남아있는 경우는 무시
            return
        data = audio.get_raw_data()
        if data not in data_queue.queue:  # 동일 데이터가 큐에 존재하지 않을 때만 추가
            data_queue.put(data)            

    def find_closest_word(transcribed_text, word_list):
        matches = get_close_matches(transcribed_text, word_list, n=1, cutoff=0.7)
        return matches[0] if matches else None
    
    def speak_text_espeak(text):
        os.system(f'espeak -v ko "{text}"')

    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)
    print(f"Model '{model}' loaded. Say a word...")

    try:
        while not rospy.is_shutdown():
            if not data_queue.empty():
                # data_queue에 저장된 모든 데이터를 합칩니다.
                all_data = b''.join([data_queue.get() for _ in range(data_queue.qsize())])
                audio_data = sr.AudioData(all_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # 내 목소리를 녹음하여 파일로 저장합니다.
                #save_audio_recording(wav_data)

                # Whisper 전사를 위해 임시 파일에 저장
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                result = audio_model.transcribe(temp_file)
                words = result['text'].strip().lower().split()                

                if words:
                    word = words[0]
                    closest_word = find_closest_word(word, WORD_LIST)
                    print(f"Recognized Word: {word}")                    
                    if closest_word:
                        print(f"Matched Word: {closest_word}")          
                        if closest_word not in OMIT_WORD_LIST:              
                            speak_text_espeak(f"{closest_word}")                                                
                        command_value = WORD_LIST[closest_word]                                                                                                                        
                        pub.publish(command_value)
                    else:
                        print("No match found")
                    print("")

            # feedback 값에 따라 추가 명령어를 처리
            if feedback == 1:
                speak_text_espeak("please calibrate")                        
                feedback = 0
            elif feedback == 2:
                speak_text_espeak("mobile start")                        
                feedback = 0
            elif feedback == 3:
                speak_text_espeak("mobile end")                        
                feedback = 0
            elif feedback == 4:
                speak_text_espeak("closed")
                feedback = 0
            elif feedback == 5:
                speak_text_espeak("released")
                feedback = 0
            elif feedback == 6:
                speak_text_espeak("direct teach start")
                feedback = 0
            elif feedback == 7:
                speak_text_espeak("direct teach end")
                feedback = 0
            elif feedback == 8:
                speak_text_espeak("calibration finished")
                feedback = 0
            elif feedback == 11:
                speak_text_espeak("parallel")
                feedback = 0
            elif feedback == 12:
                speak_text_espeak("spin")
                feedback = 0
                    
            while not data_queue.empty():
                data_queue.get()

            sleep(0.5)
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt received")
        print("Shutting down node.")
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()
