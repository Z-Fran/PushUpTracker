import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
for i in range(101):
    engine.save_to_file(str(i),f'voice/{i}.wav')
    engine.runAndWait()

engine.save_to_file('Ensure proper arm posture',f'voice/arm.wav')
engine.runAndWait()

engine.save_to_file('Keep your body straight.',f'voice/body.wav')
engine.runAndWait()
