from pydub import AudioSegment

sound = AudioSegment.from_mp3("C:/Users/cpsla/PycharmProjects/LSTM/iwaizako/audio/earthquake.mp3")
sound.export("output.wav", format="wav")
