sr = 44100  # 22050
preemphasis = 0.97

frame_shift = 30 / 1000  # 0.0125 # seconds
frame_length = 60 / 1000  # 0.05 # seconds

hop_length = int(sr * frame_shift)  # samples.
win_length = int(sr * frame_length)  # samples.

n_fft = win_length  # 2048 win_length:int <= n_fft [scalar]

n_mels = 80  # Number of Mel banks to generate
power = 1.2  # Exponent for amplifying the predicted magnitude

max_db = 100
ref_db = 20

n_iter = 60

sample_path = "C:/Users/PKU/Desktop/seq2seq"
