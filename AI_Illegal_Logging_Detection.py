#Importing Library for Sound Processing
import numpy as np
import scipy
from scipy.io import wavfile
from python_speech_features import mfcc
import wave

#Importing Library for Artificial Intelligence
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#Importing Library for operating the Respeaker
from pixel_ring import pixel_ring
from gpiozero import LED
import time

#Turn the LED on Respeaker
power = LED(5)
power.on()

# MFCC Coefficient Extraction from Sound Database for Learning Process
print("Feature extraction for learning started")
filename = ('record1.wav', 'record2.wav', 'record3.wav', 'record4.wav', 'record5.wav',
            'record6.wav', 'record7.wav', 'record8.wav', 'record9.wav', 'record10.wav',
            'record11.wav', 'record12.wav', 'record13.wav', 'record14.wav', 'record15.wav')
combinecoef = []

#Setting parameter for MFCC extraction
for i in range(15):
    sample_rate, audio = wavfile.read(filename[i])
    obj = wave.open(filename[i], 'r')
    channel = obj.getnchannels()
    rate = obj.getframerate()
    if channel > 1:
        audio = audio[:, 0]
    if rate != 16000:
        sample_rate = 16000
    audio = audio[0:int(2 * sample_rate)]
    audio_data = audio - np.mean(audio)
    audio_data = audio_data / np.max(audio_data)
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.02, winstep=0.01, numcep=13, nfilt=26,
                     nfft=1024, lowfreq=0, highfreq=8000, preemph=0.95, ceplifter=22, appendEnergy=True)
    coef = np.array(mfcc_feat)
    coef = coef.reshape(1, -1)
    combinecoef.append(coef)
    
#Splitting each features from sound database (2587 = 13*200 frame from each sound)
mfcc_coef = np.array(combinecoef)
mfcc_coef = mfcc_coef.reshape(1, -1)
mfcc_coef = mfcc_coef.reshape(-1, 2587)
print("Feature Extraction for learning Done")

# Setting target for output node (15 bit for 15 sounds)
target = []
sample_target = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
for x in range(15):
    target.append(sample_target)
    sample_target = np.roll(sample_target, 1)
targetend = np.array(target)
targetend = targetend.reshape(1, -1)
targetend = targetend.reshape(-1, 15)
print("Target have been set")

# Neural Network using Multi Layer Perceptron
print("Learning process")
clf = MLPClassifier(activation='relu', solver='adam', learning_rate="adaptive",
                    learning_rate_init=0.01, random_state=1, hidden_layer_sizes=(256), max_iter=2000)
clf.fit(mfcc_coef, targetend)

#Learning process visualization
plt.ylabel('Loss Function')
plt.xlabel('Iterations')
plt.title('Loss using Adam algorithm, learning rate =0.01')
plt.plot(clf.loss_curve_)
plt.axis((0, 18, None, None))
plt.show()
time.sleep(4)
plt.close()
print("Learning end")

#Probability and Testing
print("Feature extraction start")
filename1 = 'record1.wav'
sample_rate1, audio1 = wavfile.read(filename1)
obj = wave.open(filename1, 'r')
channel = obj.getnchannels()
rate = obj.getframerate()
if channel > 1:
    audio1 = audio1[:, 0]
if rate != 16000:
    sample_rate1 = 16000
audio1 = audio1[0:int(2 * sample_rate1)]
audio_data1 = audio1 - np.mean(audio1)
audio_data1 = audio_data1 / np.max(audio_data1)
mfcc_feat1 = mfcc(audio_data1, sample_rate1, winlen=0.02, winstep=0.01, numcep=13, nfilt=26,
                  nfft=1024, lowfreq=0, highfreq=8000, preemph=0.95, ceplifter=22, appendEnergy=True)
test1 = np.array(mfcc_feat1)
test1 = test1.reshape(1, -1)
print("Feature extraction end")
# z=clf.predict_proba(test1)
print("Predicting...please wait")
z1 = clf.predict(test1)
# result=np.array(z)
#resultend=(z > 0.9).sum()
#position=np.where(z > 0.9)
pos = np.argmax(z1)
if pos < 10:
    print('Alert!')
    pixel_ring.set_brightness(10)
    pixel_ring.set_color(r=255, g=0, b=0)
    time.sleep(5)
    pixel_ring.off()
    time.sleep(1)
    power.off()
elif pos >= 10:
    print('Not saw machine')
    pixel_ring.set_brightness(10)
    pixel_ring.set_color(r=0, g=255, b=0)
    time.sleep(5)
    pixel_ring.off()
    time.sleep(1)
    power.off()
else:
    print('not detected')
