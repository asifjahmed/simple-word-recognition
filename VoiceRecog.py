#Version 4

#Created by:
#   Fernando Santamarina
#   Tyler Cloke
#   Zach Stiggelbout

import wave
import struct
import scipy
import pyaudio
from numpy import *
import Tkinter as Tk
import sys
import os
import random as r


import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#-------------------------------------------------------
# Variables
#-------------------------------------------------------
global myCanvas
global toolbar
global f

TRAIN_COUNT = 0
VECTORS_TO_TRAIN = []

#sample dictionary, make sure to have numOutputs = length(dictionary)
dictionary = ['yes', 'no', 'maybe']

weightsYes = []
weightsNo = []
weightsMaybe = []
currentMagBin = []

learning_rate = 0.07
inputSize = 20
numOutputs = len(dictionary)

#-------------------------------------------------------
# Perceptron definitions
#-------------------------------------------------------

#returns the sum of the sigmoid function tanh(Xi*Wi)
#@param1 frequency bin values
#@param2 'yes', 'no', or 'maybe'
def sum_sig(values, word):
   if(word == 'yes'):
      return sum(math.tanh(value*weightsYes[index]) for index, value in enumerate(values))
   elif(word == 'no'):
      return sum(math.tanh(value*weightsNo[index]) for index, value in enumerate(values))
   else:
      return sum(math.tanh(value*weightsMaybe[index]) for index, value in enumerate(values))


#Called to train the word that the perceptron
#should actualy guess
def trainActual():
   word = entry.get().lower()
   if word not in dictionary:
      return
   else:
      if word == 'yes':
         trainYes([[currentMagBin, 1]])
         trainNo([[currentMagBin, 0]])
         trainMaybe([[currentMagBin, 0]])
      elif word == 'no':
         trainYes([[currentMagBin, 0]])
         trainNo([[currentMagBin, 1]])
         trainMaybe([[currentMagBin, 0]])
      else:
         trainYes([[currentMagBin, 0]])
         trainNo([[currentMagBin, 0]])
         trainMaybe([[currentMagBin, 1]])

      entry.delete(0, len(word))
      entry.insert(0, word + ' learned')

#trains all the perceptrons using the initial data collected
#by calling their respective train method
def train(training_set):
   trainYes([[training_set[0], 1], [training_set[1], 1], [training_set[2], 0], [training_set[3], 0], [training_set[4], 0], [training_set[5], 0]])
   trainNo([[training_set[0], 0], [training_set[1], 0], [training_set[2], 1], [training_set[3], 1], [training_set[4], 0], [training_set[5], 0]])
   trainMaybe([[training_set[0], 0], [training_set[1], 0], [training_set[2], 0], [training_set[3], 0], [training_set[4], 1], [training_set[5], 1]])

#trains the yes perceptron to return 1 for 'yes'
#and 0 for both 'no' and 'maybe'
def trainYes(training_set):
   global weightsYes
   for k in range(400):
      errors = 0
      for input_vector, desired_output in training_set:
         result = 1 if sum_sig(input_vector, 'yes') > 0.5 else 0
         err = desired_output - result
         if err != 0:
            errors += 1
            for index, value in enumerate(input_vector):
               weightsYes[index] += learning_rate * err * value
      if errors == 0:
         break

#trains the no perceptron to return 1 for 'no'
#and 0 for both 'yes' and 'maybe'
def trainNo(training_set):
   global weightsNo
   for k in range(400):
      errors = 0
      for input_vector, desired_output in training_set:
         result = 1 if sum_sig(input_vector, 'no') > 0.5 else 0
         err = desired_output - result
         if err != 0:
            errors += 1
            for index, value in enumerate(input_vector):
               weightsNo[index] += learning_rate * err * value
      if errors == 0:
         break

#trains the maybe perceptron to return 1 for 'maybe'
#and 0 for both 'yes' and 'no'
def trainMaybe(training_set):
   global weightsMaybe
   for k in range(400):
      errors = 0
      for input_vector, desired_output in training_set:
         result = 1 if sum_sig(input_vector, 'maybe') > 0.5 else 0
         err = desired_output - result
         if err != 0:
            errors += 1
            for index, value in enumerate(input_vector):
               weightsMaybe[index] += learning_rate * err * value
      if errors == 0:
         break


# A method to get a guess for what word
# was said during recording
def parseNewData(input_vector):
   summed = [sum_sig(input_vector, 'yes'), sum_sig(input_vector, 'no'), sum_sig(input_vector, 'maybe')]
   result = []
   for val in summed:
      result.append(1 if val > 0.5 else 0)
   entry.delete(0, len(entry.get()))
   entryMW.delete(0, len(entryMW.get()))
   entryMW.insert(0, dictionary[result.index(max(result))])


#call this function at the start of the program to initialize all weights
#to a random value between 0 and 1
def initWeights():
   global weightsYes
   global weightsMaybe
   global weightsNo
               
   for index in range(inputSize):
      weightsYes.append(r.random())
      weightsNo.append(r.random())
      weightsMaybe.append(r.random())   


#Call to initialize the weights
initWeights()



#-------------------------------------------------------
# Audio Analysis and Display
#-------------------------------------------------------

# a tk.DrawingArea
def displayHist():
   global f
   global currentMagBin
   global TRAIN_COUNT
   global VECTORS_TO_TRAIN
   TRAIN_COUNT += 1
   #Test wave
   #wavFile = wave.open('C:\Users\Tyler\Desktop\warning.wav')

   #Record wave file
   paud = pyaudio.PyAudio()
   chunk = 1024
   bitStr = paud.open(format = pyaudio.paInt16, channels = 1, 
                                  rate = 44100, input = True, output = True,
                                  frames_per_buffer = 1024)

   print 'Recording word'

   #ATTN: Change this to control how many seconds it
   #records for
   secsToRecord = 1

   all = []
   for i in range(0, 44100 / chunk * secsToRecord):
      data = bitStr.read(chunk)
      all.append(data)
          
   print 'Done recording...analyzing'

   bitStr.stop_stream()
   bitStr.close()
   paud.terminate()

   waveFile = os.getcwd() + 'curr_word.wav'
   data = ''.join(all)
   wf = wave.open(waveFile, 'wb')
   wf.setnchannels(1)
   wf.setsampwidth(paud.get_sample_size(pyaudio.paInt16))
   wf.setframerate(44100)
   wf.writeframes(data)
   wf.close()

   wavFile = wave.open(waveFile,'r')
   (nchannels, sampwidth, framerate, nframes, comptype, compname) = wavFile.getparams()
   frames = wavFile.readframes (nframes * nchannels)
   out = struct.unpack_from ("%dh" % nframes * nchannels, frames)
   wavFile.close()


   # Convert 2 channles to numpy arrays
   if nchannels == 2:
      left = array(list(out[0::2]))
      right = array(list(out[1::2]))
   else:
      left = array (out)
      right = left


   xfft = abs(scipy.fft(left))
   where(xfft,0,1e-17)
   mag = 20*scipy.log10(xfft)

   fn2 = scipy.linspace(0,44100,mag.size)

   minIdx = nonzero(fn2 < 2000)

   mag = mag[minIdx]
   fn2 = fn2[minIdx]

   #ATTN: This controls how many bins you have
   binSize = 20

   freqInBin = 2000/binSize
   j = 0

   #ATTN: This contains the data you want to input into
   #the neural network. It is the average of the magnitude of the
   #frequencies over each histogram range
   magBin = zeros(len(range(0,binSize)))


   magLabel = list()
   minFreqBin = 10000000
   for i in range(0,binSize):
      if i == 0:
         minIdx = nonzero(fn2 < (i+1)*freqInBin)
      else:
         minIdx = nonzero(logical_and(fn2 < (i+1)*freqInBin,fn2 > (i)*freqInBin))
      if mag[minIdx].size == 0:
         magBin[i] = 0
      else:   
         magBin[i] = average(mag[minIdx])
         baseStr = str(int((i+1)*freqInBin)) 
         magLabel.append(baseStr + ' Hz')
      if magBin[i] < minFreqBin:
         minFreqBin = magBin[i]

   magBin = magBin / minFreqBin
   magBin = map(lambda x: x/min(magBin), magBin)

   a = f.add_subplot(121)
   a.clear()
   a2 = f.add_subplot(122)
   a2.clear()

   a.plot(fn2,mag)
   a.set_title('FFT Coefficient Magnitude vs. Frequency')

   width = 0.35
   a2.bar(range(0,binSize), magBin, width)
   a2.set_xticklabels(magLabel)
   a2.set_title('FFT Frequencies')

   myCanvas.show()
   myCanvas.get_tk_widget().pack()

   toolbar.update()
   myCanvas._tkcanvas.pack()

   currentMagBin = magBin

# If in training phase then add to training array list
#   when you have all 6 arrays, then train
# else find a guess for the word
   if(TRAIN_COUNT <=6):
      VECTORS_TO_TRAIN.append(magBin)
      if(TRAIN_COUNT == 6):
         train(VECTORS_TO_TRAIN)
   else:
      parseNewData(magBin)





#-------------------------------------------------------
# Code for different window's and their UI
#-------------------------------------------------------

#Window 1 for training instructions
learn = Tk.Tk()
learn.wm_title('Training Instructions')
learn.geometry('450x600')
w = Tk.Canvas(learn, width=400, height=400)
w.pack()
w.create_text(135,150, text = '\n\
                               First train the perceptron by:\n   \
                                  1) Push record and say yes. Repeat a second time\n   \
                                  2) Push record and say no. Repeat a second time\n   \
                                  3) Push record and say maybe. Repeat a second time\n\n\
                                Once you have finished the training, then you can test by:\n   \
                                  1) Push record and say one of yes, no, or maybe\n\n\
                                Close this window to continue.', font=("Helvetica", 14))

closebutton = Tk.Button(learn, text="Close", command = learn.destroy)
closebutton.grid(row=0, column=0)
closebutton.pack(padx=8, pady=8)

learn.mainloop()




#Window 2 for testing
root = Tk.Tk()
root.wm_title("Voice Recognition")

frame = Tk.Frame(root)
frame.grid(row = 0, column = 0)
frame.pack()

frame2 = Tk.Frame(root)
frame2.grid(row = 0, column = 1)
frame2.pack()

f = Figure(figsize=(12,6), dpi=100)
myCanvas = FigureCanvasTkAgg(f, master=frame)
toolbar = NavigationToolbar2TkAgg( myCanvas, frame )


button = Tk.Button(root, text="Record", command=displayHist)
button.grid(row = 1, column = 0)
button.pack(side = Tk.LEFT, padx=8,pady=8)

labelMW = Tk.Label(root, text='Machine word:')
labelMW.pack(side = Tk.LEFT, padx = 12,pady=8)

entryMW = Tk.Entry(root, width = 20)
entryMW.pack(side = Tk.LEFT, padx=0,pady=8)

label = Tk.Label(root, text='Correct word:')
label.pack(side = Tk.LEFT, padx = 12,pady=8)

entry = Tk.Entry(root, width = 20)
entry.pack(side = Tk.LEFT, padx=0,pady=8)

quitbutton = Tk.Button(root, text="Quit", command=root.destroy)
quitbutton.pack(side = Tk.RIGHT)

button2 = Tk.Button(root, text="Learn", command=trainActual)
button2.grid(row = 1, column = 1)
button2.pack(side = Tk.RIGHT, padx=8,pady=8)

root.mainloop()






