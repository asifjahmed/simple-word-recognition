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


global myCanvas
global toolbar
global f

weightsYes = []
weightsNo = []
weightsMaybe = []
currentMagBin = []

learning_rate = 0.1
inputSize = 20
numOutputs = 3


#sample dictionary, make sure to have numOutputs = length(dictionary)
dictionary = ['yes', 'no', 'maybe']






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

   
#trains all the perceptrons by calling their respective train method
def train(training_set):
   trainYes([[training_set[0], 1], [training_set[1], 0], [training_set[2], 0]])
   trainNo([[training_set[0], 0], [training_set[1], 1], [training_set[2], 0]])
   trainMaybe([[training_set[0], 0], [training_set[1], 0], [training_set[2], 1]])

#trains the yes perceptron
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

#trains the no perceptron
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

#trains the maybe perceptron
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



initWeights()
#arrayYes = [1.0, 1.403776924836857, 1.5832014471721996, 1.5795352152008399, 1.7825630564406683, 1.747930765862082, 1.5957430879058976, 1.5383704542634984, 1.5877580060817162, 1.6224476028114385, 1.5511625459877452, 1.5682342029972915, 1.5322415558295053, 1.5043744421406737, 1.6430015480546698, 1.7049854628432752, 1.770062804302114, 1.9101468167439477, 1.7861959843622164, 1.7233614620126285]
arrayYes = [1.0, 1.3532338157620698, 1.5116653349808271, 1.3993980969544908, 1.5723993896152104, 1.5687563353039942, 1.5676748630378194, 1.6059230112792973, 1.6250753356888303, 1.5827371695632746, 1.5164957878980967, 1.5397440142163061, 1.5635007659738493, 1.6153040143077775, 1.6259028726751992, 1.7664829439686527, 1.7974609616429447, 1.7270493032267749, 1.6596417111942952, 1.6182082776561468]
arrayNo = [1.0, 1.4196126432372365, 1.5612856283427641, 1.4958372768117594, 1.6384810328379353, 1.616265107117357, 1.5951359148022854, 1.668970086392076, 1.7417600939019255, 1.7053395738522497, 1.6856778987297718, 1.705714291735585, 1.7354264974170133, 1.8014927943073413, 1.8157216045498219, 1.8088849025166387, 1.6776419103943221, 1.5789968962321614, 1.5944152620306697, 1.6005728986139767]
arrayMaybe = [1.0, 1.4375446235350784, 1.632546867782761, 1.5690259252239347, 1.6575212653481173, 1.4945567882781847, 1.5397837909802856, 1.5196210164098789, 1.5061753940747045, 1.6152381352203309, 1.5365281582730554, 1.5426723946849941, 1.5851742218950928, 1.5581901527597386, 1.5091823769069017, 1.578406479697386, 1.5768756399258264, 1.6555194839403042, 1.7321073160591185, 1.7674706904649178]

train([arrayYes, arrayNo, arrayMaybe])












# a tk.DrawingArea
def displayHist():
	global f
	global currentMagBin
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
	
	# transformedData = scipy.fft(left)
	# totalTransformed = transformedData.size
	# magnitude = zeros(totalTransformed)
	# i = 0
	# for datum in transformedData:
		# magnitude[i] = sqrt(square(datum.real) + square(datum.imag))
		# i += 1
	
	# magnitudeHalved = magnitude[0:totalTransformed/2]
	# totalSum =  sum(magnitudeHalved)
	# relMag = zeros(magnitudeHalved.size)
	
	# Xdb = 20*scipy.log10(scipy.absolute(transformedData))
	# fn = scipy.linspace(0, 44100, magnitudeHalved.size)
	
	
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
	
	a = f.add_subplot(121)
	a.clear()
	a2 = f.add_subplot(122)
	a2.clear()

	# fucnt = scipy.linspace(0, 44100, num=transformedData2.size/2)
	# print(fucnt)
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
	
	#ATTN: this is the call to the function you should
	#fill in with AI code, with the variable you want included.
	currentMagBin = magBin
	parseNewData(magBin)
	
#ATTN: fill in with AT code :P
def parseNewData(input_vector):
	#TODO: word recog
   print input_vector

   input_vector = map(lambda x: x/min(input_vector), input_vector)
##   print input_vector
   summed = [sum_sig(input_vector, 'yes'), sum_sig(input_vector, 'no'), sum_sig(input_vector, 'maybe')]
   result = []
   for val in summed:
      result.append(1 if val > 0.5 else 0)
   print input_vector
   print result
   
   entry.delete(0, len(entry.get()))
   entryMW.delete(0, len(entryMW.get()))
   entryMW.insert(0, dictionary[result.index(max(result))])

#sets all weights in a weight array to a random number
#between -1, and 1
def set_random_weights(weight_vector):
  for index in range(len(weight_vector)):
    weight_vector[index] = r.uniform(-1,1)
  return weight_vector
  
root = Tk.Tk()
root.wm_title("Embedding in TK")

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


button2 = Tk.Button(root, text="Learn", command=trainActual)
button2.grid(row = 1, column = 1)
button2.pack(side = Tk.RIGHT, padx=8,pady=8)

root.mainloop()



# button = Tk.Button(frame, text="QUIT", command="frame.quit")
# button.pack(side=LEFT)







