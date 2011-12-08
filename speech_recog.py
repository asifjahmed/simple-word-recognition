import wave
import struct
import scipy
import pyaudio
from numpy import *
import Tkinter as Tk
import sys
import os


import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


global myCanvas
global toolbar
global f

# a tk.DrawingArea
def displayHist():
	global f
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
	
	magBin = magBin - minFreqBin
	
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

label = Tk.Label(root, text='Machine word:')
label.pack(side = Tk.LEFT, padx = 12,pady=8)

entry = Tk.Entry(root, width = 20)
entry.pack(side = Tk.LEFT, padx=0,pady=8)

label = Tk.Label(root, text='Correct word:')
label.pack(side = Tk.LEFT, padx = 12,pady=8)

entry = Tk.Entry(root, width = 20)
entry.pack(side = Tk.LEFT, padx=0,pady=8)


button2 = Tk.Button(root, text="Learn")
button2.grid(row = 1, column = 1)
button2.pack(side = Tk.RIGHT, padx=8,pady=8)

root.mainloop()



# button = Tk.Button(frame, text="QUIT", command="frame.quit")
# button.pack(side=LEFT)


