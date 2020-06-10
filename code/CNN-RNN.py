'''
Jasper R. Sunga
2015-01771
CNN vs RNN Model Application

This application is used to extract faces from an image, data augment images, train CNN/RNN Models, and evaluate CNN/RNN Models
'''

'''
import APIs and pre-existing libraries used for face detection/recognition and Neural Network models
'''

import tkinter as tk
from tkinter import *
import tkinter.filedialog as filedialog

import random
from random import randrange

from PIL import Image, ImageTk

import numpy as np
import cv2

import face_recognition

import os
import os.path
from os import path
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LSTM, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import pickle


button_foreground = "blue"
button_background = "gray82"

if(not (path.exists("Checker.txt") and path.exists("Known Faces") and path.exists("Unknown Group Pictures") and path.exists("Unknown Individual Faces"))):
	print("Either Known Faces, Unknown Group Pictures, Unknown Individual Faces, Known Faces, or Checker.txt does not exist")
	quit()

checker = open("Checker.txt","r")
cnn_loaded, rnn_loaded, filled_known_folder, number_unknown_faces = checker.read().split("\n")
checker.close()


#GLOBALS
cnn_loaded = int(cnn_loaded)
rnn_loaded = int(rnn_loaded)
filled_known_folder = int(filled_known_folder)
number_unknown_faces = int(number_unknown_faces)
current_face_shown = 0
image = None
image_size = 150

root_window = Tk()
root_window.title("CNN vs RNN Face Recognition")
root_window.configure(background = "white smoke")
root_window.geometry("400x500")
root_window.update()
root_window.minsize(root_window.winfo_width(),root_window.winfo_height())


def Save_Checker():
	checker = open("Checker.txt","w")

	checker.write(str(cnn_loaded)+"\n"+str(rnn_loaded)+"\n"+str(filled_known_folder)+"\n"+str(number_unknown_faces))

	checker.close()

def Pop_Up(msg):
	popup = tk.Tk()
	popup.wm_title("Pop Up!")
	label = Label(popup, text=msg)
	label.pack(side="top", fill="x", pady=10)
	B1 = Button(popup, text="Okay", command = popup.destroy)
	B1.pack()
	popup.mainloop()

def Separate_Group_Pictures(root):
	global number_unknown_faces
	group_folder = path.join(os.getcwd(),"Unknown Group Pictures")

	for img in os.listdir(group_folder):
		print(img)
		image = face_recognition.load_image_file(path.join(group_folder,img))

		face_locations = face_recognition.face_locations(image)

		print("I found {} face(s) in this photograph.".format(len(face_locations)))
		for face_location in face_locations:

			# Print the location of each face in this image
			top, right, bottom, left = face_location

			# You can access the actual face itself like this:
			face_image = image[top:bottom, left:right]
			pil_image = Image.fromarray(face_image)

			pil_image.save("Unknown Individual Faces/"+str(number_unknown_faces)+".png","PNG")
			number_unknown_faces += 1
	Save_Checker()

	os.rename(group_folder,path.join(os.getcwd(),"Group Pictures {}").format(int(time.time())))
	os.mkdir(group_folder)

	Pop_Up("Successfully Separated Faces")

def Delete_Image(root,image_label,classification, classification_text, unknown_faces,faces_path,labeled_path):
	global current_face_shown,image

	os.remove(path.join(faces_path,unknown_faces[current_face_shown]))

	classification_text.delete(first=0,last=len(classification.get()))

	if(len(unknown_faces)-1==current_face_shown):
		root.destroy()
		root.update()
		return

	current_face_shown += 1

	image = ImageTk.PhotoImage(Image.open(path.join(faces_path,unknown_faces[current_face_shown])))
	image_label.configure(image=image)

def Move_Image(root,image_label,classification, classification_text, unknown_faces,faces_path,labeled_path):
	global current_face_shown,image

	save_folder = path.join(labeled_path,classification.get())

	if(not os.path.exists(save_folder)):
		os.mkdir(save_folder)

	os.rename(path.join(faces_path,unknown_faces[current_face_shown]),path.join(save_folder,unknown_faces[current_face_shown]))

	classification_text.delete(first=0,last=len(classification.get()))


	if(len(unknown_faces)-1==current_face_shown):
		root.destroy()
		root.update()
		return

	current_face_shown += 1

	image = ImageTk.PhotoImage(Image.open(path.join(faces_path,unknown_faces[current_face_shown])))
	image_label.configure(image=image)

def Label_Unknown_Faces(root):

	current_path = os.getcwd()
	labeled_path = path.join(current_path,"Known Faces")
	faces_path = path.join(current_path,"Unknown Individual Faces")

	unknown_faces = os.listdir(faces_path)

	if(len(unknown_faces)==0):
		Pop_Up("There is no image in the Folder named Unknown Individual Faces")
		label_window.destroy()
		label_window.update()
		return
		
	label_window = Toplevel()
	label_window.configure(background = "white smoke")
	label_window.geometry("500x500")
	label_window.update()
	label_window.minsize(label_window.winfo_width(),label_window.winfo_height())

	label_window.wm_title("Label Unknown Faces")

	global current_face_shown, filled_known_folder, image
	current_face_shown = 0


	image = ImageTk.PhotoImage(Image.open(path.join(faces_path,unknown_faces[current_face_shown])))
	image_label = Label(label_window, image=image)
	image_label.pack(pady=20)


	label = Label(label_window,text="Classification:")
	label.pack(pady=50,padx=20,side="left")

	classification = StringVar()
	classification_text = Entry(label_window, width = 20, textvariable = classification)
	classification_text.pack(pady=20,padx=20, side="left")

	button = Button(label_window, text="Include",fg= button_foreground, bg = button_background,command = lambda: Move_Image(
		label_window,image_label,classification, classification_text, unknown_faces,faces_path,labeled_path))
	button.pack(pady=50,padx=20,side="right")

	button = Button(label_window, text="Exclude",fg=button_foreground,bg=button_background,command=lambda: Delete_Image(
		label_window,image_label,classification,classification_text,unknown_faces,faces_path,labeled_path))
	button.pack(pady=50,side="bottom")

	label_window.mainloop()

	filled_known_folder = 1
	Save_Checker()

def Augment_Classes(root, data_num):

	num = int(data_num.get())

	save_folder = path.join(os.getcwd(),"Known Faces")

	categories = sorted(os.listdir(save_folder))

	datagen = ImageDataGenerator(rotation_range = 40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

	for category in categories:
		class_path = path.join(save_folder,category)

		images = os.listdir(class_path)

		
		image_location = filedialog.askopenfilename(initialdir = class_path,title="Choose an Image",filetypes=[("image files",(".png",".jpg"))])

		while not image_location:
			image_location = filedialog.askopenfilename(initialdir = class_path,title="Choose an Image",filetypes=[("image files",(".png",".jpg"))])			

		dirpath = os.getcwd()

		common_prefix = os.path.commonprefix([dirpath,image_location])

		image_location = os.path.relpath(image_location, common_prefix)

		image = load_img(image_location)

		image_array = img_to_array(image)

		image_array = image_array.reshape((1,)+ image_array.shape)

		i = len(images)

		if(i >= num):
			continue

		for batch in datagen.flow(image_array, batch_size = 1, save_to_dir = class_path,save_prefix= category+"{}".format(int(time.time())), save_format="png"):
			i += 1
			if i == num:
				break

	root.destroy()
	root.update()
	Pop_Up("Data Augmented!")

def Image_Augment_Classes(root):

	input_window = Toplevel()

	label = Label(input_window,text="How many data will be the result on each class?")
	label.pack(pady=20)

	data_num = StringVar()
	data_num_text = Entry(input_window, width = 20, textvariable = data_num)
	data_num_text.pack(pady=20,padx=20, side = "left")

	button = Button(input_window, text="Enter", fg = button_foreground, bg = button_background, command = lambda: Augment_Classes(input_window,data_num))
	button.pack(pady=20,side="right")

	input_window.mainloop()

def Train_CNN(root):
	training_data = []
	data_path = path.join(os.getcwd(),"Known Faces")

	categories = sorted(os.listdir(data_path))

	for category in categories:
		category_path = path.join(data_path,category)
		category_num = categories.index(category)
		for img in os.listdir(category_path):
			img_array = cv2.imread(path.join(category_path,img),cv2.IMREAD_GRAYSCALE)
			new_img = cv2.resize(img_array, (image_size, image_size))

			training_data.append([new_img,category_num])

	random.shuffle(training_data)

	X = []
	y = []

	for features, label in training_data:
		X.append(features)
		y.append(label)

	X = np.array(X).reshape(-1, image_size, image_size, 1)

	X = X/255.0

	y = np.array(y)
	print(y)

	if(path.exists("CNN-Model.model")):
		cnn_model = tf.keras.models.load_model("CNN-Model.model")
		cnn_model.save("CNN-Model-{}.model".format(int(time.time())))
	else:
		cnn_model = Sequential()

		cnn_model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
		cnn_model.add(Activation("relu"))
		cnn_model.add(MaxPooling2D(pool_size=(2,2)))

		for i in range(0,2):
			cnn_model.add(Conv2D(64, (3,3)))
			cnn_model.add(Activation("relu"))
			cnn_model.add(MaxPooling2D(pool_size=(2,2)))
		
		cnn_model.add(Flatten())
		cnn_model.add(Dense(len(categories)))
		cnn_model.add(Activation("softmax"))

		cnn_model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


	earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')

	cnn_model.fit(X, y, batch_size=16, epochs=1000,validation_split=0.1, callbacks= [earlyStopping])

	print(K.get_value(cnn_model.optimizer.lr))

	cnn_model.save("CNN-Model.model")


def Train_RNN(root):
	training_data = []
	data_path = path.join(os.getcwd(),"Known Faces")

	categories = sorted(os.listdir(data_path))

	for category in categories:
		category_path = path.join(data_path,category)
		category_num = categories.index(category)
		for img in sorted(os.listdir(category_path)):
			img_array = cv2.imread(path.join(category_path,img),cv2.IMREAD_GRAYSCALE)
			new_img = cv2.resize(img_array, (image_size, image_size))

			training_data.append([new_img,category_num])

	random.shuffle(training_data)

	X = []
	y = []

	for features, label in training_data:
		X.append(features)
		y.append(label)

	print(len(X))

	X = np.array(X).reshape(-1, image_size, image_size)

	X = X/255.0

	y = np.array(y)

	print(X.shape)

	if(path.exists("RNN-Model.model")):
		rnn_model = tf.keras.models.load_model("RNN-Model.model")
		rnn_model.save("RNN-Model-{}.model".format(int(time.time())))
	else:
		rnn_model = Sequential()

		rnn_model.add(LSTM(64,input_shape=X.shape[1:],activation="relu",return_sequences=True))
		rnn_model.add(Dropout(0.3))

		rnn_model.add(LSTM(64, activation="relu"))
		rnn_model.add(Dropout(0.3))

		rnn_model.add(Dense(16,activation="relu"))
		rnn_model.add(Dropout(0.3))

		rnn_model.add(Dense(len(categories),activation="softmax"))

		optimizer = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

		rnn_model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

	earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')

	rnn_model.fit(X, y, batch_size=16, epochs=1000,validation_split=0.1, callbacks= [earlyStopping])

	rnn_model.save("RNN-Model.model")

def Classify_CNN(root):
	group_folder = path.join(os.getcwd(),"Testing Pictures")

	categories = sorted(os.listdir(group_folder))

	if(path.exists("CNN-Model.model")):
		cnn_model = tf.keras.models.load_model("CNN-Model.model")
	else:
		print("No CNN Model yet.")
		return

	for category in categories:

		category_path = path.join(group_folder,category)
		category_num = categories.index(category)

		length = len(os.listdir(category_path))
		right = 0

		print("Currently testing: "+ category)
		for img in os.listdir(category_path):
			img_array = cv2.imread(path.join(category_path,img),cv2.IMREAD_GRAYSCALE)
			new_img = cv2.resize(img_array, (image_size, image_size))

			testing_data = []


			testing_data.append(new_img)

			testing_data = np.array(testing_data).reshape(-1, image_size, image_size, 1)

			testing_data = testing_data/255.0

			prediction = cnn_model.predict(testing_data)

			print(prediction[0])

			print(img + " is "+str(prediction[0][np.argmax(prediction[0])])+"% sure that it is "+categories[np.argmax(prediction[0])])

			if(categories[np.argmax(prediction[0])]==category):
				right += 1

		print("Testing Done: "+ str(right)+ " out of " + str(length)+" or "+str(right/length))

def Classify_RNN(root):
	group_folder = path.join(os.getcwd(),"Testing Pictures")

	categories = sorted(os.listdir(group_folder))

	if(path.exists("RNN-Model.model")):
		rnn_model = tf.keras.models.load_model("RNN-Model.model")
	else:
		print("No RNN Model yet.")
		return

	for category in categories:

		category_path = path.join(group_folder,category)
		category_num = categories.index(category)

		length = len(os.listdir(category_path))
		right = 0

		print("Currently testing: "+ category)
		for img in os.listdir(category_path):
			img_array = cv2.imread(path.join(category_path,img),cv2.IMREAD_GRAYSCALE)
			new_img = cv2.resize(img_array, (image_size, image_size))

			testing_data = []


			testing_data.append(new_img)

			testing_data = np.array(testing_data).reshape(-1, image_size, image_size)

			testing_data = testing_data/255.0

			prediction = rnn_model.predict(testing_data)

			print(prediction[0])

			print(img + " is "+str(prediction[0][np.argmax(prediction[0])])+"% sure that it is "+categories[np.argmax(prediction[0])])

			if(categories[np.argmax(prediction[0])]==category):
				right += 1

		print("Testing Done: "+ str(right)+ " out of " + str(length)+" or "+str(right/length))


def Call_Main_Menu(root):

	button = Button(root,text="Separate Group Pictures",fg = button_foreground,bg=button_background,command = lambda: Separate_Group_Pictures(root))
	button.grid(row=0,column=1,pady=20)

	button = Button(root,text="Train CNN Model",fg = button_foreground,bg=button_background,command = lambda: Train_CNN(root))
	button.grid(row=0,column=0,pady=20)

	button = Button(root,text="Train RNN Model",fg = button_foreground,bg=button_background,command = lambda: Train_RNN(root))
	button.grid(row=1,column=0,pady=20)

	button = Button(root,text="Classify Testing Data - CNN",fg = button_foreground,bg=button_background,command = lambda: Classify_CNN(root))
	button.grid(row=2,column=0,pady=20)

	button = Button(root,text="Classify Testing Data - RNN",fg = button_foreground,bg=button_background,command = lambda: Classify_RNN(root))
	button.grid(row=3,column=0,pady=20)

	button = Button(root,text="Label Unknown Faces",fg = button_foreground,bg=button_background,command = lambda: Label_Unknown_Faces(root))
	button.grid(row=1,column=1,pady=20)
	
	button = Button(root,text="Image Augment Classes",fg = button_foreground,bg=button_background,command = lambda: Image_Augment_Classes(root))
	button.grid(row=2,column=1,pady=20)


	button = Button(root,text="Exit",fg = button_foreground,bg=button_background,command = root.destroy)
	button.grid(row=3,column=1,pady=20)


Call_Main_Menu(root_window)
root_window.mainloop()