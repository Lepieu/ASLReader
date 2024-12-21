# ASLReader

## About the Project

This project was created for an internship at Mangofarm Assets LLC. The goal was to create a Convolutional Neural Network to recognize images (or break a video into images and recognize them). The model currently is able to read American Sign Language (ASL), with 29 catgories representing each letter, as well as a few special characters, but it can be configured to recognize any type of image with any number of categories.

## About the Repository

The repository contains two Convolutional Neural Networks -- one made using the Keras library, and the other created from scratch, both of which are made using Python. The Keras model is more accurate, but the other model is completely programmed by me, with guidance from my mentor. 

## Prerequisites

A collection of images used for training and testing must be inputted into the folder titled "Database," and ordered into separate categorical files. These files can be given any name. They do not need to be split into separate training and testing folders, as that is done by the program. 

## Usage

The "lettertonumber" dictionary must be edited within the source code, as well as the "outputs" global variable. "outputs" must be equal to the number of categories of images inputted. It must be the same as the number of files inputted in the Prerequisites section. The "lettertonumber" dictionary must be edited to include each file name instead of letters. Do not change the name of the dictionary itself, as it will mess something else up down the line. I plan on fixing this by changing it to "categorytonumber" at a later time (it was called that because of the dataset that was used when writing the code).

Example: 
Let's say I have 3 files as my categories, with the following titles: "Bee", "Three", "Misc." 

I would edit the "outputs" variable to equal 3, as there are 3 files.
I would also edit the "lettertonumber" dictionary to be:

lettertonumber = {
  "Bee" : 0
  "Three" : 1
  "Misc" : 2
}

## Future Implimentations


1. The keras model is currently much more accurate and faster than my own model. To get them to be closer in speed and accuracy, I plan to:
-    Change iteration during convolution and pooling steps, so it is faster than O(n^2) -- potentially by using a hashmap
-    Add more filters to increase accuracy
-    Standardize my methods of storing data to avoid redundancy
2. I need to change "lettertonumber" to "categorytonumber" (quick fix)
3. Once the neural network is trained, give the ability to save it and transfer to other images or videos -- I have already implimented a way to break a video into images, but it currently does not have much use due to the fact that the AI can't automatically categorize.
