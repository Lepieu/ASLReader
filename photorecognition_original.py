import cv2 as cv
import os
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

import Model

images = list()
labels = list()

height = 128
width = 128
FPS = 10 
outputs = 29 #29 outputs for 26 letters, nothing, space, delete

EPOCHS = 10
TEST_SIZE = .2

lettertonumber = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
    "del": 26,
    "nothing": 27,
    "space": 28 
}

"""
Main method
issue: accuracy, but it works crudely now, which is good
"""
def main():

    print("Loading data...")
    (images, labels, numbered_labels) = load_data(os.path.join(os.getcwd(), "database", "ASL_Alphabet_Dataset", "train"))     

    print("Splitting data...")
    numbered_labels = tf.keras.utils.to_categorical(numbered_labels)
    

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(numbered_labels), test_size=TEST_SIZE
    )
    
    #compiles the neural network to extract features from each frame of the video
    print("Prepping neural network...")
    model = PrepNeuralNetwork()
    
    print("Training...")
    model.train(x_train, y_train, x_test, y_test, 10, 0.01)
    
    print("evaluating...")
    model.evaluate(x_test, y_test)
    
    
    
    """
    #processes the broad underlying features of the images
    feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    broad_features = feature_extractor.predict(images)
    
    #cluster features together to predict which letter it is
    cluster_labels = Cluster(Normalize(broad_features))
    """
    
    """_summary_
    Loads images to memory
    """
def load_data(path):
    
    #labels and data are parallel arrays, where labels contains the type ("a", "b", "space", etc.) while data contains the images
    data = list() 
    labels = list()
    numbered_labels = list()
    
    
    for subdir in os.listdir(path):
        if subdir == ".DS_Store":
            continue
        
        #remove_files(os.path.join(path, subdir), 200)
        
        for file in os.listdir(os.path.join(path, subdir)):
            
            img = cv.imread(os.path.join(path, subdir, file))
            img = TransformImage(img)
            
            #adds each image, along with its label, to the end of each respective array
            data.append(img)
            labels.append(subdir)
            numbered_labels.append(lettertonumber[subdir])
            
    return (data, labels, numbered_labels)

def remove_files(path, target):
    
    files = os.listdir(path)
    i = len(files)
    
    print("removing", i - target, "files...")
    
    for file in files:
        if i <= target:
            break
        os.remove(os.path.join(path, file))
        i -=1
        
    
"""
Returns images frame by frame from video
"""
def OpenVideo(filename: str):
    #TODO: impliment frame delay to standardize video FPS

    path = os.path.join(os.getcwd(), "database", filename) #path to the video in directory
    frames = list()
    
    video = cv.VideoCapture(path) #opens video capture
    
    #set frame delay
    original_fps = video.get(cv.CAP_PROP_FPS)
    skip = round(original_fps / min(FPS, original_fps)) #skips frames to make the video run at a certain FPS
    
    while video.isOpened(): 
        frameNumber = int(video.get(cv.CAP_PROP_POS_FRAMES)) #current frame
        
        ret, frame = video.read() #reads frame from 
        if ret == False:
            break
        
        frame = TransformImage(frame)
        frames.append(frame)
        
        """
        #play video (may delete later)
        cv.imshow("Frame", frame) #shows current frame as an image
        if (cv.waitKey(round(1000/FPS)) & 0xFF == ord('q')): #delays to match with FPS
            break
        """
        
        #next frame = current frame + skip
        video.set(cv.CAP_PROP_POS_FRAMES, frameNumber + skip)
    
    #if the video is closed or it cannot receive the frame data, release capture and terminate program if necessary
    video.release()
    cv.destroyAllWindows()
    
    return frames



def TransformImage(img):
    #TODO: Figure out a size to change the image to, and then plug it into the neural network
    
    img = cv.resize(img, (height, width), interpolation=cv.INTER_LINEAR) #Resize to height x width size
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Black and White
    
    return img

    """
Compiles neural network to process the images into [outputs] different categories
    """
def PrepNeuralNetwork():
    #TODO:     
    
    layers = [
        Model.setLayer("conv", (3, 3), 1),
        Model.setLayer("pool", (2, 2), 2),
        Model.setLayer("conv", (3, 3), 1),
        Model.setLayer("pool", (2, 2), 2)
    ]
    
    model = Model.Convolutional_NN((height, width), outputs, layers)
    
    return model


#Main method (idk if i'll use this just test with it)
if __name__ == "__main__":
    main()
    
    
