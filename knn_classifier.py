import cv2
import os
import math
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import sys
import matplotlib.pyplot as plt
import itertools

global folder,imageDict,accuracy,best_k

if len(sys.argv) > 1:
    folder = sys.argv[1] # hhd_dataset
else:
    folder = "hhd_dataset"
imageDict = {}
splitDataDict = {}
accuracy = -1
best_k = -1


#Function to load all the images to a dict of lists, key = folder name, value = list of  grayscaled images.
def loadRawData():
    global folder,imageDict
    if folder == None:
        folder = "hhd_dataset"
    for root, subdirectories, files in os.walk(folder):
        for subdirectory in subdirectories:
            subdirectoryList = []
            for file in os.listdir((os.path.join(root,subdirectory))):
                subdirectoryList.append(cv2.imread(os.path.join(root,subdirectory,file)))
            imageDict[subdirectory] = subdirectoryList

#function to floor & ceil float numbers, convert them to int and randomliy assign to x,y.
def floor_and_ceil(x):
    variables = [int(math.floor(x)),int(math.ceil(x))]
    x = random.choice(variables)
    variables.remove(x)
    return x,variables[0]

#Preproccessing function, padding,resize.
def pre_processing():
    global imageDict

    total = 0
    for key, value in imageDict.items():
        total += len(value)

    white = [255,255,255]
    for key in imageDict.keys():
        newImageList = []
        for image in imageDict[key]:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height,width = image.shape[:2]
            if height > width:
                paddingRange = (height - width)/2
                x,y = floor_and_ceil(paddingRange)
                dst = cv2.copyMakeBorder(image,0,0,x,y,cv2.BORDER_CONSTANT,None,white)
            else:
                paddingRange = (width - height)/2
                x,y = floor_and_ceil(paddingRange)
                dst = cv2.copyMakeBorder(image,x,y,0,0,cv2.BORDER_CONSTANT,None,white)
            resized_image = cv2.resize(dst, (32, 32)).flatten()
            newImageList.append(resized_image)
        imageDict[key] = newImageList

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def splitData():
    global imageDict,splitDataDict,tempDict
    tempDict = {}
    total = 0

    for key, value in imageDict.items():
              total += len(value)

    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    train_images_per_class = int(math.floor(total * train_size / len(imageDict.keys())))
    val_images_per_class = int(math.floor(total * val_size / len(imageDict.keys())))
    test_images_per_class = int(math.floor(total * test_size / len(imageDict.keys())))

    train_x,test_x,val_x = [],[],[]
    train_y,test_y,val_y = [],[],[]


    for key in imageDict.keys():
        res = imageDict[key]
        sampleList = random.sample(res,test_images_per_class)
        for sample in sampleList:
            removearray(res,sample)
        test_x = [*test_x,*sampleList]
        test_y = [*test_y,*[int(key) for i in range(0,len(sampleList))]]

        sampleList = random.sample(res, val_images_per_class)
        for sample in sampleList:
            removearray(res,sample)
        val_x = [*val_x, *sampleList]
        val_y = [*val_y, *[int(key) for i in range(0, len(sampleList))]]

        sampleList = random.sample(res,min(len(res),train_images_per_class))
        for sample in sampleList:
            removearray(res,sample)
        train_x = [*train_x, *sampleList]
        train_y = [*train_y, *[int(key) for i in range(0, len(sampleList))]]

        tempDict[key] = res
    for key in tempDict.keys():
        res = imageDict[key]
        while len(res) > 0:
            choice = [0,1,2]
            x = random.sample(choice,1)
            x = x[0]

            sampleList = random.sample(res, 1)

            if int(math.floor(train_size * total) >= len(train_x)) and x == 0:
                train_x = [*train_x, *sampleList]
                train_y = [*train_y, *[int(key) for i in range(0, len(sampleList))]]
                for sample in sampleList:
                    removearray(res, sample)

            elif int(math.floor(val_size * total) >= len(val_x)) and x == 1:
                val_x = [*val_x, *sampleList]
                val_y = [*val_y, *[int(key) for i in range(0, len(sampleList))]]
                for sample in sampleList:
                    removearray(res, sample)

            elif int(math.floor(test_size * total) >= len(test_x)) and x == 2:
                test_x = [*test_x, *sampleList]
                test_y = [*test_y, *[int(key) for i in range(0, len(sampleList))]]
                for sample in sampleList:
                    removearray(res, sample)

    splitDataDict['train_x'] = train_x
    splitDataDict['test_x'] = test_x
    splitDataDict['val_x'] = val_x

    splitDataDict['train_y'] = train_y
    splitDataDict['test_y'] = test_y
    splitDataDict['val_y'] = val_y

    print("train_x size:", len((splitDataDict['train_x'])), "test_x size:", len(splitDataDict['test_x']), "val_x size:",
          len(splitDataDict['val_x']))
    print("train_y size:", len((splitDataDict['train_y'])), "test_y size:", len(splitDataDict['test_y']), "val_y size:",
          len(splitDataDict['val_y']))

def evaluateKNN():
    global accuracy,best_k,splitDataDict
    bestModel = None
    for k in range(1,16,2):
        knn = KNeighborsClassifier(metric='euclidean',n_neighbors=k)
        knn.fit(splitDataDict['train_x'],splitDataDict['train_y'])
        y_pred = knn.predict(splitDataDict['val_x'])

        if accuracy_score(splitDataDict['val_y'], y_pred) > accuracy:
            accuracy = accuracy_score(splitDataDict['val_y'], y_pred)
            best_k = k
            bestModel = knn

    model_and_accuracy = [bestModel,accuracy,splitDataDict]
    knnPickle = open('Best_KNN_Model', 'wb')
    pickle.dump(model_and_accuracy, knnPickle)

def loadModel():
    return  pickle.load(open('Best_KNN_Model', 'rb'))

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def calculateAccPerClass(y_pred,test_y):
    y_pred = y_pred.tolist()
    accuracyDict = {}
    for i in range(0, 27):
        accuracyDict[i] = [0, 0]
    for index in range(0, len(test_y)):
        if y_pred[index] == test_y[index]:
            accuracyDict[test_y[index]][0] += 1
            accuracyDict[test_y[index]][1] += 1
        else:
            accuracyDict[test_y[index]][1] += 1
    for key, value in accuracyDict.items():
        accuracyDict[key] = float("{0:.3f}".format(value[0]/value[1]))
    return accuracyDict

def writeResults(results,best_KNN):
    with open('results.txt', 'w') as f:
        f.write("Best K for KNN is:")
        f.write(str(best_KNN[0].n_neighbors))
        print("Best K for KNN is:",str(best_KNN[0].n_neighbors))
        print("Model accuracy is:",str(best_KNN[1]))
        f.write('\n')
        f.write("Model accuracy is:")
        f.write(str(best_KNN[1]))
        f.write('\n')
        f.write('\n')
        f.write("Letter    Accuracy")
        f.write('\n')
        print("Letter    Accuracy")

        for key in results.keys():
            f.write('  ')
            f.write(str(key))
            f.write('        ')
            f.write(str(results[key]))
            f.write('\n')
            print('  '+ str(key),'       ',results[key])

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',  cmap=None,normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    accuracy = float("{:.2f}".format(accuracy))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    plt.show()
    cm = trunc(cm,decs=3)

    pd.DataFrame(cm).to_csv('confusion_matrix.csv')

def saveData():
    global splitDataDict
    f = open('data_resized_padded_greyscale', 'wb')
    pickle.dump(splitDataDict, f)

def loadData():
    '''
    splitDataDict['train_x'] = train_x
    splitDataDict['test_x'] = test_x
    splitDataDict['val_x'] = val_x

    splitDataDict['train_y'] = train_y
    splitDataDict['test_y'] = test_y
    splitDataDict['val_y'] = val_y
    '''
    return pickle.load(open('data_resized_padded_greyscale', 'rb'))

#load the model, evaluate test data, write results.
def KNN_Model():
    best_KNN = loadModel()
    test_x,test_y = best_KNN[2]["test_x"],best_KNN[2]["test_y"]
    y_pred = best_KNN[0].predict(test_x)
    accuracy_per_letter = calculateAccPerClass(y_pred,test_y)
    writeResults(accuracy_per_letter,best_KNN)
    cm = confusion_matrix(y_pred=y_pred, y_true=test_y)
    plot_confusion_matrix(cm,normalize=True,target_names=[i for i in range(0,27)])




#pre conditions to read,pre-proccess,split and evaluate the KNN model.
#loadRawData()
#pre_processing()
#splitData()
#saveData()
#evaluateKNN()
KNN_Model()






