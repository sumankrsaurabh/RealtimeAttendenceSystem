import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

# first we're going to import faces and then were going to encode it & then we're going to dump it using the pickle library


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "",
    'storageBucket': ""
})


# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)  #this will give extension as well ex- d12.png

#now the list that will contain all the modes
imgList = []
studentIds = []  #importing ids
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))  #aise were going to add img
    studentIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


    # print(path)    #this will print d12.png
    # to remove it
    #  print(os.path.splitext(path[0]))
    # this will only print d12

    # print(os.path.splitext(path)[0])

    # now we have to put it all in studentIds
print(studentIds) #this will print out student ids without extentions
#now we will create a function where we will send list in the functions where itwill do all the encodings and spit out the images


def findEncodings(imagesList):
    encodeList = []

    # we will now loop through all the images and encode them one byy one

    for img in imagesList:
        # opencv user bgr and face recog uses rgb , we will convert the color

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  #we will encode pictures one by one
        encodeList.append(encode)

    return encodeList


print("Encoding Started ...")
# this will generate all of our encodings

encodeListKnown = findEncodings(imgList)   #find encoding for all the known list
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

#now we need to store it in a pickle file so that we import it whenever we need it
# when we store it in the pickle lib , we need to store 1. encodings , and 2. id's
file = open("EncodeFile.p", 'wb') #wb is the permission
pickle.dump(encodeListKnownWithIds, file) #were dumping it in pickle lib
file.close()
print("File Saved")