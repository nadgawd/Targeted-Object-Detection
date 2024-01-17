import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
import keras

np.random.seed(20)

class Detector:
    def __init__(self):
        pass 
    
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        print(len(self.classesList), len(self.colorList))
    
    def downloadModel(self, modelurl):
        fileName = os.path.basename(modelurl)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName, origin=modelurl, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model "+self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Model "+ self.modelName + " loaded successfully")


    def createBoundBox(self, image, threshold=0.5):
        inpTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inpTensor = tf.convert_to_tensor(inpTensor, dtype=tf.uint8)
        inpTensor = inpTensor[tf.newaxis,...]
        detection = self.model(inpTensor)
        bboxes = detection['detection_boxes'][0].numpy()
        classIndexes = detection['detection_classes'][0].numpy().astype(np.int32)
        classScores = detection['detection_scores'][0].numpy()
        

        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxes, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
        print(bboxIdx)
        if(len(bboxes)) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxes[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]
                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
        return image
    
    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundBox(image, threshold)
        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened() == False):
            print("Error opening file")
            return
        (success, image) = cap.read()
        startTime = 0
        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            bboxImage = self.createBoundBox(image, threshold)

            cv2.putText(bboxImage, "FPS: "+str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", bboxImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()

    
   
    