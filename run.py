from main import *

modelurl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"      #less objects less confidence
# modelurl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"               #more objects less confidence
# modelurl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz"                #less objects high confidence

classFile = "coco.names"

imagePath = "bjh.jpg"
videoPath = "vid2.mp4"
# videoPath = 0    #for webcam
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelurl)
detector.loadModel()
# detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)