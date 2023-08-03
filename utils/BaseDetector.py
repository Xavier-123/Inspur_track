from tracker import update_tracker, update_tracker_api
import cv2
from configs.cfg import cfg


class baseDet(object):

    def __init__(self):

        # self.img_size = 640
        # self.threshold = 0.3
        # self.stride = 1
        self.img_size = cfg["det"]["img_size"]
        self.threshold = cfg["det"]["threshold"]
        self.stride = cfg["det"]["stride"]

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1

        im, faces, face_bboxes = update_tracker(self, im)

        retDict['frame'] = im                      # 原始图像
        retDict['faces'] = faces                   # 目标图像，trackId
        retDict['face_bboxes'] = face_bboxes       # bbox

        return retDict

    def feedCap_api(self, im):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1

        im, faces, face_bboxes = update_tracker_api(self, im)

        retDict['frame'] = im                      # 原始图像
        retDict['faces'] = faces                   # 目标图像，trackId
        retDict['face_bboxes'] = face_bboxes       # bbox

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self, im):
        raise EOFError("Undefined model type.")

    def detect(self, im):
        raise EOFError("Undefined model type.")

    def v8_preprocess(self, im):
        raise EOFError("Undefined model type.")