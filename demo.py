from AIDetector_pytorch import Detector
from configs.cfg import DEFAULT_CFG
from utils import LOGGER
import imutils
import cv2

def main():

    name = 'demo'
    det = Detector(True)
    cap = cv2.VideoCapture(DEFAULT_CFG.det.source)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None
    frame_id = 0

    # 清空文件
    if DEFAULT_CFG.track.save:
        path = str(DEFAULT_CFG.track.save_path) + "/save_result.txt"
        with open(path, 'w') as f:
            f.truncate(0)
        LOGGER.info("清空save_result.txt")

    while True:

        # try:
        _, im = cap.read()
        # print(_)
        if im is None:
            break
        result = det.feedCap(im, frame_id)
        # print("result------>:", result["faces"], result["face_bboxes"])
        result = result['frame']
        result = imutils.resize(result, height=500)

        # 保存跟踪视频
        if DEFAULT_CFG.track.save:
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
            videoWriter.write(result)


        if DEFAULT_CFG.track.show:
            cv2.imshow(name, result)
            cv2.waitKey(t)

            if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


# 调用api
def run_api():
    name = 'demo'
    det = Detector(False)
    cap = cv2.VideoCapture(DEFAULT_CFG.det.source)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None
    frame_id = 0

    # 清空文件
    if DEFAULT_CFG.track.save:
        with open(str(DEFAULT_CFG.track.save_path) + "/save_result.txt") as f:
            f.truncate(0)
        LOGGER.info("清空save_result.txt")

    while True:

        _, im = cap.read()
        if im is None:
            break
        result = det.feedCap_api(im, frame_id)
        # print("result------>:", result["faces"], result["face_bboxes"])
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
            videoWriter.write(result)

        if DEFAULT_CFG.track.show:
            cv2.imshow(name, result)
            cv2.waitKey(t)

            if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    if DEFAULT_CFG.track.api:
        # 调用api实现目标检测
        run_api()
    else:
        # 本地运行目标检测
        main()

