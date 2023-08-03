from AIDetector_pytorch import Detector
from configs.cfg import cfg
import imutils
import cv2

def main():

    name = 'demo'
    det = Detector()
    # cap = cv2.VideoCapture(r'E:\work\AI_Project\ComputerVision\target_track\deep_tracking\examples\cat.mp4')
    cap = cv2.VideoCapture(cfg["det"]["source"])
    # cap = cv2.VideoCapture(0)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im)
        # print("result------>:", result["faces"], result["face_bboxes"])
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
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
    det = Detector()
    cap = cv2.VideoCapture(r'E:\work\AI_Project\ComputerVision\target_track\deep_tracking\examples\cat.mp4')
    # cap = cv2.VideoCapture(0)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None

    while True:

        _, im = cap.read()
        if im is None:
            break
        result = det.feedCap_api(im)
        # print("result------>:", result["faces"], result["face_bboxes"])
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    # 本地运行目标检测
    # main()

    # 调用api实现目标检测
    run_api()