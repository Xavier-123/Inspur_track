import requests
import json
import base64
import cv2
from AIDetector_pytorch import Detector
import imutils
import yaml
import os

def img_to_imgbs64(img_np):
    """
    convert img_np to img_bs64
    """
    img_str = cv2.imencode('.jpg', img_np)[1].tobytes() #将图片编码成流数据，放到内存缓存中，然后转化成byte格式
    img_byte = base64.b64encode(img_str) # 编码为base64
    imgbs64 = img_byte.decode('ascii')
    return imgbs64

def getB64strByFilepath(filepath):
    with open(filepath, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str

def post_test(encoded_string):
    get_test_url = f"http://localhost:9048/inspur_track"
    request_example = {
        "imgbase64": encoded_string
    }
    # get_test_url = f"http://192.168.12.84:9003/solr/search_topic/?group=guzhangdingyi&q=业务类全专业网络故障橙色预警发布标准是什么？"

    test_post_response = requests.post(get_test_url, json=request_example)
    # print(json.loads(test_post_response.content.decode('utf-8')))
    return json.loads(test_post_response.content.decode('utf-8'))

def main():
    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture(r'E:\work\AI_Project\ComputerVision\target_track\deep_tracking\examples\cat.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None

    while True:
        _, im = cap.read()
        print(type(im))
        # break
        if im is None:
            break

        # result = det.feedCap(im)
        result = det.feedCap_api(im)

        print("result------>:", result["faces"], result["face_bboxes"])
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter('result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()



def read_yaml_all(yaml_path):
    try:
        # 打开文件
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except:
        return None

yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/configs.yaml")
cfg = read_yaml_all(yaml_path)
print(cfg)