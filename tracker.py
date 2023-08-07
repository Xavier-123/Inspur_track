from MMT.deep_sort.utils.parser import get_config
from MMT.deep_sort.deep_sort import DeepSort
from configs.cfg import DEFAULT_CFG
from configs.cfg import url
import torch
import cv2
import requests
import json
import base64


# save_result = "./save_reslut"
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("MMT/deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['person']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def img_to_imgbs64(img_np):
    """
    convert img_np to img_bs64
    """
    img_str = cv2.imencode('.jpg', img_np)[1].tobytes()  # 将图片编码成流数据，放到内存缓存中，然后转化成byte格式
    img_byte = base64.b64encode(img_str)  # 编码为base64
    imgbs64 = img_byte.decode('ascii')
    return imgbs64


def post_test(encoded_string):
    request_example = {
        "imgbase64": encoded_string
    }
    test_post_response = requests.post(url, json=request_example)
    return json.loads(test_post_response.content.decode('utf-8'))


def update_tracker(target_detector, image):
    new_faces = []
    _, bboxes = target_detector.detect(image)

    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)
        # if cls_ == 'cat':  # xaw
        if cls_ in DEFAULT_CFG.track.track_target:  # xaw
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )

        # 保存txt
        with open(str(DEFAULT_CFG.track.save_path) + "/save_result.txt", 'a') as f:
            # print("frame_id---> ", deepsort.frame_id)
            f.write(str(deepsort.frame_id) + ', ' + str(value) + '\n')
    deepsort.frame_id += 1

    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        # if target_detector.faceTracker[history_id] < -5:
        if target_detector.faceTracker[history_id] < -int(DEFAULT_CFG.track.max_history_id):
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw)

    return image, new_faces, face_bboxes


def update_tracker_api(target_detector, image):
    img_base64 = img_to_imgbs64(image)
    test_post_response = post_test(img_base64)

    new_faces = []
    bboxes = test_post_response["res"]["pred_boxes"]

    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)
        # if cls_ in ['cat', "car", "person"]:  # xaw
        if cls_ in DEFAULT_CFG.track.track_target:  # xaw
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )

        # 保存txt
        with open(str(DEFAULT_CFG.track.save_path) + "/save_result.txt", 'a') as f:
            f.write(str(deepsort.frame_id) + ', ' + str(value) + '\n')
    deepsort.frame_id += 1

    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -int(DEFAULT_CFG.track.max_history_id):
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw)

    return image, new_faces, face_bboxes
