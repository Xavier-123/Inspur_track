from pydantic import BaseModel, Field
from typing import Any
from fastapi import FastAPI, Request, status
from AIDetector_pytorch import Detector
from configs.cfg import cfg
import uvicorn
import logging
import base64
import numpy as np
import cv2

app = FastAPI()
logger = logging.getLogger()
# 设置日志等级
logger.setLevel(logging.INFO)


class result(object):
    def __init__(self, isSuc, msg, res, code):
        self.isSuc = isSuc
        self.code = code
        self.msg = msg
        self.res = res


# 返回值结构及样例定义
class Response(BaseModel):
    isSuc: bool = Field(..., example=True)
    code: int = Field(..., example=0)
    msg: str = Field(..., example="success")
    res: Any = Field(..., example="hello, <text>")


# post body参数结构定义
class Train_Data(BaseModel):
    imgbase64: str  # 图像


def _b642cv(img_b64):
    img_b64 = img_b64.encode('ascii')
    img_base64 = base64.b64decode(img_b64)
    img_array = np.frombuffer(img_base64, np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img_cv


# 处理目标检测
@app.post("/inspur_track",
          summary="post body参数模式，json格式，针对长参数",  # 接口基本简介，swagger显示，非必须
          # response_model=Response,  # 返回值样例，swagger显示，非必须
          tags=["AN推理接口"])  # 标签，swagger显示，非必须
def inspur_track(trainData: Train_Data):
    try:
        det = Detector()

        # base64转图像
        img = _b642cv(trainData.imgbase64)

        im, pred_boxes = det.detect(img)
        results = {"outInfo": "Reasoning is OK", "pred_boxes": pred_boxes}  # 推理接口返回数据，可为空
        return result(True, "success", results, 0).__dict__

    except:
        return result(False, "decode str error", None, -1).__dict__


if __name__ == '__main__':
    # uvicorn.run(app, host='0.0.0.0', port=cfg.port, log_config=None, access_log=False)
    print(cfg["api"]["port"])
    uvicorn.run(app, host='0.0.0.0', port=cfg["api"]["port"], log_config=None, access_log=False)
