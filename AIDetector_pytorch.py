import torch
import numpy as np
from utils.general import non_max_suppression
from utils.BaseDetector import baseDet
from utils.torch_utils import select_device
from utils.dataloaders import letterbox
from pathlib import Path
from models.model import YOLO

from models.tasks import attempt_load_one_weight
from yolo.data.augment import LetterBox
from yolo.engine.results import Results
from utils import ops
from configs.cfg import cfg


class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _load(self, weights: str, task=None, device="cpu"):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights, device=device)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    def init_model(self):
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)

        # model = YOLO(r'D:/Inspur/base_model/yolo8/yolov8s.pt')  # load a custom model
        # # model.model.to(self.device).eval()
        # model.model.eval()
        # model.model.float()  # half() or float()
        # self.m = model
        # self.names = model.module.names if hasattr(model, 'module') else model.names

        # 脱离ultralytics
        self._load(r'D:/Inspur/base_model/yolo8/yolov8s.pt', device=self.device)
        self.m = self.model.eval()

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def v8_preprocess(self, im):
        img0 = im.copy()
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform([im]))
            if len(im.shape) == 3:
                im = im[np.newaxis, :]
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.float()  # half() or float()
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img0, img

    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes
        return [LetterBox(self.img_size, auto=auto)(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds,
                                    self.threshold,
                                    0.4,
                                    agnostic=False,
                                    max_det=300,
                                    classes=None)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = ""
            results.append(Results(orig_img=orig_img, path=img_path, names=self.m.names, boxes=pred))
        return results

    def detect(self, im):
        # im0, img = self.preprocess(im)
        im0, img = self.v8_preprocess(im)

        pred = self.m(img, augment=False, visualize=False)

        pred = self.postprocess(pred, img, im0)

        pred_boxes = []
        for det in pred:

            det = det.boxes.data  # xaw

            if det is not None and len(det):

                for *x, conf, cls_id in det:
                    lbl = self.m.names[int(cls_id)]  # xaw
                    _conf = conf.item()

                    # if not lbl in ['person', 'car', 'truck', 'cat']:  # ['person', 'car', 'truck', 'cat']需要追踪的类别
                    if not lbl in cfg["det"]["target"]:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        # (x1, y1, x2, y2, lbl, conf))
                        (x1, y1, x2, y2, lbl, _conf))

        return im, pred_boxes
