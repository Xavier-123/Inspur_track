import contextlib
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, LOGGER, ROOT, USER_CONFIG_DIR,
                                    IterableSimpleNamespace, __version__, checks, colorstr, deprecation_warn,
                                    get_settings, yaml_load, yaml_print)

# Define valid tasks and modes
MODES = 'train', 'val', 'predict', 'export', 'track', 'benchmark'
TASKS = 'detect', 'segment', 'classify', 'pose'
TASK2DATA = {
    'detect': 'coco128.yaml',
    'segment': 'coco128-seg.yaml',
    'classify': 'imagenet100',
    'pose': 'coco8-pose.yaml'}
TASK2MODEL = {
    'detect': 'yolov8n.pt',
    'segment': 'yolov8n-seg.pt',
    'classify': 'yolov8n-cls.pt',
    'pose': 'yolov8n-pose.pt'}
TASK2METRIC = {
    'detect': 'metrics/mAP50-95(B)',
    'segment': 'metrics/mAP50-95(M)',
    'classify': 'metrics/accuracy_top1',
    'pose': 'metrics/mAP50-95(P)'}

CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['yolo'] + sys.argv[1:])}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

    4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    5. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

CFG_FLOAT_KEYS = 'warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear'
CFG_FRACTION_KEYS = ('dropout', 'iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr',
                     'label_smoothing', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud',
                     'fliplr', 'mosaic', 'mixup', 'copy_paste', 'conf', 'iou', 'fraction')  # fraction floats 0.0 - 1.0
CFG_INT_KEYS = ('epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride',
                'line_width', 'workspace', 'nbs', 'save_period')
CFG_BOOL_KEYS = ('save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val',
                 'save_json', 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop',
                 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks', 'boxes', 'keras',
                 'optimize', 'int8', 'dynamic', 'simplify', 'nms', 'v5loader', 'profile')

def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg

def _handle_deprecation(custom):
    """
    Hardcoded function to handle deprecated config keys
    """

    for key in custom.copy().keys():
        if key == 'hide_labels':
            deprecation_warn(key, 'show_labels')
            custom['show_labels'] = custom.pop('hide_labels') == 'False'
        if key == 'hide_conf':
            deprecation_warn(key, 'show_conf')
            custom['show_conf'] = custom.pop('hide_conf') == 'False'
        if key == 'line_thickness':
            deprecation_warn(key, 'line_width')
            custom['line_width'] = custom.pop('line_thickness')

    return custom


def check_cfg_mismatch(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (Dict): a dictionary of custom configuration options
        base (Dict): a dictionary of base configuration options
    """
    custom = _handle_deprecation(custom)
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    if mismatched:
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base)  # key list
            matches = [f'{k}={DEFAULT_CFG_DICT[k]}' if DEFAULT_CFG_DICT.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_cfg_mismatch(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':  # assign model to 'name' arg
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. "
                                     f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")

    # Return instance
    return IterableSimpleNamespace(**cfg)