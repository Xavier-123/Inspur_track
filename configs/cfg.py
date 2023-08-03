import yaml
import os
import re

def read_yaml_all(yaml_path):
    try:
        # 打开文件
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except:
        return None

def yaml_load(file='data.yaml', append_filename=False):
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")



from MMT.deep_sort.utils.parser import *
DEFAULT_CFG = get_config()
DEFAULT_CFG.merge_from_file(yaml_path)
# url = "http://localhost:" + str(DEFAULT_CFG.api.port) + "/inspur_track"  # http://localhost:9045/inspur_track
url = "http://" + str(DEFAULT_CFG.api.host) + ":" + str(DEFAULT_CFG.api.port) + "/inspur_track"  # http://localhost:9045/inspur_track

# print(cfg)
# print(cfg.api.port)
# print(cfg.det.model_path)
# print(cfg.track)