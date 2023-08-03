import yaml
import os

def read_yaml_all(yaml_path):
    try:
        # 打开文件
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except:
        return None

yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
# print(yaml_path)
cfg = read_yaml_all(yaml_path)
url = "http://localhost:" + str(cfg["api"]["port"]) + "/inspur_track"  # http://localhost:9045/inspur_track
print(cfg)
# print(cfg["det"]["img_size"])