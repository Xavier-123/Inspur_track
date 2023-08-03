from pathlib import Path

GITHUB_ASSET_NAMES = [f'yolov8{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '6', '-cls', '-seg', '-pose')] + \
                     [f'yolov5{k}u.pt' for k in 'nsmlx'] + \
                     [f'yolov3{k}u.pt' for k in ('', '-spp', '-tiny')] + \
                     [f'yolo_nas_{k}.pt' for k in 'sml'] + \
                     [f'sam_{k}.pt' for k in 'bl'] + \
                     [f'rtdetr-{k}.pt' for k in 'lx']
GITHUB_ASSET_STEMS = [Path(k).stem for k in GITHUB_ASSET_NAMES]