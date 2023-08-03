from utils import TryExcept, colorstr, LOGGER, emojis, USER_CONFIG_DIR

from matplotlib import font_manager
from pathlib import Path
import pkg_resources as pkg
import subprocess
import os
import glob
import urllib
import re

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    import socket

    for host in '1.1.1.1', '8.8.8.8', '223.5.5.5':  # Cloudflare, Google, AliDNS:
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            # If the connection was successful, close it to avoid a ResourceWarning
            test_connection.close()
            return True
    return False

@TryExcept()
def check_requirements(requirements=ROOT.parent / 'requirements.txt', exclude=(), install=True, cmds=''):
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.

    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.
    """
    prefix = colorstr('red', 'bold', 'requirements:')
    # check_python()  # check python version
    file = None
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f'{prefix} {file} not found, check failed.'
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''  # console string
    n = 0  # number of packages updates
    for r in requirements:
        rmin = r.split('/')[-1].replace('.git', '')  # replace git+https://org/repo.git -> 'repo'
        try:
            pkg.require(rmin)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            try:  # attempt to import (slower but more accurate)
                import importlib
                importlib.import_module(next(pkg.parse_requirements(rmin)).name)
            except ImportError:
                s += f'"{r}" '
                n += 1

    if s:
        if install and AUTOINSTALL:  # check environment variable
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
            try:
                assert is_online(), 'AutoUpdate skipped (offline)'
                LOGGER.info(subprocess.check_output(f'pip install --no-cache {s} {cmds}', shell=True).decode())
                s = f"{prefix} {n} package{'s' * (n > 1)} updated per {file or requirements}\n" \
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                LOGGER.info(s)
            except Exception as e:
                LOGGER.warning(f'{prefix} ❌ {e}')
                return False
        else:
            return False

    return True


def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'

def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if ('yolov3' in file or 'yolov5' in file) and 'u' not in file:
        original_file = file
        file = re.sub(r'(.*yolov5([nsmlx]))\.pt', '\\1u.pt', file)  # i.e. yolov5n.pt -> yolov5nu.pt
        file = re.sub(r'(.*yolov5([nsmlx])6)\.pt', '\\1u.pt', file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
        file = re.sub(r'(.*yolov3(|-tiny|-spp))\.pt', '\\1u.pt', file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
        if file != original_file and verbose:
            LOGGER.info(f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                        f'trained with https://github.com/ultralytics/ultralytics and feature improved performance vs '
                        f'standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n')
    return file

def clean_url(url):

    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return urllib.parse.unquote(url).split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth

def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name

def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = []
        for d in 'models', 'datasets', 'tracker/cfg', 'yolo/cfg':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file

def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)


def check_font(font='Arial.ttf'):
    name = Path(font).name

    # Check USER_CONFIG_DIR
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # Download to USER_CONFIG_DIR if missing
    url = f'https://ultralytics.com/assets/{name}'
    if downloads.is_url(url):
        downloads.safe_download(url=url, file=file)
        return file


def check_version(current: str = '0.0.0',
                  minimum: str = '0.0.0',
                  name: str = 'version ',
                  pinned: bool = False,
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        (bool): True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f'WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(warning_message)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result


def is_ascii(s) -> bool:
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)

def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
        # else:
        #     downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = []
        for d in 'models', 'datasets', 'tracker/cfg', 'yolo/cfg':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file