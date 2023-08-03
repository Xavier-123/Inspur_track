import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

import requests
from tqdm import tqdm

from utils import (ENVIRONMENT, LOGGER, ONLINE, RANK, SETTINGS, TESTS_RUNNING, TQDM_BAR_FORMAT,
                                    TryExcept, __version__, colorstr, get_git_origin_url, is_colab, is_git_dir,
                                    is_pip_package)

PREFIX = colorstr('Ultralytics HUB: ')
HELP_MSG = 'If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance.'
HUB_API_ROOT = os.environ.get('ULTRALYTICS_HUB_API', 'https://api.ultralytics.com')


def requests_with_progress(method, url, **kwargs):
    """
    Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        **kwargs (dict): Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        (requests.Response): The response object from the HTTP request.

    Note:
        If 'progress' is set to True, the progress bar will display the download progress
        for responses with a known content length.
    """
    progress = kwargs.pop('progress', False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get('content-length', 0))  # total size
    pbar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, bar_format=TQDM_BAR_FORMAT)
    for data in response.iter_content(chunk_size=1024):
        pbar.update(len(data))
    pbar.close()
    return response

def smart_request(method, url, retry=3, timeout=30, thread=True, code=-1, verbose=True, progress=False, **kwargs):
    retry_codes = (408, 500)  # retry only these codes

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        """Make HTTP requests with retries and timeouts, with optional progress tracking."""
        r = None  # response
        t0 = time.time()  # initial time for timer
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)  # i.e. get(url, data, json, files)
            if r.status_code < 300:  # return codes in the 2xx range are generally considered "good" or "successful"
                break
            try:
                m = r.json().get('message', 'No JSON message.')
            except AttributeError:
                m = 'Unable to read JSON.'
            if i == 0:
                if r.status_code in retry_codes:
                    m += f' Retrying {retry}x for {timeout}s.' if retry else ''
                elif r.status_code == 429:  # rate limit
                    h = r.headers  # response headers
                    m = f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). " \
                        f"Please retry after {h['Retry-After']}s."
                if verbose:
                    LOGGER.warning(f'{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})')
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2 ** i)  # exponential standoff
        return r

    args = method, url
    kwargs['progress'] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)

class Events:

    url = 'https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw'

    def __init__(self):
        """
        Initializes the Events object with default values for events, rate_limit, and metadata.
        """
        self.events = []  # events list
        self.rate_limit = 60.0  # rate limit (seconds)
        self.t = 0.0  # rate limit timer (seconds)
        self.metadata = {
            'cli': Path(sys.argv[0]).name == 'yolo',
            'install': 'git' if is_git_dir() else 'pip' if is_pip_package() else 'other',
            'python': '.'.join(platform.python_version_tuple()[:2]),  # i.e. 3.10
            'version': __version__,
            'env': ENVIRONMENT,
            'session_id': round(random.random() * 1E15),
            'engagement_time_msec': 1000}
        self.enabled = \
            SETTINGS['sync'] and \
            RANK in (-1, 0) and \
            not TESTS_RUNNING and \
            ONLINE and \
            (is_pip_package() or get_git_origin_url() == 'https://github.com/ultralytics/ultralytics.git')

    def __call__(self, cfg):
        """
        Attempts to add a new event to the events list and send events if the rate limit is reached.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
        """
        if not self.enabled:
            # Events disabled, do nothing
            return

        # Attempt to add to events
        if len(self.events) < 25:  # Events list limited to 25 events (drop any events past this)
            params = {**self.metadata, **{'task': cfg.task}}
            if cfg.mode == 'export':
                params['format'] = cfg.format
            self.events.append({'name': cfg.mode, 'params': params})

        # Check rate limit
        t = time.time()
        if (t - self.t) < self.rate_limit:
            # Time is under rate limiter, wait to send
            return

        # Time is over rate limiter, send now
        data = {'client_id': SETTINGS['uuid'], 'events': self.events}  # SHA-256 anonymized UUID hash and events list

        # POST equivalent to requests.post(self.url, json=data)
        smart_request('post', self.url, json=data, retry=0, verbose=False)

        # Reset events and rate limit timer
        self.events = []
        self.t = t

events = Events()