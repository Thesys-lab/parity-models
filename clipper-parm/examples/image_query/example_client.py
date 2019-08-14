from __future__ import print_function
import base64
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer
import json
import requests
from datetime import datetime
import time
import numpy as np
import signal
import sys
import argparse

"""def predict(addr, filename):
    url = "http://%s/image-example/predict" % addr
    req_json = json.dumps({
        "input":
        base64.b64encode(open(filename, "rb").read()).decode()
    })
    headers = {'Content-type': 'application/json'}
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (r.text, latency))

def image_size(img):
    import base64, io, os, PIL.Image, tempfile
    tmp = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.jpg')
    tmp.write(io.BytesIO(img).getvalue())
    tmp.close()
    size = PIL.Image.open(tmp.name, 'r').size
    os.unlink(tmp.name)
    return [size]


def image_size_v2(img):
    return [(2, 2)]"""


def query(addr, filename):
    url = "http://%s/image-example/predict" % addr
    req_json = json.dumps({
        "input":
        base64.b64encode(open(filename, "rb").read()).decode() # bytes to unicode
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    print(r.json())


def image_size(imgs):
    """
    Input: 
    - imgs : (np.ndarray) of shape (n, d). n is the number of data in this batch
             d is the length of the bytes as numpy int8 array.  
    Output:
    - sizes : List[Tuple(int, int),...]
    """
    import base64
    import io
    import os
    import PIL.Image
    import tempfile
  
    num_imgs = len(imgs)
    sizes = []
    for i in range(num_imgs):
        # Create a temp file to write to
        tmp = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.png')
        tmp.write(io.BytesIO(imgs[i]).getvalue())
        tmp.close()
        
        # Use PIL to read in the file and compute size
        size = PIL.Image.open(tmp.name, 'r').size
        
        # Remove the temp file
        os.unlink(tmp.name) 

        sizes.append(size)
    return sizes


# Stop Clipper on Ctrl-C
def signal_handler(signal, frame):
    print("Stopping Clipper...")
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description='Use Clipper to Query Images.')
    parser.add_argument('image', nargs='+', help='Path to an image')
    imgs = parser.parse_args().image

    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.start_clipper()
    python_deployer.create_endpoint(
            clipper_conn=clipper_conn, 
            name="image-example", 
            input_type="bytes", 
            func=image_size, 
            pkgs_to_install=['pillow']
            )
    time.sleep(2)
    try:
        for f in imgs:
            if f.endswith('.jpg') or f.endswith('.png'):
                query(clipper_conn.get_query_addr(), f)
    except Exception as e:
        print("exception")
    clipper_conn.get_clipper_logs()
    clipper_conn.stop_all()
