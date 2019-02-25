import urllib.request
import urllib.error
import socket
import time
import numpy as np
import cv2
import os

def download(url, timeout=150, retry=3, sleep=1, verbose=True):
    """Downloads a file at given URL."""
    count = 0
    while True:
        try:
            user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
            headers={'User-Agent':user_agent,}
            req = urllib.request.Request(url,None,headers)
            f = urllib.request.urlopen(req, timeout=timeout)
            if f is None:
                raise Exception('Cannot open URL {0}'.format(url))
            content = f.read()
            f.close()
            break
        except urllib.error.HTTPError as e:
            if 500 <= e.code < 600:
                if verbose:
                    print('Error: HTTP with code {0}\n'.format(e.code))
                count += 1
                if count > retry:
                    if verbose:
                        print('Error: too many retries on {0}\n'.format(url))
                    return None
            else:
                if verbose:
                    print('Error: HTTP with code {0}\n'.format(e.code))
                return None
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.gaierror):
                count += 1
                time.sleep(sleep)
                if count > retry:
                    if verbose:
                        print('Error: too many retries on {0}\n'.format(url))
                    return None
            else:
                if verbose:
                    print('Error: URLError {0}\n'.format(e))
                return None
        except Exception as e:
            if verbose:
                print('Error: unknown during download: {0}\n'.format(e))
            return None
    return content

def get_names_url(i):
    """Fetches the urls and the names from a particular training set"""
    urls = list()
    with open('./urls/fall11_urls_train_'+str(i)+'.txt','r',encoding="Latin-1") as f:
        for line in f:
            urls.append(line)
    urls = [url.strip('\n') for url in urls]
    urls1 = [url.split('\t')[1] for url in urls]
    names = [url.split('\t')[0] for url in urls]
    return urls1,names 

def download_img(url,name):
    """Downloads and saves the image as input name"""
    resp = download(url)
    if (resp!=None):
        image = np.asarray(bytearray(resp), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite(name,image)
    return

def del_imgs(path):
    """Deletes all the images in the path location"""
    files = os.listdir(path)
    for file in files:
        if file.endswith(".jpg"):
            os.remove(os.path.join(path, file))
