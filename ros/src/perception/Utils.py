import os
import sys
import errno
import uuid
import time
import urllib
import shutil
import cv2
import tarfile
from zipfile import ZipFile
import datetime

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def uuid_file_name(extension, dir="."):
    return dir + '/' + str(uuid.uuid4()) + "." + extension


def date_file_name(extension=None, dir="."):
    new_file_name = datetime.datetime.now().isoformat('_')
    new_file_name = new_file_name.replace(":","-")

    if extension is not None:
        new_file_name += "." + extension
    if dir != ".":
        new_file_name = dir + '/' + new_file_name

    return new_file_name


def save_screenshot(cv_img):
    path = "screenshots"
    mkdir(path)
    filename = uuid_file_name("png", dir=path)
    cv2.imwrite(filename, cv_img)
    print("saving image %s..." % filename)



# Taken from https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    info = "%d/%d MB, %d KB/s" % (progress_size / (1024 * 1024), total_size / (1024 * 1024), speed)
    progress_bar(count * block_size, total_size, unit="MB")


def download(url, filename):
    urllib.urlretrieve(url, filename, reporthook)


def progress_bar(i, n, message=None, length=40, absolute_numbers=True, unit=""):
    percent = float(i) / n
    dots = int(percent * length)
    head = "" if message is None else message + ' ... '
    if percent < 1.0:
        bar_length = max(dots - 1,1)
        bar = "[" + '='*(bar_length) + '>' + '.'*(length - bar_length - 1) + ']'
    else:
        bar = '[' + '='*length + ']'

    bar += " %3.d%%" % (percent*100)

    if absolute_numbers:
        bar += "  %d/%d %s" % (i,n,unit)

    sys.stdout.write('\r' + head + bar)
    sys.stdout.flush()
    if i == n:
        print("")


def display_image(img_file):
    img = cv2.imread(img_file)
    cv2.imshow(img_file, img)
    cv2.waitKey(100)

# from: http://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def download_and_extract(dir_name, archive_name, url):
    if not os.path.exists(dir_name):
        if not os.path.exists(archive_name):
            print("Downloading %s ..." % url)
            download(url, archive_name)
        else:
            print("Found archive %s" % archive_name)

        print("Upacking %s..." % archive_name)
        if archive_name.endswith(".zip"):
            zf = ZipFile(archive_name)
            zf.extractall()
        elif archive_name.endswith(".tar.gz"):
            tf = tarfile.open(archive_name)
            if "/" in dir_name:
                extract_path = os.path.dirname(dir_name)
            else:
                extract_path = "."
            tf.extractall(extract_path)
