import time


def timef():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def tprint(msg):
    print(timef(), end=": ")
    print(msg)
