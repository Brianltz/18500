import cv2
import socket
import pickle
from time import time

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 10)
# 172.26.176.135
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(('172.26.212.117',8888))
s.connect(("172.26.176.135", 8888))
try:
    while True:
        ret, frame = cap.read()
        ret, pack = cv2.imencode(".jpg", frame)
        packen = pickle.dumps(pack, protocol=2)
        print(len(packen))
        s.sendall(str(len(packen)).zfill(8).encode("ascii"))
        s.sendall(
            int(time() * 1000).to_bytes(8, "little", signed=False)
        )  # send unix time
        s.sendall(packen)
except Exception as e:
    print(e)
    cap.release()
