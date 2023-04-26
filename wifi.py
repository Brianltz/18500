import cv2
import socket
import pickle
from time import time

width = 1280
height = 720

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 172.26.176.134
s.bind(("", 8888))
s.listen(30)

conn, addr = s.accept()
out = cv2.VideoWriter(
    str(int(time())) + "test.mp4",
    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
    10,
    (width, height),
)

try:
    while True:
        data = b""
        data = conn.recv(8)
        print(data)
        imlen = int(data.decode("ascii"))
        print(imlen)
        frame = b""
        while len(frame) < imlen:
            frame += conn.recv(imlen - len(frame))
            # print(len(frame))
        framede = pickle.loads(frame, encoding="bytes")
        framefinal = cv2.imdecode(framede, 1)
        cv2.imshow("frame", framefinal)
        out.write(framefinal)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(e)
    s.close()
    out.release()
    cv2.destroyAllWindows()
