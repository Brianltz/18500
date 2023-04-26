#frame grabber from https://github.com/jhanmtl/squirrel-squirter/blob/master/raspberry_pi/detection_client/utils.py
import requests
from threading import Thread
import cv2
import numpy as np
import time
import datetime
class frameGrabber:
    
    def __init__(self,url):
        self.r=requests.get(url,stream=True)
        self.img=None
        self.stopped=False
        #self.outFile = open("debug.txt", 'w')
        self.count = 0
        print(self.r.status_code)       
                  
        
    def start(self):
        print('starting thread')
        t=Thread(target=self.update,args=())
        t.setDaemon(True)
        t.start()
        print('thread started')
        return self
       
    def update(self):
        while True:
            if self.stopped==True:
                print("daemon stopping")
                return
            else:
                img_bytes=bytes()
                for chunk in self.r.iter_content(chunk_size=1024):
                    img_bytes+=chunk
                    a=img_bytes.find(b'\xff\xd8')
                    b=img_bytes.find(b'\xff\xd9')
                    if a!=-1 and b!=-1:
                        
                        jpg=img_bytes[a:b+2]
                        #jpg=img_bytes[a+2:b]
                        img_bytes=img_bytes[b+2:]
                        #if (self.count < 1):
                        #    self.outFile.write(str(jpg))
                        #    self.outFile.write("\n")
                        #    self.outFile.write(str(img_bytes))
                        self.img=cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8), cv2.IMREAD_COLOR)
                        #self.img = jpg.decode()
                        self.count += 1
    def read(self):
        return self.img
    
    def stop(self):
        self.stopped=True

framex, framey = (640,480)
img_url = "https://d473-128-2-149-250.ngrok-free.app/video_feed"

out = cv2.VideoWriter("ngrokTest1.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 30, (framex, framey))
time.sleep(5)

img_bytes=bytes()
completeimg = False

img_cap=frameGrabber(img_url).start()
time.sleep(5)
outFile = open("debug.txt", 'w')
print("starting")
start = time.time() * 1000
while True:
    curr = time.time() * 1000
    change = curr - start
    print(int(change))
    if (int(change) > 10000):
        print("stopping")
        img_cap.stop()
        out.release()
        break
    img=img_cap.read()
    #print(img)
    outFile.write("frame:" + str(img) + "\n\n")
    cv2.imshow('',img)
    cv2.waitKey(1)
    cv2.imwrite("testimg1.jpg", img)
    out.write(img)

