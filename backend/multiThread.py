
import cv2
import torch
import sqlite3
import pickle
import sched, time
import socket
from datetime import datetime
from pytz import timezone
import numpy as np
from threading import Thread
import requests
import sqlite3


from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.models as tmod


def getPriorCount(dt, id, cursor):
    #takes in datetime.now return and queries db for prior count
    #returns int of prior count
    retval = []
    step = 15
    day = dt.month * 100 + dt.day #month|day
    tablename = "room" + str(id)
    for i in range(1,5):
        currstep = 15 * i
        if (dt.minute >= currstep):
            prior_time = dt.hour * 100 + dt.minute
        else:
            hour = dt.hour
            if hour > 0:
                hour -= 1
            minute = dt.minute + (60 - currstep)
            prior_time = hour * 100 + minute

        query = f"select count from {tablename} where time={prior_time} and day={day}"
        res = cursor.execute(query)
        priorcount = res.fetchone()
        if (priorcount is None):
            print(f"cannot find prior count for time {prior_time}, day {day} from db, append 0")
            retval.append(0)
        else:
            print(f"found prior count for time {prior_time}, day {day} from db, append {priorcount}")
            retval.append(int(priorcount[0]))
    return retval

def calculateCategory(count, totalCapacity):
    capacity = (count / totalCapacity) * 100
    if capacity < 0.25:
        res = 'almost_empty'
    elif capacity < 0.5:
        res = 'not_busy'
    elif capacity < 0.75:
        res = 'busy'
    else:
        res = 'almost_full'
    return res

class outputData:
    # class for capacity data
    # id is room id, cursor is db cursor, totalCap is total cap for the room
    def __init__(self, count, id, totalCapacity, cursor):
        self.dbcursor = cursor
        self.count = count
        self.cat = calculateCategory(count, totalCapacity)
        self.id = id
        t = datetime.now()
        self.dayofweek = t.isoweekday()
        self.time = t.hour * 100 + t.minute # time in format hour|minute as int
        self.day = t.month * 100 + t.day
        self.prior_count = getPriorCount(t, id, cursor)
        #lines below need change
        self.class_in_session = False
        self.is_peak_hours = False
        self.is_240 = False
        self.is_500 = False
    
    def storeToDb(self, conn):
        #db order
        """(day, time, day_in_week, class_in_session, is_peak_hours, is_240, 
        is_500, 15min_prior_count, 30min_prior_count, 45min_prior_count, 
        60min_prior_count, count, category)"""
        tableName = "room" + str(self.id)
        query = f"""INSERT INTO {tableName} (day, time, day_in_week, class_in_session, 
        is_peak_hours, is_240, is_500, prior_count_15, prior_count_30, prior_count_45, prior_count_60, count, category) 
        VALUES ({self.day}, {self.time}, {self.dayofweek}, 
        {self.class_in_session}, {self.is_peak_hours}, {self.is_240}, 
        {self.is_500}, {self.prior_count[0]}, {self.prior_count[1]}, {self.prior_count[2]}, {self.prior_count[3]}, {self.count}, '{self.cat}')"""
        #print("db query:", query)
        self.dbcursor.execute(query)
        conn.commit() #commit changes

class frameGrabber: # to get frames from url
    
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
                print("thread stopped")
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
#m = tmod.mobilnet_v2(pretrained=False)
#print("mobilnetv2", m)

#yolov5 export: 
# in yolov5 directory
# $ python export.py --weights yolov5n.pt --include onnx --opset 12
# onnx runs faster on cpu
device = 'cpu'
model = torch.hub.load('../yolov5', 'custom', '../yolov5/yolov5n.onnx', source='local')
class cvInstance:
    def __init__(self, doorOnRight):
        self.confidenceThreshold = 0.4
        self.classes = model.names
        self.deepSort = DeepSort(max_age=5,
                   n_init=2,
                   nms_max_overlap=1.0,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=True)
        self.trackMap = {}
        self.dirMap = {}
        self.doorid = 0
        self.doors = {}
        self.doorDelays = {}
        self.delayThreshold = 200
        self.inIds = {}
        self.outIds = {} # set tracking in and out ids
        self.url = ""
        self.img_cap = frameGrabber(self.url).start()
        time.sleep(2) #wait to get frames
        for i in range(self.doorid): #init doors
            self.doorCounts[i] = 0
            self.doorDelays[i] = 0
        self.maxCap = {} #init max capacity for each room tracked by this
        self.framex = 640
        self.framey = 640
        self.halfx = self.framex//2
        self.halfy = self.framey//2
        self.dooronright = doorOnRight # boolean recording if doors are on right
        self.outputFrame = None #output so others can access it
        self.tracks = [] #tracks for tracker
        self.dbFile = "rooms.sqlite"
        self.con = sqlite3.connect(self.dbFile)
        self.dbcursor = self.con.cursor()
        self.tz = timezone('US/Eastern')
        self.xbounds = []
    
    def score_frame(self, frame):
        dim = [frame]
        results = model(dim)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]
        return labels, cords

    def class_to_label(self, label):
        return self.classes[int(label)]

    def plot_box(self, labels, cords, frame):
        n = len(labels)
        x_shape, y_shape = frame.shape[0], frame.shape[1]
        detections = []
        for i in range(n):
            row = cords[i]
            if row[4] >= self.confidenceThreshold and int(labels[i]) == 0: #confidence is row[4]
                x1,y1,x2,y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'person'))
                # these three lines are for printing out detection bounding boxes
                #bgr = (0, 255, 0)
                #cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                #cv2.putText(frame, class_to_label(labels[i]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr)
        return frame, detections

    # finds the displacement of a person over last 6 frames
    # tracks is trackmap id
    def updateTrackMap(self, id, center):
        offset = 3
        numFrames = 6 
        halfNum = numFrames//2
        # center point coords
        if id in self.trackMap.keys():
            self.trackMap[id].append(center)
            if (len(self.trackMap[id]) > 6):
                #print("before modify", tracks[id])
                self.trackMap[id].pop(0)
                #print("after modify", tracks[id])
                avgpt1 = sum(self.trackMap[id][0:halfNum]) / halfNum
                #print(avgpt1)
                avgpt2 = sum(self.trackMap[id][halfNum:numFrames]) / halfNum
                #print(avgpt2)
                slope = avgpt2 - avgpt1
                #print("dx, dy", slope[0], slope[1])

                if (slope[0] < (-1 * offset)):
                    self.dirMap[id] = 'L'
                elif (slope[0] > offset):
                    self.dirMap[id] = 'R'
                else:
                    self.dirMap[id] = 'N'
                # add up down
                prevDir = self.dirMap[id]
                if (slope[1] < (-1 * offset)):
                    self.dirMap[id] = prevDir + 'U'
                elif (slope[1] > offset):
                    self.dirMap[id] = prevDir + 'D'
                else:
                    self.dirMap[id] = prevDir + 'n'
        else:
            self.trackMap[id] = [center]
            self.dirMap[id] = 'Nn'
        return None

    #coords is the coordinates of the bounding box, ndarray of shape (4,): x1, y1, x2, y2
    def checkCrossDoor(self, track_id, center, coords, f, left, ids_in, ids_out, dirMap, track):
         # model xy coordinates with slope
    
        #if (track_id in ids_in):
            #print("track id already counted", track_id)
            #return 0
        xl, yt, xr, yb = coords
        #print(xl, yt, xr, yb)
        xbounds.clear()
        if (center[0] < leftDoorThreshold and left == True):
            #case 1
            for id in leftDoors.keys():
                pt0, pt1 = leftDoors[id]
                slope = (pt0[0] - pt1[0]) / (pt0[1] - pt1[1])#slope from top point to bottom point
                assert slope < 0

                '''
                logic for left door was modified to acomodate new camera setups, 
                might fail on older camera footages
                '''
                # if bottom left corner passes the line defining the door
                if pt1[1] <= yb and yb <= pt0[1]: #bottom y coordinate in range of door y coords, pt1 is lower y val bound, pt0 is higher y val bound
                    # line below needs work?
                    # if bottom left of bbox crosses bottom left door corner X and moving left
                    xboundary = pt1[0] + slope*(yb - pt1[1])
                    xbounds.append((xboundary, yb))
                    #print("xbound", xboundary)
                    currTime = time.time() * 1000
                    if (xl <= xboundary):
                        #print("person in x boundary")
                        if (track_id not in ids_in 
                            and ('U' in dirMap[track_id] or 'L' in dirMap[track_id]) 
                            and (currTime > doorDelays[id] + delayThreshold)):
                            doorCounts[id] += 1
                            doorDelays[id] = currTime #time in ms
                            f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " + str(datetime.now()) + "\n")
                            print(f'leftdoor:{id} increased count to {doorCounts[id]}')
                            ids_in.append(track_id)
                            print("in ids", ids_in)
                        elif (track_id not in ids_out 
                              #and ('R' in dirMap[track_id] or "D" in dirMap[track_id] or "Nn" in dirMap[track_id]) 
                              and ("Nn" in dirMap[track_id] or "D" in dirMap[track_id])
                              and (currTime > doorDelays[id] + delayThreshold)
                              and len(track[track_id]) < 5): #this line needed

                            #print("trackid", track_id, "map", track[track_id], "tracks", tracks, "tracks1", tracks1)
                            doorCounts[id] -= 1
                            doorDelays[id] = currTime #time in ms
                            f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " +  str(datetime.now()) + "\n")
                            print(f'leftdoor:{id} decreased count to {doorCounts[id]}')
                            ids_out.append(track_id)
                            print("out ids", ids_in)

        if (center[0] > rightDoorThreshold and left == False):
            #case 2
            for id in rightDoors.keys():
                pt0, pt1 = rightDoors[id]
                dxdy = (pt0[0] - pt1[0]) / (pt0[1] - pt1[1]) 
                assert dxdy > 0
                if pt1[1] <= yb and yb <= pt0[1]: #bottom y coordinate in range of door y coords
                    #print("bottom match")
                    xboundary = pt1[0] + dxdy*(yb - pt1[1])
                    xbounds.append((xboundary, yb))
                    #print("xbound", xboundary)
                    if (xl < xboundary and xboundary <= xr):
                        #print("check cross door called for right door boundary match")
                        currTime = time.time() * 1000
                        if (track_id not in ids_in 
                            and ('U' in dirMap[track_id] or 'R' in dirMap[track_id]) 
                            and (currTime > doorDelays[id] + delayThreshold)):
                            doorCounts[id] += 1
                            doorDelays[id] = time.time() * 1000 #time in ms
                            f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " + str(datetime.now()) + "\n")
                            print(f'rightdoor:{id} increased count to {doorCounts[id]}')
                            ids_in.append(track_id)
                            print("in ids", ids_in)
                        if (track_id not in ids_out 
                              and ('L' in dirMap[track_id] or 'Nn' in dirMap[track_id]) 
                              and (currTime > doorDelays[id] + delayThreshold)):
                            doorCounts[id] -= 1
                            doorDelays[id] = currTime #time in ms
                            f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " +  str(datetime.now()) + "\n")
                            print(f'rightdoor:{id} decreased count to {doorCounts[id]}')
                            ids_out.append(track_id)
                            print("out ids", ids_in)
        return 0


    def processFeed(self, frame, outfile, i) -> np.ndarray:
        #i = 0 -> right door
        #i = 1 -> left door
        labels, cords = score_frame(frame)
        f, bbs = plot_box(labels, cords, frame)

        if (i == 0): # first camera
            tracks = deepSort.update_tracks(bbs, frame=f)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()

                bbox = ltrb
                center = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
                coords = np.array(bbox)
                val1 = (int(coords[0]),int(coords[3]))
                val2 = (int(coords[2]), int(coords[3]))
                updateTrackMap(track_id, center, trackMap, dirMap)
                checkCrossDoor(track_id, center, coords, outfile, False, inIds, outIds, dirMap=dirMap, track=trackMap)
                cv2.rectangle(f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
                cv2.line(f, val1, val2, (213, 255, 52), 2)
                cv2.putText(f, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
                #lines for testing dimensions
                #cv2.line(f, (halfx, 0),(halfx, framey), (213, 255, 52), 1)
                #cv2.line(f, (0, 10), (10, 10), (255,0,0), 1)
                cv2.circle(f, (int(center[0]), int(center[1])), 1, (255, 255, 0), -1)
                if (track_id in dirMap.keys()):
                    cv2.putText(f, dirMap[track_id], (int(bbox[2]) - 10, int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
        else: # second camera
            tracks1 = self.deepSort.update_tracks(bbs, frame=f)
            for track in tracks1:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()

                bbox = ltrb
                center = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
                coords = np.array(bbox)
                val1 = (int(coords[0]),int(coords[3]))
                val2 = (int(coords[2]), int(coords[3]))
                updateTrackMap(track_id, center, self.trackMap, self.dirMap)
                checkCrossDoor(track_id, center, coords, outfile, True, self.inIds, outIds1, dirMap=self.dirMap, track=self.trackMap)
                cv2.rectangle(f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
                cv2.line(f, val1, val2, (213, 255, 52), 2)
                cv2.putText(f, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
                #lines for testing dimensions
                #cv2.line(f, (halfx, 0),(halfx, framey), (213, 255, 52), 1)
                #cv2.line(f, (0, 10), (10, 10), (255,0,0), 1)
                cv2.circle(f, (int(center[0]), int(center[1])), 1, (255, 255, 0), -1)
                if (track_id in self.dirMap.keys()):
                    cv2.putText(f, self.dirMap[track_id], (int(bbox[2]) - 10, int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)

        # draw doors
        if (i == 1):
            for id in leftDoors.keys():
                coords = leftDoors[id]
                cv2.line(f, coords[0], coords[1], (213, 255, 52), 1)
                textx = coords[0][0]
                texty = coords[0][1] + 10
                cv2.putText(f, str(doorCounts[id]), (textx, texty), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

        if (i == 0):
            for id1 in rightDoors.keys():
                coords1 = rightDoors[id1]
                cv2.line(f, coords1[0], coords1[1], (213, 255, 52), 1)
                textx = coords1[0][0]
                texty = coords1[0][1] + 10
                cv2.putText(f, str(doorCounts[id1]), (textx, texty), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

        return f

def dbUpdate(dbFile):
    connection = sqlite3.connect(dbFile)
    cursor = connection.cursor()
    for i in range(len(doorCounts)):
        data = outputData(doorCounts[i], i, maxcap[i], cursor)
        data.storeToDb(connection) #insert update to dbfile


outputFrames = {0:None, 1:None}
Limit = 60000 #number of ms to run
outfile = open("test1.txt", 'w')
outfile1 = open("test2.txt", "w")
outfiles = (outfile, outfile1)




def threadOutput(startTime, outputFrames, lastUpdate, dbPath):
  # this thread handles writing to file and updating db
    out = cv2.VideoWriter("/content/drive/MyDrive/500/test5.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1280, 640))
    while True:
      currTime = time.time() * 1000
      start = time.time()
      if (int(currTime - lastUpdate) > 60000):#if its been 60 sec since last update
          lastUpdate = currTime
          dbUpdate(dbPath)
      if (int(currTime - startTime) > Limit):
        if (outputFrames[0] is None or outputFrames[1] is None):
            continue # only do stack if there are output images
        f = np.hstack((outputFrames[0], outputFrames[1]))
        end = time.time()
        fps = 1/np.round(end - start, 2)
        cv2.putText(f, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        print("fps", fps)
        #print("fshape", f.shape)
        out.write(f)#output code
        time.sleep(0.1)
      else: #exit loop
        break
    #end of loop
    print("update thread exiting")
    out.release()


def main():
    #path = '/Users/bli/Desktop/500/CV/backend/footages/1680725348test.mp4' 
    #path1 = '/Users/bli/Desktop/500/CV/backend/footages/1680725378test.mp4'
    out = cv2.VideoWriter("test5.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1280, 640))
    lastupdate = 0 #varaible tracking last update to db
    startTime = time.time() * 1000
    #start threads
    #end of loop
    print("ending task")
    #videoInput.release()
    cv2.destroyAllWindows()
    outfile.close()
    return 0

   

if __name__ == "__main__":
    main()