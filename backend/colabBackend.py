
import cv2
import torch
import sqlite3
import pickle
import sched, time
from datetime import datetime
from pytz import timezone
import numpy as np
from threading import Thread
import requests
import sqlite3


from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.models as tmod


tz = timezone('US/Eastern') #timezone for datetime
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
            prior_time = dt.hour * 100 + dt.minute - currstep
        else:
            hour = dt.hour
            if hour > 0:
                hour -= 1 # this part need change
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
    capacity = (count / totalCapacity)
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
    def __init__(self, count, id, cursor):
        self.dbcursor = cursor
        self.count = count
        self.cat = calculateCategory(count, maxcap[id])
        self.id = id
        t = datetime.now(tz)
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

confidenceThreshold = 0.4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(str(device), "name", torch.cuda.get_device_name())
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True, device=device) 
print("model running on cuda: ", next(model.parameters()).is_cuda)
#model = torch.hub.load('yolov5', 'yolov5/yolov5n.onnx', source='local', device=device)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5n') # using onnx model is much faster on cpu, im getting 1-2 fps on this
classes = model.names

#goTurnTracker = cv2.TrackerGOTURN_create()
deepSort = DeepSort(max_age=5,
                   n_init=2,
                   nms_max_overlap=1.0,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=True)

deepSort1 = DeepSort(max_age=5,
                   n_init=2,
                   nms_max_overlap=1.0,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=True)
#deepSort = DeepSort(max_age=5) #instantiate deepsort
#track track ids and bboxes with map
trackMap = {}
trackMap1 = {}
dirMap1 = {}
#track track ids and direction
dirMap = {} 
#used to label entrance line of each door in format id : [(x1,y1), (x2,y2)]
doorid = 0


leftDoors = {}
rightDoors = {}
doorCounts = {}

doorDelays = {}
delayThreshold = 200 #time in milisecond since someone entered a door


doorid = 4

# the doors looking down the hallway (39 pi)
rightDoors[0] = [(446, 486), (346, 385)] # this is the big room, id 0
rightDoors[1] = [(229, 271), (213, 250)] # mid room, id 1


#134 pi
leftDoors[3] = [(92, 517), (226, 404)] #big room id 3
leftDoors[2] =  [(353, 309), (378, 286)] # small room id 2
doorCounts[0] = 0
doorCounts[1] = 10
doorCounts[2] = 3
doorCounts[3] = 25

##init max capacity for each room
maxcap = [10, 30, 50, 50]
## #init count for each room
## 
url = 'https://2f14-128-2-149-254.ngrok-free.app/video_feed' #left cam, 39 pi
url1 = 'https://347b-128-2-149-250.ngrok-free.app/video_feed' #right cam, 134 pi
img_cap = frameGrabber(url).start()
img_cap1 = frameGrabber(url1).start()
time.sleep(3)
for i in range(doorid):
   
    doorDelays[i] = 0



#keeps track of tracks that have been counted to prevent double count
inIds = []
outIds = []

inIds1 = []
outIds1 = []

#video frame dimension globals
framex = 640
framey = 640
halfx = framex//2
halfy = framey//2
rightDoorThreshold = 0
leftDoorThreshold = framex
xbounds = []

tracks = []
tracks1 = []

#db connection
dbFile = "/content/drive/MyDrive/rooms.sqlite"
con = sqlite3.connect(dbFile)
dbcursor = con.cursor()


def score_frame(frame):
    dim = [frame]
    results = model(dim)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]
    return labels, cords

def class_to_label(label):
    return classes[int(label)]

def plot_box(labels, cords, frame):
    n = len(labels)
    x_shape, y_shape = frame.shape[0], frame.shape[1]
    detections = []
    for i in range(n):
        row = cords[i]
        if row[4] >= confidenceThreshold and int(labels[i]) == 0: #confidence is row[4]
            x1,y1,x2,y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            
            detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'person'))
            # these three lines are for printing out detection bounding boxes
            #bgr = (0, 255, 0)
            #cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
            #cv2.putText(frame, class_to_label(labels[i]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr)
    return frame, detections

# finds the displacement of a person over last 6 frames
# tracks is trackmap id
def updateTrackMap(id, center, track, dirs):
    offset = 3
    numFrames = 6 
    halfNum = numFrames//2
    # center point coords
    if id in track.keys():
        track[id].append(center)
        if (len(track[id]) > 6):
            #print("before modify", tracks[id])
            track[id].pop(0)
            #print("after modify", tracks[id])
            avgpt1 = sum(track[id][0:halfNum]) / halfNum
            #print(avgpt1)
            avgpt2 = sum(track[id][halfNum:numFrames]) / halfNum
            #print(avgpt2)
            slope = avgpt2 - avgpt1
            #print("dx, dy", slope[0], slope[1])
            
            if (slope[0] < (-1 * offset)):
                dirs[id] = 'L'
            elif (slope[0] > offset):
                dirs[id] = 'R'
            else:
                dirs[id] = 'N'
            # add up down
            prevDir = dirs[id]
            if (slope[1] < (-1 * offset)):
                dirs[id] = prevDir + 'U'
            elif (slope[1] > offset):
                dirs[id] = prevDir + 'D'
            else:
                dirs[id] = prevDir + 'n'
    else:
        track[id] = [center]
        dirs[id] = 'Nn'

    return None

def selectDoors(event, x, y, flags, param):
    global doorid
    global halfx
    if (event == cv2.EVENT_LBUTTONDOWN):
        print("down coord", x, y)
        if (x < halfx):
            leftDoors[doorid] = [(int(x), int(y))]
        else:
            rightDoors[doorid] = [(int(x), int(y))]
    elif (event == cv2.EVENT_LBUTTONUP):
        print("up coord", x, y)
        if (x < halfx):
            leftDoors[doorid].append((int(x), int(y)))
        else:
            rightDoors[doorid].append((int(x), int(y)))
        doorCounts[doorid] = 0
        doorid += 1
        print("doorid", doorid)
        print("doorCounts", doorCounts)
        print("leftDoors", leftDoors)
        print("rightDoors,", rightDoors)

#coords is the coordinates of the bounding box, ndarray of shape (4,): x1, y1, x2, y2
def checkCrossDoor(track_id, center, coords, f, left, ids_in, ids_out, dirMap, track):
     # model xy coordinates with slope
    global halfx # this is the middle of the frame
    global doorid # highest doorid
    global doorCounts #map containing doorid and corresponding counts
    global leftDoors #coordinates of doors, index 0 is bottom coord, index 1 is top coord
    global rightDoors
    global tracks
    global tracks1
   
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
                        and ('LU' in dirMap[track_id] or 'LD' in dirMap[track_id] or 'Ln' in dirMap[track_id]) 
                        and (currTime > doorDelays[id] + delayThreshold)
                        and len(track[track_id]) > 5):
                        doorCounts[id] += 1
                        doorDelays[id] = currTime #time in ms
                        f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " + str(datetime.now(tz)) + "\n")
                        print(f'leftdoor:{id} increased count to {doorCounts[id]}')
                        ids_in.append(track_id)
                        print("in ids", ids_in)
                    elif (track_id not in ids_out 
                          #and ('R' in dirMap[track_id] or "D" in dirMap[track_id] or "Nn" in dirMap[track_id]) 
                          and (("Nn" in dirMap[track_id] and len(track[track_id]) < 5) or ("R" in dirMap[track_id] and len(track[track_id]) > 5))
                          and (currTime > doorDelays[id] + delayThreshold)
                          and len(track[track_id]) < 5): #this line needed
                        
                        #print("trackid", track_id, "map", track[track_id], "tracks", tracks, "tracks1", tracks1)
                        doorCounts[id] -= 1
                        doorDelays[id] = currTime #time in ms
                        f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " +  str(datetime.now(tz)) + "\n")
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
                        and ('RU' in dirMap[track_id] or 'RD' in dirMap[track_id] or 'Rn' in dirMap[track_id]) 
                        and (currTime > doorDelays[id] + delayThreshold)
                        and len(track[track_id]) > 5):
                        doorCounts[id] += 1
                        doorDelays[id] = time.time() * 1000 #time in ms
                        f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " + str(datetime.now(tz)) + "\n")
                        print(f'rightdoor:{id} increased count to {doorCounts[id]}')
                        ids_in.append(track_id)
                        print("in ids", ids_in)
                    if (track_id not in ids_out 
                          and (("Nn" in dirMap[track_id] and len(track[track_id]) < 5) or ("L" in dirMap[track_id] and len(track[track_id]) > 5))
                          and (currTime > doorDelays[id] + delayThreshold)
                          and len(track[track_id]) < 5):
                        doorCounts[id] -= 1
                        doorDelays[id] = currTime #time in ms
                        f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " +  str(datetime.now(tz)) + "\n")
                        print(f'rightdoor:{id} decreased count to {doorCounts[id]}')
                        ids_out.append(track_id)
                        print("out ids", ids_in)
    return 0

#live video to run on local code
def liveVideo(conn, addr):
    try:
        #while(True):
        data = b''
        data = conn.recv(8)
        #print(data)
        imlen = int(data.decode('ascii'))
          #print(imlen)
        data = conn.recv(8)
        timestamp = int.from_bytes(data, 'little', signed=False)
        frame = b''
        while (len(frame) < imlen):
            frame += conn.recv(imlen - len(frame))
             # print(len(frame))
        framede = pickle.loads(frame, encoding='bytes')
        framefinal = cv2.imdecode(framede,1)
        return framefinal
        #cv2.imshow('frame', framefinal)
          #out.write(framefinal)
      
    except Exception as e:
        print(e)
           #s.close()
           #out.release()
        cv2.destroyAllWindows()

def processFeed(frame, outfile, i) -> np.ndarray:
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
        tracks1 = deepSort1.update_tracks(bbs, frame=f)
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
            updateTrackMap(track_id, center, trackMap1, dirMap1)
            checkCrossDoor(track_id, center, coords, outfile, True, inIds1, outIds1, dirMap=dirMap1, track=trackMap1)
            cv2.rectangle(f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
            cv2.line(f, val1, val2, (213, 255, 52), 2)
            cv2.putText(f, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
            #lines for testing dimensions
            #cv2.line(f, (halfx, 0),(halfx, framey), (213, 255, 52), 1)
            #cv2.line(f, (0, 10), (10, 10), (255,0,0), 1)
            cv2.circle(f, (int(center[0]), int(center[1])), 1, (255, 255, 0), -1)
            if (track_id in dirMap1.keys()):
                cv2.putText(f, dirMap1[track_id], (int(bbox[2]) - 10, int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
    
    # draw doors
    if (i == 1):
        for id in leftDoors.keys():
            coords = leftDoors[id]
            cv2.line(f, coords[0], coords[1], (213, 255, 52), 1)
            textx = coords[0][0]
            texty = coords[0][1] + 10
            count = str(doorCounts[id])
            if (id == 0 or id == 3):
              count = str(doorCounts[0] + doorCounts[3])
            cv2.putText(f, count, (textx, texty), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

    if (i == 0):
        for id1 in rightDoors.keys():
            coords1 = rightDoors[id1]
            cv2.line(f, coords1[0], coords1[1], (213, 255, 52), 1)
            textx = coords1[0][0]
            texty = coords1[0][1] + 10
            count = str(doorCounts[id1])
            if (id1 == 0 or id1 == 3):
              count = str(doorCounts[0] + doorCounts[3])
            cv2.putText(f, count, (textx, texty), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    
    return f

def dbUpdate():
    for i in range(len(doorCounts)):
        data = outputData(doorCounts[i], i, dbcursor)
        data.storeToDb(con) #insert update to dbfile


def main():
    outfile = open("test1.txt", 'w')
    outfile1 = open("test2.txt", "w")
    outfiles = (outfile, outfile1)

    #path = '/Users/bli/Desktop/500/CV/backend/footages/1680725348test.mp4' 
    #path1 = '/Users/bli/Desktop/500/CV/backend/footages/1680725378test.mp4'
    out = cv2.VideoWriter("/content/drive/MyDrive/500/4-22-1850.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1280, 640))

    
    lastupdate = 0 #varaible tracking last update to db
    try:
        frameCounter = 0
        startTime = time.time() * 1000
        while True:
            #ret, frame = videoInput.read()
            #ret1, frame1 = videoInput1.read()
            #code for live feed
            #frame = liveVideo(conn, addr)
            #frame1 = liveVideo(conn1, addr1)
            
            #if ret == True or ret1 == True: #use this line for local video testing
            currTime = time.time() * 1000

            
            if (int(currTime - lastupdate) > 60000):#if its been 60 sec since last update
                lastupdate = currTime
                dbUpdate() #code to update db

            if (int(currTime - startTime) < 600000): #for 300 seconds
                frame = img_cap.read()
                frame1 = img_cap1.read()
                frameCounter += 1
                #if (frameCounter < 100): continue
                start = time.time()
                outputFrames = [0, 0]
                for i in range(2):
                    subFrame = None
                    if i == 1:
                        #subFrame = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
                        subFrame = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        subFrame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) #for video in incorrect dimension
                    subFrame = cv2.resize(subFrame, (framex,framey))
                    outputFrames[i] = processFeed(subFrame, outfiles[i], i)

                    
                    #print("key", key)
                ## end of frames loop

                
                f = np.hstack((outputFrames[0], outputFrames[1]))
                end = time.time()
                fps = 1/np.round(end - start, 2)
                cv2.putText(f, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                print("fps", fps)
                #print("fshape", f.shape)
                out.write(f)#output code

                time.sleep(0.033) #for ngrok stream
            else:
                print("ending loop by time")
                img_cap.stop()
                img_cap1.stop()
                out.release()
                break
    except KeyboardInterrupt:
        print("ending task by interrupt")
        #videoInput.release()
        cv2.destroyAllWindows()
        outfile.close()
        return 0
    #end of loop
    print("ending task")
    #videoInput.release()
    cv2.destroyAllWindows()
    outfile.close()
    return 0

   

if __name__ == "__main__":
    main()