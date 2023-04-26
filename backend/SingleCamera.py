
#used the following resources to setup yolov5s to process live video feed from camera
#https://github.com/ultralytics/yolov5
#https://www.youtube.com/watch?v=Cof7CNjDppo&t=640s
import cv2
import torch
import sqlite3
import pickle
import sys
import sched, time
import socket
from datetime import datetime
from pytz import timezone
import numpy as np
from threading import Thread
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.models as tmod
#m = tmod.mobilnet_v2(pretrained=False)
#print("mobilnetv2", m)

#yolov5 export: 
# in yolov5 directory
# $ python export.py --weights yolov5n.pt --include onnx --opset 12
# onnx runs faster on cpu

confidenceThreshold = 0.4
device = 'cpu'
model = torch.hub.load('../yolov5', 'custom', '../yolov5/yolov5n.onnx', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device="cpu") # using onnx model is much faster on cpu, im getting 1-2 fps on this
classes = model.names
dbFile = "rooms.sqlite"
#goTurnTracker = cv2.TrackerGOTURN_create()
deepSort = DeepSort(max_age=5,
                   n_init=2,
                   nms_max_overlap=1.0,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=False)

#track track ids and bboxes with map
trackMap = {}


#track track ids and direction
dirMap = {} 
#used to label entrance line of each door in format id : [(x1,y1), (x2,y2)]
doorid = 0
tz = timezone('US/Eastern')

leftDoors = {}
rightDoors = {}
doorCounts = {}

doorDelays = {}
delayThreshold = 200 #time in milisecond since someone entered a door

## #init count for each room
## 
# leftDoors[0] = [(215, 565), (334, 391)]
#doorid = 2
doorid = 0
#0: [(487, 555), (320, 470)], 1: [(134, 361), (110, 343)]

#for camera looking at second doors

#rightDoors[0] =  [(487, 555), (320, 470)]
#rightDoors[1] = [(134, 361), (110, 343)]

# for camera looking at first two doors 
#
#leftDoors[0] = [(96, 617), (291, 525)]
#leftDoors[1] = [(493, 408), (522, 388)]
for i in range(doorid):
    doorCounts[i] = 0
    doorDelays[i] = 0

# for camera looking at later doors
#[(96, 617), (271, 535)]
#[(494, 408), (515, 392)]

#keeps track of tracks that have been counted to prevent double count
inIds = []
outIds = []

#video frame dimension globals
framex = 640
framey = 640
#halfx = framex//2 + 100 
halfx = framex
#halfx = 0
halfy = framey//2
rightDoorThreshold = 0
leftDoorThreshold = framex
xbounds = []

tracks = []

def scheduled_count_update(scheduler, f):
    scheduler.enter(10, 1, scheduled_count_update, (scheduler, f, ))
    print("updating count in database")
    for key in doorCounts.keys():
        f.write(str(key) + ", count: " + str(doorCounts[key]) + "\n")

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

# finds the displacement of a person over last 10 frames
def updateTrackMap(id, center, tracks, dirs):
    offset = 3
    numFrames = 6 
    halfNum = numFrames//2
    # center point coords
    if id in tracks.keys():
        tracks[id].append(center)
        if (len(tracks[id]) > 6):
            #print("before modify", tracks[id])
            tracks[id].pop(0)
            #print("after modify", tracks[id])
            avgpt1 = sum(tracks[id][0:halfNum]) / halfNum
            #print(avgpt1)
            avgpt2 = sum(tracks[id][halfNum:numFrames]) / halfNum
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
        tracks[id] = [center]
        dirs[id] = 'Nn'
    end = time.time() * 1000
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

def storeToDB(): # stores count info into db file
    '''
    day, time, day_in_week
    class_in_session is_peak_hours, count, category
    '''
    today = datetime.date()
    return 0
#coords is the coordinates of the bounding box, ndarray of shape (4,): x1, y1, x2, y2

def withinDoorArea(x, y, x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if a point (x, y) is within a parallelogram defined by its four vertices (x1, y1), (x2, y2), (x3, y3), and (x4, y4).
    """
    # Compute the vectors corresponding to the sides of the parallelogram
    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x2, y3 - y2)
    v3 = (x4 - x3, y4 - y3)
    v4 = (x1 - x4, y1 - y4)

    # Compute the vectors corresponding to the edges between the point and the vertices of the parallelogram
    u1 = (x - x1, y - y1)
    u2 = (x - x2, y - y2)
    u3 = (x - x3, y - y3)
    u4 = (x - x4, y - y4)

    # Compute the cross product of each vector pair (v_i, u_i)
    c1 = v1[0] * u1[1] - v1[1] * u1[0]
    c2 = v2[0] * u2[1] - v2[1] * u2[0]
    c3 = v3[0] * u3[1] - v3[1] * u3[0]
    c4 = v4[0] * u4[1] - v4[1] * u4[0]

    # Check if the signs of the cross products are the same for all four pairs
    return (c1 > 0) == (c2 > 0) == (c3 > 0) == (c4 > 0)

def checkCrossDoor(track_id, center, coords, f, ids_in, ids_out):
     # model xy coordinates with slope
    global halfx # this is the middle of the frame
    global doorid # highest doorid
    global doorCounts #map containing doorid and corresponding counts
    global leftDoors #coordinates of doors, index 0 is bottom coord, index 1 is top coord
    global rightDoors
    '''
    case 1: center left of middle line
    case 2: center right of middle line
    '''
    #if (track_id in ids_in):
        #print("track id already counted", track_id)
        #return 0
    xl, yt, xr, yb = coords
    #print(xl, yt, xr, yb)
    xbounds.clear()
    if (center[0] < leftDoorThreshold):
        #case 1
        for id in leftDoors.keys():
            pt0, pt1 = leftDoors[id]
            slope = (pt0[0] - pt1[0]) / (pt0[1] - pt1[1])#slope from top point to bottom point
            #assert slope < 0
            #print(pt0, pt1)
            # if bottom left corner passes the line defining the door
            if pt1[1] <= yb and yb <= pt0[1]: #bottom y coordinate in range of door y coords, pt1 is lower y val bound, pt0 is higher y val bound
                #print("person within leftdoor y bound")
                # line below needs work?
                # if bottom left of bbox crosses bottom left door corner X and moving left
                xboundary = pt1[0] + slope*(yb - pt1[1])
                xbounds.append((xboundary, yb))
                #print("xbound", xboundary)
                currTime = time.time() * 1000
                if (xl <= xboundary):
                    print("person in x boundary")
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
                          and len(trackMap[track_id]) < 5): #this line needed
                        doorCounts[id] -= 1
                        doorDelays[id] = currTime #time in ms
                        f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " +  str(datetime.now()) + "\n")
                        print(f'leftdoor:{id} decreased count to {doorCounts[id]}')
                        ids_out.append(track_id)
                        print("out ids", ids_in)
                #end xl <= xbound case
                #else:
                ##elif (xl <= xboundary + 25):
                #    if (track_id not in ids_out 
                #          and ('D' in dirMap[track_id] or "Nn" in dirMap[track_id]) 
                #          and (currTime > doorDelays[id] + delayThreshold)
                #          and len(trackMap[track_id]) < 5
                #          and withinDoorArea(xl, yb, pt0[0], pt0[1], pt1[0], pt1[1], pt1[0], pt1[1] + 25, pt0[0], pt0[1] + 25) == True):
                #        doorCounts[id] -= 1
                #        doorDelays[id] = currTime #time in ms
                #        f.write(str(id) + ", count: " + str(doorCounts[id]) + "  " +  str(datetime.now()) + "\n")
                #        print(f'leftdoor:{id} decreased count to {doorCounts[id]}')
                #        ids_out.append(track_id)
                #        print("out ids", ids_in)
                    

    #else:
    if (center[0] > rightDoorThreshold):
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

def startScheduler(scheduler, f):
    scheduler.enter(10, 1, scheduled_count_update, (scheduler, f, ))
    scheduler.run()

def processFeed(frame, outfile):
    #i = 0 -> right door
    #i = 1 -> left door
    labels, cords = score_frame(frame)
    f, bbs = plot_box(labels, cords, frame)


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
        #print("coords", coords)
        #print("point", point)
        # update map containing each track and the centerpoint in past 10 frames 
        

        updateTrackMap(track_id, center, trackMap, dirMap)
        checkCrossDoor(track_id, center, coords, outfile, inIds, outIds)
        cv2.rectangle(f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
        cv2.line(f, val1, val2, (213, 255, 52), 2)
        cv2.putText(f, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
        #lines for testing dimensions
        #cv2.line(f, (halfx, 0),(halfx, framey), (213, 255, 52), 1)
        #cv2.line(f, (0, 10), (10, 10), (255,0,0), 1)
        cv2.circle(f, (int(center[0]), int(center[1])), 1, (255, 255, 0), -1)
        if (track_id in dirMap.keys()):
            cv2.putText(f, dirMap[track_id], (int(bbox[2]) - 10, int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
    
    # draw doors
    for id in leftDoors.keys():
        coords = leftDoors[id]
        cv2.line(f, coords[0], coords[1], (213, 255, 52), 1)
        pts = np.array([coords[0], [coords[0][0],coords[0][1] + 30], [coords[1][0], coords[1][1] + 30], coords[1]])
        pts = pts.reshape(-1,1,2)

        cv2.polylines(f, [pts], True, (213, 255, 52), 1)

    for id1 in rightDoors.keys():
        coords1 = rightDoors[id1]
        cv2.line(f, coords1[0], coords1[1], (213, 255, 52), 1)

    for bound in xbounds:
        cv2.circle(f, (int(bound[0]), int(bound[1])), 2, (255, 255, 255), -1)
    
    return f

def main():
    outfile = open("test1.txt", 'w')
   

    path = '/Users/bli/Desktop/500/CV/backend/footages/1680725348test.mp4' 
    #path = '/Users/bli/Desktop/500/CV/backend/footages/1680709161test.mp4'
    videoInput = cv2.VideoCapture(path)
    #videoInput1 = cv2.VideoCapture(path1)

    #code for live testing
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('',int(sys.argv[1])))
    s.listen(30)
    conn, addr = s.accept()
#
    #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.bind(('',8889))
    #s.listen(30)
    #conn, addr = s.accept()
    try:
        frameCounter = 0
        while True:
            ret, frame = videoInput.read()
            #code for live feed
            frame = liveVideo(conn, addr)
            if (1): #live feed code
            #if ret == True: #use this line for local video testing
            
                start = time.time()
                #if (0 in leftDoors.keys()):
                #
                
                #elif (0 in rightDoors.keys()):
                frame = cv2.resize(frame, (framex,framey))
                #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                
                f = processFeed(frame, outfile)
                end = time.time()
                fps = 1/np.round(end - start, 2)
                cv2.putText(f, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.imshow('test', f)
                cv2.setMouseCallback('test', selectDoors)
                key = cv2.pollKey()
                if (key & 0xFF == ord('p')): #logic to pause video with keyboard input
                    key = cv2.waitKey(0)
                    if (key & 0xFF == ord('c')):
                        key = cv2.waitKey(1)
                    elif (key & 0xFF == ord('q')):
                        break
                key = cv2.waitKey(1)
            else:
                break
    except KeyboardInterrupt:
        print("ending task by interrupt")
        videoInput.release()
        cv2.destroyAllWindows()
        outfile.close()
        return 0
    #end of loop
    #print("trackMap", trackMap)
    print("ending task")
    videoInput.release()
    cv2.destroyAllWindows()
    outfile.close()
    return 0

   

if __name__ == "__main__":
    main()