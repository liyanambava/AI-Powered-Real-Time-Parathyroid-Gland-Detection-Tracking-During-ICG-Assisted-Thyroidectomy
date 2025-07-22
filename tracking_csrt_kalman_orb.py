import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

#Utility Functions
def detect_green_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    return None

#Kalman filter setup
def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],  
                                        [0, 1, 0, 1],  
                                        [0, 0, 1, 0],  
                                        [0, 0, 0, 1]], dtype=np.float32)
    kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.statePre = np.zeros((4, 1), dtype=np.float32)
    return kalman

#check If NIR Mode is Active
def is_nir_mode(mask, threshold=30):
    return np.mean(mask) < threshold

#ORB feature matching
def match_features(prev_frame, current_frame, prev_bbox):
    orb = cv2.ORB_create(nfeatures=3500)
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(current_frame, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) > 10:  #require at least 10 good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            dx, dy = np.mean(dst_pts[mask.ravel() == 1] - src_pts[mask.ravel() == 1], axis=0).ravel()

        x, y, w, h = prev_bbox
        new_x, new_y = int(x + dx), int(y + dy)

        if abs(new_x - x) > 0.3 * w or abs(new_y - y) > 0.3 * h:
            return None 

        return new_x, new_y, w, h

    return None

#IOU Calculation
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

#Video input and output
video_path = r"C:\Users\hp\Documents\VIT\SEMESTERS\8\thyroid\thyroid_cropped.mp4"
cap = cv2.VideoCapture(video_path)
output_path = 'output_orb_kalman.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#frame resize
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 800
resize_height = int((resize_width / frame_width) * frame_height) if frame_width > 0 else 480  
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

#Initializing Variables
iou_history = []
frame_times = []
area_history = []
green_start_time = None
bounding_box = None
last_good_bbox = None  
prev_frame = None  
prev_keypoints = None  
prev_descriptors = None  
box_set = False
tracker_initialized = False
frame_count_no_green = 0
prev_gray = None

#Initialize Trackers
csrt_tracker = None  
kalman = initialize_kalman()

#to make the output video run at normal speed
#fps = int(cap.get(cv2.CAP_PROP_FPS)) 
#frame_delay = 1 / fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time() 

    #convert to grayscale & detect ICG green
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = detect_green_color(frame)
    is_nir = is_nir_mode(mask)

    if not box_set:
        if np.any(mask):
            if green_start_time is None:
                green_start_time = time.time()
                print("Green detected, starting timer...")
            elif time.time() - green_start_time >= 2:
                bounding_box = find_largest_contour(mask)
                if bounding_box:
                    x, y, w, h = bounding_box
                    frame_area = frame.shape[0] * frame.shape[1]
                    box_area = w * h
                    if box_area > frame_area * 0.08:
                        box_set = True
                        last_good_bbox = bounding_box 
                        prev_frame = gray
                        print(f"Gland locked at: {last_good_bbox}")

                        #extract ORB features
                        orb = cv2.ORB_create(nfeatures=500)
                        keypoints, descriptors = orb.detectAndCompute(gray, None)
                        prev_keypoints = keypoints
                        prev_descriptors = descriptors

                        #initialize CSRT tracker
                        csrt_tracker = cv2.TrackerCSRT_create()
                        csrt_tracker.init(frame, (x, y, w, h))
                        tracker_initialized = True
                        print("CSRT Tracker Initialized")

                        #initialize Kalman filter
                        kalman.statePre[:2] = np.array([[x], [y]], dtype=np.float32)
                        kalman.statePre[2:] = np.zeros((2, 1), dtype=np.float32)

        else:
            green_start_time = None

    #maintain tracking across mode changes
    if tracker_initialized:
        success, bbox = csrt_tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            if last_good_bbox is not None:
                iou = calculate_iou(last_good_bbox, (x, y, w, h))
                iou_history.append(iou)
                
            last_good_bbox = bbox
            frame_count_no_green = 0

            bounding_box = (x, y, w, h)
            
            iou_history.append(iou)
            area_history.append(w * h)  

        elif is_nir and prev_frame is not None and last_good_bbox is not None:
            print("Tracking lost in NIR, using ORB Feature Matching.")
            matched_bbox = match_features(prev_frame, gray, last_good_bbox)
            if matched_bbox:
                x, y, w, h = matched_bbox
                csrt_tracker.init(frame, (x, y, w, h))
                print("Reinitialized tracker with ORB.")

    #ensure bb is Drawn
    if tracker_initialized:
        try:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except:
            print("Error drawing bounding box.")

    #display resize
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    resized_mask = cv2.resize(mask, (resize_width, resize_height))

    #input video normal speed
    #elapsed_time = time.time() - start_time
    #sleep_time = max(0, frame_delay - elapsed_time)
    #time.sleep(sleep_time) 

    #save output
    out.write(frame)

    processing_time = time.time() - start_time
    frame_times.append(processing_time)

    cv2.imshow("Processed Frame", resized_frame)
    cv2.imshow("Green Mask", resized_mask)

    #time delay
    #elapsed_time = time.time() - start_time
    #if elapsed_time < frame_delay:
    #    time.sleep(frame_delay - elapsed_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plot IOU
if iou_history:
    mean_iou = sum(iou_history) / len(iou_history)
    print(f"Mean IoU: {mean_iou:.4f}")
else:
    print("No IoU values recorded.")

plt.figure()
plt.plot(iou_history)
plt.title("IoU over Time")
plt.xlabel("Frame")
plt.ylabel("IoU")
plt.show()

# Plot Bounding Box Area over time
plt.figure()
plt.plot(area_history)
plt.title("Bounding Box Area over Time")
plt.xlabel("Frame")
plt.ylabel("Area")
plt.show()

# Plot Processing Time per Frame
plt.figure()
plt.plot(frame_times)
plt.title("Processing Time per Frame")
plt.xlabel("Frame")
plt.ylabel("Time (s)")
plt.show()