import os
import cv2
import time
import math
import numpy as np
from pathlib import Path
from openvino.inference_engine import IECore
from collections import deque
from playsound import playsound
from PIL import Image, ImageDraw, ImageFont

import pygame
pygame.mixer.init()

score_thresh = 0.2
COUNT = 0
LAST_COUNT = 0
TIP_FRAME = 200
COUNT_FRAME = 1

# https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}
LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    """
    Draw complex characters.
    """
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def pad_and_resize(frame, pad_w, pad_h, input_w, input_h):
    """
    Resize the frame to the input size of the model.
    """
    padded = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    padded = cv2.resize(padded, (input_w, input_h), interpolation=cv2.INTER_AREA)
    return padded


def get_bodies(results, padded_w, padded_h, img_w, img_h):
    """
    Extract the information to calculate the push-up posture from the model's output.
    """
    results = np.squeeze(results["Identity"])
    bodies = []
    for i in range(6):
        kps = results[i][:51].reshape(17,-1)
        bbox = results[i][51:55].reshape(2,2)          
        score = results[i][55]
        if score > score_thresh:
            ymin, xmin, ymax, xmax = (bbox * [padded_h, padded_w]).flatten().astype(np.int32)
            kp_xy =kps[:,[1,0]]
            keypoints = kp_xy * np.array([padded_w, padded_h])
            # body = Body(score=score, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, 
            #             keypoints_score = kps[:,2], 
            #             keypoints = keypoints.astype(np.int),
            #             keypoints_norm = keypoints / np.array([img_w, img_h]))
            body = {'keypoints_score': kps[:,2], 'keypoints': keypoints.astype(np.int32)}
            bodies.append(body)
    return bodies

def get_angle(frame, p1, p2, p3, type, drawText=False, voice=False):
    """
    Calculate the angle between three points.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # use the trigonometric formula to get the angle value 
    # between 3 points p1-p2-p3, with p2 as the angle, between 0-180 degrees
    angle = int(math.degrees(math.atan2(y1 - y2, x1 - x2) -
                math.atan2(y3 - y2, x3 - x2)))
    if angle < 0:
        angle = angle + 360
    if angle > 180:
        angle = 360 - angle
    if drawText:
        cv2.putText(frame, str(angle), (x2 - 20, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    ok = 1
    global TIP_FRAME
    if type==1:
        if angle<160:
            cv2.polylines(frame, [np.array([[x1,y1],[x2,y2],[x3,y3]])], False, (0,0,255), 2, cv2.LINE_AA)
            ok = 0
            if TIP_FRAME==0:
                pygame.mixer.music.load(f'voice/body.wav')
                pygame.mixer.music.play()
                TIP_FRAME=100
    elif type==2:
        global COUNT
        # print(angle,COUNT)
        if angle<80:
            if COUNT == int(COUNT):
                COUNT += 0.5
        elif angle>160:
            if COUNT!=int(COUNT):
                # TIP_FRAME=200
                COUNT += 0.5
                pygame.mixer.music.load(f'voice/{int(COUNT)}.wav')
                pygame.mixer.music.play()
        else:
            cv2.polylines(frame, [np.array([[x1,y1],[x2,y2],[x3,y3]])], False, (0,0,255), 2, cv2.LINE_AA)
            ok = 0

    elif type==3:
            if angle>80 or angle<160:
                cv2.polylines(frame, [np.array([[x1,y1],[x2,y2],[x3,y3]])], False, (0,0,255), 2, cv2.LINE_AA)
                ok = 0
    return ok

def draw_pose(frame, bodies):
    """
    Draw the pose on the frame and judge whether the posture is standard.
    """
    text1=' √ '+'Keep your body straight.'
    color1=(0,255,0)
    text2=' √ '+'Arms straight when up and body close to the ground when down.'
    color2=(0,255,0)
    text3=' √ '+'Shoulders perpendicular to your upper arms when up, and level with them when down.'
    color3=(0,255,0)
    LAST_COUNT = COUNT
    for body in bodies:
        x1,y1 = body['keypoints'][11]
        x2,y2 = body['keypoints'][12]
        if (x1-x2)**2 + (y1-y2)**2 > 25**2 and abs(x1-x2) > 50*abs(y1-y2):
            type = 1 # in the mirror
        else:
            type = 2 

        lines = [np.array([body['keypoints'][point] for point in line]) 
                    for line in LINES_BODY 
                    if body['keypoints_score'][line[0]] > score_thresh and body['keypoints_score'][line[1]] > score_thresh
                ]
        cv2.polylines(frame, lines[4:], False, (0,255,0), 1, cv2.LINE_AA)

        # judge whether the posture is standard
        for i,x_y in enumerate(body['keypoints']):
            if i<5:
                continue
            if body['keypoints_score'][i] > score_thresh:
                color = (0,255,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

        if type == 2:
            ok1 = get_angle(frame,body['keypoints'][6],body['keypoints'][12],body['keypoints'][14],1,True)
            ok2 = get_angle(frame,body['keypoints'][5],body['keypoints'][11],body['keypoints'][13],1)
            ok3 = get_angle(frame,body['keypoints'][12],body['keypoints'][14],body['keypoints'][16],1,True)
            ok4 = get_angle(frame,body['keypoints'][11],body['keypoints'][13],body['keypoints'][15],1)
            if not((ok1 or ok2) and (ok3 or ok4)):
                text1=' × '+'Keep your body straight.'
                color1=(255,0,0)
        else:
            ok1 = get_angle(frame,body['keypoints'][5],body['keypoints'][7],body['keypoints'][9],2,True,True)
            ok2 = get_angle(frame,body['keypoints'][6],body['keypoints'][8],body['keypoints'][10],2,True)
            ok3 = get_angle(frame,body['keypoints'][5],body['keypoints'][6],body['keypoints'][8],3,True)
            ok4 = get_angle(frame,body['keypoints'][6],body['keypoints'][5],body['keypoints'][7],3)
            if not ok1 or not ok2:
                text2=' × '+'Arms straight when up and body close to the ground when down.'
                color2=(255,0,0)
            if not ok3 or not ok4:
                text3=' × '+'Shoulders perpendicular to your upper arms when up, and level with them when down.'
                color3=(0,255,0)

    frame = cv2AddChineseText(frame,text1, (10, 10),color1, 10)
    frame = cv2AddChineseText(frame,text2, (10, 30),color2, 10)
    frame = cv2AddChineseText(frame,text3, (10, 50),color3, 10)
    frame = cv2AddChineseText(frame,f'Counter: {int(COUNT)}', (10, 70),(0,255,0), 30)

    # voice controller
    global TIP_FRAME, COUNT_FRAME
    if TIP_FRAME>0:
        TIP_FRAME-=1
    if COUNT == LAST_COUNT:
        COUNT_FRAME+=1
    if COUNT_FRAME % 200 == 0:
        pygame.mixer.music.load(f'voice/arm.wav')
        pygame.mixer.music.play()
        TIP_FRAME=100
    return frame


if __name__ == "__main__":
    # load the model
    from openvino.inference_engine import IECore
    ie = IECore()
    root_path = Path(__file__).resolve().parent
    xml_path = root_path / "models/movenet_multipose_lightning_256x256_FP32.xml"
    # args.xml = SCRIPT_DIR / f"models/movenet_multipose_lightning_{args.res}_FP32.xml"
    bin_path = os.path.splitext(xml_path)[0] + '.bin'
    print(xml_path,bin_path)
    model = ie.read_network(model=xml_path, weights=bin_path)
    # Input blob: input:0 - shape: [1, 3, 256, 256] (lightning)
    # Output blob: Identity - shape: [1, 6, 56]
    input_blob = next(iter(model.input_info))
    _, _, input_h, input_w = model.input_info[input_blob].input_data.shape
    device = 'CPU'
    model = ie.load_network(network=model, num_requests=1, device_name=device)


    path = input("Please enter the video path (no input and press Enter to use the camera):")
    if path == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video_fps,img_w,img_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(f'result_{path}.mp4', fourcc, video_fps, (img_w, img_h))

    if img_w / img_h > input_w / input_h:
        pad_w = 0
        pad_h = int(img_w * input_h / input_w - img_h)
        padded_w = img_w
        padded_h = img_h + pad_h
    else:
        pad_w = int(img_h * input_w / input_h - img_w)
        pad_h = 0
        padded_w = img_w + pad_w
        padded_h = img_h


    while True:     
        readSuccess, frame = cap.read()
        if not readSuccess:
            break

        padded = pad_and_resize(frame, pad_w, pad_h, input_w, input_h)
        frame_input = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)[None,] 
        results = model.infer(inputs={input_blob: frame_input})

        bodies = get_bodies(results, padded_w, padded_h, img_w, img_h)
        frame=draw_pose(frame, bodies)

        cv2.imshow("Push Up Tracker", frame)
        output.write(frame)

        key = cv2.waitKey(1) 
        if key == ord('q') or key == 27:
            break
        elif key == 32:
            cv2.waitKey(0)

    output.release()
    cap.release()
    cv2.destroyAllWindows()
