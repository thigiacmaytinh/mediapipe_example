import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

count_squat = 0
current_state = ""

###################################################################################################

def make_landmark_timestep(results):
    """
    It takes the results of the pose estimation and returns a list of the x and y coordinates of the
    landmarks
    
    :param results: the output of the pose estimation
    :return: A list of lists of the x and y coordinates of the landmarks.
    """
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append([lm.x, lm.y])
    return c_lm

###################################################################################################

def getDegree(pos_shoulder, pos_elbow, pos_wrist):
    """
    It takes the coordinates of the shoulder, elbow, and wrist, and returns the angle between the
    shoulder and elbow, and the elbow and wrist
    
    :param pos_shoulder: The position of the shoulder
    :param pos_elbow: The position of the elbow
    :param pos_wrist: The position of the wrist
    :return: The degree of the angle between the shoulder and the wrist.
    """
    vecto1 = np.array([
        (pos_shoulder[0] - pos_elbow[0]),
        (pos_shoulder[1] - pos_elbow[1])
    ])

    vector2 = np.array([
        (pos_wrist[0] - pos_elbow[0]),
        (pos_wrist[1] - pos_elbow[1])
    ])
    cos_degree = vecto1.dot(vector2) / \
        (np.linalg.norm(vecto1) * np.linalg.norm(vector2))
    degree = np.arccos(cos_degree)
    return int(degree * 180 / np.pi)

###################################################################################################

def draw_class_on_image(label, img):
    """
    It draws the class label on the image
    
    :param label: The label to be drawn on the image
    :param img: The image on which you want to draw the text
    :return: The image with the label drawn on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 30)
    fontScale = 1
    fontColor = (0,0, 255)
    thickness = 2
    lineType = 2
    cv2.rectangle(img, (0, 0), (100, 40), (255,255,255), -1)
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                cv2.LINE_AA)
    return img

###################################################################################################

def draw_num_squat(num, img):
    """
    It draws the class label on the image
    
    :param label: The label to be drawn on the image
    :param img: The image on which you want to draw the text
    :return: The image with the label drawn on it.
    """
    im_height, im_width = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (im_width - 90, 30)
    fontScale = 1
    fontColor = (0,0, 255)
    thickness = 2    
    label = str(num)
    cv2.rectangle(img, (im_width - 100, 0), (im_width, 40), (255,255,255), -1)
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                cv2.LINE_AA)
    return img

###################################################################################################

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            pos_right_hip = c_lm[24]
            pos_right_knee = c_lm[26]
            pos_right_ankle = c_lm[28]

            pos_left_hip = c_lm[23]
            pos_left_knee = c_lm[25]
            pos_left_ankle = c_lm[27]

            degree_right = getDegree(pos_shoulder=pos_right_hip,
                                        pos_elbow=pos_right_knee,
                                        pos_wrist=pos_right_ankle)

            degree_left = getDegree(pos_shoulder=pos_left_hip,
                                    pos_elbow=pos_left_knee,
                                    pos_wrist=pos_left_ankle)

            if(degree_left < 100 and degree_right < 100):
                draw_class_on_image("down", image)
                if(current_state == "up"):
                    current_state = "down"                    
            elif(degree_left > 150 and degree_right > 150):
                draw_class_on_image("up", image)
                if(current_state == ""):
                    current_state = "up"
                elif(current_state == "down"):
                    count_squat += 1                    
                    current_state = "up"
            else:
                draw_class_on_image("", image)
                
            draw_num_squat(count_squat, image)
            # else:
            #     draw_class_on_image(str(degree_left) + ' ' + str(degree_right), image)



        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
