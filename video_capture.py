import cv2, time, pandas
from datetime import datetime

first_frame = None
video = cv2.VideoCapture(0)
status_list = [None, None]
time = []
df = pandas.DataFrame(columns=['Start', 'End'])  # using panda dataframe to store date and time of motion.
while True:
    check, frame = video.read()
    status = 0

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting frame image to grey scaled version.
    grey = cv2.GaussianBlur(grey, (21, 21), 0)

    if first_frame is None:
        first_frame = grey  # storing the first frame.
        continue

    delta_frame = cv2.absdiff(first_frame,
                              grey)  # calculating the absolute difference between the initial frame and changed image.
    tresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    tresh_delta = cv2.dilate(tresh_delta, None, iterations=2)
    (cnts, _) = cv2.findContours(tresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:  # comparing different contours.
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)  # indicating changes withe a green rectangle.
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 3)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:  # storing the time of motion in a list.
        time.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time.append(datetime.now())

    cv2.imshow("Delta Frame", delta_frame)  # displaying the frames.
    cv2.imshow("Threshhold Frame", tresh_delta)
    cv2.imshow("Colour Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # defining letter to close the window.
        if status == 1:
            time.append(datetime.now())
        break

for i in range(0, len(time), 2):
    df = df.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)  # inserting values to data frame.

df.to_csv("Times.csv")  # creating .csv executable file.
video.release()
cv2.destroyAllWindows
# PRESS 'q' ON KEYBOARD TO STOP THE PROGRAM.