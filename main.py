import cv2

VIDEO_SOURCE = "traffic.mp4"

video = cv2.VideoCapture(VIDEO_SOURCE)

if not video.isOpened():
    print("ERROR: Could not open the video.")
    exit()

print("Video opened successfully!")
print("Press Q to stop.")

while True:
    success, frame = video.read()

    if not success:
        print("End of video.")
        break

    cv2.imshow("Smart Traffic Violation Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()