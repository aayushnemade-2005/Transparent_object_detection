import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {i}: not available")
        continue
    ok, frame = cap.read()
    if ok:
        print(f"Index {i}: works, {frame.shape[1]}x{frame.shape[0]}  (press any key for the next one)")
        cv2.imshow(f"Camera index {i}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()