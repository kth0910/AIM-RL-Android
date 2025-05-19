import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 좌우 반전
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 피부색 HSV 범위 (조정 가능)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 잡음 제거
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))

                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                if angle <= np.pi / 2:  # 90도 이하만 인식
                    cnt += 1
                    cv2.circle(frame, far, 4, [0, 0, 255], -1)

            # 손가락 수 출력
            cv2.putText(frame, f"Fingers: {cnt+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
