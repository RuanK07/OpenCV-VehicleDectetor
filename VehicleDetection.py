import cv2
import numpy as np

class VehicleDetection:
    def __init__(self, video_path):
        self.min_contour_width = 50
        self.min_contour_height = 50
        self.screen_width = 1920
        self.screen_height = 1080
        self.offset = 10
        self.line_height = 380
        self.matches = []
        self.cars = 0

        self.video_path = video_path
        self.cap = None

    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(3, self.screen_width)
        self.cap.set(4, self.screen_height)

        if not self.cap.isOpened():
            raise ValueError("Could not open video file.")

    def close_video(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_centroid(self, x, y, width, height):
        x1 = int(width / 2)
        y1 = int(height / 2)

        centroid_x = x + x1
        centroid_y = y + y1

        return centroid_x, centroid_y

    def process_frame(self, frame):
        ret, frame1 = self.cap.read()
        ret, frame2 = self.cap.read()
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        kernel_size = (5, 5)
        variation_Gaussian_function = 0
        blur = cv2.GaussianBlur(gray, kernel_size, variation_Gaussian_function)

        limit_to_white = 20
        ret, th = cv2.threshold(blur, limit_to_white, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contours, height = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for (i, contour) in enumerate(contours):
            (x, y, width, height) = cv2.boundingRect(contour)
            contour_valid = (width >= self.min_contour_width) and (
                    height >= self.min_contour_height)

            if not contour_valid:
                continue

            cv2.rectangle(frame1, (x - 10, y - 10), (x + width + 10, y + height + 10), (255, 0, 0), 2)

            color_frame = (0, 255, 0)
            cv2.line(frame1, (0, self.line_height), (2600, self.line_height), color_frame, 2)

            centroid = self.get_centroid(x, y, width, height)
            self.matches.append(centroid)

            circle_color = (0, 255, 0)
            circle_radius = 5
            fill_circle = -1
            cv2.circle(frame1, centroid, circle_radius, circle_color, fill_circle)
            cx, cy = self.get_centroid(x, y, width, height)

            for (x, y) in self.matches:
                if (self.line_height + self.offset) > y > (self.line_height - self.offset):
                    self.cars = self.cars + 1
                    self.matches.remove((x, y))

        cv2.putText(frame1, "Total Cars Detected: " + str(self.cars), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Vehicle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Vehicle Detection", frame1)


from VehicleDetection import VehicleDetection
video_path = "car_traffic2.mp4"

vehicle_detection = VehicleDetection(video_path)

vehicle_detection.open_video()

while True:
    ret, frame = vehicle_detection.cap.read()
    if not ret:
        break

    vehicle_detection.process_frame(frame)

    if cv2.waitKey(1) == 27:
        break

vehicle_detection.close_video()
