import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt

class FacialRehabTracker:
    def __init__(self):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.symmetry_data = []
        self.start_time = time.time()
        self.csv_path = "facial_progress.csv"

        # Facial landmark index pairs for symmetry check (left-right)
        self.symmetric_pairs = [
            (234, 454),  # Cheekbones
            (133, 362),  # Outer eye corners
            (130, 359),  # Inner eye corners
            (61, 291),   # Mouth corners
            (78, 308),   # Mid-lips
            (172, 397)   # Chin region
        ]

    def calculate_symmetry(self, face_landmarks, w, h):
        left_points, right_points = [], []
        for l, r in self.symmetric_pairs:
            left = np.array([face_landmarks[l][0], face_landmarks[l][1]])
            right = np.array([w - face_landmarks[r][0], face_landmarks[r][1]])  # mirror x
            left_points.append(left)
            right_points.append(right)

        left_points, right_points = np.array(left_points), np.array(right_points)
        diff = np.linalg.norm(left_points - right_points, axis=1)
        symmetry_score = max(0, 100 - np.mean(diff) / 3)  # Convert to %
        return round(symmetry_score, 2)

    def track(self):
        cap = cv2.VideoCapture(0)
        pTime = 0
        print("Press 's' for smile, 'b' for blink, 'r' for rest, 'q' to quit.")

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)

            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    ih, iw, ic = img.shape
                    face_points = [(int(lm.x * iw), int(lm.y * ih)) for lm in faceLms.landmark]

                    symmetry = self.calculate_symmetry(face_points, iw, ih)
                    cv2.putText(img, f"Symmetry: {symmetry}%", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec
                    )

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Facial Rehabilitation Tracker", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('s'), ord('b'), ord('r')]:
                label = 'Smile' if key == ord('s') else 'Blink' if key == ord('b') else 'Rest'
                t = round(time.time() - self.start_time, 2)
                self.symmetry_data.append((t, symmetry, label))
                print(f"Recorded {label} | Symmetry: {symmetry}%")

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()
        self.plot_progress()

    def save_data(self):
        if not self.symmetry_data:
            print("No data to save.")
            return
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Time (s)", "Symmetry (%)", "Expression"])
            writer.writerows(self.symmetry_data)
        print(f"Data saved to {self.csv_path}")

    def plot_progress(self):
        if not self.symmetry_data:
            print("No progress data to plot.")
            return

        times = [t for t, _, _ in self.symmetry_data]
        symmetry_scores = [s for _, s, _ in self.symmetry_data]
        labels = [e for _, _, e in self.symmetry_data]

        plt.figure(figsize=(8, 5))
        plt.plot(times, symmetry_scores, '-o', label="Symmetry %")
        for i, label in enumerate(labels):
            plt.text(times[i], symmetry_scores[i] + 0.5, label, fontsize=8)
        plt.xlabel("Time (s)")
        plt.ylabel("Facial Symmetry (%)")
        plt.title("Facial Rehabilitation Progress Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    tracker = FacialRehabTracker()
    tracker.track()
