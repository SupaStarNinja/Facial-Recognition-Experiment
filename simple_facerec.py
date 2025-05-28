import cv2
import numpy as np
import face_recognition
import os

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        images_path = os.listdir(images_path)
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(img_path)[0]
                
                img = face_recognition.load_image_file(f"images/{img_path}")
                
                face_encodings = face_recognition.face_encodings(img)
                
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded encoding for {name}")
                else:
                    print(f"No face found in {img_path}")

    def detect_known_faces(self, frame):
        """
        Detect faces in frame and compare with known faces
        :param frame:
        :return: face_locations, face_names
        """
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]

        return face_locations, face_names
