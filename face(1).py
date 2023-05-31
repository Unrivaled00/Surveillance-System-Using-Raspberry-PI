import face_recognition
import os
import dlib
import cv2
import os
import numpy as np
import cv2
import imutils
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
sender_email = "write your email here"
receiver_email = "winniethepooh20023@gmail.com"
def message():
   subject = "Unknown Face Detected!"
   message = "An unknown face has been detected."
   msg = MIMEMultipart()
   msg["Subject"] = subject
   msg["From"] = sender_email
   msg["To"] = receiver_email
   msg.attach(MIMEText(message))
   with open("unknown_face.jpg",'rb') as img:
    image=MIMEImage(img.read())
    msg.attach(image)
   with smtplib.SMTP("smtp.gmail.com", 587) as server:
      server.starttls()
      server.login(sender_email, "jpiaywsmnfmnyrcp")
      server.sendmail(sender_email, receiver_email, msg.as_string())
                 

person_directories = {
    "person 1": r"person 1 path",
    "person 2": r"person 2 path",
    "person 3": r"person 3 path",
    "person 4": r"person 4 path"
}

person_face_encodings = {}
person_names = []

for person, image_directory in person_directories.items():
    # Define a list to store the face encodings for the current person
    face_encodings = []
    
    # Iterate over the image files in the current person's directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Load the image file
            image_path = os.path.join(image_directory, filename)
            image = face_recognition.load_image_file(image_path)

            # Compute the face encodings for the image
            face_encoding = face_recognition.face_encodings(image)[0]

            # Append the face encoding to the list
            face_encodings.append(face_encoding)

    # Only store the face encodings if there is at least one encoding
    if face_encodings:
        # Store the face encodings for the current person in the dictionary
        person_face_encodings[person] = face_encodings
        
        # Append the person name to the list
        person_names.append(person)

# Initialize an empty list to store the face encodings
Our_faces = []
person_names = []

# Iterate over the person names and face encodings
for person, face_encodings in person_face_encodings.items():
    # Append the face encodings to the list
    Our_faces.extend(face_encodings)
    
    # Append the person name to the list
    person_names.extend([person] * len(face_encodings))

face_locations = []
face_encodings = []
face_names = []
frame_number = 0
i=0
vidcap = cv2.VideoCapture(0)
vidcap.set(3,640)
vidcap.set(4,480)
if vidcap.isOpened():
    while True:
        # Read a frame from the video capture
        ret, frame = vidcap.read()

        if ret:
            # Find face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Reset face names for each frame
            face_names = []

            for face_encoding in face_encodings:
                # Compare the current face encoding with the known faces
                match = face_recognition.compare_faces(Our_faces, face_encoding, tolerance=0.6)

                name = None
                if any(match):
                # Find the first matched person
                 index = np.argmax(match)
                 name = person_names[index]
                 face_names.append(name)
                 for (top, right, bottom, left), name in zip(face_locations, face_names):
                  if not name:
                     continue
                  cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                  cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                  font = cv2.FONT_HERSHEY_DUPLEX
                  cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2) 
                else:
                    for (top, right, bottom, left) in face_locations:
                     unknown="unknown"
                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                     font = cv2.FONT_HERSHEY_DUPLEX
                     cv2.putText(frame, unknown, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)
                     cv2.imwrite("unknown_face.jpg", frame)
                     image = cv2.imread('unknown_face.jpg')
                     message()

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        

        else:
            print("Frame not captured")
else:
    print("Cannot open camera")

# Release the video capture and close the OpenCV windows
vidcap.release()
cv2.destroyAllWindows()
