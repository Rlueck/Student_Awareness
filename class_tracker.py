import face_recognition
from PIL import Image

#classphoto imports a screenshot from one of our recorded class sessions
classphoto = face_recognition.load_image_file(r'C:\Users\Randy\Desktop\IST718_class_photo.JPG')

#variable to print out the 4-digit pixel grid of each face location from the photo
face_locations = face_recognition.face_locations(classphoto)
face_locations

#let's see how many faces the program found
print(len(face_locations))

#let's save a copy of each of the faces the program recognizes

#counter var
i = 0

for face_location in face_locations:

    top, right, bottom, left = face_location
    print('''A face located at pixel location
          top: {}, left: {}, bottom: {}, right: {}'''.format(top, left, bottom, right))

    face_image = classphoto[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.save('face-{}.jpg'.format(i))
    i = i + 1

