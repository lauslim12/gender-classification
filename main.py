#!usr/bin/env python

# Python Imports
import os
import time
from tkinter import Tk, Button, Label, filedialog

# Third party modules
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN

# Developer metadata.
__author__ = "Nicholas Dwiarto W."
__copyright__ = "Copyright 2020, Nicholas Dwiarto W."
__credits__ = "Nicholas Dwiarto W."

__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Nicholas Dwiarto W."
__status__ = "Prototype"

# Global Functions.
# I dislike doing this, but Tkinter leaves me no choice.
root = Tk()
image_panel = Label(root, image='')
female = Label(root, text='')
male = Label(root, text='')

# Constants
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def analyze_face():
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join('models', 'gender-recognition.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check the type of the input tensor.
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    image = Image.open('tmp.jpg').resize((width, height))

    # add N dim
    input_data = np.expand_dims(image, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(os.path.join('models', 'labels.txt'))
    print(labels)

    for i in top_k:
        if floating_model:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
    print('Time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    return results


def get_highlighted_face(filename, faces):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    x, y, width, height = faces[0]['box']
    rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)

    plt.savefig('plot.png', bbox_inches='tight')


def get_cropped_faces(filename, faces):
    """
        Only gets the first detected face!
        Returns cropped data.
    """
    try:
        # data = plt.imread(filename)
        # x1, y1, width, height = faces[0]['box']
        # x2, y2 = x1 + width, y1 + height
        # plt.axis('off')
        # plt.imshow(data[y1:y2, x1:x2])
        # plt.imsave('tmp.jpg', data[y1:y2, x1:x2])
        # get_highlighted_face(filename, faces)
        bounding_box = faces[0]['box']
        cv2.rectangle(filename,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2],
                       bounding_box[1] + bounding_box[3]),
                      (255, 0, 0),
                      5)
        cropped_image = filename[bounding_box[1]:bounding_box[1] +
                                 bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
        cv2.imwrite('tmp.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite('plot.png', cv2.cvtColor(filename, cv2.COLOR_RGB2BGR))

    except Exception as ex:
        print(ex)
        Label(root, text='No face detected!').pack()
        return


def detect_faces(image):
    """
        Detect faces using MTCNN.
    """
    face_detector_mtcnn = MTCNN()
    # image_to_detect = plt.imread(image)
    image_to_detect = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    try:
        faces = face_detector_mtcnn.detect_faces(image_to_detect)
        print(faces)
    except Exception as ex:
        print(ex)
        Label(root, text='Something is wrong! Please try again!').pack()
        return

    get_cropped_faces(image_to_detect, faces)


def open_file():
    filename = filedialog.askopenfilename(title='Select an image!')
    return filename


def open_image():
    image_panel.pack_forget()
    male.pack_forget()
    female.pack_forget()

    image_to_open = open_file()
    detect_faces(image_to_open)
    results = analyze_face()

    # Show full image in the tkinter's screen.
    image = Image.open('plot.png')
    image.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    image.save('plot.png')

    image = Image.open('plot.png')
    rendered_image = ImageTk.PhotoImage(image)

    # image_panel = Label(root, image=rendered_image)
    # image_panel.image = rendered_image
    # image_panel.pack()

    female_str = 'Female: ' + str(results[0])
    male_str = 'Male: ' + str(results[1])
    print(male_str, female_str)

    image_panel.image = rendered_image
    image_panel.config(image=rendered_image)
    female.configure(text=female_str)
    male.configure(text=male_str)

    image_panel.pack()
    female.pack()
    male.pack()

    os.remove('tmp.jpg')
    os.remove('plot.png')


def clear_screen(image_panel, female, male):
    image_panel.destroy()
    female.destroy()
    male.destroy()


def define_user_interface(root):
    root.title('Face Classifier Application')
    root.geometry('800x800')
    root.resizable(width=True, height=True)
    Label(root, text='Welcome to the Face Recognition Application!').pack()
    Label(root, text='Powered by MTCNN, TensorFlow, and Deep Learning!').pack()
    Button(root, text='Classify your face!', command=open_image).pack()


def main():
    define_user_interface(root)
    root.mainloop()


if __name__ == '__main__':
    main()
