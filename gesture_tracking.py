from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import arange

def load_pretrain_model():
    # load json and hdf5, json: sturcture of model, hdf5: weights of model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("weights.hdf5")
    print("Model load success.")
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def main():
    cv2.namedWindow("INSE6610")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    
    gesture_list = ["peace","punch","stop","thumbs_up"]
    
    while ret:
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
        cv2.imshow("INSE6610", frame)
        frame = frame[200:400,300:500]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.reshape((1,)+frame.shape + (1,))
        datagen = ImageDataGenerator(rescale=1./255)
        something = datagen.flow(frame,batch_size=32)
        classifier_result=model.predict_generator(something, steps = 1)

        new_gesture={'PEACE': classifier_result[0][0],
                     'PUNCH': classifier_result[0][1],
                     'STOP': classifier_result[0][2],
                     'Thumbs Up': classifier_result[0][3]}

        update(new_gesture)
        ret, frame = cap.read()
        key = cv2.waitKey(20)
        if key == 27:
            break
    cv2.destroyWindow("INSE6610")
    cap=None
    
def update(new_gesture):
    global gesture_dict
    gesture_dict=new_gesture


def update_value(a):
    xar= ["PEACE", "PUNCH", "STOP", "Thumbs Up"]
    yar = []
    for gesture in gesture_dict:
        yar.append(gesture_dict[gesture])

    ax1.clear()
    plt.bar(xar,yar)


gesture_dict = {'PEACE':0, 'PUNCH':0, 'STOP': 0, 'Thumbs Up':0}


if __name__ == "__main__":
    model = load_pretrain_model()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ani = animation.FuncAnimation(fig, update_value)
    fig.show()
    main()




# reference:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://www.kaggle.com/benenharrington/hand-gesture-recognition-database-with-cnn
# http: // yangguang2009.github.io / 2016 / 11 / 27 / deeplearning / develop - neural - network - model - with-keras - step - by - step /
# https: // keras.io /

# dataset:
# https://www.idiap.ch/resource/gestures/