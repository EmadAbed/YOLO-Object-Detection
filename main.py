import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

model = YOLO('yolov8s.pt')

class_names = model.names

st.set_page_config(page_title="Detect Object In Image",page_icon="🤖",layout="centered")
st.title("Detect Object In Image")


object_detect = st.selectbox('Select Object :',options= list(class_names.values()))
image_upload = st.file_uploader("Upload Image",type=['png','jpg','jpeg','webp'])

if image_upload is not None:
    image = Image.open(image_upload)
    image = np.array(image)
    result = model.predict(image)[0].boxes.data
    predict = pd.DataFrame(result).astype(float)

    for idx, row in predict.iterrows():
        x1 ,y1,x2,y2 = int(row[0]),int(row[1]),int(row[2]),int(row[3])
        cls = class_names[int(row[5])]
        conf = row[4]

        if object_detect == cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image,(x1,y1),(x2,y1 + 30),(0,255,0),-1)
            cv2.putText(image,f'{cls}',(x1,y1 + 15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
            cv2.putText(image, f'{round(conf * 100,2)}', (x1, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
    if st.button('Show Image',key='show_image'):
        st.image(image,caption='Image',use_column_width=True)