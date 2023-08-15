from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
from playsound import playsound 

file = r"C:\Users\Jay\Downloads\alert_msg.mp3"

#initialise Tkinter
root = tkinter.Tk()
root.withdraw()


model = load_model("C:/Users/Jay/Downloads/Vhac.h5")

face_cascade = cv2.CascadeClassifier(r"C:\Users\Jay\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

source=cv2.VideoCapture(0,cv2.CAP_DSHOW)

labels_dict={0:' MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,img=source.read() 
    
    img = np.array(img, dtype=np.uint8)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
        if label == 1:
            playsound(file)
            messagebox.showwarning("warning","please wear a face mask")
   
        else:
            
            break
		
        
    cv2.imshow('LIVE',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        
cv2.destroyAllWindows()
source.release()