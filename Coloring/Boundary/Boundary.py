import cv2
import numpy as np

def ImgBoundarySketch(img, output_data):
    if img is None:
        print("로딩 실패")
    for detection in output_data:
        class_id = detection[0]
        start_x = detection[1]
        start_y = detection[2]
        end_x = detection[3]
        end_y = detection[4]
        
        if class_id == 1:
            cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,192,203), thickness=cv2.FILLED)
        elif class_id == 2:
            cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (153,204,255), thickness=cv2.FILLED)
            # cv2.putText(image, str(class_id), (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    
    return img
    
    
    