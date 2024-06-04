    #   =================================================
    #
    #   이미지내부의 AI 결과값을 토대로 바운더리 박스 스케치
    #   
    #   스케치된 Boundary Box 
    #
    #   =================================================

import cv2
import numpy as np

def ImgBoundarySketch(img, output_data):
    img_Width, img_Height = img.shape[:2]
    print(img_Width, img_Height, "<=========shape")
    cv2.rectangle(img, (0, 0), (1920, 2000), (255, 255, 255), thickness = cv2.FILLED)
    if img is None:
        print("로딩 실패")
    for detection in output_data:
        class_id = detection[0]
        start_x = detection[1]
        start_y = detection[2]
        end_x = detection[3]
        end_y = detection[4]
        
        if class_id == 1:
            # cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 255, 255), thickness=cv2.FILLED)
            # cv2.ellipse(img, (int((start_x + end_x) / 2), int((start_y + end_y) / 2)), (int((end_x - start_x) / 2), int((end_y - start_y) / 2)), 0, 0, 360,(105, 105, 105), thickness=1)
            # cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (220, 220, 200), thickness=1)
            pass
        elif class_id == 2:
            # cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 255, 255), thickness=cv2.FILLED)
            cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (105, 105, 105), thickness=cv2.FILLED)
            # cv2.putText(image, str(class_id), (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    
    return img


#====================================
#
# Boundary_remake
#
#====================================

# def ImgBoundaryRemake(img, output_data):
    
#     base_width = 1920
#     base_height = 1280
    
#     img_Width, img_Height = img.shape[:2]
    
#     print("Currnet Image Size : ", img_Width, img_Height, "<=========shape")
    
#     x_enlarge_ratio = base_width / img_Width
#     y_enlarge_ratio = base_height / img_Height
    

#     # NOTE : 이미지를 재생성하고 할지 위에 얹고 할지 리소스 체크
#     drawing_paper = np.full((base_width, base_height, 1), 128, dtype=np.uint8)
#     # cv2.rectangle(img, (0, 0), (1920, 1280), (255, 255, 255), thickness = cv2.FILLED)
    
#     if img is None:
#         print("로딩 실패")
    
#     image_boxes = []
#     text_boxes = []
    
#     for box in output_data:
#         if box[0] == 1:
#             text_boxes.append(box)
#         else :
#             image_boxes.append(box)
    
    
#     num_text = len(text_boxes)
#     num_image = len(image_boxes)
    
#     enlarged_image_boxes = []
    
#     for i, (class_id, min_x, min_y, max_x, max_y, _) in enumerate(image_boxes):
#         enlarged_image_boxes.append([class_id, 
#                                      int(min_x * x_enlarge_ratio), 
#                                      int(min_y * y_enlarge_ratio),
#                                      int(max_x * x_enlarge_ratio), 
#                                      int(max_y * y_enlarge_ratio)])
    
#     enlarged_image_boxes.sort(key=lambda x: (x[1], x[2]))
    
#     # extract_start_x = [image_box[1] for image_box in enlarged_image_boxes]
    
#     min_value_of_start_x = enlarged_image_boxes[0][1]
    
#     min_start_x_boxes = []
    
#     for enlarged_image_box in enlarged_image_boxes:
#         if enlarged_image_box[1] == min_value_of_start_x:
#             min_start_x_boxes.append(enlarged_image_box)
    
#     for test in min_start_x_boxes:
#         cv2.rectangle(drawing_paper, test[0], test[1], test[2], test[3], (105, 105, 105), thickness=cv2.FILLED)
    
#     # 이게 가장 작은 x 값을 갖는 인덱스 박스 image_boxes[min_index_of_start_x]
    `   `
#     # for detection in output_data:
#     #     class_id = detection[0]
#     #     start_x = detection[1]
#     #     start_y = detection[2]
#     #     end_x = detection[3]
#     #     end_y = detection[4]
        
#     #     if class_id == 1:
#     #         # cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 255, 255), thickness=cv2.FILLED)
#     #         # cv2.ellipse(img, (int((start_x + end_x) / 2), int((start_y + end_y) / 2)), (int((end_x - start_x) / 2), int((end_y - start_y) / 2)), 0, 0, 360,(105, 105, 105), thickness=1)
#     #         # cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (220, 220, 200), thickness=1)
#     #         pass
#     #     elif class_id == 2:
#     #         # cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 255, 255), thickness=cv2.FILLED)
#     #         cv2.rectangle(img, start_x, start_y, end_x, end_y,(105, 105, 105), thickness=cv2.FILLED)
#     #         # cv2.putText(image, str(class_id), (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    
#     return drawing_paper