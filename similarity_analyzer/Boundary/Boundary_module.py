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
    
    
def create_patterned_image( rect_params ):
    """
    rect_params: 리스트 [(label, min_x, min_y, max_x, max_y, score)] 형태로 각 사각형의 좌표를 전달
    """
    # 빈 이미지 생성 (회색 배경)
    image = np.full((1280,1920, 1), 0, dtype=np.uint8)
    rect_params = [item for item in rect_params if item[0] == 2]
    rect_params.sort(key=lambda x: (x[2], x[1]))

    data = []
    for i in rect_params:
        data.append(i[1:5])

    data.sort(key=lambda x: (x[0], x[1]))
    
    grouped_data = []

    for i in range(len(data)):
        group = [data[i]]
        for j in range(i, len(data)):
            if i != j:
                if abs(data[i][0] - data[j][0]) <= 10 and abs(data[i][2] - data[j][2]) <= 10:
                    group.append(data[j])
        if len(group) > 1:
            grouped_data.append(group)
    
    flattened_grouped_data = [item for sublist in grouped_data for item in sublist]

    not_grouped = [item for item in data if item not in flattened_grouped_data]
    not_grouped = sorted(not_grouped, key=lambda x: x[0])
    interval = 15
    interval_x_num = 0

    combined_groups = []
    
    not_grouped = sorted(not_grouped, key=lambda x: x[0])

    grouped_not_grouped = []
    current_group = []

    for i, item in enumerate(not_grouped):
        if not current_group:
            current_group.append(item)
        else:
            if item[0] <= current_group[-1][0] + 2:
                current_group.append(item)
            else:
                current_group.sort(key=lambda x: x[1])  
                grouped_not_grouped.append(current_group)
                current_group = [item]
    if current_group:
        current_group.sort(key=lambda x: x[1])
        grouped_not_grouped.append(current_group)

    combined_groups = grouped_data + grouped_not_grouped

    def median_xmin(group):
        xmins = [item[0] for item in group]
        return np.median(xmins)

    combined_groups.sort(key=median_xmin)

    for group in combined_groups:
        median_value = int(median_xmin(group))
        for item in group:
            item[0] = median_value
    
    control_position = []
    for i, d in enumerate(combined_groups):
        d.sort(key=lambda d: (d[0], d[1]))
        for j, x in enumerate(d):
            if j == 0:
                x[0] = x[0] + (interval * interval_x_num)
                x[1] = x[1]
                x[2] = x[2] + (interval * interval_x_num)
                x[3] = x[3]
                control_position.append(x)
            else:
                x[0] = x[0] + (interval * interval_x_num)
                x[1] = x[1] + (interval * j)
                x[2] = x[2] + (interval * interval_x_num)
                x[3] = x[3] + (interval * j)
                control_position.append(x)
                             

        interval_x_num += 1

    first_rect = control_position[0]
    offset_x = 100 - first_rect[0]
    offset_y = 100 - first_rect[1]

    adjusted_rectangles = [
        [rect[0] + offset_x, rect[1] + offset_y, rect[2] + offset_x, rect[3] + offset_y]
        for rect in control_position
    ]
    for rect in adjusted_rectangles:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (245, 245, 245), thickness=1)


    return image

    # # NOTE : 이미지를 재생성하고 할지 위에 얹고 할지 리소스 체크
    # drawing_paper = np.full((base_width, base_height, 1), 128, dtype=np.uint8)
    # # cv2.rectangle(img, (0, 0), (1920, 1280), (255, 255, 255), thickness = cv2.FILLED)
