    #   ======================================================
    #
    #   AI 결과값에 대한 추출 과정 (minX, minY, maxX, maxY)
    #
    #   ======================================================

def classbox(results):
    dataList = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        names = result.names
        origImg = result.orig_shape
        imgWidth = origImg[1]
        imgHeight = origImg[0]

        data = []
        for box in boxes:
            boxResult = box.xywhn.tolist()[0]
            classNum = int(box.cls.tolist()[0])
            className = names[classNum]
            conf = box.conf.tolist()[0]
            centerX = boxResult[0]
            centerY = boxResult[1]
            width = boxResult[2]
            height = boxResult[3]

            left = centerX - (width / 2)
            top = centerY - (height / 2)

            minX = left
            minY = top
            maxX = left + width
            maxY = top + height
            
            origin_minX = minX * imgWidth
            origin_minY = minY * imgHeight
            origin_maxX = maxX * imgWidth
            origin_maxY = maxY * imgHeight
            boxData = [classNum, int(origin_minX), int(origin_minY), int(origin_maxX), int(origin_maxY), conf]
            data.append(boxData)

        dataList.append(data)

    return dataList

def process_boxes(boxes_list, isImage=None):                     
    processed_boxes = []
    for boxes in boxes_list:
        for box in boxes:
        # if isImage == True :
        #     print( boxes )
            processed_boxes.append(box)
        
    return processed_boxes

