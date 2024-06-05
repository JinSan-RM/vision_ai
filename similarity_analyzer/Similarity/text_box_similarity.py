def text_box_similarity_calculator(in_t_boxes_num, out_t_boxes_num):
    
    temp_list = []
    temp_list.append(in_t_boxes_num)
    temp_list.append(out_t_boxes_num)
    
    text_box_similarity_ratio = min(temp_list) / max(temp_list)
    
    return text_box_similarity_ratio