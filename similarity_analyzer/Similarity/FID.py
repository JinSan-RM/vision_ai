import pytorch_fid_wrapper as pfw
import torch  
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageOps
import numpy as np
import cv2
from torchvision.transforms.functional import resize

def FID_score(img1, img2, output_size=299):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 이미지 객체 변환 확인 및 변환
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)
    # 객체 타입 검증
    if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
        raise ValueError("One or both provided objects are not PIL.Image.Image objects")

    # 이미지 전처리 설정
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지 타입 변환 (Tensor)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])

    # 리사이즈 및 패딩
    img1 = resize_keep_aspect_ratio(img1, (output_size, output_size))
    img2 = resize_keep_aspect_ratio(img2, (output_size, output_size))

    # cv2.imwrite('/code/Img/FID1.jpg', img1)
    # cv2.imwrite('/code/Img/FID2.jpg', img2)
    # 전처리 적용
    img1 = transform(img1)
    img2 = transform(img2)

    # 배치 생성
    batch = torch.stack([img1, img2]).to(device)

    # FID 설정 및 계산
    pfw.set_config(batch_size=2, device=device)
    fid_score = pfw.fid(batch, batch)
    print("FID score:", fid_score)
    return fid_score

def resize_keep_aspect_ratio(image, output_size):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:  # 너비가 높이보다 큰 경우
        new_height = output_size[1]
        new_width = int(new_height * aspect_ratio)
    else:  # 높이가 너비보다 큰 경우
        new_width = output_size[0]
        new_height = int(new_width / aspect_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 이미지를 중앙에 배치하고 주어진 크기로 패딩 추가
    padding_l = (output_size[0] - new_width) // 2
    padding_t = (output_size[1] - new_height) // 2
    padding_r = output_size[0] - new_width - padding_l
    padding_b = output_size[1] - new_height - padding_t

    image = ImageOps.expand(image, border=(padding_l, padding_t, padding_r, padding_b), fill=0)  # 검은색으로 패딩
    return image