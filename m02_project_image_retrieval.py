import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Tạo file_path
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))

folder = 'data/train/'
list_dir = [folder + name for name in os.listdir(folder)]

# tạo một hàm tiền xử lý các ảnh
# đọc ảnh - chuyển đổi ảnh sang RGB - chỉnh kích thước của ảnh


def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def folder2image(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    # tạo một ma trận zero chứa một list các ảnh
    image_np = np.zeros(shape=(len(list_dir), *size, 3))
    image_path = []
    for i, path in enumerate(list_dir):
        image_np[i] = read_image_from_path(path, size)
        image_path.append(path)
    image_path = np.array(image_path)
    return image_np, image_path


def plot_results(querquery_pathy, ls_path_score, reverse):
    fig = plt.figure(figsize=(15, 9))
    # ảnh đầu tiên ở vị trí 1 sẽ là ảnh dùng để truy vấn
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(querquery_pathy, size=(448, 448)))
    plt.title(f"Query Image: {querquery_pathy.split('/')[2]}", fontsize=16)
    plt.axis("off")

    # ảnh ở vị trí thứ 2 đến vị trí thứ 6 sẽ là ảnh được truy vấn
    for i, path in enumerate(sorted(ls_path_score, key=lambda x: x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448, 448)))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis("off")
    plt.show()

# ----------------------------------------------------------------------


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    # cái này là tuple (1,2,3)
    # axis = axis_batch_size mục đích là cộng tất cả các chiều của một bức ảnh
    return np.sum(np.abs(query - data), axis=axis_batch_size)


def get_l1_score(root_img_path, query_path, size):
    # ảnh dùng để truy vấn
    query = read_image_from_path(query_path, size)
    # tạo một list path_l1_score sẵn
    path_l1_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            image_np, image_path = folder2image(path, size)
            rates = absolute_difference(query, image_np)
            path_l1_score.extend(list(zip(image_path, rates)))
    return query, path_l1_score

# ----------------------------------------------------------------------


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    # cái này là tuple (1,2,3)
    # axis = axis_batch_size mục đích là cộng tất cả các chiều của một bức ảnh
    return np.mean((query - data)**2, axis=axis_batch_size)


def get_l2_score(root_img_path, query_path, size):
    # ảnh dùng để truy vấn
    query = read_image_from_path(query_path, size)
    # tạo một list path_l1_score sẵn
    path_l2_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            image_np, image_path = folder2image(path, size)
            rates = mean_square_difference(query, image_np)
            path_l2_score.extend(list(zip(image_path, rates)))
    return query, path_l2_score


# ----------------------------------------------------------------------


def cosine_similarty(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    # cái này là tuple (1,2,3)
    # axis = axis_batch_size mục đích là cộng tất cả các chiều của một bức ảnh
    # np.finfo(float).eps tránh trường hợp chia cho không
    return np.sum((query * data), axis=axis_batch_size)/(np.multiply(query_norm, data_norm) + np.finfo(float).eps)


def get_cosine_similarity_score(root_img_path, query_path, size):
    # ảnh dùng để truy vấn
    query = read_image_from_path(query_path, size)
    # tạo một list path_l1_score sẵn
    path_cosine_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            image_np, image_path = folder2image(path, size)
            rates = cosine_similarty(query, image_np)
            path_cosine_score.extend(list(zip(image_path, rates)))
    return query, path_cosine_score


# ----------------------------------------------------------------------

def correlaton_coef_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_cor = query - np.mean(query)
    data_cor = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum((query - np.mean(query))**2))
    data_norm = np.sqrt(
        np.sum((data - np.mean(data))**2, axis=axis_batch_size))
    # cái này là tuple (1,2,3)
    # axis = axis_batch_size mục đích là cộng tất cả các chiều của một bức ảnh
    # np.finfo(float).eps tránh trường hợp chia cho không
    return np.sum((query_cor * data_cor), axis=axis_batch_size)/(np.multiply(query_norm, data_norm) + np.finfo(float).eps)


def get_correlaton_coef_score(root_img_path, query_path, size):
    # ảnh dùng để truy vấn
    query = read_image_from_path(query_path, size)
    # tạo một list path_l1_score sẵn
    path_cor_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            image_np, image_path = folder2image(path, size)
            rates = correlaton_coef_difference(query, image_np)
            path_cor_score.extend(list(zip(image_path, rates)))
    return query, path_cor_score


if __name__ == '__main__':
    root_img_path = f"{ROOT}/train/"
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    size = (448, 448)
    query, ls_path_score = get_l1_score(
        root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    query, ls_path_score = get_l2_score(
        root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    query, ls_path_score = get_cosine_similarity_score(
        root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=True)

    query, ls_path_score = get_correlaton_coef_score(
        root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=True)
