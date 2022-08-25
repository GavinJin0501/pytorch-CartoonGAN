import cv2 as cv
import os
import numpy as np
from tqdm import tqdm


def rename_according_to_order(path):
    photos = os.listdir(path)
    n = len(photos)
    count = 0

    for i in range(n):
        photo = photos[i]
        if photo.split(".")[-1] == "jpg":
            old_name = os.path.join(path, photo)
            new_name = os.path.join(path, "oldold_%d.jpg" % count)
            os.rename(old_name, new_name)
            count += 1
        else:
            print("this file is not valid:", photo)

    photos = os.listdir(path)
    n = len(photos)
    for i in range(n):
        photo = photos[i]
        if photo.split(".")[-1] == "jpg":
            old_name = os.path.join(path, photo)
            new_name = os.path.join(path, "%s.jpg" % photo.split(".")[0].split("_")[-1])
            os.rename(old_name, new_name)
            count += 1
        else:
            print("this file is not valid:", photo)

    print("Total valid photos:", count)


def sharpen_photo(root, save):
    # Create the save dir if it does not exist
    if not os.path.isdir(save):
        os.mkdir(save)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    photo_list = os.listdir(root)
    for photo in tqdm(photo_list):
        img0 = cv.resize(cv.imread(os.path.join(root, photo)), (256, 256))
        img1 = reduce_color(img0)
        # img1 = img0
        img2 = cv.filter2D(img1, -1, kernel)
        res = np.concatenate((img0, img2), 1)
        cv.imwrite(os.path.join(save, photo), res)

        cv.imshow("image", res)
        cv.waitKey()
        cv.destroyWindow("image")


def animate_photo(root, save):
    # Create the save dir if it does not exist
    if not os.path.isdir(save):
        os.mkdir(save)

    line_size = 3
    blur_value = 5
    photo_list = os.listdir(root)
    for photo in tqdm(photo_list):
        img = cv.imread(os.path.join(root, photo))
        edges = edge_mask(img, line_size, blur_value)
        img1 = reduce_color(img)
        # img2 = cv.bilateralFilter(img1, d=1, sigmaColor=250, sigmaSpace=250)
        img2 = img1
        img3 = cv.bitwise_and(img2, img2, mask=edges)
        res = np.concatenate((img, img3), 1)
        cv.imwrite(os.path.join(save, photo), res)


def edge_mask(img, line_size, blur_value):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blur = cv.medianBlur(gray, blur_value)
    edges = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, line_size, blur_value)
    return edges


def reduce_color(img):
    k = 8
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


if __name__ == "__main__":
    REAL_ANIME = "C:\\Users\\WIN10\\Downloads\\real_anime"
    save = "C:\\Users\\WIN10\\Downloads\\paired_dataset\\anime_after_smoothing"
    save1 = "C:\\Users\\WIN10\\Downloads\\paired_dataset\\real_being_animated"
    root = "C:\\Users\\WIN10\\Downloads\\paired_dataset\\real"
    # rename_according_to_order(REAL_ANIME)

    # sharpen_photo(root, save)

    animate_photo(root, save1)
