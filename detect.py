import cv2 as cv
import numpy as np


class Detector:
    def __init__(self, img_loc: str, blur_ksize: int, dilate_ksize: int, thresh_decr: int = 20, thresh_max: int = 0):
        """
        传入待检测图像进行初始化
        :param img_loc: 传入图像地址
        :param blur_ksize: 高斯模糊核大小，应为奇数
        :param dilate_ksize:膨胀核大小
        :param thresh_decr:二值化阈值求值时的减少量
        :param thresh_max:二值化阈值的最大值限制
        """
        if blur_ksize <= 1 or blur_ksize >= 16 or blur_ksize // 2 == 0:
            raise Exception("kernel size between 1 and 16!")
        if dilate_ksize <= 1 or dilate_ksize >= 16:
            raise Exception("kernel size between 1 and 16!")
        if thresh_decr < 0 or thresh_decr > 255:
            raise Exception("thresh_decr between 0 and 255!")
        if thresh_max < 0 or thresh_max > 255:
            raise Exception("thresh_max between 0 and 255!")

        self.res = []
        self.img = cv.imread(img_loc)
        if self.img is None:
            raise Exception("open image failed!")
        self.img_blur = cv.GaussianBlur(self.img, (blur_ksize, blur_ksize), 0)
        self.img_gray = cv.cvtColor(self.img_blur, cv.COLOR_BGR2GRAY)
        self.threshold = int(np.average(self.img_gray)) - thresh_decr
        if self.threshold < 0:
            self.threshold = 0
        if thresh_max != 0:
            self.threshold = self.threshold if self.threshold < thresh_max else thresh_max
        self.threshed = cv.threshold(self.img_gray, self.threshold, 255, cv.THRESH_BINARY_INV)[1]
        self.dilated = cv.dilate(self.threshed, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1)

    def detect(self, area: int):
        """
        检测图像中的缺陷并返回检测结果图像，检测结果序列存储进self.res
        :param area: 检测面积阈值
        :return: 检测结果图像
        """
        if area < 0 or area > self.img.shape[0] * self.img.shape[1]:
            raise Exception("area too large or small!")
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(self.dilated, connectivity=8)
        detected_img = self.img.copy()
        for i in range(1, num_labels):
            x, y, w, h, s = stats[i]
            if s < area:
                continue
            box = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)]
            self.res.append(box)
            cv.drawContours(detected_img, box, 0, (0, 0, 255), 2)
        return detected_img


