import detect
import cv2 as cv

AREA = 450
# THRESHOLD = 130
START_IMG = 1
END_IMG = 1000
BLUR_KSIZE = 7
DILATE_KSIZE = 5

if __name__ == '__main__':
    img_name = START_IMG
    while True:
        img_loc = f'./SurfaceCrack/{img_name:05}.jpg'
        img_detector = detect.Detector(img_loc, BLUR_KSIZE, DILATE_KSIZE)
        cv.imshow('detected', img_detector.detect(AREA))
        cv.imshow('threshed', img_detector.threshed)
        cv.imshow('dilated', img_detector.dilated)

        print(f'\r{img_name:05}.jpg Threshold: {img_detector.threshold}', end='')

        key = cv.waitKey(0) & 0xFF
        if key == ord('a'):             # 按a切换到下一张图
            img_name = img_name - 1 if img_name > START_IMG else START_IMG
        elif key & 0xFF == ord('d'):    # 按d切换到上一张图
            img_name = img_name + 1 if img_name < END_IMG else END_IMG
        elif key & 0xFF == ord('q'):    # 按q退出程序
            break
        # 鼠标聚焦到cv.imshow()窗口，输入法为英文小写
    cv.destroyAllWindows()
