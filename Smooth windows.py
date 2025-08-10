import cv2
import os
import shutil

def sliding_window(image, stepSize, windowSize):
    # 滑动窗口遍历图像
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # 产生当前窗口
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':
    # 读取图像
    dir_path = 'dataset/text feature/bind_Train2.0_reverse'
    output_dir_path = 'dataset/text feature/bind_Train2.0_reverse_full_and_slice'
    for person in os.listdir(dir_path):
        person_path = os.path.join(dir_path, person)
        output_path=os.path.join(output_dir_path, person)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image_path in os.listdir(person_path):
            output_path_1=os.path.join(output_path, image_path)
            img_path=os.path.join(person_path,image_path)
            os.makedirs(output_path_1)
            image = cv2.imread(img_path)
            shutil.copy(img_path, output_path_1)
            w = image.shape[1]
            h = image.shape[0]
            (winW, winH) = (255,255)
            stepSize = (255, 255)
            cnt = 0
            for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
                slice_img = image[y:y + winH, x:x + winW]
                slice_filename = os.path.join(output_path_1,image_path+'-slice'+str(cnt)+'.png')
                cv2.imwrite(slice_filename, slice_img)
                print(f'Saved: {slice_filename}')
                cnt += 1
            print(f'Total {cnt} slices saved.')

