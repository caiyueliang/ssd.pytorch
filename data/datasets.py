# encoding:utf-8
import glob
import random
import os
import numpy as np

import cv2
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def is_image(file_path):
    if '.jpg' in file_path or '.jpeg' in file_path or '.png' in file_path or '.bmp' in file_path:
        return True
    else:
        return False


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=300, transform=None):
        self.img_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if is_image(file):
                    self.img_files.append(os.path.join(root, file))
        print("get image number", len(self.img_files))

        self.img_shape = img_size
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        if self.transform is not None:
            img, boxes, labels = self.transform(img, None, None)

            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            return transforms.ToTensor()(img), target, height, width, img_path

    def pull_item(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        if self.transform is not None:
            img, boxes, labels = self.transform(img, None, None)

            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            return transforms.ToTensor()(img), target, height, width, img_path

    def __len__(self):
        return len(self.img_files)


class SSDDataset(Dataset):
    def __init__(self, root_path, image_file, train=False, img_size=300, transform=None):
        self.transform = transform

        self.root_path = root_path
        self.image_file = image_file
        print('ListDataset', os.path.join(self.root_path, self.image_file))
        with open(os.path.join(self.root_path, self.image_file), 'r') as file:
            self.img_files = file.readlines()
        self.img_files = [os.path.join(self.root_path, files.replace('\n', '')) for files in self.img_files]
        # self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt') for path in self.img_files]
        self.label_files = [path.replace('.bmp', '.txt').replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt') for path in self.img_files]
        print('img_files len: %d' % len(self.img_files))
        print('label_files len: %d' % len(self.label_files))
        self.name = "things"

        self.img_shape = img_size
        self.train = train
        self.max_objects = 50                       # 每张图片最多支持多少个标签

    def __getitem__(self, index):
        # ==============================================================================================
        #  Label
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        new_labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # print('labels', labels)
            x_center = labels[:, 1:2]
            y_center = labels[:, 2:3]
            w = labels[:, 3:4]
            h = labels[:, 4:]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            # print('x1', x1)
            # print('y1', y1)
            # print('x2', x2)
            # print('y2', y2)

            c = labels[:, :1]
            # print('labels_2', labels_2)
            # new_labels = [labels_1, labels_2]
            new_labels = np.concatenate((x1, y1, x2, y2, c), axis=1)
            # new_labels = torch.from_numpy(new_labels)
            # print('new_labels', new_labels)

        # ==============================================================================================
        #  Image
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)

        # h, w, c = img.shape
        # show_img = img.copy()
        # for label in new_labels:
        #     cv2.rectangle(show_img, (int(label[0] * w), int(label[1] * h)),
        #                   (int(label[2] * w), int(label[3] * h)), (0, 255, 0))
        # cv2.imshow('old_image', show_img)

        if self.transform is not None:
            target = np.array(new_labels)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            # print('target_2', target)
            # h, w, c = img.shape
            # show_img = img.copy()
            # for label in target:
            #     cv2.rectangle(show_img, (int(label[0] * w), int(label[1] * h)),
            #                   (int(label[2] * w), int(label[3] * h)), (0, 255, 0))
            # cv2.imshow('new_image', show_img)
            # cv2.waitKey()

            # TODO
            # return torch.from_numpy(img).permute(2, 0, 1), target
            return transforms.ToTensor()(img), target

    def pull_item(self, index):
        # ==============================================================================================
        #  Label
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        new_labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            x_center = labels[:, 1:2]
            y_center = labels[:, 2:3]
            w = labels[:, 3:4]
            h = labels[:, 4:]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            c = labels[:, :1]
            new_labels = np.concatenate((x1, y1, x2, y2, c), axis=1)

        # ==============================================================================================
        #  Image
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # h, w, c = img.shape
        # show_img = img.copy()
        # for label in new_labels:
        #     cv2.rectangle(show_img, (int(label[0] * w), int(label[1] * h)),
        #                   (int(label[2] * w), int(label[3] * h)), (0, 255, 0))
        # cv2.imshow('old_image', show_img)

        if self.transform is not None:
            target = np.array(new_labels)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            # print('target_2', target)
            # h, w, c = img.shape
            # show_img = img.copy()
            # for label in target:
            #     cv2.rectangle(show_img, (int(label[0] * w), int(label[1] * h)),
            #                   (int(label[2] * w), int(label[3] * h)), (0, 255, 0))
            # cv2.imshow('new_image', show_img)
            # cv2.waitKey()

            # TODO
            # return torch.from_numpy(img).permute(2, 0, 1), target
            return transforms.ToTensor()(img), target, height, width, img_path

    def __len__(self):
        return len(self.img_files)

    # 随机高斯模糊
    def random_gaussian(self, img, max_n=5):
        k = random.randrange(1, max_n, 2)
        # print(k)
        if k != 1:
            img = cv2.GaussianBlur(img, ksize=(k, k), sigmaX=1.5)
        return img

    # 随机翻转
    def random_flip(self, img, labels):
        if random.random() <= 0.5:
            img_lr = np.fliplr(img).copy()
            labels[:, 1] = 1 - labels[:, 1]
            return img_lr, labels
        return img, labels

    # def random_crop(self, img, labels, prob=0.95, log=False):
    #     if random.random() > prob:
    #         return img, labels
    #
    #     h, w, c = img.shape
    #     if log is True:
    #         print('======================================================================')
    #         print('old w, h, c', w, h, c)
    #         print('old labels', labels, type(labels))
    #
    #     top_crop = random.randint(0, int(h * 0.4))
    #     bottom_crop = h - random.randint(0, int(h * 0.4))
    #     left_crop = random.randint(0, int(w * 0.4))
    #     right_crop = w - random.randint(0, int(w * 0.4))
    #     if log is True:
    #         print('top_crop, bottom_crop, left_crop, right_crop', top_crop, bottom_crop, left_crop, right_crop)
    #
    #     x1s = w * (labels[:, 1] - labels[:, 3] / 2)
    #     y1s = h * (labels[:, 2] - labels[:, 4] / 2)
    #     x2s = w * (labels[:, 1] + labels[:, 3] / 2)
    #     y2s = h * (labels[:, 2] + labels[:, 4] / 2)
    #     if log is True:
    #         print('x1s, y1s, x2s, y2s', x1s, y1s, x2s, y2s)
    #
    #     new_labels = []
    #     new_x1s = []
    #     new_y1s = []
    #     new_x2s = []
    #     new_y2s = []
    #     for i, x1 in enumerate(x1s):
    #         if right_crop - x1 <= ((x2s[i] - x1) / 10):
    #             if log is True:
    #                 print('right_crop - x1 <= (x2s[i] - x1) / 10')
    #                 print('%f - %f <= %f' % (right_crop, x1, (x2s[i] - x1) / 10))
    #             continue
    #         if bottom_crop - y1s[i] <= ((y2s[i] - y1s[i]) / 10):
    #             if log is True:
    #                 print('bottom_crop - y1s[i] <= (y2s[i] - y1s[i]) / 10')
    #                 print('%f - %f <= %f' % (bottom_crop, y1s[i], (y2s[i] - y1s[i])/10))
    #             continue
    #         if x2s[i] - left_crop <= ((x2s[i] - x1) / 10):
    #             if log is True:
    #                 print('x2s[i] - left_crop <= ((x2s[i] - x1) / 10)')
    #                 print('%f - %f <= %f' % (x2s[i], left_crop, (x2s[i] - x1) / 10))
    #             continue
    #         if y2s[i] - top_crop <= ((y2s[i] - y1s[i]) / 10):
    #             if log is True:
    #                 print('y2s[i] - top_crop <= (y2s[i] - y1s[i]) / 10')
    #                 print('%f - %f <= %f' % (y2s[i], top_crop, (y2s[i] - y1s[i]) / 10))
    #             continue
    #
    #         # if x1 >= right_crop:
    #         #     continue
    #         # if y1s[i] >= bottom_crop:
    #         #     continue
    #         # if x2s[i] <= left_crop:
    #         #     continue
    #         # if y2s[i] <= top_crop:
    #         #     continue
    #
    #         new_labels.append(labels[i])
    #         new_x1s.append(x1)
    #         new_y1s.append(y1s[i])
    #         new_x2s.append(x2s[i])
    #         new_y2s.append(y2s[i])
    #
    #     # 过滤后label为空，则放弃本次的裁剪
    #     if len(new_labels) == 0:
    #         return img, labels
    #
    #     new_labels = np.array(new_labels, dtype=np.float)
    #     new_x1s = np.array(new_x1s, dtype=np.float)
    #     new_y1s = np.array(new_y1s, dtype=np.float)
    #     new_x2s = np.array(new_x2s, dtype=np.float)
    #     new_y2s = np.array(new_y2s, dtype=np.float)
    #     if log is True:
    #         print('new labels', new_labels)
    #         print('new_x1s, new_y1s, new_x2s, new_y2s', new_x1s, new_y1s, new_x2s, new_y2s)
    #
    #     new_x1s[new_x1s < left_crop] = left_crop
    #     new_y1s[new_y1s < top_crop] = top_crop
    #     new_x2s[new_x2s > right_crop] = right_crop
    #     new_y2s[new_y2s > bottom_crop] = bottom_crop
    #     if log is True:
    #         print('new_x1s, new_y1s, new_x2s, new_y2s', new_x1s, new_y1s, new_x2s, new_y2s)
    #
    #     new_x1s = new_x1s - left_crop
    #     new_x2s = new_x2s - left_crop
    #     new_y1s = new_y1s - top_crop
    #     new_y2s = new_y2s - top_crop
    #     if log is True:
    #         print('new_x1s, new_y1s, new_x2s, new_y2s', new_x1s, new_y1s, new_x2s, new_y2s)
    #
    #     img = img[top_crop:bottom_crop, left_crop:right_crop]
    #     h, w, c = img.shape
    #     if log is True:
    #         print('new w, h, c', w, h, c)
    #
    #     new_labels[:, 1] = ((new_x1s + new_x2s) / 2) / float(w)   # 第1列表示：x轴中心点（比例） # 第0列表示：类别
    #     new_labels[:, 2] = ((new_y1s + new_y2s) / 2) / float(h)   # 第2列表示：y轴中心点（比例）
    #     new_labels[:, 3] = (new_x2s - new_x1s) / float(w)         # 第3列表示：w（比例）
    #     new_labels[:, 4] = (new_y2s - new_y1s) / float(h)         # 第4列表示：h（比例）
    #
    #     return img, new_labels

    # 随机裁剪
    def random_crop(self, img, labels, prob=0.9):
        h, w, c = img.shape
        # print('old w, h, c', w, h, c)
        # print('old labels', labels)

        x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        y2 = h * (labels[:, 2] + labels[:, 4] / 2)
        # print(x1, y1, x2, y2)

        min_left = min(min(x1), min(x2))
        min_top = min(min(y1), min(y2))
        min_lt = min(min_left, min_top)

        min_right = w - max(max(x1), max(x2))
        min_bottom = h - max(max(y1), max(y2))
        min_rb = min(min_right, min_bottom)
        # print('min_left, min_top, min_right, min_bottom', min_left, min_top, min_right, min_bottom)

        crop_left = 0
        crop_top = 0
        crop_right = w
        crop_bottom = h

        # random crop left and top
        if random.random() < prob:
            rate = random.random()
            crop = int(min_lt * rate)

            x1 = x1 - crop
            x2 = x2 - crop
            crop_left = crop
            # print('crop_left', crop_left, rate, x1, x2)

            y1 = y1 - crop
            y2 = y2 - crop
            crop_top = crop
            # print('crop_top', crop_top, rate, y1, y2)

        # random crop right
        if random.random() < prob:
            rate = random.random()
            crop = int(min_rb * rate)

            crop_right = crop_right - crop
            # print('crop_right', crop_right, rate)

            crop_bottom = crop_bottom - crop
            # print('crop_bottom', crop_bottom, rate)

        img = img[crop_top:crop_bottom, crop_left:crop_right]
        h, w, c = img.shape
        # print('new w, h, c', w, h, c)

        labels[:, 1] = ((x1 + x2) / 2) / float(w)                # 第1列表示：x轴中心点（比例） # 第0列表示：类别
        labels[:, 2] = ((y1 + y2) / 2) / float(h)                # 第2列表示：y轴中心点（比例）
        labels[:, 3] = (x2 - x1) / float(w)                      # 第3列表示：w（比例）
        labels[:, 4] = (y2 - y1) / float(h)                      # 第4列表示：h（比例）
        # print('new labels', labels)

        return img, labels

    # 随机调亮
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha >= 0.4 and alpha <= 0.9:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

    def padding(self, img, labels):
        h, w, _ = img.shape
        # print(img.shape)
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # print('pad', pad)
        input_img = np.pad(img, pad, 'constant', constant_values=0)
        padded_h, padded_w, _ = input_img.shape
        # print(input_img.shape)

        # print(labels)
        if pad[0][0] != 0:
            labels[:, 2] = (labels[:, 2] * h + pad[0][0]) / float(padded_h)
            labels[:, 4] = labels[:, 4] * h / float(padded_h)
        elif pad[1][0]:
            labels[:, 1] = (labels[:, 1] * w + pad[1][0]) / float(padded_w)
            labels[:, 3] = labels[:, 3] * w / float(padded_w)
        # print(labels)
        # print("")
        return input_img, labels


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(SSDDataset("../../Data/yolo/yolo_data_new/car_detect_train", "image_path.txt", train=True),
                                shuffle=False, batch_size=4, num_workers=0)
    for batch_i, (_, imgs, targets) in enumerate(train_loader):
        print(batch_i)
