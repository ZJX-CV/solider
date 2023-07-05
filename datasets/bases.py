from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError

"""
market、dukemtmc等等数据集都继承自次数
输出统计信息的功能抽离出来，除此之外没用
"""
class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("{} Dataset statistics:".format(self.dataset_dir))
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

"""
实际后面构造dataloader的时候使用的数据集
此处应该支持多个数据集的sum
"""
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, num_train_pids=0, num_train_cams=0, num_train_vids=0):
        self.num_train_pids = num_train_pids
        self.num_train_cams = num_train_cams
        self.view_num = num_train_vids
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path
        #  return img, pid, camid, trackid,img_path.split('/')[-1]
    
    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.dataset)

        # 此处默认所有id都是从0开始，并且连续排列的！
        for img_path, pid, camid, _ in other.dataset:
            pid += self.num_train_pids
            camid += self.num_train_cams
            # dsetid += self.num_datasets
            train.append((img_path, pid, camid, 1))

        ###################################
        # Note that
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset; setting it to True will
        #    create new IDs that should have already been included
        ###################################
        return ImageDataset(
            train,
            self.transform,
            self.num_train_pids+other.num_train_pids,
            self.num_train_cams+other.num_train_cams
            # mode=self.mode,
            # combineall=False,
            # verbose=False
        )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)
