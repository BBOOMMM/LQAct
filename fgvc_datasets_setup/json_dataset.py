#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter
import json

_DATA_SUBDIR = {
    "CUB": "CUB_200_2011",
    'OxfordFlowers': "OxfordFlower",
    'StanfordCars': "Stanford-cars",
    'StanfordDogs': "Stanford-dogs",
    "nabirds": "nabirds",
}

def read_json(filename):
    """read json files"""
    with open(filename, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, processor):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, args.dataset_name)
        print("Constructing {} dataset {}...".format(
            args.dataset_name, split))

        self.args = args
        self._split = split
        self.name = args.dataset_name
        self.data_dir = os.path.join(args.data_dir, _DATA_SUBDIR[args.dataset_name])
        # self.data_percentage = cfg.DATA.PERCENTAGE
        self._construct_imdb(args)
        self.transform = processor

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)
        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        
        encoding = self.transform(images=im, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, args, split, processor):
        super(CUB200Dataset, self).__init__(args, split, processor)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, args, split, processor):
        super(CarsDataset, self).__init__(args, split, processor)

    def get_imagedir(self):
        return self.data_dir


class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, args, split, processor):
        super(DogsDataset, self).__init__(args, split, processor)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")


class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, args, split, processor):
        super(FlowersDataset, self).__init__(args, split, processor)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, args, split, processor):
        super(NabirdsDataset, self).__init__(args, split, processor)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

