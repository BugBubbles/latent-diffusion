"""copy from https://github.com/danier97/LDMVFI"""

import numpy as np
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
from PIL import Image
import ldm.data.vfitransforms as vt


class KaguyuDTM(Dataset):
    def __init__(
        self, db_dir, channels, crop_sz=(512, 512), augment_s=True, augment_t=True
    ):
        self.crop_sz = crop_sz
        self.channels = channels
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.db_dir = db_dir
        self.data = self._prepare()
        # print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return len(self.data)

    def _prepare(self):
        raise NotImplementedError

    def _reader(self, path):
        cat = Image.open(path)
        cat = cat.resize(self.crop_sz)
        if self.augment_s:
            cat = vt.rand_flip(cat, p=0.5)
        if self.augment_t:
            cat = vt.rand_reverse(cat, p=0.5)
        cat = np.array(cat, dtype=np.float32).squeeze()[:, :, : self.channels]
        cat = cat / 127.5 - 1.0
        return {"image": cat}

    def __getitem__(self, index):
        return self._reader(self.data[index])


class DTM_Train(KaguyuDTM):

    def _prepare(self):
        data = []
        with open(join(self.db_dir, "train.txt")) as f:
            for line in map(lambda x: x.strip(), f):
                data.append(join(self.db_dir, line))
        return data


class DTM_Validate(KaguyuDTM):

    def _prepare(self):
        data = []
        with open(join(self.db_dir, "val.txt")) as f:
            for line in map(lambda x: x.strip(), f):
                data.append(join(self.db_dir, line))
        return data


if __name__ == "__main__":
    dataset = DTM_Validate(
        db_dir="/disk527/sdb1/a804_cbf/datasets/lunar_crater/textures"
    )
    for i, data in enumerate(dataset):
        print(data.shape)
        if i > 10:
            break
