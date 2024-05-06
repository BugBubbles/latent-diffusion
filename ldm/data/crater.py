"""copy from https://github.com/danier97/LDMVFI"""

import numpy as np
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
from PIL import Image
import ldm.data.vfitransforms as vt


class KaguyuDTM(Dataset):
    def __init__(
        self,
        db_dir,
        channels,
        shape=(960, 960),
        crop_sz=(480, 480),
        augment_s=True,
        augment_t=True,
    ):
        self.h_crop = shape[0] // crop_sz[0]
        self.w_crop = shape[1] // crop_sz[1]
        self.crop_sz = crop_sz
        self.channels = channels
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.db_dir = db_dir
        self.data = self._prepare()

    def __len__(self):
        return self.h_crop * self.w_crop * len(self.data)

    def _prepare(self):
        raise NotImplementedError

    def _reader(self, path, h_idx, w_idx):
        cat = Image.open(path)
        cat_crop = cat.crop(
            (
                h_idx * self.crop_sz[0],
                w_idx * self.crop_sz[1],
                (h_idx + 1) * self.crop_sz[0],
                (w_idx + 1) * self.crop_sz[1],
            )
        )
        cat_crop = cat_crop.resize(self.crop_sz)
        if self.augment_s:
            cat_crop = vt.rand_flip(cat_crop, p=0.5)
        if self.augment_t:
            cat_crop = vt.rand_reverse(cat_crop, p=0.5)
        cat_crop = np.array(cat_crop, dtype=np.float32).squeeze()[:, :, : self.channels]
        cat_crop = cat_crop / 127.5 - 1.0
        return {"image": cat_crop, "h_idx": h_idx, "w_idx": w_idx, "path": path}

    def __getitem__(self, index):
        i = index % (self.h_crop * self.w_crop)
        h = i % self.h_crop
        w = i // self.h_crop
        path_idx = index // (self.h_crop * self.w_crop)
        return self._reader(self.data[path_idx], h, w)


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


class DTM_Test(KaguyuDTM):

    def _prepare(self):
        data = []
        with open(join(self.db_dir, "val.txt")) as f:
            for line in map(lambda x: x.strip(), f):
                data.append(join(self.db_dir, line))
        with open(join(self.db_dir, "train.txt")) as f:
            for line in map(lambda x: x.strip(), f):
                data.append(join(self.db_dir, line))
        return data


if __name__ == "__main__":
    dataset = DTM_Validate(
        db_dir="/disk527/sdb1/a804_cbf/datasets/lunar_crater/textures",
        channels=3,
        augment_s=False,
        augment_t=False,
    )
    for i, data in enumerate(dataset):
        print(data["image"].shape)
        print(data["h_idx"], data["w_idx"])
        print(data["path"])
        import matplotlib.pyplot as plt

        plt.imsave(f"test_{i}.png", (data["image"] + 1) / 2)
        if i > 10:
            break
