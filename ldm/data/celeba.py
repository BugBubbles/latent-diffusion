import numpy as np
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
from PIL import Image
import ldm.data.vfitransforms as vt


class Celeba(Dataset):
    def __init__(
        self,
        db_dir,
        size=256,
    ):
        self.crop_sz = (size, size)
        self.db_dir = db_dir
        self.data = self._prepare()

    def __len__(self):
        return len(self.data)

    def _prepare(self):
        raise NotImplementedError

    def _reader(self, path):
        cat = Image.open(path)
        cat_crop = cat.resize(self.crop_sz)
        cat_crop = np.array(cat_crop, dtype=np.float32).squeeze()
        cat_crop = cat_crop / 127.5 - 1.0
        return {"image": cat_crop, "path": path}

    def __getitem__(self, index):
        return self._reader(self.data[index])


class Celeba_Train(Celeba):
    def _prepare(self):
        data = []
        with open(join(self.db_dir, "list_eval_partition.txt")) as f:
            for line in map(lambda x: x.strip().split(".jpg"), f):
                if int(line[1]) < 2:
                    data.append(join(self.db_dir, "img_align_celeba", line[0] + ".jpg"))
        return data


class Celeba_Validate(Celeba):

    def _prepare(self):
        data = []
        with open(join(self.db_dir, "list_eval_partition.txt")) as f:
            for line in map(lambda x: x.strip().split(".jpg"), f):
                if int(line[1]) == 2:
                    data.append(join(self.db_dir, "img_align_celeba", line[0] + ".jpg"))
        return data


class Celeba_Test(Celeba):

    def _prepare(self):
        data = []
        with open(join(self.db_dir, "list_eval_partition.txt")) as f:
            for line in map(lambda x: x.strip().split(".jpg"), f):
                data.append(join(self.db_dir, "img_align_celeba", line[0] + ".jpg"))
        return data


if __name__ == "__main__":
    dataset = Celeba_Validate(
        db_dir="/disk527/Datadisk/xdy_cbf/datasets/celeba",
    )
    for i, data in enumerate(dataset):
        print(data["image"].shape)
        print(data["path"])
