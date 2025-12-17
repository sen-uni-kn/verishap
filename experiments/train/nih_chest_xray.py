# Copyright 2025 David Boetius
import hashlib
import urllib.request
from pathlib import Path

# Install dependencies as needed:
# pip install kagglehub[hf-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
hf_dataset = kagglehub.load_dataset(
  KaggleDatasetAdapter.HUGGING_FACE,
  "nih-chest-xrays/data",
  file_path,
  # Provide any additional arguments like 
  # sql_query, hf_kwargs, or pandas_kwargs. See 
  # the documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterhugging_face
)

print("Hugging Face Dataset:", hf_dataset)


class NIHChestXrayDataset:
    links = (
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
        "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
        "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
        "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
        "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
        "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
        "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
        "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
        "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
        "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
        "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
    )
    md5_checksums = {
        "images_001.tar.gz": "fe8ed0a6961412fddcbb3603c11b3698",
        "images_002.tar.gz": "ab07a2d7cbe6f65ddd97b4ed7bde10bf",
        "images_003.tar.gz": "2301d03bde4c246388bad3876965d574",
        "images_004.tar.gz": "9f1b7f5aae01b13f4bc8e2c44a4b8ef6",
        "images_005.tar.gz": "1861f3cd0ef7734df8104f2b0309023b",
        "images_006.tar.gz": "456b53a8b351afd92a35bc41444c58c8",
        "images_007.tar.gz": "1075121ea20a137b87f290d6a4a5965e",
        "images_008.tar.gz": "b61f34cec3aa69f295fbb593cbd9d443",
        "images_009.tar.gz": "442a3caa61ae9b64e61c561294d1e183",
        "images_010.tar.gz": "09ec81c4c31e32858ad8cf965c494b74",
        "images_011.tar.gz": "499aefc67207a5a97692424cf5dbeed5",
        "images_012.tar.gz": "dc9fda1757c2de0032b63347a7d2895c",
    }

    def __init__(self, root, train=True, download=False):
        self.dataset_dir = Path(root) / "nih_chest_xray"
        if download:
            self.download(self.dataset_dir)

    @classmethod
    def download(cls, dataset_dir: Path):
        print("Downloading NIH Chest X-ray dataset...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for idx, link in enumerate(cls.links):
            fn = dataset_dir / f"images_{idx + 1:03d}.tar.gz"
            print("downloading" + fn + "...")
            urllib.request.urlretrieve(link, fn)
            if cls.md5_checksums[fn] != hashlib.md5(open(fn, "rb").read()).hexdigest():
                raise ValueError(f"Checksum mismatch for {fn}")

        print("Download complete.")