"""Including functions：
1. make `merge-subjects` dataset

"""

import os
import shutil
from pathlib import Path


def copy_folder(src, dst, overwrite=False) -> None:
    """
    Copies the contents of the src folder to the dst folder.
    If overwrite is True, it will overwrite files in dst with files from src.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            copy_folder(src_path, dst_path, overwrite)
        else:
            if overwrite or not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(
                    f"Skipped copying {src_path} to {dst_path} (already exists in destination)."
                )


def merge_tibkat_subjects() -> None:
    all_folder = "./data/shared-task-datasets/TIBKAT/all-subjects/data"
    core_folder = "./data/shared-task-datasets/TIBKAT/tib-core-subjects/data"
    merged_folder = "./data/shared-task-datasets/TIBKAT/merged-subjects/data"

    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # First, copy everything from folder_a
    copy_folder(all_folder, merged_folder, overwrite=True)
    copy_folder(core_folder, merged_folder, overwrite=True)

    all_train_files = list(Path(all_folder, "train").glob("**/*.jsonld"))
    print("all train files: ", len(all_train_files))
    all_dev_files = list(Path(all_folder, "dev").glob("**/*.jsonld"))
    print("all dev files: ", len(all_dev_files))

    merged_train_files = list(Path(merged_folder, "train").glob("**/*.jsonld"))
    print("merge train files: ", len(merged_train_files))
    merged_dev_files = list(Path(merged_folder, "dev").glob("**/*.jsonld"))
    print("merged dev files: ", len(merged_dev_files))

    # 上面处理完毕的结果中，存在en和de两个文件夹里面同名，而文件名称和内容
    # 完全一样，此时，删除en文件夹中的同名文件
    merged_de_files = list(Path(merged_folder).glob("**/de/*.jsonld"))

    n_cleaned = 0
    for f in merged_de_files:
        en_dir = Path(f.parent.parent, "en")
        en_file = Path(en_dir, f.name)
        if en_file.exists():
            print(f)
            n_cleaned += 1
    print("cleaned: ", n_cleaned)
    
    merged_train_files = list(Path(merged_folder, "train").glob("**/*.jsonld"))
    print("final merge train files: ", len(merged_train_files))
    merged_dev_files = list(Path(merged_folder, "dev").glob("**/*.jsonld"))
    print("final merged dev files: ", len(merged_dev_files))

def remove_duplicate():
    merged_folder = "./data/shared-task-datasets/TIBKAT/merged-subjects/data"

    merged_de_files = list(Path(merged_folder).glob("**/de/*.jsonld"))

    n_cleaned = 0
    for f in merged_de_files:
        en_dir = Path(f.parent.parent, "en")
        en_file = Path(en_dir, f.name)
        if en_file.exists():
            en_file.unlink()
            n_cleaned += 1
    print("cleaned: ", n_cleaned)


if __name__ == "__main__":
    merge_tibkat_subjects()
