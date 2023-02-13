import json
from glob import glob

import numpy as np
from fire import Fire
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def get_json_files(split: str = "valid_seen"):
    if "test" in split:
        return glob("{}/*/traj_data.json".format(split))
    else:
        return glob("{}/*/*/traj_data.json".format(split))


def draw_sentence_length(split: str = "valid_seen"):
    json_files = get_json_files(split)
    desc_length = []
    desc_diversity = []
    task_length = []
    task_diversity = []
    num_description = []
    for json_file in tqdm(json_files):
        with open(json_file, "r") as fin:
            annotations = json.load(fin)["turk_annotations"]["anns"]
        for annotation in annotations:
            for desc in annotation["high_descs"]:
                tokenized = word_tokenize(desc)
                desc_length.append(len(tokenized))
                desc_diversity.append(len(set(tokenized)) / len(tokenized))
            num_description.append(len(annotation["high_descs"]))
            tokenized = word_tokenize(annotation["task_desc"])
            task_length.append(len(tokenized))
            task_diversity.append(len(set(tokenized)) / len(tokenized))
    desc_length = np.array(desc_length)
    desc_diversity = np.array(desc_diversity)
    task_length = np.array(task_length)
    task_diversity = np.array(task_diversity)
    num_description = np.array(num_description)

    plt.hist(desc_length, bins=40)
    plt.xlabel("description length")
    plt.title("mean: {:.2f}, std: {:.2f}".format(desc_length.mean(), desc_length.std()))
    plt.savefig("{}_description_length.png".format(split))
    plt.close()

    plt.hist(desc_diversity, bins=40)
    plt.xlabel("description diversity")
    plt.title(
        "mean: {:.2f}, std: {:.2f}".format(desc_diversity.mean(), desc_diversity.std())
    )
    plt.savefig("{}_description_diversity.png".format(split))
    plt.close()

    plt.hist(task_length, bins=40)
    plt.xlabel("task length")
    plt.title("mean: {:.2f}, std: {:.2f}".format(task_length.mean(), task_length.std()))
    plt.savefig("{}_task_length.png".format(split))
    plt.close()

    plt.hist(task_diversity, bins=40)
    plt.xlabel("task diversity")
    plt.title(
        "mean: {:.2f}, std: {:.2f}".format(task_diversity.mean(), task_diversity.std())
    )
    plt.savefig("{}_task_diversity.png".format(split))
    plt.close()

    plt.hist(num_description, bins=20)
    plt.xlabel("number of description per task")
    plt.title(
        "mean: {:.2f}, std: {:.2f}".format(
            num_description.mean(), num_description.std()
        )
    )
    plt.savefig("{}_per_task.png".format(split))
    plt.close()


if __name__ == "__main__":
    Fire()
