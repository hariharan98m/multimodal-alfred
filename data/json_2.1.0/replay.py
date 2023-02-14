import json
import os
from glob import glob

import cv2
from ai2thor.controller import Controller
from tqdm import tqdm

controller = Controller()
controller.start()


def replay(data_dir: str):
    with open(os.path.join(data_dir, "traj_data.json"), "r") as fin:
        traj_data = json.load(fin)
    rendered_dir = os.path.join(data_dir, "rendered")
    if not os.path.isdir(rendered_dir):
        os.makedirs(rendered_dir)
    controller.reset(traj_data["scene"]["floor_plan"])
    controller.step(
        dict(action="Initialize", renderClassImage=True, renderObjectImage=True)
    )
    controller.step(
        dict(
            action="SetObjectPoses",
            objectPoses=traj_data["scene"]["object_poses"],
        )
    )
    event = controller.step(traj_data["scene"]["init_action"])
    objects = [[x["objectType"] for x in event.metadata["objects"] if x["visible"]]]
    for i, action in enumerate(tqdm(traj_data["plan"]["low_actions"])):
        cv2.imwrite(
            os.path.join(rendered_dir, "seg_{:09d}.jpg".format(i)),
            event.instance_segmentation_frame,
        )
        cv2.imwrite(
            os.path.join(rendered_dir, "cls_{:09d}.jpg".format(i)),
            event.class_segmentation_frame,
        )
        cv2.imwrite(
            os.path.join(rendered_dir, "org_{:09d}.jpg".format(i)), event.cv2img
        )
        try:
            event = controller.step(action["api_action"])
        except Exception as e:
            print("Exeption raised")
            raise e
        objects.append(
            [x["objectType"] for x in event.metadata["objects"] if x["visible"]]
        )
    with open(os.path.join(data_dir, "objects.json"), "w") as fout:
        json.dump(objects, fout)
    cv2.imwrite(
        os.path.join(rendered_dir, "seg_{:09d}.jpg".format(i)),
        event.instance_segmentation_frame,
    )
    cv2.imwrite(
        os.path.join(rendered_dir, "cls_{:09d}.jpg".format(i)),
        event.class_segmentation_frame,
    )
    cv2.imwrite(os.path.join(rendered_dir, "org_{:09d}.jpg".format(i)), event.cv2img)


if __name__ == "__main__":
    if os.path.isfile("completed.json"):
        with open("completed.json", "r") as fin:
            completed = set(json.load(fin))
    else:
        completed = set()
    for i, files in enumerate(tqdm(glob("valid_seen/*/*/traj_data.json"))):
        if i not in completed:
            replay(files[:-14])
            with open("completed.json", "w") as fout:
                json.dump(list(completed), fout)
            completed.add(i)
