import json


def json_parser(json_path):
    with open(json_path, "r") as fp:
        json_config = json.load(fp)

    if "tinkerboard_ID" in json_config:
        tinkerboard_id = json_config["tinkerboard_ID"]
    else:
        tinkerboard_id = "Default"

    if "skip_frame" in json_config:
        skip_frame = json_config["skip_frame"]
    else:
        skip_frame = 0

    if "display" in json_config:
        display_flag = json_config["display"]
    else:
        display_flag = False

    if "write_video" in json_config:
        write_video = json_config["write_video"]
    else:
        write_video = False

    if "remote_upload" in json_config:
        remote_upload = json_config["remote_upload"]
    else:
        remote_upload = False
    if "running_time" in json_config:
        running_time = json_config["running_time"]
    else:
        running_time = 8
    if "min_face" in json_config:
        min_face = json_config["min_face"]
    else:
        min_face = [30, 30]
    if "max_face" in json_config:
        max_face = json_config["max_face"]
    else:
        max_face = []


    return tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, running_time, min_face, max_face