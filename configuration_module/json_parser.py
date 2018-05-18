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
    if "csv_write_frequency" in json_config:
        csv_write_frequency = json_config["csv_write_frequency"]
    else:
        csv_write_frequency = 3600

    return tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, csv_write_frequency