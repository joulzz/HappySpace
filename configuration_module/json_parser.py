import json


def json_parser(json_path):

    """ Description

    Functions loads all variables from the configuration.json and returns them to be unpacked in the main script

    :return: Returning variables to the main script set in the configuration.json
    """
    with open(json_path, "r") as fp:
        json_config = json.load(fp)

    if "tinkerboard_ID" in json_config:
        tinkerboard_id = json_config["tinkerboard_ID"]
    else:
        tinkerboard_id = "Default"

    if "skip_frames" in json_config:
        skip_frame = json_config["skip_frames"]
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

    if "dongle_connection" in json_config:
        dongle_connection = json_config["dongle_connection"]
    else:
        dongle_connection = False

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

    if "write_images" in json_config:
        write_images = json_config["write_images"]
    else:
        write_images = False

    if "blur_images" in json_config:
        blur_images = json_config["blur_images"]
    else:
        blur_images = False

    if "calibration_smile" in json_config:
        calibration_smile = json_config["calibration_smile"]
    else:
        calibration_smile = False

    if "calibration_nonsmile" in json_config:
        calibration_nonsmile = json_config["calibration_nonsmile"]
    else:
        calibration_nonsmile = False


    if "face_vector_display" in json_config:
        face_vector_display = json_config["face_vector_display"]
    else:
        face_vector_display = False


    return tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, dongle_connection, running_time, min_face, max_face, write_images, blur_images, calibration_smile, calibration_nonsmile, face_vector_display