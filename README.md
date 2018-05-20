Running the code


Edit configuration.json to first set parameters for running the code.

{
  "tinkerboard_ID": "Nkdoo-Mark-I",
  "skip_frames": 0,
  "display": 0,
  "write_video":1,
  "remote_upload": 0,
  "running_time": 8

}

tinkerboard_ID: AWS Identifier
skip_frames: Number of frames to skip, 0 means run every frame, 1 means skip one frame and so on
display: Set to 0 for no display and 1 for display
write_video: 0 to not write the output to a video file and 1 to write output to a video file
remote_upload: 0 to upload to AWS after writing csv and 1 to skip remote upload
running_time: Set value in number of hours to run the script




To run the code once:
For tracking:
python smile_detection_demo.py configuration.json

For non-tracking:
python no_tracking_demo.py configuration.json


To run the code daily with a cron scheduler

python nkdoo_service.py your_username "30 1 * * *"

The last argument depicts time at which the code needs to be run daily: Use the following link to set times: (https://crontab.guru/#30_1_*_*_*)[https://crontab.guru/#30_1_*_*_*]