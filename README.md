Running the code


Edit configuration.json to first set parameters for running the code.

```
{
  "tinkerboard_ID": "Nkdoo-Mark-II",
  "skip_frames": 3,
  "display": 1,
  "write_video":0,
  "remote_upload": 0,
  "min_face": [30, 30],
  "max_face": [400, 400],
  "running_time": 8

}
```

*tinkerboard_ID*: AWS Identifier
*skip_frames*: Number of frames to skip, 0 means run every frame, 1 means skip one frame and so on [ Takes only integer values]
*display*: Set to 0 for no display and 1 for display
*write_video*: 0 to not write the output to a video file and 1 to write output to a video file
*remote_upload*: 0 to upload to AWS after writing csv and 1 to skip remote upload
"min_face": pixel values for width and height in the following format [width, height]
"max_face": pixel values for width and height in the following format [width, height]
*running_time*: Set value in number of hours to run the script (Min. 1 hr and max 24 hrs) due to integration with the daily service code



To run the code once:
For tracking:
```
python smile_detection_demo.py configuration.json
```

For non-tracking:
```
python no_tracking_demo.py configuration.json
```

To run the code daily with a cron scheduler
```
python nkdoo_service.py your_username "30 1 * * *"
```

The last argument depicts time at which the code needs to be run daily: Use the following link to set times: [https://crontab.guru/#30_1_*_*_*](https://crontab.guru/#30_1_*_*_*)