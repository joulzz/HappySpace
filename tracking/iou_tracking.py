import cv2
import numpy as np

class Tracker:
    def __init__(self):
        print "IOU Tracker created"

    def iou_tracker(self, source_bbox, dest_bbox):
        x1 = max(source_bbox[0][0], dest_bbox[0][0])
        y1 = max(source_bbox[0][1], dest_bbox[0][1])
        x2 = min(source_bbox[1][0], dest_bbox[1][0])
        y2 = min(source_bbox[1][1], dest_bbox[1][1])
        source_width = source_bbox[1][0] - source_bbox[0][0]
        source_height = source_bbox[1][1] - source_bbox[0][1]
        source_area = source_width * source_height
        iou_area = (x2 - x1) * (y2 - y1)
        overlap_percentage = float(iou_area)/ source_area
        return overlap_percentage