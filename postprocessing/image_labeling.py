import argparse
import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np

class LabelImages(object):
  def __init__(self, args):
    self.calib_data_dir = args.calib_data_dir
    # Drawing states
    self.image = None
    self.moving_xy = (0, 0)
    self.buttondown_xy = (0, 0)

  def get_click(self, event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
      self.moving_xy = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
      self.buttondown_xy = (x, y)

  def label(self):
    for experiment_reldir in os.listdir(self.calib_data_dir):
      experiment_dir = os.path.join(self.calib_data_dir, experiment_reldir)
      frame_path = os.path.join(experiment_dir, "image.png")
      if not os.path.isfile(frame_path):
        continue
      image = cv2.imread(frame_path)
      cv2.namedWindow('mouse')
      cv2.setMouseCallback('mouse', self.get_click)
      self.moving_xy = (0, 0)
      self.buttondown_xy = (0, 0)
      print("Accept this result? (y/n)")
      while True:
        h, w, _ = image.shape
        draw_image = image.copy()
        cv2.circle(draw_image, (self.moving_xy[0],self.moving_xy[1]), 4, (0,0,255),-1)
        cv2.circle(draw_image, (self.buttondown_xy[0],self.buttondown_xy[1]), 4, (0,255,0),-1)
        cv2.imshow('mouse', draw_image)
        key = cv2.waitKey(1)
        if key == ord('y') or key == ord('n'):
          if key == ord('n'):
            print(experiment_reldir + " not accepted")
            self.buttondown_xy = (0, 0)
          else:
            print("Experiment saved: %s"%experiment_reldir)
            x = self.buttondown_xy[0]
            y = self.buttondown_xy[1]
            print(x, y)
            np.save(os.path.join(experiment_dir, "point_in_camera.npy"), (x, y))
            cv2.circle(image, (self.buttondown_xy[0],self.buttondown_xy[1]), 4, (0,255,0),-1)
            cv2.imwrite(os.path.join(experiment_dir, "label.png"), image)
          break
      cv2.destroyAllWindows()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Label images")
  parser.add_argument('-d', '--calib_data_dir', type=str,
      default="/Users/joehuang/LocalDocuments/16833/Project/tactilevision/data/cam_calib_data")
  args = parser.parse_args()
  # Label the images
  label_images = LabelImages(args)
  label_images.label()
