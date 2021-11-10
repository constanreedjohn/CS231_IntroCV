import numpy as np
import cv2 as cv
import os

# Harris Corner Detection
def harris_corner(path):

  # Read image then grayscale
  img = cv.imread(path)
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  # Conver to float type
  gray = np.float32(gray)
  
  # Harris Corner
  blockSize = 2 # The size of the neighbourhood considered for corner detection
  ksize = 3 # Aperture param for Sobel derivative used
  k = 0.04 # Harris detector tree param in the equation
  dst = cv.cornerHarris(gray, blockSize, ksize, k)

  dst = cv.dilate(dst, None)
  num_corner = num_corner = np.sum(dst > 0.01 * dst.max())

  img[dst > 0.01 * dst.max()] = [0, 0, 255]

  return img, num_corner

def shitomasi_corner(path):
  
  # Read image then grayscale
  img = cv.imread(path)
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  # Conver to float type
  gray = np.float32(gray)

  num_corners = 200
  quality = 0.01
  min_dist = 10
  corners = cv.goodFeaturesToTrack(gray, num_corners, quality, min_dist)
  corners = np.int0(corners)

  radius = 3
  color = (255, 0, 0)
  thickness = -1
  for i in corners:
      x, y = i.ravel()
      cv.circle(img, (x, y), radius, color, thickness)

  return img, len(corners)

def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)

def main():
  """
  So sánh giữa Harris và Shitomasi Corner Detection:
  - Harris corner có độ chính xác cao, nhưng bị lặp lại kết quả, suy ra số lượng corner lại lớn hơn shitomasi.
  - Shitomasi vẫn cho ra được độ chính xác giống với harris, nhưng lại bỏ xót vài trường hợp.
  - Vì shitomasi tính khoảng cách nhỏ nhất giữa các corner phát hiện được nên số lượng corner ít hơn so với harris,
  đồng thời ít bị trùng lặp.
  """
  img_path = 'path/to/folder/of/images'
  ext = ['.jpg', '.png', '.jpeg']

  img_lst = [os.path.join(img_path, i) \
    for i in os.listdir(img_path) if i[i.rfind('.'):] in ext]
  
  for image in img_lst:
      harris, h_corner = harris_corner(image)
      shito, s_corner = shitomasi_corner(image)
      print(f'Number of corner in harris/shitomasi: {h_corner}/{s_corner}')
      cv.imshow('Result Harris(left) - Shitomasi(right)', hconcat_resize_min([harris, shito]))
      cv.waitKey(0)
  cv.destroyAllWindows()

if __name__ == '__main__':
  main()