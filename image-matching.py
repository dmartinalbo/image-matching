from __future__ import division
import sys
import os
import argparse
import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt

# load global logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s')

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

def calculate_SIFT(img):
  # Find the keypoints and descriptors using SIFT features
  kp, des = sift.detectAndCompute(img,None)
  return kp, des

def knn_match(des1, des2, nn_ratio=0.7):
  
  # FLANN parameters
  index_params = dict(algorithm = 0, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)
  
  # Match features from each image
  matches = flann.knnMatch(des1, des2, k=2)

  # store only the good matches as per Lowe's ratio test.
  good = []
  for m, n in matches:
    if m.distance < nn_ratio * n.distance:
      good.append(m)

  return good

# calculate the angle with the horizontal
def angle_horizontal(v):
    return -np.arctan2(v[1],v[0])

def knn_clasif(good_matches):
  best_template, highest_logprob = None, 0.0

  sum_good_matches = sum([len(gm) for gm in good_matches])
  for i, gm in enumerate(good_matches):
    logprob = len(gm)/sum_good_matches
    # save highest
    if logprob > highest_logprob:
      highest_logprob = logprob
      best_template = i
    logger.info('p(t_{} | x) = {:.4f}'.format(i, logprob))
  return best_template

def main(argv=None):

  try:
    parser = argparse.ArgumentParser(description='Classifies (and geometrically corrects) an image between DNIv3 (head or tail).')
    parser.add_argument('-t', dest='template_names', nargs='+', required=True, help='Image to be used as tail template')
    parser.add_argument('-q', dest='query_names', nargs='+', required=True, help='Preprocessed image to query')
    parser.add_argument('-c', dest='bbs', nargs='*', help='Bounding boxes to crop. (Format="WxH+X+Y).')
    parser.add_argument('-n', dest='nn_ratio', type=float, default=0.85, help='Nearest neighbor matching ratio')
    parser.add_argument('-v', dest='verbosity', action='store_true', help='Increase output verbosity')
    parser.add_argument('-p', dest='photocopied', action='store_true', help='Use only if the image is scanned or photocopied. Do not with photos!')
    parser.add_argument('--matches', dest='view_matches', action='store_true')
    parser.add_argument('-o', dest='output_path', help='Output path', default='.')

    parser.set_defaults(view_matches=False)
    parser.set_defaults(photocopied=False)

    args = parser.parse_args()

  except Exception, e:
    logger.error('Error', exc_info=True)
    return 2

  # logging stuff
  if args.verbosity:
    logger.setLevel(logging.DEBUG)

  # load template images
  templates = []
  for name in args.template_names:
    logger.info('Loading template image {}'.format(name))
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    logger.info('  Calculating SIFT features ...')
    kp, des = calculate_SIFT(img)
    templates.append( [name, img, kp, des])

  # load query
  for name in args.query_names:
    logger.info('Loading query image {}'.format(name))
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    logger.info('  Calculating SIFT features ...'.format(name))
    query_kp, query_des = calculate_SIFT(img)

    # for each template, calculate the best match
    list_good_matches = []
    for templ_name, _, _, templ_des in templates:
      logger.info('Estimating match between {} and {}'.format(templ_name, name))
      gm = knn_match(templ_des, query_des)
      list_good_matches.append(gm)

    # Get closer template using k-nn
    best_template = knn_clasif(list_good_matches)

    # Keep the best result the best result
    template_kp = templates[best_template][2]
    good_matches = list_good_matches[best_template]
    
    # data massaging
    src_pts = np.float32([ template_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ query_kp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    logger.info('Estimating homography between {} and {}'.format(templates[best_template][0], name))
    # find the matrix transformation M
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
    matchesMask = mask.ravel().tolist()

    # Make it affine
    M[2,2] = 1.0
    M[2,0] = 0.0
    M[2,1] = 0.0

    # Calculate the rectangle enclosing the query image
    h,w = templates[best_template][1].shape

    # Define the rectangle in the coordinates of the template image
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

    # transform the rectangle from the template "coordinates" to the query "coordinates"
    dst = cv2.perspectiveTransform(pts,M)

    if args.photocopied:
      logger.info('Simplifying transformation matrix ...'.format(templates[best_template][0], name))
      # if the image is a photocopy or scanned we can assume that there is no shear in x or y.
      # Thus we can simplify the transformation matrix M with only: rotation, scale and tranlation.

      # calculate template "world" reference vectors
      w_v = np.array([w-1,0])
      h_v = np.array([h-1,0])

      # calculate query "world" reference vectors
      w_vp = (dst[3]-dst[0])[0]
      h_vp = (dst[1]-dst[0])[0]

      # We lost the angle and the scale given that scalation shares the same position in M with the shear transformation
      # see https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
      
      # estimate the angle using the top-horizontal line
      angle = angle_horizontal(w_vp)
      
      # estimate the scale using the top-horizontal line and left-vertical line
      scale_x = np.linalg.norm(w_vp) / np.linalg.norm(w_v)
      scale_y = np.linalg.norm(h_vp) / np.linalg.norm(h_v)

      # retrieve translation from original matrix M
      M = np.matrix([[ scale_x * np.cos(angle) , np.sin(angle)           , M[0,2] ],
                     [ -np.sin(angle)          , scale_y * np.cos(angle) , M[1,2] ],
                     [ 0                       , 0                       , 1.     ]])

      # retransform the rectangle with the new matrix
      dst = cv2.perspectiveTransform(pts,M)
    
    # if bbs crop those areas
    bn, ext = os.path.splitext(os.path.basename(name))
    # using M^{-1} we go from query coordinates to template coordinates.
    img_templ_coords = cv2.warpPerspective(img, np.linalg.inv(M), (w,h))
    if args.bbs:
      cont_bbs = 0
      for bb in args.bbs:
        # parse bb string to variables
        width, height, x_ini, y_ini = (int(c) for c in bb.replace('x','+').split('+'))
        #
        logger.info('Cropping "{}x{}+{}+{}" from {}'.format(width, height, x_ini, y_ini, name))
        # crop image
        img_templ_coords_crop = img_templ_coords[y_ini:y_ini+height, x_ini:x_ini+width]
        # write it
        cv2.imwrite('{}/{}_crop_{}{}'.format(args.output_path, bn, cont_bbs, ext), img_templ_coords_crop)
        logger.info('  Saved in {}/{}_crop_{}{}'.format(args.output_path, bn, cont_bbs, ext))
        cont_bbs += 1
    else:
      cv2.imwrite('{}/{}_fix{}'.format(args.output_path, bn, ext), img_templ_coords)

    if args.view_matches:
      # draw the rectangle in the image
      out = cv2.polylines(img,[np.int32(dst)],True,0,2, cv2.LINE_AA)
      # show the matching features
      params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
      ## draw the matches image 
      out = cv2.drawMatches(templates[best_template][1], template_kp,
                            img, query_kp,
                            good_matches, 
                            None, **params)

      ## show result
      plt.imshow(out, 'gray')
      plt.show()

if __name__ == "__main__":
  sys.exit(main())
