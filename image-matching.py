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

def keypoints_matches(template, query, nn_ratio=0.7):

  # Initiate SIFT detector
  sift = cv2.xfeatures2d.SIFT_create()

  # Find the keypoints and descriptors using SIFT features
  kp1, des1 = sift.detectAndCompute(template,None)
  kp2, des2 = sift.detectAndCompute(query,None)

  # FLANN parameters
  index_params = dict(algorithm = 0, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)
  
  # Match features from each image
  matches = flann.knnMatch(des1, des2, k=2)

  # store only the good matches as per Lowe's ratio test.
  good = []
  for m,n in matches:
    if m.distance < nn_ratio * n.distance:
      good.append(m)

  return kp1, kp2, good

# calculate the angle with the horizontal
def angle_horizontal(v):
    return -np.arctan2(v[1],v[0])

def main(argv=None):

  try:
    parser = argparse.ArgumentParser(description='Classifies (and geometrically corrects) an image between DNIv3 (head or tail).')
    parser.add_argument('-t', dest='template_names', nargs='+', required=True, help='Image to be used as tail template')
    parser.add_argument('-q', dest='query', required=True, help='Preprocessed image to query')
    parser.add_argument('-i', dest='orig_query', help='Original image to query')
    parser.add_argument('-n', dest='nn_ratio', type=float, default=0.85, help='Nearest neighbor matching ratio')
    parser.add_argument('-v', dest='verbosity', action='store_true', help='Increase output verbosity')
    parser.add_argument('-p', dest='photocopied', action='store_true', help='Use only if the image is scanned or photocopied. Do not with photos!')
    parser.add_argument('--matches', dest='view_matches', action='store_true')
    parser.add_argument('-o', dest='output_path', help='Output path')

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
  for t in args.template_names:
    logger.info('Loading template image {}'.format(t))
    templates.append( (t, cv2.imread(t, cv2.IMREAD_GRAYSCALE) ))

  # load query
  logger.info('Loading query image %s', args.query)
  query = cv2.imread(args.query, cv2.IMREAD_GRAYSCALE)

  keypoints = []

  # Calculate keypoints for each template
  for i,(name, img) in enumerate(templates):
    logger.info('Finding keypoints for template {}'.format(name))
    template_kp, query_kp, good_matches = keypoints_matches(img, query, args.nn_ratio)
    keypoints.append( (template_kp, query_kp, good_matches) )

    number_good_matches = len(good_matches)

  # Estimate post prob using k-nn
  best_template, highest_logprob = None, 0.0
  
  sum_good_matches = sum([len(gm) for _,_, gm in keypoints])
  for i,(template_kp, query_kp, good_matches) in enumerate(keypoints):
    logprob = len(good_matches)/sum_good_matches
    # save highest
    if logprob > highest_logprob:
      highest_logprob = logprob
      best_template = i
    logger.info('p(t_{} | x) = {:.4f}'.format(i, logprob))

  # Show the best result
  template_kp, query_kp, good_matches = keypoints[best_template]

  # data massaging
  src_pts = np.float32([ template_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
  dst_pts = np.float32([ query_kp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

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

  # transform the rectangle from the template "world" to the query "world"
  dst = cv2.perspectiveTransform(pts,M)

  if args.photocopied:
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
    M = np.matrix([[ scale_x * np.cos(angle) , np.sin(angle)           , M[0,2 ]],
                   [ -np.sin(angle)          , scale_y * np.cos(angle) , M[1,2 ]],
                   [ 0                       , 0                       , 1.     ]])

    # transform the rectangle
    dst = cv2.perspectiveTransform(pts,M)

  # draw it  
  out = cv2.polylines(query,[np.int32(dst)],True,0,2, cv2.LINE_AA)
  
  if args.view_matches:
    # if enabled, show the matching features
    params = dict(matchColor = (0,255,0), # draw matches in green color
                  singlePointColor = None,
                  matchesMask = matchesMask, # draw only inliers
                  flags = 2)
    # draw 
    out = cv2.drawMatches(templates[best_template][1], template_kp,
                          query, query_kp,
                          good_matches, 
                          None, **params)
  if args.orig_query:
    logger.info('Loading original query image %s', args.orig_query)
    orig_query = cv2.imread(args.orig_query, cv2.IMREAD_COLOR)
    #
    img_out = cv2.warpPerspective(orig_query, np.linalg.inv(M), (w,h)) #[90:345,250:525]
    #
    output_name = '{}/{}'.format(args.output_path, os.path.basename(args.orig_query))
    logger.info('Saving output query image %s', output_name)
    cv2.imwrite(output_name,img_out)

  # # # show result
  plt.imshow(out, 'gray')
  plt.show()

if __name__ == "__main__":
  sys.exit(main())
