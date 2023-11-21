import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def measure_performance(matcher, image1, image2):
    start_time = time.time()
    keypoints1, descriptors1 = matcher.find_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = matcher.find_keypoints_and_descriptors(image2)
    good_matches = matcher.match(descriptors1, descriptors2)
    end_time = time.time()

    execution_time = end_time - start_time
    num_matches = len(good_matches)

    return num_matches, execution_time


class SIFTMatcher:
    def __init__(self, ratio_thresh=0.6):
        self.ratio_thresh = ratio_thresh
        self.sift = cv2.SIFT_create()

    def find_keypoints_and_descriptors(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(grayscale, None)
        return keypoints, descriptors

    def match(self, descriptors1, descriptors2):
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]
        return good


class RANSACFilter:
    def __init__(self, reprojection_thresh=2.0):
        self.reprojection_thresh = reprojection_thresh

    def apply(self, keypoints1, keypoints2, matches):
        if not matches:
            return None, None
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.reprojection_thresh)
        return M, mask

    def check_geometric_consistency(self, keypoints1, keypoints2, good_matches, M, threshold=5):
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2_proj = cv2.perspectiveTransform(pts1, M)
        distances = np.sqrt(np.sum((pts2 - pts2_proj) ** 2, axis=2))
        inlier_count = np.sum(distances < threshold)
        return inlier_count, distances


class ORBMatcher:
    def __init__(self, number_of_points=1000, fast_threshold=20):
        self.orb = cv2.ORB_create(nfeatures=number_of_points, fastThreshold=fast_threshold)

    def find_keypoints_and_descriptors(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(grayscale, None)
        return keypoints, descriptors

    def match(self, descriptors1, descriptors2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches



class AKAZEMatcher:
    def __init__(self):
        self.akaze = cv2.AKAZE_create()

    def find_keypoints_and_descriptors(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.akaze.detectAndCompute(grayscale, None)
        return keypoints, descriptors

    def match(self, descriptors1, descriptors2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches