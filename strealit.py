import cv2
import numpy as np
import streamlit as st

def display(img, cmap='gray'):
    if img is None:
        st.write("Image not loaded or invalid.")
    else:
        st.image(img, use_column_width=True, channels=cmap)

st.title("Image Matching with OpenCV")

st.write("## Upload Images")
uploaded_helmet = st.file_uploader("Upload Helmet Image", type=["jpg", "png", "jpeg"])
uploaded_input = st.file_uploader("Upload Input Image", type=["jpg", "png", "jpeg"])

if uploaded_helmet and uploaded_input:
    # Read the uploaded images using OpenCV
    helmet = cv2.imdecode(np.frombuffer(uploaded_helmet.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    input_img = cv2.imdecode(np.frombuffer(uploaded_input.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    st.header("Original Images")
    st.subheader("Helmet Image")
    display(helmet)
    st.subheader("Input Image")
    display(input_img)

    # ORB Matcher
    st.header("ORB Matcher")

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(helmet, None)
    kp2, des2 = orb.detectAndCompute(input_img, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 25 matches.
    helmet_matches = cv2.drawMatches(
        helmet, kp1, input_img, kp2, matches[:25], None, flags=2)

    st.image(helmet_matches, use_column_width=True)

    # SIFT Matcher
    st.header("SIFT Matcher")

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(helmet, None)
    kp2, des2 = sift.detectAndCompute(input_img, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for match1, match2 in matches:
        if match1.distance < 0.75 * match2.distance:
            good.append([match1])

    sift_matches = cv2.drawMatchesKnn(helmet, kp1, input_img, kp2, good, None, flags=2)

    st.image(sift_matches, use_column_width=True)

    # FLANN Matcher
    st.header("FLANN Matcher")

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(helmet, None)
    kp2, des2 = sift.detectAndCompute(input_img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []

    # ratio test
    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7 * match2.distance:
            good.append([match1])

    flann_matches = cv2.drawMatchesKnn(helmet, kp1, input_img, kp2, good, None, flags=0)

    st.image(flann_matches, use_column_width=True)
