import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Template Matching App")

# Initialize variables
full = None
face = None

uploaded_full = st.file_uploader("Upload the Full Image", type=["jpg", "jpeg", "png"])
uploaded_face = st.file_uploader("Upload the Template Image", type=["jpg", "jpeg", "png"])

if uploaded_full and uploaded_face:
    # Read the uploaded images using OpenCV
    full = cv2.imdecode(np.frombuffer(uploaded_full.read(), np.uint8), cv2.IMREAD_COLOR)
    full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

    face = cv2.imdecode(np.frombuffer(uploaded_face.read(), np.uint8), cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize the template image if needed
    if face.shape[0] > full.shape[0] or face.shape[1] > full.shape[1]:
        face = cv2.resize(face, (full.shape[1], full.shape[0]))

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    st.write("## Template Matching Results")

    for m in methods:
        # Create a copy of the image
        full_copy = full.copy()

        # Get the actual function instead of the string
        method = eval(m)

        # Apply template Matching with the method
        res = cv2.matchTemplate(full_copy, face, method)

        # Normalize the result to [0, 1] range
        res_normalized = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)

        # Convert to 8-bit for visualization
        res_normalized_8u = (res_normalized * 255).astype(np.uint8)

        # Convert to RGB (3 channels)
        res_normalized_rgb = cv2.cvtColor(res_normalized_8u, cv2.COLOR_GRAY2RGB)

        # Display the images and results
        st.write(f"### Method: {m}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(res_normalized_rgb, caption="Result of Template Matching")

        with col2:
            st.image(full_copy, caption="Detected Point")

        st.write('\n\n')

# Display the original images if available
st.write("## Original Images")
if full is not None:
    st.write("### Full Image")
    st.image(full, use_column_width=True)

if face is not None:
    st.write("### Template Image")
    st.image(face, use_column_width=True)
