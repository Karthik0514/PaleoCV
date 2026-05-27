import os
import cv2
import tempfile
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from reconstruct_open3d import generate_fossil_mesh
from infer import classify_fossil
from gradcam import generate_gradcam_from_path
from segment import segment_fossil
from shape_analysis import compute_shape_features


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PaleoCV",
    layout="wide"
)

st.title("PaleoCV – AI Fossil Analysis System")

st.write(
    "Upload or capture a fossil image for AI-assisted classification and morphology analysis."
)


# =====================================================
# IMAGE INPUT
# =====================================================

uploaded = st.file_uploader(
    "Upload Fossil Image",
    type=["jpg", "jpeg", "png"]
)

camera_img = st.camera_input(
    "Or capture using camera"
)

img_file = None

if uploaded is not None:
    img_file = uploaded

elif camera_img is not None:
    img_file = camera_img


# =====================================================
# MAIN PIPELINE
# =====================================================

if img_file is not None:

    os.makedirs("outputs", exist_ok=True)

    # =============================================
    # SAVE TEMP IMAGE
    # =============================================

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(img_file.read())

    img_path = tfile.name

    # =============================================
    # DISPLAY IMAGE
    # =============================================

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Input Fossil", width=450)

    # =============================================
    # CLASSIFICATION
    # =============================================

    st.header("1. Fossil Classification")

    predictions = classify_fossil(img_path)

    best = predictions[0]

    st.success(
        f"Predicted Fossil: {best['class']} "
        f"({best['confidence']:.2f}%)"
    )
    
    
    st.subheader("Top Predictions")

    for pred in predictions:

        st.write(
            f"{pred['class']} — {pred['confidence']:.2f}%"
        )

    # =============================================
    # GRAD-CAM
    # =============================================

    st.header("2. Grad-CAM Explainability")

    gradcam_path = generate_gradcam_from_path(img_path)

    st.image(
        gradcam_path,
        caption="Grad-CAM Attention Map",
        width=500
    )

    # =============================================
    # SEGMENTATION
    # =============================================

    st.header("3. Fossil Segmentation")

    result = segment_fossil(img_path)

    mask_path = "outputs/segment_mask.png"
    segmented_path = "outputs/segment_result.png"

    cv2.imwrite(mask_path, result["mask"])
    cv2.imwrite(segmented_path, result["segmented"])

    col1, col2 = st.columns(2)

    with col1:
        st.image(mask_path, caption="Segmentation Mask")

    with col2:
        st.image(segmented_path, caption="Segmented Fossil")

    # =============================================
    # SHAPE ANALYSIS
    # =============================================

    st.header("4. Morphological Analysis")

    features = compute_shape_features(result["contour"])

    for k, v in features.items():
        st.write(f"{k}: {v:.4f}")

   # =============================================
# INTERACTIVE 3D RECONSTRUCTION
# =============================================

st.header("5. Interactive 3D Reconstruction")

if st.button("Generate 3D Fossil"):

    mesh = generate_fossil_mesh(img_path)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]

    i = triangles[:,0]
    j = triangles[:,1]
    k = triangles[:,2]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=1.0
        )
    ])

    fig.update_layout(
        title="Interactive Fossil Reconstruction",
        width=900,
        height=700,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    st.plotly_chart(fig, use_container_width=True)