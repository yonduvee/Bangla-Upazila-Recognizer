import base64
import importlib.util
import io
import mimetypes
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import streamlit as st
from PIL import Image


# =========================================================
# LOAD INFERENCE MODULE FROM EXACT FILE PATH
# =========================================================

APP_DIR = Path(__file__).resolve().parent
INFERENCE_FILE = APP_DIR / "inference.py"

if not INFERENCE_FILE.is_file():
    raise FileNotFoundError(
        f"inference.py was not found at: {INFERENCE_FILE}"
    )

module_spec = importlib.util.spec_from_file_location(
    "bangla_upazila_model_inference",
    INFERENCE_FILE,
)

if module_spec is None or module_spec.loader is None:
    raise ImportError(
        f"Could not load inference module from: {INFERENCE_FILE}"
    )

inference_module = importlib.util.module_from_spec(module_spec)
sys.modules["bangla_upazila_model_inference"] = inference_module
module_spec.loader.exec_module(inference_module)

if not hasattr(inference_module, "predict_upazila_district"):
    raise ImportError(
        "predict_upazila_district() was not found "
        f"inside {INFERENCE_FILE}"
    )

predict_upazila_district = inference_module.predict_upazila_district

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bangla Handwritten Upazila–District Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PROJECT PATHS
# =========================================================

ASSETS_DIR = APP_DIR / "assets"

VILLAGE_IMAGE = ASSETS_DIR / "village.png"
LALBAGH_IMAGE = ASSETS_DIR / "lalbagh_fort.png"
MEMORIAL_IMAGE = ASSETS_DIR / "memorial.png"


# =========================================================
# HELPERS
# =========================================================

def render_html(content: str) -> None:
    """Render custom HTML safely without Markdown code formatting."""

    html_content = dedent(content).strip()

    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(
            html_content,
            unsafe_allow_html=True,
        )


def image_to_data_uri(image_path: Path) -> str:
    """Convert a local image to a Base64 data URI."""

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    mime_type, _ = mimetypes.guess_type(
        image_path.name
    )

    mime_type = mime_type or "image/png"

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def recognize_handwritten_image(
    image: Image.Image,
) -> dict:
    """Run the real ensemble model prediction."""

    return predict_upazila_district(
        image
    )


def clear_prediction() -> None:
    """Clear the previous recognition result."""

    st.session_state.prediction = None

# =========================================================
# CHECK AND LOAD ARTWORKS
# =========================================================

missing_files = [
    path
    for path in (
        VILLAGE_IMAGE,
        LALBAGH_IMAGE,
        MEMORIAL_IMAGE,
    )
    if not path.is_file()
]


if missing_files:

    missing_text = "\n".join(
        f"• {path}"
        for path in missing_files
    )

    st.error(
        "The following required image file(s) "
        "were not found:\n\n"
        f"{missing_text}"
    )

    st.code(
        """
Bangla-Upazila-Recognizer/
├── app.py
└── assets/
    ├── village.png
    ├── lalbagh_fort.png
    └── memorial.png
        """.strip()
    )

    st.stop()


village_uri = image_to_data_uri(
    VILLAGE_IMAGE
)

lalbagh_uri = image_to_data_uri(
    LALBAGH_IMAGE
)

memorial_uri = image_to_data_uri(
    MEMORIAL_IMAGE
)


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# =========================================================
# CUSTOM CSS
# =========================================================

render_html(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;500;600;700&family=Noto+Serif+Bengali:wght@500;600;700&family=Libre+Baskerville:wght@400;700&display=swap'
    );


    :root {
        --dark-green: #064126;
        --green: #095c35;
        --red: #a51920;
        --cream: #f7efdc;
        --paper: #fffdf7;
        --border: #ddcfad;
        --text: #202820;
    }


    html,
    body,
    .stApp,
    [class*="css"] {
        font-family:
            "Noto Sans Bengali",
            Arial,
            sans-serif;
    }


    .stApp {
        background:
            radial-gradient(
                circle at 8% 5%,
                rgba(165, 25, 32, 0.06),
                transparent 24%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(6, 65, 38, 0.06),
                transparent 24%
            ),
            linear-gradient(
                180deg,
                #fbf7ec 0%,
                #f5ecd8 100%
            );

        color: var(--text);
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        display: none;
    }


    .block-container {
        max-width: 1280px;

        padding-top: 0.35rem;
        padding-left: 1.15rem;
        padding-right: 1.15rem;
        padding-bottom: 1rem;
    }


    /* =====================================================
       TOP DECORATIVE PATTERN
       ===================================================== */

    .top-pattern {
        height: 15px;

        border-radius: 6px 6px 0 0;

        border-bottom:
            2px solid
            #d3c39d;

        background:
            repeating-linear-gradient(
                45deg,
                #063a22 0,
                #063a22 9px,
                #0b5933 9px,
                #0b5933 18px
            );
    }


    /* =====================================================
       HERO HEADER
       ===================================================== */

    .hero {
        position: relative;

        min-height: 292px;

        overflow: hidden;

        isolation: isolate;

        margin-bottom: 24px;

        border:
            1px solid
            #e2d4b6;

        border-top: none;

        border-radius:
            0
            0
            14px
            14px;

        background: #fbf5e7;

        box-shadow:
            0 7px 18px
            rgba(32, 43, 35, 0.13);
    }


    .hero::before {
        content: "";

        position: absolute;
        inset: 0;

        z-index: 2;

        pointer-events: none;

        background:
            linear-gradient(
                90deg,
                rgba(251, 245, 231, 0.07) 0%,
                rgba(251, 245, 231, 0.76) 31%,
                rgba(251, 245, 231, 0.98) 43%,
                rgba(251, 245, 231, 0.98) 57%,
                rgba(251, 245, 231, 0.76) 69%,
                rgba(251, 245, 231, 0.07) 100%
            );
    }


    .hero-art {
        position: absolute;

        top: 0;

        width: 36%;
        height: 100%;

        z-index: 1;

        object-fit: cover;

        mix-blend-mode: multiply;
    }


    .hero-village {
        left: 0;

        object-position:
            33%
            center;

        -webkit-mask-image:
            linear-gradient(
                90deg,
                #000 67%,
                transparent 100%
            );

        mask-image:
            linear-gradient(
                90deg,
                #000 67%,
                transparent 100%
            );
    }


    .hero-lalbagh {
        right: 0;

        object-position:
            67%
            center;

        -webkit-mask-image:
            linear-gradient(
                270deg,
                #000 67%,
                transparent 100%
            );

        mask-image:
            linear-gradient(
                270deg,
                #000 67%,
                transparent 100%
            );
    }


    .hero-content {
        position: relative;

        z-index: 5;

        max-width: 760px;

        margin:
            0
            auto;

        padding:
            31px
            20px
            22px;

        text-align: center;
    }


    .hero-title-one,
    .hero-title-two,
    .hero-subtitle {
        font-family:
            "Libre Baskerville",
            serif;

        font-weight: 700;
    }


    .hero-title-one {
        margin: 0;

        color: var(--dark-green);

        font-size:
            clamp(
                31px,
                4vw,
                48px
            );

        line-height: 1.15;
    }


    .hero-title-two {
        margin:
            10px
            0;

        color: var(--red);

        font-size:
            clamp(
                25px,
                3.35vw,
                39px
            );

        line-height: 1.2;
    }


    .hero-subtitle {
        margin-top: 8px;

        color: var(--dark-green);

        font-size:
            clamp(
                19px,
                2.1vw,
                26px
            );
    }


    .hero-description {
        margin-top: 17px;

        color: #27342d;

        font-size: 17px;
    }


    .hero-divider {
        position: relative;

        width:
            min(
                500px,
                75%
            );

        height: 20px;

        margin:
            16px
            auto
            0;
    }


    .hero-divider::before,
    .hero-divider::after {
        content: "";

        position: absolute;

        top: 9px;

        width: 44%;
        height: 2px;

        background:
            linear-gradient(
                90deg,
                transparent,
                var(--red),
                var(--dark-green)
            );
    }


    .hero-divider::before {
        left: 0;
    }


    .hero-divider::after {
        right: 0;

        transform:
            scaleX(-1);
    }


    .hero-flower {
        position: absolute;

        left: 50%;
        top: -4px;

        transform:
            translateX(-50%);

        padding:
            0
            8px;

        color: var(--red);

        background: #fbf5e7;

        font-size: 25px;
    }


    /* =====================================================
       MAIN CARDS
       ===================================================== */

    div[data-testid="stHorizontalBlock"] {
        gap: 1.35rem;

        align-items: stretch;
    }


    div[data-testid="stColumn"] {
        overflow: hidden;

        padding:
            0
            19px
            19px;

        border:
            1px solid
            var(--border);

        border-radius: 15px;

        background:
            rgba(
                255,
                253,
                247,
                0.98
            );

        box-shadow:
            0 7px 18px
            rgba(33, 44, 36, 0.12);
    }


    .panel-header {
        display: flex;

        align-items: center;

        gap: 11px;

        margin:
            0
            -19px
            20px;

        padding:
            16px
            21px;

        border-radius:
            14px
            14px
            0
            0;

        color: white;

        background:
            linear-gradient(
                135deg,
                #063d24,
                #0a6037
            );

        font-family:
            "Libre Baskerville",
            serif;

        font-size: 19px;

        font-weight: 700;
    }


    .panel-number {
        display: inline-flex;

        align-items: center;
        justify-content: center;

        width: 26px;
        height: 26px;

        border-radius: 50%;

        color: var(--dark-green);

        background: #fff8e5;

        font-family:
            Arial,
            sans-serif;

        font-size: 14px;

        font-weight: 700;
    }


    .panel-description {
        margin:
            4px
            12px
            18px;

        color: #29322d;

        text-align: center;

        font-size: 16px;

        line-height: 1.65;
    }


    .highlight-red {
        color: var(--red);

        font-weight: 700;
    }


    [data-testid="stImage"] {
        margin-top: 8px;

        padding: 7px;

        border:
            1px solid
            #ddd1b7;

        border-radius: 10px;

        background: white;
    }


    .or-divider {
        display: flex;

        align-items: center;
        justify-content: center;

        gap: 14px;

        margin:
            10px
            0
            2px;

        color: #46534b;

        font-family:
            Arial,
            sans-serif;
    }


    .or-divider::before,
    .or-divider::after {
        content: "";

        width: 45px;
        height: 1px;

        background: #9ba59e;
    }


    /* =====================================================
       RECOGNIZE BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;

        min-height: 51px;

        border: none;

        border-radius: 9px;

        color: white;

        background:
            linear-gradient(
                135deg,
                #063e24,
                #0b6339
            );

        box-shadow:
            0 5px 11px
            rgba(7, 63, 37, 0.2);

        font-family:
            "Libre Baskerville",
            serif;

        font-size: 17px;

        font-weight: 700;
    }


    .stButton > button:hover {
        border: none;

        color: white;

        background:
            linear-gradient(
                135deg,
                #8d151b,
                #b5242a
            );
    }


    .stButton > button:disabled {
        color: #f3f3f3;

        background: #91a097;

        opacity: 0.75;
    }


    /* =====================================================
       NOTE
       ===================================================== */

    .note-box {
        display: flex;

        align-items: flex-start;

        gap: 13px;

        margin-top: 17px;

        padding:
            14px
            16px;

        border:
            1px solid
            #d2dbc1;

        border-radius: 10px;

        color: #27342c;

        background:
            linear-gradient(
                135deg,
                #f3f7e8,
                #eaf0de
            );

        line-height: 1.55;
    }


    .note-icon {
        display: flex;

        align-items: center;
        justify-content: center;

        min-width: 38px;
        height: 38px;

        border-radius: 50%;

        background: #faf0c7;

        font-size: 21px;
    }


    /* =====================================================
       RESULT
       ===================================================== */

    .result-introduction {
        margin:
            4px
            0
            17px;

        color: #2a332e;

        font-size: 16px;
    }


    .result-card {
        display: flex;

        align-items: center;

        gap: 18px;

        min-height: 112px;

        padding:
            17px
            19px;

        border:
            1px solid
            #ddcda7;

        background:
            linear-gradient(
                135deg,
                #fffdf8,
                #f9f0dc
            );
    }


    .result-card.top {
        border-radius:
            13px
            13px
            0
            0;
    }


    .result-card.bottom {
        margin-bottom: 20px;

        border-radius:
            0
            0
            13px
            13px;
    }


    .result-icon {
        display: flex;

        align-items: center;
        justify-content: center;

        min-width: 74px;
        height: 74px;

        border-radius: 50%;

        color: white;

        font-size: 34px;

        box-shadow:
            0 4px 10px
            rgba(0, 0, 0, 0.13);
    }


    .result-icon.green {
        background:
            radial-gradient(
                circle at 30% 20%,
                #157445,
                #063d24
            );
    }


    .result-icon.red {
        background:
            radial-gradient(
                circle at 30% 20%,
                #cf2831,
                #991017
            );
    }


    .result-label {
        margin-bottom: 4px;

        color: #202720;

        font-family:
            "Libre Baskerville",
            serif;

        font-size: 17px;

        font-weight: 700;
    }


    .bangla-result {
        font-family:
            "Noto Serif Bengali",
            serif;

        font-size: 37px;

        font-weight: 600;

        line-height: 1.25;
    }


    .bangla-result.green {
        color: var(--dark-green);
    }


    .bangla-result.red {
        color: var(--red);
    }


    .confidence-heading {
        display: flex;

        align-items: flex-end;
        justify-content: space-between;

        gap: 20px;

        margin:
            5px
            0
            8px;
    }


    .confidence-label {
        color: #273129;

        font-size: 17px;

        font-weight: 700;
    }


    .confidence-value {
        color: var(--dark-green);

        font-family:
            "Libre Baskerville",
            serif;

        font-size: 31px;

        font-weight: 700;
    }


    .confidence-track {
        width: 100%;
        height: 15px;

        margin-bottom: 20px;

        overflow: hidden;

        border-radius: 30px;

        background: #e3e1d6;

        box-shadow:
            inset
            0 1px 3px
            rgba(0, 0, 0, 0.11);
    }


    .confidence-fill {
        height: 100%;

        border-radius: 30px;

        background:
            linear-gradient(
                90deg,
                #064025,
                #0e6b3d
            );
    }


    .model-box {
        display: flex;

        align-items: center;

        gap: 14px;

        padding:
            15px
            17px;

        border:
            1px solid
            #d0dcc6;

        border-radius: 10px;

        color: #26332b;

        background:
            linear-gradient(
                135deg,
                #edf4e7,
                #e6eede
            );

        font-size: 14.5px;

        line-height: 1.55;
    }


    .model-shield {
        display: flex;

        align-items: center;
        justify-content: center;

        min-width: 42px;
        height: 45px;

        color: white;

        background: #4d914c;

        clip-path:
            polygon(
                50% 0%,
                93% 17%,
                88% 70%,
                50% 100%,
                12% 70%,
                7% 17%
            );

        font-size: 21px;

        font-weight: 700;
    }


    .result-placeholder {
        min-height: 404px;

        display: flex;

        align-items: center;
        justify-content: center;

        padding: 35px;

        border:
            2px dashed
            #d4c8ae;

        border-radius: 13px;

        color: #617068;

        text-align: center;

        background:
            rgba(
                250,
                246,
                235,
                0.75
            );

        line-height: 1.6;
    }


    .placeholder-icon {
        margin-bottom: 12px;

        color: var(--green);

        font-size: 54px;
    }


    /* =====================================================
       ABOUT
       ===================================================== */

    .about-section {
        margin-top: 23px;

        overflow: hidden;

        border:
            1px solid
            #decfb0;

        border-radius: 15px;

        background:
            linear-gradient(
                100deg,
                #fffdf7,
                #f8efdc
            );

        box-shadow:
            0 7px 18px
            rgba(33, 43, 36, 0.1);
    }


    .about-grid {
        display: grid;

        grid-template-columns:
            1.12fr
            1fr
            0.9fr;

        min-height: 250px;
    }


    .about-content {
        padding:
            27px
            25px;
    }


    .about-title {
        color: var(--red);

        font-family:
            "Libre Baskerville",
            serif;

        font-size: 22px;

        font-weight: 700;
    }


    .about-line {
        width: 190px;
        height: 2px;

        margin:
            12px
            0
            17px;

        background:
            linear-gradient(
                90deg,
                var(--red),
                transparent
            );
    }


    .about-content p {
        margin: 0;

        color: #29332d;

        font-size: 15px;

        line-height: 1.75;
    }


    .memorial-section {
        position: relative;

        min-height: 250px;

        overflow: hidden;

        background: #f8f1e1;
    }


    .memorial-image {
        display: block;

        width: 100%;
        height: 100%;

        min-height: 90px;

        object-fit: cover;

        object-position: center;

        mix-blend-mode: multiply;
    }


    .features-box {
        position: relative;

        margin: 17px;

        padding:
            19px
            17px;

        border:
            2px solid
            #58745e;

        border-radius: 9px;

        background:
            rgba(
                255,
                253,
                246,
                0.94
            );
    }


    .features-title {
        margin-bottom: 14px;

        color: var(--dark-green);

        font-size: 17px;

        font-weight: 700;
    }


    .feature-item {
        display: flex;

        align-items: flex-start;

        gap: 10px;

        margin:
            12px
            0;

        color: #253029;

        font-size: 14px;
    }


    .feature-icon {
        width: 22px;

        color: var(--green);

        font-weight: 700;
    }


    .custom-footer {
        position: relative;

        overflow: hidden;

        padding:
            17px
            15px;

        color: white;

        text-align: center;

        background:
            linear-gradient(
                135deg,
                #063d24,
                #0a5732
            );

        font-family:
            "Libre Baskerville",
            serif;

        font-size: 14px;
    }


    /* =====================================================
       RESPONSIVE
       ===================================================== */

    @media (max-width: 1000px) {

        .hero-art {
            width: 42%;

            opacity: 0.48;
        }


        .about-grid {
            grid-template-columns: 1fr;
        }


        .memorial-image {
            max-height: 340px;
        }
    }


    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }


        .hero {
            min-height: 340px;
        }


        .hero-art {
            width: 65%;

            opacity: 0.20;
        }


        .hero-content {
            padding-top: 45px;
        }


        .bangla-result {
            font-size: 30px;
        }


        .result-icon {
            min-width: 61px;
            height: 61px;
        }
    }

    </style>
    """
)


# =========================================================
# HERO SECTION
# =========================================================

render_html(
    f"""
    <div class="top-pattern"></div>

    <section class="hero">

        <img
            src="{village_uri}"
            class="hero-art hero-village"
            alt="Bangladeshi village sketch"
        >

        <img
            src="{lalbagh_uri}"
            class="hero-art hero-lalbagh"
            alt="Lalbagh Fort sketch"
        >

        <div class="hero-content">

            <h1 class="hero-title-one">
                Bangla Handwritten
            </h1>

            <h2 class="hero-title-two">
                Upazila–District Pair Name Recognition
            </h2>

            <div class="hero-subtitle">
                ← Using Ensemble Learning →
            </div>

            <div class="hero-description">
                A Research Project |
                AI Powered Recognition System
            </div>

            <div class="hero-divider">

                <span class="hero-flower">
                    ✤
                </span>

            </div>

        </div>

    </section>
    """
)


# =========================================================
# MAIN SECTION
# =========================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large",
)


# =========================================================
# LEFT — IMAGE INPUT
# =========================================================

with left_column:

    render_html(
        """
        <div class="panel-header">

            <span style="font-size:25px;">
                ✎
            </span>

            <span class="panel-number">
                1
            </span>

            <span>
                Input Handwritten Image
            </span>

        </div>


        <div class="panel-description">

            Write a Bangla

            <span class="highlight-red">
                Upazila–District
            </span>

            pair name on paper

            <br>

            and upload the image here.

        </div>
        """
    )


    # =========================================================
    # IMAGE UPLOADER
    # =========================================================

    uploaded_image = st.file_uploader(
        "Choose a handwritten image",
        type=None,
        accept_multiple_files=False,
        key="handwritten_image_upload",
        help=(
            "Upload PNG, JPG, JPEG, JFIF, WEBP, "
            "BMP or TIFF image."
        ),
    )


    preview_image = None


    if uploaded_image is not None:

        try:

            uploaded_bytes = uploaded_image.getvalue()

            if not uploaded_bytes:
                raise ValueError(
                    "The selected file is empty."
                )

            st.info(
                f"Selected file: {uploaded_image.name} | "
                f"Size: {len(uploaded_bytes) / 1024:.1f} KB"
            )

            preview_image = Image.open(
                io.BytesIO(uploaded_bytes)
            ).convert("RGB")

            preview_image.load()

            upload_signature = (
                uploaded_image.name,
                len(uploaded_bytes),
            )

            if (
                st.session_state.get(
                    "last_upload_signature"
                )
                != upload_signature
            ):
                st.session_state.prediction = None
                st.session_state.last_upload_signature = (
                    upload_signature
                )

            st.success(
                "Image uploaded successfully."
            )

            st.image(
                preview_image,
                caption=uploaded_image.name,
                use_container_width=True,
            )

        except Exception as error:

            preview_image = None
            st.session_state.prediction = None

            st.error(
                "The selected file could not be "
                "opened as an image."
            )

            st.exception(error)


    recognize_button = st.button(
        "🔍 Recognize Name",
        disabled=preview_image is None,
        use_container_width=True,
        key="recognize_name_button",
    )


    if (
        recognize_button
        and preview_image is not None
    ):

        try:

            with st.spinner(
                "Loading models and recognizing "
                "the image..."
            ):

                result = recognize_handwritten_image(
                    preview_image
                )

                st.session_state.prediction = result

            st.success(
                "Recognition completed successfully."
            )

        except Exception as error:

            st.session_state.prediction = None

            st.error(
                "Recognition failed."
            )

            st.exception(error)


    render_html(
        """
        <div class="note-box">

            <div class="note-icon">
                💡
            </div>

            <div>

                <strong>
                    Note:
                </strong>

                Please write clearly.
                The system works best
                with neat handwritten text.

            </div>

        </div>
        """
    )


# =========================================================
# RIGHT — RESULT
# =========================================================

with right_column:

    render_html(
        """
        <div class="panel-header">

            <span style="font-size:24px;">
                ✓
            </span>

            <span class="panel-number">
                2
            </span>

            <span>
                Recognition Result
            </span>

        </div>


        <div class="result-introduction">

            The system has recognized
            the following:

        </div>
        """
    )


    prediction = st.session_state.prediction


    if prediction is None:

        render_html(
            """
            <div class="result-placeholder">

                <div>

                    <div class="placeholder-icon">
                        ✍️
                    </div>

                    <strong>
                        No recognition result yet
                    </strong>

                    <br><br>

                    Upload a handwritten Bangla
                    Upazila–District pair image
                    and click

                    <br>

                    <strong>
                        Recognize Name
                    </strong>.

                </div>

            </div>
            """
        )


    else:

        upazila = prediction["upazila"]

        district = prediction["district"]

        confidence = float(
            prediction["confidence"]
        )

        model_name = prediction["model"]


        render_html(
            f"""
            <div class="result-card top">

                <div class="result-icon green">
                    🏛
                </div>

                <div>

                    <div class="result-label">
                        Upazila
                    </div>

                    <div class="bangla-result green">
                        {upazila}
                    </div>

                </div>

            </div>


            <div class="result-card bottom">

                <div class="result-icon red">
                    📍
                </div>

                <div>

                    <div class="result-label">
                        District
                    </div>

                    <div class="bangla-result red">
                        {district}
                    </div>

                </div>

            </div>


            <div class="confidence-heading">

                <div class="confidence-label">
                    Confidence Score
                </div>

                <div class="confidence-value">
                    {confidence:.1f}%
                </div>

            </div>


            <div class="confidence-track">

                <div
                    class="confidence-fill"
                    style="width: {confidence}%;"
                ></div>

            </div>


            <div class="model-box">

                <div class="model-shield">
                    ✓
                </div>

                <div>

                    This recognition is generated
                    by the

                    <strong>
                        Ensemble Learning model
                    </strong>.

                    <br>

                    <strong>
                        Model:
                    </strong>

                    {model_name}

                </div>

            </div>
            """
        )


# =========================================================
# ABOUT SECTION
# =========================================================

current_year = datetime.now().year


render_html(
    f"""
    <section class="about-section">

        <div class="about-grid">


            <div class="about-content">

                <div class="about-title">
                    ❈ About This Research
                </div>

                <div class="about-line"></div>

                <p>

                    This system recognizes
                    handwritten Bangla
                    Upazila–District pair names
                    using an

                    <strong>
                        Ensemble Learning
                    </strong>

                    approach.

                    <br><br>

                    The proposed framework combines
                    image processing, deep learning
                    and machine learning techniques
                    for accurate administrative
                    name recognition.

                </p>

            </div>


            <div class="memorial-section">

                <img
                    src="{memorial_uri}"
                    class="memorial-image"
                    alt="National Martyrs Memorial with Bangladesh flag"
                >

            </div>


            <div class="features-box">

                <div class="features-title">
                    ❈ Key Features
                </div>


                <div class="feature-item">

                    <span class="feature-icon">
                        ✎
                    </span>

                    <span>
                        Handwritten Image Recognition
                    </span>

                </div>


                <div class="feature-item">

                    <span class="feature-icon">
                        ●
                    </span>

                    <span>
                        Upazila–District Pair Extraction
                    </span>

                </div>


                <div class="feature-item">

                    <span class="feature-icon">
                        ↗
                    </span>

                    <span>
                        High Accuracy with Ensemble Model
                    </span>

                </div>


                <div class="feature-item">

                    <span class="feature-icon">
                        ◆
                    </span>

                    <span>
                        Research-Based AI System
                    </span>

                </div>

            </div>

        </div>


        <div class="custom-footer">

            ❈ &nbsp;

            © {current_year} |

            Bangla Handwritten
            Upazila–District Pair Name
            Recognition System

            &nbsp; ❈

        </div>

    </section>
    """
)
