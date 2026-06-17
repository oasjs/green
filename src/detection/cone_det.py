# =============================================================================
# Cone Detection Pipeline — Formula Student Classic CV
# =============================================================================
#
# Main References:
# Libs:
#   https://docs.opencv.org/3.4/
#   https://numpy.org/doc/
#   https://learnopencv.com/
# 
# Online articles:
#   https://www.geeksforgeeks.org/computer-vision/image-processing-algorithms-in-computer-vision/
#   https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
#   https://www.geeksforgeeks.org/computer-vision/comprehensive-guide-to-edge-detection-algorithms/
#   https://www.geeksforgeeks.org/computer-graphics/hsv-color-model-in-computer-graphics/
#   https://www.geeksforgeeks.org/python/filter-color-with-opencv/
#   https://github.com/computervisionpro/yt/blob/main/hsv_masking/color-nemo.py
#   https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/
#   https://learnopencv.com/contour-detection-using-opencv-python-c/
#   https://www.geeksforgeeks.org/python/find-co-ordinates-of-contours-using-opencv-python/
#   https://docs.opencv.org/3.4.20/d3/dc1/tutorial_basic_linear_transform.html
#   https://learnopengl.com/Advanced-Lighting/Gamma-Correction
#   https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
# 
# Online videos:
#   https://youtu.be/Wl11eloYVm8
#
# MOD 1: Cap. 1.2.1 - morphological operations
# MOD 1: Cap. 1.2.2 - cone border/geometry identification
# MOD 1: Cap  1.3.1 - HSV filter normalisation
#
# Secondary References:
#   https://github.com/HongchenNa/ImageLuminanceAnalyzer
#   https://revisitingmiwb.github.io/
#   https://github.com/PooyaNasiri/Object_Detection-OpenCV
#   VÖDISCH, N.; DODEL, D.; SCHÖTZ, M. FSOCO: The Formula Student Objects in
#   Context Dataset. SAE International Journal of Connected and Automated
#   Vehicles, v. 5, n. 1, 2022. DOI: 10.4271/12-05-01-0003.
# =============================================================================

# =============================================================================
# ALTERNATIVE METHOD: finding the middle stripes of the cones and getting their information from it
# This would be done finding white areas between blue/yellow/orange areas (white-blue and blue-white borders)
# For this, maybe a dilating method would be helpful for getting the blue/yellow/orange to be uniform for the cone.
#
# The saturation and luminosity problems are still present in this method, making it necessary for
# the increase in the white balance or any other processing method that high lights the white/black
# of the middle stripes.
# =============================================================================

from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

# HSV colour ranges  (OpenCV scale: H 0-179, S 0-255, V 0-255)
YELLOW_LOW  = np.array([ 18, 120,  80])
YELLOW_HIGH = np.array([ 35, 255, 255])

BLUE_LOW    = np.array([ 95, 120,  50])
BLUE_HIGH   = np.array([135, 255, 255])

# Orange wraps around H=0 in OpenCV → two sub-ranges merged later
ORANGE_LOW_A  = np.array([  0, 150,  80])
ORANGE_HIGH_A = np.array([  8, 255, 255])
ORANGE_LOW_B  = np.array([170, 150,  80])
ORANGE_HIGH_B = np.array([179, 255, 255])

# Minimum solidity per colour (blue/yellow have stripes that fragment the blob)
SOLIDITY_MIN = {
    "yellow": 0.55,
    "blue":   0.50,
    "orange": 0.75,
}

PATH_TO_IMAGES = '../../assets/images/'

# Not used yet
def find_middle_stripes(img: cv2.typing.MatLike, low_color: np.typing.ArrayLike, high_color: np.typing.ArrayLike) -> cv2.typing.MatLike:
    pass

# Not used yet
# Put here to think about using segmentation methods to get more uniform colors while
# filtering the color of the cone (due to variations in luminosity)
def uniform_colors(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    pass

def search_upwards_conic_structure(img: cv2.typing.MatLike, contours: cv2.typing.MatLike) -> cv2.typing.MatLike:
    pass

# Utilities
def display_image(title: str, image: cv2.typing.MatLike) -> None:
    # Show an image and wait for a key press
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Saves image on permanent storage
def save_image(title: str, image: cv2.typing.MatLike) -> None:
    cv2.imwrite(str(title + ".jpg"), image)


def plot_hsv(img_rgb: cv2.typing.MatLike, hsv: cv2.typing.MatLike) -> None:
    # Scatter-plot every pixel in the HSV space, coloured by its RGB value
    pixel_colors = img_rgb.reshape((-1, 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(h.flatten(), s.flatten(), v.flatten(),
               facecolors=pixel_colors, marker=".")
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    plt.show()

# Pre-processing
def percentile_stretch(img_bgr: cv2.typing.MatLike,
                        percentile: float = 95) -> cv2.typing.MatLike:
    
    # White-balance via per-channel percentile stretch.
    # Keeps hue stable across different times of day / sky conditions.
    # Must be applied BEFORE HSV conversion.
    
    img = img_bgr.astype(np.float32)
    for i in range(3):
        p = np.percentile(img[:, :, i], percentile)
        if p > 0:
            img[:, :, i] = img[:, :, i] * (255.0 / p)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_clahe_contrast(hsv: cv2.typing.MatLike,
                          clip_limit: float = 2.0,
                          tile_grid: tuple[int, int] = (8, 8)
                          ) -> cv2.typing.MatLike:
    
    # Apply CLAHE only to the V (brightness) channel of an HSV image.
    # Normalises local contrast without distorting H or S.
    # FIX: previously received BGR by mistake; now correctly receives HSV.
    # FIX: preview is converted back to BGR before display.
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    # Convert to BGR only for the visual preview — do not alter the return value
    preview = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    display_image("CLAHE preview", preview)
    save_image("CLAHE_preview", preview)

    return hsv

# Applies gamma correction to the image
def apply_gamma_correction(src, gamma) -> cv2.typing.MatLike:
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def preprocess(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    
    # Full pre-processing chain:
    #   1. White balance (percentile stretch) — operates in BGR
    #   2. BGR → HSV conversion
    #   3. CLAHE on V channel — operates in HSV
    # Returns an HSV image ready for colour segmentation.
    # FIX: apply_clahe_contrast now receives hsv, not img.
    
    # Gamma value is arbitrary -> a parameter to be tested
    gamma = 1
    
    img = percentile_stretch(img)
    # display_image("Percentile", img)
    img = apply_gamma_correction(img, gamma)
    display_image("Gamma corrected", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # FIX: was apply_clahe_contrast(img)
    # hsv = apply_clahe_contrast(hsv)
    display_image("Pre-processed", hsv)
    return hsv

# Colour segmentation
def build_masks(hsv: cv2.typing.MatLike
                ) -> dict[str, cv2.typing.MatLike]:
    
    # Build one binary mask per cone colour.
    # Orange uses two sub-ranges because its hue wraps around H=0 in OpenCV.
    # Returns a dict with keys 'yellow', 'blue', 'orange'.
    
    mask_yellow = cv2.inRange(hsv, YELLOW_LOW,  YELLOW_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW,    BLUE_HIGH)

    mask_or_a   = cv2.inRange(hsv, ORANGE_LOW_A, ORANGE_HIGH_A)
    mask_or_b   = cv2.inRange(hsv, ORANGE_LOW_B, ORANGE_HIGH_B)
    mask_orange = cv2.bitwise_or(mask_or_a, mask_or_b)

    display_image("yellow_mask", mask_yellow)
    display_image("blue_mask", mask_blue)

    return {
        "yellow": mask_yellow,
        "blue":   mask_blue,
        "orange": mask_orange,
    }

# Morphological cleaning
def clean_mask(mask: cv2.typing.MatLike,
               color: str = "orange") -> cv2.typing.MatLike:
    
    # Remove noise and close stripe-induced gaps in the binary mask.

    # Blue and yellow cones have white/black stripes that fragment the blob.
    # A larger closing kernel (7×7) is used for those colours so the gaps
    # between colour regions are bridged before contour extraction.

    # Pipeline:
    #   1. Opening  (3×3) — removes isolated noise pixels
    #   2. Closing  (5×5 or 7×7) — closes internal holes / stripe gaps
    #   3. Dilation (3×3, 1 iter) — softly reconnects near-separated regions
    
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 10))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

    # Opening: kill small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

    # Closing: bridge stripe gaps — larger kernel for striped cones
    # k_close = k7 if color in ("yellow", "blue") else k5
    k_close = k7
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Mild dilation to reconnect nearby fragments
    mask = cv2.dilate(mask, k3, iterations=1)
    
    display_image("mascara", mask)

    return mask

# Filters the luminosity out of the image, keeping the correct color
# PROBABLY DOES NOT WORK THIS WAY -> to be experimented
def apply_luminosity_filter(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    
    return img

# Contour extraction
def find_cone_contours(mask: cv2.typing.MatLike) -> list:
    
    # Extract external contours from a binary mask.

    # FIX: removed the erroneous cvtColor(BGR2GRAY) call — the mask is already
    #      single-channel binary after cv2.inRange + morphological ops.
    # FIX: removed the unnecessary threshold step for the same reason.
    # FIX: switched from RETR_TREE to RETR_EXTERNAL so stripe holes are not
    #      returned as child contours.
    # FIX: switched from CHAIN_APPROX_NONE to CHAIN_APPROX_SIMPLE to reduce
    #      the number of points and speed up downstream processing.
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = mask
    
    for c in contours:
        cv2.drawContours(contours_img, c, -1, 90, 5)
    
    display_image("contours", contours_img)
    
    return list(contours)

# Contour filtering
def filter_contours_geometry(contours: list, img_height: int, color: str) -> list[dict]:
    
    # Reject contours that are geometrically incompatible with a cone silhouette.

    # Filters applied in order (cheapest first):
    #   1. Area bounds      — removes dust and full-frame blobs
    #   2. Aspect ratio     — cones are taller than wide (w/h < 1.2)
    #   3. Solidity         — ratio of contour area to convex-hull area;
    #                         threshold is looser for striped cones (blue/yellow)
    #   4. Extent           — ratio of contour area to bounding-box area;
    #                         rejects sparse/thin shapes

    # Returns a list of dicts with geometry features for downstream stages.
    
    sol_min = SOLIDITY_MIN.get(color, 0.65)
    candidates = []

    for cnt in contours:

        # # 1. Area
        # area = cv2.contourArea(cnt)
        # if area < 80 or area > 80_000:
        #     continue

        # # 2. Aspect ratio
        # x, y, w, h = cv2.boundingRect(cnt)
        # if h == 0:
            # continue
        # aspect = w / h
        # if aspect < 0.2 or aspect > 1.2:
        #     continue

        # # 3. Solidity (via convex hull)
        # hull = cv2.convexHull(cnt)
        # hull_area = cv2.contourArea(hull)
        # if hull_area == 0:
        #     continue
        # solidity = area / hull_area
        # if solidity < sol_min:
        #     continue

        # # 4. Extent
        # extent = area / (w * h)
        # if extent < 0.30:
        #     continue

        # candidates.append({
        #     "contour":  cnt,
        #     "hull":     hull,
        #     "bbox":     (x, y, w, h),
        #     "area":     area,
        #     "solidity": solidity,
        #     "aspect":   aspect,
        #     "extent":   extent,
        # })
        a = 0

    return candidates


# Shape validation (approxPolyDP)
def validate_cone_shape(candidate: dict) -> bool:
    # Use the Douglas-Peucker algorithm (cv2.approxPolyDP) to check whether
    # the convex hull of the candidate approximates a triangular silhouette.
    
    # The hull is used instead of the raw contour so that stripe-induced
    # concavities do not inflate the vertex count.
    
    # A cone should simplify to 3–6 vertices at epsilon = 4% of perimeter.
    # More vertices -> jagged / non-cone shape.
    
    hull = candidate["hull"]
    perimeter = cv2.arcLength(hull, closed=True)
    if perimeter == 0:
        return False

    epsilon = 0.04 * perimeter
    approx  = cv2.approxPolyDP(hull, epsilon, closed=True)
    n = len(approx)

    candidate["approx"] = approx
    candidate["n_vertices"] = n

    return 3 <= n <= 6

# # Stripe verification (uses markings as positive evidence)
# def has_stripe(bbox: tuple[int, int, int, int],
#                hsv_frame: cv2.typing.MatLike,
#                color: str) -> bool:

#     # Check whether the ROI contains the characteristic stripe of the cone:
#     #   - Blue cone  → white stripe  (low S, high V)
#     #   - Yellow cone → black stripe (low V)
#     #   - Orange cone → no stripe   (always returns True)
    
#     # Converts a limitation (fragmented mask) into positive evidence, reducing
#     # false positives from other blue/yellow objects in the scene.
    
#     if color == "orange":
#         return True

#     x, y, w, h = bbox
#     roi = hsv_frame[y:y + h, x:x + w]

#     if color == "blue":
#         # White pixels: very low saturation, high brightness
#         stripe_mask = cv2.inRange(roi,
#                                   np.array([  0,   0, 180]),
#                                   np.array([179,  50, 255]))
#     else:  # yellow
#         # Black pixels: very low brightness
#         stripe_mask = cv2.inRange(roi,
#                                   np.array([  0,   0,   0]),
#                                   np.array([179, 255,  60]))
    
#     display_image("stripe", stripe_mask)

#     return int(stripe_mask.sum()) > 100

# Size discrimination (orange only)
def classify_orange_size(candidate: dict,
                          small_max_height: int = 60) -> str:
    
    # Separate small orange cones from big orange cones using bounding-box
    # height in pixels.

    # 'small_max_height' is camera- and resolution-dependent and must be
    # calibrated empirically. A good starting point for 640×480 is 60 px.
    _, _, _, h = candidate["bbox"]
    return "orange_small" if h <= small_max_height else "orange_big"

# Visualisation helper
def draw_detections(img: cv2.typing.MatLike,
                    results: dict[str, list[dict]]) -> cv2.typing.MatLike:
     
    # Draw bounding boxes and labels on a copy of the original image.
    # Colours match the cone type for quick visual verification.
    COLOR_BGR = {
        "yellow":       (  0, 220, 220),
        "blue":         (220,  80,   0),
        "orange":       (  0, 140, 255),
        "orange_small": (  0, 140, 255),
        "orange_big":   (  0,  60, 200),
    }

    out = img.copy()
    for label, candidates in results.items():
        bgr = COLOR_BGR.get(label, (200, 200, 200))
        for c in candidates:
            x, y, w, h = c["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(out, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1,
                        cv2.LINE_AA)
            # Draw approx polygon if available
            if "approx" in c:
                cv2.polylines(out, [c["approx"]], True, bgr, 1, cv2.LINE_AA)

    return out

# Main pipeline
def detect_cones(img: cv2.typing.MatLike,
                 hsv: cv2.typing.MatLike) -> dict[str, list[dict]]:
    
    # Full detection pipeline for a single frame.
    # Returns a dict with one list of candidate dicts per cone class.
    
    masks = build_masks(hsv)

    results: dict[str, list[dict]] = {
        "yellow":       [],
        "blue":         [],
        "orange_small": [],
        "orange_big":   [],
    }

    for color, mask in masks.items():

        # 1. Morphological cleaning
        clean = clean_mask(mask, color)

        # 2. Contour extraction
        contours = find_cone_contours(clean)

        # 3. Geometric filtering
        candidates = filter_contours_geometry(contours, hsv.shape[0], color)

        # 4. Shape validation + stripe check
        valid = []
        for c in candidates:
            if not validate_cone_shape(c):
                continue
            # if not has_stripe(c["bbox"], hsv, color):
            #     continue
            valid.append(c)

        # 5. Size discrimination (orange only) / direct assignment
        if False:
        # if color == "orange":
            for c in valid:
                size_label = classify_orange_size(c)
                results[size_label].append(c)
        else:
            results[color] = valid

    return results


def main() -> None:
    # Load image
    img = cv2.imread(PATH_TO_IMAGES + 'cone.jpg')
    if img is None:
        raise FileNotFoundError(f"Image not found at {PATH_TO_IMAGES}cone.jpg")

    display_image("Original image", img)
    save_image("original_image", img)

    # Pre-processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = preprocess(img)

    # Detection
    results = detect_cones(img, hsv)

    # Summary
    for label, candidates in results.items():
        print(f"[{label}] {len(candidates)} cone(s) detected")

    # Visualisation
    annotated = draw_detections(img, results)
    display_image("Detections", annotated)
    save_image("Detections", annotated)

    print("Done.")


if __name__ == "__main__":
    main()