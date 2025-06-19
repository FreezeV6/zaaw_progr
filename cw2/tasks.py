import cv2
import imutils
import numpy as np
from util import load_image, show_image

# Paths to images (adjust as needed)
BOTTLE_PATH = 'images/fanta_bottle.png'
LOGO_PATH = 'images/fanta_logo.png'
SCREENSHOT_PATH = 'images/app_screenshot.png'
ICON_PATH = 'images/icon_template.png'
OBJECTS_PATH = 'images/multiple_objects.png'
ONE_OBJECT_PATH = 'images/one_object_template.jpg'

# Helper function for template detection
def detect_template(image, template, method=cv2.TM_CCOEFF_NORMED):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray_img.shape
    h_tmpl, w_tmpl = gray_tmpl.shape
    if h_img < h_tmpl or w_img < w_tmpl:
        raise ValueError(f"Szablon ({w_tmpl}×{h_tmpl}) jest większy niż obraz ({w_img}×{h_img})!")
    res = cv2.matchTemplate(gray_img, gray_tmpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return res, (min_val, max_val, min_loc, max_loc)

# 1. Wykrywanie logo w butelce Fanty
def task_1():
    bottle = load_image(BOTTLE_PATH)
    logo = load_image(LOGO_PATH)
    print("Bottle:", bottle.shape if bottle is not None else "None")
    print("Logo:", logo.shape if logo is not None else "None")
    res, (min_val, max_val, min_loc, max_loc) = detect_template(bottle, logo)
    h, w = logo.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(bottle, top_left, bottom_right, (0, 255, 0), 2)
    show_image('Logo Detection', bottle)
    print(f'Logo coordinates: {top_left}, match value: {max_val}')

# 2. Wrażliwość na obrót obrazu
def task_2():
    for angle in [30, 45]:
        bottle = load_image(BOTTLE_PATH)
        rotated = imutils.rotate(bottle, angle)
        logo = load_image(LOGO_PATH)
        _, (_, max_val, _, max_loc) = detect_template(rotated, logo)
        print(f'Rotation {angle}° -> maxVal: {max_val}, location: {max_loc}')
        show_image(f'Rotated {angle}°', rotated)

# 3. Wrażliwość na skalowanie obrazu
def task_3():
    for scale in [0.5, 1.5]:
        bottle = load_image(BOTTLE_PATH)
        scaled = cv2.resize(bottle, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        logo = load_image(LOGO_PATH)
        _, (_, max_val, _, max_loc) = detect_template(scaled, logo)
        print(f'Scaling {scale}x -> maxVal: {max_val}, location: {max_loc}')
        show_image(f'Scaled {scale}x', scaled)

# 4. Porównanie metod matchTemplate
def task_4():
    methods = {
        'TM_CCOEFF': cv2.TM_CCOEFF,
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
        'TM_CCORR': cv2.TM_CCORR,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
        'TM_SQDIFF': cv2.TM_SQDIFF,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
    }
    bottle = load_image(BOTTLE_PATH)
    logo = load_image(LOGO_PATH)
    for name, method in methods.items():
        res, (min_val, max_val, min_loc, max_loc) = detect_template(bottle, logo, method)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            match_val = min_val
            match_loc = min_loc
        else:
            match_val = max_val
            match_loc = max_loc
        top_left = match_loc
        h, w = logo.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        img_draw = bottle.copy()
        cv2.rectangle(img_draw, top_left, bottom_right, (255, 0, 0), 2)
        show_image(name, img_draw)
        print(f'{name}: match value = {match_val}, location = {match_loc}')

# 5. Detekcja małych ikon interfejsu
def task_5():
    screenshot = load_image(SCREENSHOT_PATH)
    icon = load_image(ICON_PATH)
    _, (_, max_val, _, max_loc) = detect_template(screenshot, icon)
    h, w = icon.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(screenshot, top_left, bottom_right, (0, 255, 255), 2)
    show_image('Icon Detection', screenshot)
    print(f'Icon detected at {top_left} with match value {max_val}')

# 6. Test fałszywych trafień
def task_6():
    objects = load_image(OBJECTS_PATH)
    template = load_image(ONE_OBJECT_PATH)
    res = cv2.matchTemplate(objects, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    img_draw = objects.copy()
    detections = []
    h, w = template.shape[:2]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_draw, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        detections.append(pt)
    show_image('False Hits', img_draw)
    print(f'Detected {len(detections)} occurrences at {detections}')
    print('Consider lowering threshold or using non-maximum suppression to reduce false positives.')

# 7. Połączenie konturów i template matching
def task_7():
    objects = load_image(OBJECTS_PATH)
    template = load_image(ONE_OBJECT_PATH)
    gray = cv2.cvtColor(objects, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = objects.copy()
    h_tmpl, w_tmpl = template.shape[:2]
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        # Skip ROIs smaller than the template
        if w_cnt < w_tmpl or h_cnt < h_tmpl:
            continue
        roi = objects[y:y+h_cnt, x:x+w_cnt]
        try:
            _, (min_val, max_val, _, max_loc) = detect_template(roi, template)
            if max_val > 0.7:
                cv2.rectangle(img_draw, (x, y), (x+w_cnt, y+h_cnt), (0, 255, 0), 2)
        except ValueError:
            continue
    show_image('Contours + Template', img_draw)
    print('Objects similar to template are marked in green.')
