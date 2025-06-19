import os
import cv2
import numpy as np
from util import load_image, preprocess_image, show_image


def task1(image_path, thresholds=[100, 140, 180], output_dir='results'):
    img = load_image(image_path)
    h, w = img.shape[:2]
    resized = cv2.resize(img, (300, int(h * 300 / w)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    for t in thresholds:
        _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        path = os.path.join(output_dir, f'threshold_{t}.png')
        cv2.imwrite(path, th)
        print(f'Saved threshold image: {path}')
        show_image(f'Threshold {t}', th)


def task2(image_path, threshold_value=140, output_dir='results'):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    modes = dict(EXTERNAL=cv2.RETR_EXTERNAL, TREE=cv2.RETR_TREE, LIST=cv2.RETR_LIST)
    for name, mode in modes.items():
        cnts, _ = cv2.findContours(th.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
        out = img.copy()
        cv2.drawContours(out, cnts, -1, (0,0,255), 2)
        path = os.path.join(output_dir, f'contours_{name}.png')
        cv2.imwrite(path, out)
        print(f'Saved contours_{name}.png: {len(cnts)} contours')
        show_image(f'Contours {name}', out)


def task3(image_path):
    img = load_image(image_path)
    h, w = img.shape[:2]
    for size in (150, 600):
        resized = cv2.resize(img, (size, int(h * size / w)))
        th = preprocess_image(resized)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'Resolution {size}px: {len(cnts)} contours')
        show_image(f'Resz {size}', th)


def task4(image_path, output_dir='results'):
    img = load_image(image_path)
    th = preprocess_image(img)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numbered = img.copy()
    for i, c in enumerate(cnts, 1):
        # numeracja
        x,y,w,h = cv2.boundingRect(c)
        cv2.putText(numbered, str(i), (x+w//2, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    path = os.path.join(output_dir, 'numbered.png')
    cv2.imwrite(path, numbered)
    print(f'Saved numbered image: {path}')
    show_image('Numbered', numbered)
    return cnts


def task5(image_path, output_dir, contours):
    img = load_image(image_path)
    out = img.copy()
    dims = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        dims.append((w,h))
        cv2.rectangle(out, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(out, f'{w}x{h}px', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
    path = os.path.join(output_dir, 'dimensions.png')
    cv2.imwrite(path, out)
    print(f'Saved dimensions image: {path}')
    show_image('Dimensions', out)
    return dims


def task6(image_path, min_area, max_area, output_dir):
    img = load_image(image_path)
    th = preprocess_image(img)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filt = [c for c in cnts if min_area < cv2.contourArea(c) < max_area]
    out = img.copy()
    cv2.drawContours(out, filt, -1, (255,0,255),2)
    path = os.path.join(output_dir, 'filtered.png')
    cv2.imwrite(path, out)
    print(f'Saved filtered image: {path} ({len(filt)} remain)')
    show_image('Filtered', out)
    return filt


def task7(contours):
    import numpy as np
    ws, hs = zip(*(cv2.boundingRect(c)[2:] for c in contours))
    print(f'Liczba kostek: {len(contours)}')
    print(f'Śr. szer.: {np.mean(ws):.1f}px, Śr. wys.: {np.mean(hs):.1f}px')
    print(f'Min rozmiar: {min(ws)}x{min(hs)}px, Max: {max(ws)}x{max(hs)}px')


def task_inspect(image_path, contours, output_dir='results'):
    img = load_image(image_path)
    ratio = 1.0  # assuming same size
    for i, c in enumerate(contours, 1):
        c_float = c.astype('float') * ratio
        c_int = c_float.astype('int')
        x,y,w,h = cv2.boundingRect(c_int)
        # mask
        mask = np.zeros(img.shape[:2], dtype='uint8')
        cv2.drawContours(mask, [c_int], -1, 255, -1)
        segmented = cv2.bitwise_and(img, img, mask=mask)
        roi = segmented[y:y+h, x:x+w]
        # save and show
        m_path = os.path.join(output_dir, f'mask_{i:02d}.png')
        s_path = os.path.join(output_dir, f'segmented_{i:02d}.png')
        r_path = os.path.join(output_dir, f'roi_{i:02d}.png')
        cv2.imwrite(m_path, mask)
        cv2.imwrite(s_path, segmented)
        cv2.imwrite(r_path, roi)
        print(f'Brick {i}: mask->{m_path}, segmented->{s_path}, roi->{r_path}')
        show_image(f'Mask {i}', mask)
        show_image(f'Segmented {i}', segmented)
        show_image(f'ROI {i}', roi)
