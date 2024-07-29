import numpy as np
import cv2 as cv
from random import random, randint, uniform
from scipy.ndimage import zoom


def _rotate_img(img, angle):
    '''
    Rotates img by angle
    :param img: Image to be rotated
    :param angle: Extent of rotation
    :return: Rotated image
    '''
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    angle = angle % 360
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    img = cv.warpAffine(img, M, (w, h))
    if len(img.shape) == 2:
        img = img.reshape((*img.shape, 1))
    return img


def _flip_img(img, label):
    '''
    Flips img horizontally and vertically
    :param img: Image to be flipped
    :param label: Has to be adjusted to meet flip
    :return: List of tuples containing flipped image and new label
    '''
    if len(img.shape) == 3:
        img = img.reshape(img.shape[1], img.shape[2])
    hor = np.flip(img, 0).reshape(1, img.shape[0], img.shape[1])
    ver = np.flip(img, 1).reshape(1, img.shape[0], img.shape[1])

    label_hor = abs(label - 360) % 360
    label_ver = (abs((label - 90 % 360) - 360) + 90) % 360
    flip_data = [(hor, np.int32(label_hor)), (ver, np.int32(label_ver))]
    return flip_data


def _shift_img(img, tx, ty):
    '''
    Shifts img by tx and ty
    :param img: Image to be shifted
    :param tx: Shift in x dim
    :param ty: Shift in y dim
    :return: Shifted image
    '''
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv.warpAffine(img, M, (h, w))
    if len(img.shape) == 2:
        img = img.reshape((*img.shape, 1))
    return img


def _zoom_img(img, zoom_factor):
    '''
    Zooms img by zoom_factor
    :param img: Image to be zoomed
    :param zoom_factor: Factor to zoom, can be any real number
    :return: Zoomed image
    '''
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple)

    elif zoom_factor > 1:
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple)
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]
    else:
        out = img
    return out

def augment_images(images,
                   gt,
                   rotate_angle=None,
                   rotate_90degree=None,
                   relative_shift=None,
                   zoom_factor=None,
                   flip=False,
                   bg_fill=False,
                   _augment=0.5):
    '''
    The function makes various changes to images and multiplies them in the process.
    :param imgs: Dataset to be processed
    :param gt: GT of Dataset has to be processed
    :param rotate_angle: Extend of the rotation
    :param rotate_90degree: If True rotates every image 3 times by 90 degree
    :param relative_shift: Extend of the shift
    :param zoom_factor: Decides extend of zoom (positive real number, sign will be randomly chosen)
    :param flip: if True images will be multiplied by flipping them horizontally and vertically
    :param: bg_fill: If True zero padding of images post rotation, zoom or shift will be replaced by mean value of image
    :param _augment: Proportion of images that are processed (real number in [0, 1])
    :return: Returns processed dataset
    '''
    if len(images.shape) < 4:
        images = images.reshape((*images.shape, 1))

    if rotate_90degree:
        new_images = []
        new_gt = []
        for idx, img in enumerate(images):
            new_images.append(img)
            new_gt.append(gt[idx])
            for i in range(1, 4):
                angle = i * 90
                new_img = _rotate_img(img, angle)
                new_images.append(new_img)
                new_gt.append((gt[idx] + angle) % 360)
        images = np.array(new_images)
        gt = np.array(new_gt)
    if flip:
        new_images = []
        new_gt = []
        for idx, img in enumerate(images):
            new_images.append(img)
            new_gt.append(gt[idx])

            hor = np.flip(img, 0).reshape(1, img.shape[0], img.shape[1])[0]
            hor = hor.reshape((*hor.shape, 1))
            gt_hor = abs(gt[idx] - 360) % 360
            ver = np.flip(img, 1).reshape(1, img.shape[0], img.shape[1])[0]
            ver = ver.reshape((*ver.shape, 1))
            gt_ver = (abs((gt[idx] - 90 % 360) - 360) + 90) % 360

            new_images.append(hor)
            new_gt.append(gt_hor)
            new_images.append(ver)
            new_gt.append(gt_ver)
        images = np.array(new_images)
        gt = np.array(new_gt)
    if zoom_factor:
        new_images = []
        new_gt = []
        for idx, img in enumerate(images):
            new_images.append(img)
            new_gt.append(gt[idx])
            if random() < _augment:
                sign = -1 if randint(1, 2) == 1 else 1
                new_img = _zoom_img(img, 1 + sign * zoom_factor)
                if bg_fill:
                    _mean = img.mean()
                    mask = np.zeros((new_img.shape[0] + 2, new_img.shape[1] + 2), dtype=np.uint8)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(0, 0), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(0, new_img.shape[1] - 1), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(new_img.shape[0] - 1, 0), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask,
                                                    seedPoint=(new_img.shape[0] - 1, new_img.shape[1] - 1),
                                                    newVal=_mean)
                new_images.append(new_img)
                new_gt.append(gt[idx])
        images = np.array(new_images)
        gt = np.array(new_gt)
    if rotate_angle:
        new_images = []
        new_gt = []
        for idx, img in enumerate(images):
            new_images.append(img)
            new_gt.append(gt[idx])
            if random() < _augment:
                angle = randint(0, rotate_angle)
                new_img = _rotate_img(img, angle)
                if bg_fill:
                    _mean = img.mean()
                    mask = np.zeros((new_img.shape[0] + 2, new_img.shape[1] + 2), dtype=np.uint8)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(0, 0), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(0, new_img.shape[1] - 1), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(new_img.shape[0] - 1, 0), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask,
                                                    seedPoint=(new_img.shape[0] - 1, new_img.shape[1] - 1),
                                                    newVal=_mean)
                new_images.append(new_img)
                new_gt.append((gt[idx] + angle) % 360)
        images = np.array(new_images)
        gt = np.array(new_gt)
    if relative_shift:
        new_images = []
        new_gt = []
        for idx, img in enumerate(images):
            new_images.append(img)
            new_gt.append(gt[idx])
            if random() < _augment:
                (h, w) = img.shape[:2]
                sign1, sign2 = -1 if randint(1, 2) == 1 else 1, -1 if randint(1, 2) == 1 else 1
                tx, ty = sign1 * (w * uniform(0, relative_shift)), sign2 * (h * uniform(0, relative_shift))
                new_img = _shift_img(img, tx, ty)
                if bg_fill:
                    _mean = img.mean()
                    mask = np.zeros((new_img.shape[0] + 2, new_img.shape[1] + 2), dtype=np.uint8)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(0, 0), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(0, new_img.shape[1] - 1), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask, seedPoint=(new_img.shape[0] - 1, 0), newVal=_mean)
                    _, new_img, _, _ = cv.floodFill(new_img, mask,
                                                    seedPoint=(new_img.shape[0] - 1, new_img.shape[1] - 1),
                                                    newVal=_mean)
                new_images.append(new_img)
                new_gt.append(gt[idx])
        images = np.array(new_images)
        gt = np.array(new_gt)
    return images, gt

def TTA(X, y, n=4, rotations=None):
    '''
    Test-time augmentation (TTA)
    :param X: Images
    :param y: GT angles
    :param n: Number of Rotations
    :param rotations: Rotations to apply (if None, rotations will be random, default=None)
    :return: Rotated images, new gt angles, applied rotations
    '''
    new_X, new_y = np.zeros((n * len(X), *X.shape[1:])), np.zeros(n * len(y))
    applied_rotations = []
    idx = 0
    for img, gt in zip(X, y):
        new_X[n * idx] = img
        new_y[n * idx] = gt
        applied_rotations.append(0)

        for i in range(1, n):
            degree = rotations[i - 1] if rotations else randint(0, 360)
            applied_rotations.append(degree)

            img_rotated = _rotate_img(img, degree)
            gt_rotated = (gt + degree) % 360
            new_X[n * idx + i] = img_rotated
            new_y[n * idx + i] = gt_rotated

        idx += 1
    return new_X, new_y, np.array(applied_rotations)
