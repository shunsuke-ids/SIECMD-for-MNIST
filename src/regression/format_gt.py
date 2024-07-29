import numpy as np
from math import pi, acos, cos, sin
from scipy.spatial.distance import euclidean as euclidean_distance


def angles_2_unit_circle_points(angles):
    '''
    Converts angles to 2D points on unit circle
    :param angles: Array of angles
    :return: Array of 2D points
    '''
    points = []
    for a in angles:
        points.append(angle_2_unit_circle_point(a))
    return np.asarray(points)


def angle_2_unit_circle_point(a):
    '''
    Converts angle to a 2D point on unit circle
    :param a: Angle
    :return: 2D point
    '''
    return (cos(a * pi / 180), sin(a * pi / 180))


def angles_2_pseudo_circle_points(angles, d):
    '''
    Converts angles to 2D points on a pseudo circle
    :param angles: Array of angles
    :param d: Determines circle $x^d + y^d = 1$ in case d=1 it is $|x| + |y| = 1$
    :return: Array of 2D points
    '''
    points = []
    for a in angles:
        points.append(angle_2_pseudo_circle_point(a, d))
    return np.asarray(points)


def angle_2_pseudo_circle_point(a, d):
    '''
    Converts angle to a 2D point on a pseudo circle
    :param a: Angle
    :param d: Determines circle $x^d + y^d = 1$ in case d=1 it is $|x| + |y| = 1$ (default d=2 unit circle)
    :return: 2D point
    '''
    point = angle_2_unit_circle_point(a)
    point = associated_point_on_circle(*point, d)
    return point


def associated_points_on_circle(points, d=2):
    '''
    Returns associated 2D points on pseudo-circle with minimal distance to input points
    :param points: Array of 2D points
    :param d: Determines circle $x^d + y^d = 1$ in case d=1 it is $|x| + |y| = 1$ (default d=2 unit circle)
    :return: Array of 2D points
    '''
    _points = []
    for p in points:
        x, y = associated_point_on_circle(*p, d)
        # corrects floating point inaccuracies
        if x > 1:
            x, y = 1, 0
        if x < -1:
            x, y = -1, 0
        if y > 1:
            x, y = 0, 1
        if y < -1:
            x, y = 0, -1
        _points.append((x, y))
    return np.asarray(_points)


def associated_point_on_circle(x, y, d=2):
    '''
    Returns associated 2D point on pseudo-circle with minimal distance to input point (x, y)
    :param x: X coordinate of point
    :param y: Y coordinate of point
    :param d: Determines circle $x^d + y^d = 1$ in case d=1 it is $|x| + |y| = 1$ (default d=2 unit circle)
    :return: 2D point
    '''
    if x == 0 and y == 0: return (1., 0.)
    assert d == 1 or d % 2 == 0
    xsign, ysign = 1, 1
    if d == 1:
        xsign = -1 if x < 0 else 1
        ysign = -1 if y < 0 else 1
        x = abs(x)
        y = abs(y)

    _x = x / (x ** d + y ** d) ** (1 / d)
    _y = (y / x) * _x

    return xsign * _x, ysign * _y


def points_2_angles(points, d=2):
    '''
    Returns angles of points on circle
    :param points: Array of 2D points
    :param d: Determines circle $x^d + y^d = 1$ in case d=1 it is $|x| + |y| = 1$ (default d=2 unit circle)
    :return: Array of angles
    '''

    angles = []
    for p in points:
        angles.append(point_2_angle(*p, d))
    return np.asarray(angles)


def point_2_angle(x, y, d):
    '''
    Returns angle of point on circle
    :param x: X coordinate of point
    :param y: Y coordinate of point
    :return: Angle
    '''

    if d != 2:
        x, y = associated_point_on_circle(x, y, d)

    if y == 0:
        radian = 0 if x == 1 else pi
    if y < 0:
        a = abs(euclidean_distance((x, y), (1, 0)))
        b, c = 1, 1
        r = (b * b + c * c - a * a) / (2 * b * c)
        r = 1 if r > 1 else r if r > -1 else -1
        radian = acos(r)
        radian = 2 * pi - radian
    if y > 0:
        a, b = 1, 1
        c = abs(euclidean_distance((x, y), (1, 0)))
        r = (a * a + b * b - c * c) / (2 * a * b)
        r = 1 if r > 1 else r if r > -1 else -1
        radian = acos(r)
    return round(radian * 180 / pi)


def angles_to_areas(y, n_areas=4):
    '''
    Convert GT angles to areas for bin regression
    :param y: GT
    :param n_areas: Number of areas (default=4, quadrants)
    :return: Array of corresponding segments of GT angles
    '''
    area_size = 360 // n_areas
    assert isinstance(area_size, int)

    return np.floor_divide(y, area_size, out=np.zeros_like(y)).astype(np.uint8)


def radian_2_degree(radian):
    '''
    Convert radian to degree
    :param radian: Angle in [0, 2 PI)
    :return: Returns angle in [0, 360)
    '''
    degree = radian * 180.0 / np.pi
    return degree


def degree_2_radian(degree):
    '''
    Convert degree to radian
    :param degree: Angle in [0, 360)
    :return: Returns angle in [0, 2 PI)
    '''
    radian = degree * np.pi / 180.0
    return radian

def areas_to_points(samples, n_areas):
    '''
    Converts areas to points
    :param samples: set that represents bins {0, ..., n_areas-1}
    :param n_areas: number of bins
    :return: two dimensional points
    '''
    area_size = 360 / n_areas
    n_samples = len(samples)
    new_samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        angle = area_size * samples[i] + area_size / 2
        new_samples[i] = angle_2_unit_circle_point(angle)
    return new_samples