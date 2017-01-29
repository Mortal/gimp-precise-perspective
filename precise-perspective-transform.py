#!/usr/bin/env python2
# encoding: utf8

'''
Gimp plugin "Precise perspective transform"

Author:
Mathias Rav

License:

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

The GNU Public License is available at
http://www.gnu.org/copyleft/gpl.html
'''

from __future__ import print_function, unicode_literals
import numpy as np
from gimpfu import _, gimp, main, PF_DRAWABLE, PF_IMAGE, register

import gettext
gettext.install('gimp20', gimp.locale_directory, unicode=True)

# img, = gimp.image_list()


def python_fu_precise_perspective_transform(img, drawable):
    if not img.active_vectors:
        raise ValueError('Image must have a path.')

    vec = img.active_vectors

    if len(vec.strokes) != 2:
        raise ValueError(
            'Active image path should have exactly two components.')

    s1, s2 = vec.strokes
    p1, b1 = s1.points
    p2, b2 = s2.points
    if b1 is not False:
        raise ValueError('b1 is not False')
    if b2 is not False:
        raise ValueError('b2 is not False')

    q1 = quad_from_path(p1)
    q2 = quad_from_path(p2)

    x0, y0 = drawable.offsets
    w, h = drawable.width, drawable.height
    x1, y1 = x0 + w, y0
    x2, y2 = x0, y0 + h
    x3, y3 = x1, y2

    local = q1.to_local(((x0, x1, x2, x3), (y0, y1, y2, y3)))
    target = q2.to_world(local)
    (x0, x1, x2, x3), (y0, y1, y2, y3) = target.tolist()
    transform_perspective(drawable, x0, y0, x1, y1, x2, y2, x3, y3)


(GIMP_TRANSFORM_RESIZE_ADJUST,
 GIMP_TRANSFORM_RESIZE_CLIP,
 GIMP_TRANSFORM_RESIZE_CROP,
 GIMP_TRANSFORM_RESIZE_CROP_WITH_ASPECT) = range(4)

(GIMP_INTERPOLATION_NONE,
 GIMP_INTERPOLATION_LINEAR,
 GIMP_INTERPOLATION_CUBIC,
 GIMP_INTERPOLATION_LANCZOS) = range(4)

GIMP_TRANSFORM_FORWARD, GIMP_TRANSFORM_BACKWARD = range(2)


def transform_perspective(drawable, x0, y0, x1, y1, x2, y2, x3, y3,
                          transform_direction=GIMP_TRANSFORM_FORWARD,
                          interpolation=GIMP_INTERPOLATION_CUBIC,
                          supersample=False, recursion_level=3,
                          clip_result=GIMP_TRANSFORM_RESIZE_ADJUST):
    '''
    x0: The new x coordinate of upper-left corner of original bounding box.
    y0: The new y coordinate of upper-left corner of original bounding box.
    x1: The new x coordinate of upper-right corner of original bounding box.
    y1: The new y coordinate of upper-right corner of original bounding box.
    x2: The new x coordinate of lower-left corner of original bounding box.
    y2: The new y coordinate of lower-left corner of original bounding box.
    x3: The new x coordinate of lower-right corner of original bounding box.
    y3: The new y coordinate of lower-right corner of original bounding box.
    transform_direction: Direction of transformation.
    interpolation: Type of interpolation.
    supersample:
        This parameter is ignored, supersampling is performed based on the
        interpolation type.
    recursion_level:
        Maximum recursion level used for supersampling (3 is a nice value).
    clip_result: How to clip results.

    Returns: The newly mapped drawable.
    '''
    drawable.transform_perspective(
        x0, y0, x1, y1, x2, y2, x3, y3, transform_direction,
        interpolation, int(supersample), recursion_level, clip_result)


def _coeff_property(i, j):
    def fget(self):
        return self.A[i, j]

    def fset(self, v):
        self.A[i, j] = v

    return property(
        fget, fset, None,
        'The (%d, %d)-entry of the matrix' % (i, j))


def _coeff_properties(n, m):
    return tuple(_coeff_property(i, j) for i in range(n) for j in range(m))


class Quadrilateral(object):
    '''
    Transformation between world coordinates (R x R)
    and quadrilateral-local coordinates ([0, 1] x [0, 1]).
    Quadrilateral corners are x0,y0 to x3,y3 in the world,
    and the transformation of (u, v) in the unit square
    to world coordinates is (x'/w, y'/w) given by
    [x']   [a  b  c] [u]
    [y'] = [d  e  f] [v]
    [w ]   [g  h  i] [1].

    Based on "Projective Mappings for Image Warping" by Paul Heckbert, 1999.
    '''

    def __init__(self, xy):
        xy = np.asarray(xy)
        if xy.shape != (2, 4):
            raise TypeError(
                'xy must be (2, 4) with corners in columns, not %r' %
                (xy.shape,))
        x, y = xy
        self.A = np.eye(3)
        upper_left, upper_right, lower_right, lower_left = xy.T
        d1 = upper_right - lower_right
        d2 = lower_left - lower_right
        s = upper_left - upper_right + lower_right - lower_left

        self.c, self.f = upper_left
        self.i = 1

        if (s ** 2).sum() < 1e-6:
            # Parallelogram
            self.a = x[1] - x[0]
            self.b = x[2] - x[1]
            self.d = y[1] - y[0]
            self.e = y[2] - y[1]
            self.g = self.h = 0
        else:
            g1 = s[0] * d2[1] - d2[0] * s[1]
            h1 = d1[0] * s[1] - s[0] * d1[1]
            den = d1[0] * d2[1] - d2[0] * d1[1]
            self.g = g1 / den
            self.h = h1 / den
            self.a = x[1] - x[0] + self.g * x[1]
            self.b = x[3] - x[0] + self.h * x[3]
            self.d = y[1] - y[0] + self.g * y[1]
            self.e = y[3] - y[0] + self.h * y[3]

        self.A_inv = np.linalg.inv(self.A)

    def arg(self):
        return self.to_world(
            [[0, 1, 1, 0], [0, 0, 1, 1]])

    (a, b, c,
     d, e, f,
     g, h, i) = _coeff_properties(3, 3)

    @staticmethod
    def _projective_transform(A, x):
        x = np.asarray(x)
        if x.shape[0] != 2 or x.ndim != 2:
            raise TypeError(
                'data matrix must have 2 rows; invalid shape is %r'
                % (x.shape,))
        x1 = np.asarray((x[0], x[1], np.ones_like(x[0])))
        Ax = np.dot(A, x1)
        res_x = Ax[0] / Ax[2]
        res_y = Ax[1] / Ax[2]
        return np.asarray((res_x, res_y))

    def to_world(self, uv):
        '''
        Transform columns of uv in local space to columns of result in world.
        '''
        return self._projective_transform(self.A, uv)

    def to_local(self, xy):
        '''
        Transform columns of xy in world space to columns of result in local.
        '''
        return self._projective_transform(self.A_inv, xy)

    def suggested_size(self):
        '''Compute max (horizontal, vertical) side length as (w, h)-pair'''
        upper_left, upper_right, lower_right, lower_left = self.arg().T

        def dsq(p, q):
            return ((p - q) ** 2).sum()

        width = np.sqrt(max(dsq(upper_left, upper_right),
                            dsq(lower_left, lower_right)))
        height = np.sqrt(max(dsq(upper_left, lower_left),
                             dsq(upper_right, lower_right)))
        return (width, height)


def quad_from_path(path_data):
    xs, xs_, xs__ = path_data[::6], path_data[2::6], path_data[4::6]
    if not (xs == xs_ == xs__):
        raise ValueError('Path is not polygonal in x')
    ys, ys_, ys__ = path_data[1::6], path_data[3::6], path_data[5::6]
    if not (ys == ys_ == ys__):
        raise ValueError('Path is not polygonal in y')
    if not (len(xs) == len(ys) == 4):
        raise ValueError('Path must have exactly four vertices')
    return Quadrilateral((xs, ys))


if __name__ == '__main__':
    register(
        'python-fu-precise-perspective-transform',  # Function name
        '',  # Blurb / description
        _('Run perspective transform from two quads'),  # Help
        'Mathias Rav',  # Author
        '2017 Mathias Rav',  # Copyright notice
        '2017 Jan 29',  # Date
        _('Precise perspective transform'),  # Menu label
        'RGB*,GRAY*',
        [
            (PF_IMAGE,    'img',      _('Input image'),    None),
            (PF_DRAWABLE, 'drawable', _('Input drawable'), None),
        ],
        [],  # No results
        python_fu_precise_perspective_transform,  # Internal function name
        menu='<Image>/Filters/Distorts',  # Register in menu
        domain=('gimp20-template', gimp.locale_directory),
    )

    main()
