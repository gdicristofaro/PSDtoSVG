'''
* Greg DiCristofaro
* February 18th, 2017
* Ported from
* https://github.com/kilobtye/potrace
* which in turn is ported from:
* https://potrace.sourceforge.net/
* licensed under GPL-3: https://www.gnu.org/licenses/gpl-3.0.en.html
'''

import math


def new_array(n, def_val=0):
    """
    creates a list of length n with the initial value specified
    :param n: number of items in list
    :param def_val: the initial value of every item in list
    :returns: a list of length n with default value of def_val
    """
    return [def_val] * n


class Point:
    """
    defines a point with x,y coordinates
    """

    def __init__(self, x=0, y=0):
        """
        main constructor for a point
        :param x: the x coordinate
        :param y: the y coordinate
        """
        self.x = x
        self.y = y

    def copy(self):
        """
        creates a copy of the current point
        :returns: the copy of this point
        """
        return Point(self.x, self.y)

    def __str__(self):
        return "Point: ({}, {})".format(self.x, self.y)


class Bitmap:
    """
    Defines image data where images are captured as a sequence of
    boolean values.  True indicates that there is picture information
    to be encapsulated at a particular point.
    """

    def __init__(self, w, h, data=None):
        """
        Initializes the bitmap.
        :param w: the width of the bitmap
        :param h: the height of the bitmap
        :param data: the image data as a list of boolean values where true
        indicates that there is image data at the given location.  Point (x,y)
        can be found at index x + w * y in the list
        """
        self.w = int(w)
        self.h = int(h)
        self.size = w * h

        if data is None:
            self.data = new_array(self.size, False)
        else:
            self.data = data

    def at(self, x, y):
        """
        Returns whether or not image data is present at the given location
        :param x: the 0-indexed x location
        :param y: the 0-indexed y location
        :returns: true if there is image data at the given location
        """
        x = int(x)
        y = int(y)
        return (self.w > x >= 0 and self.h > y >= 0 and
                self.data[self.w * y + x])

    def index(self, i):
        """
        Returns the x,y location of the given index
        :param i: the index requested
        :returns: an x,y point of the location
        """
        y = int(i / self.w)
        x = i - y * self.w
        return Point(x, y)

    def flip(self, x, y):
        """
        Switches the boolean value at the given location.
        :param x: the x location
        :param y: the y location
        """
        x = int(x)
        y = int(y)
        if self.at(x, y):
            self.data[self.w * y + x] = False
        else:
            self.data[self.w * y + x] = True

    def copy(self):
        """
        Returns a deep copy of the bitmap data
        :returns: the copy of the bitmap
        """
        return Bitmap(self.w, self.h, self.data[:])


class Path:
    """
    the path structure is filled in with information about a given path
    as it is accumulated and passed through the different stages of the
    Potrace algorithm.
    """
    def __init__(self):
        self.area = 0
        self.len = 0
        self.curve = {}
        self.pt = []
        self.min_x = 100000
        self.min_y = 100000
        self.max_x = -1
        self.max_y = -1


class Curve:
    """
    defines elements of the curve
    Attributes:
        n: number of segments
        tag: tag[n]: is either 'CORNER' or 'CURVE'
        alphacurve: have the following fields been initialized?
        vertex: for CORNER, this equals c[1]
        alpha: only for CURVE
    """
    def __init__(self, n):
        self.n = n
        self.tag = new_array(n)
        self.c = new_array(n * 3)
        self.alpha_curve = 0
        self.vertex = new_array(n)
        self.alpha = new_array(n)
        self.alpha0 = new_array(n)
        self.beta = new_array(n)


def find_next(bm1, point):
    """
    search for next point that has image data or returns None if none left
    :param point: defines point to start search
    :returns: the next point that has image data or None if none found
    """
    i = bm1.w * point.y + point.x
    while i < bm1.size and (not bm1.data[i]):
        i += 1

    if i < bm1.size:
        return bm1.index(i)
    else:
        return None


def majority(bm1, x, y):
    """
    return the "majority" value of bitmap bm at intersection (x,y). We
    assume that the bitmap is balanced at "radius" 1.
    :param bm1: the bitmap to search
    :param x: the x coordinate
    :param y: the y coordinate
    :returns: the "majority" value
    """
    for i in range(2, 5):
        ct = 0
        for a in range(-i + 1, i):
            ct += 1 if bm1.at(x + a, y + i - 1) else -1
            ct += 1 if bm1.at(x + i - 1, y + a - 1) else -1
            ct += 1 if bm1.at(x + a - 1, y - i) else -1
            ct += 1 if bm1.at(x - i, y + a) else -1

        if ct > 0:
            return True
        elif ct < 0:
            return False

    return False


def find_path(bm, bm1, turnpolicy, point):
    """
    Computes a path in the given pixmap, separating black from white.
    Also compute the area enclosed by the path.
    :param bm: original bitmap
    :param bm1: altered bitmap copy
    :param turnpolicy: either 'right', 'black', 'white', 'majority', 'minority'
    :param point: the upper left corner of the path
    :returns: a new path object
    """
    path = Path()
    x = point.x
    y = point.y
    dirx = 0
    diry = 1

    path.sign = '+' if bm.at(point.x, point.y) else '-'
    while True:
        path.pt.append(Point(x, y))
        if x > path.max_x:
            path.max_x = x
        if x < path.min_x:
            path.min_x = x
        if y > path.max_y:
            path.max_y = y
        if y < path.min_y:
            path.min_y = y

        path.len += 1

        x += dirx
        y += diry
        path.area -= x * diry

        if x == point.x and y == point.y:
            break

        left = bm1.at(x + (dirx + diry - 1) / 2, y + (diry - dirx - 1) / 2)
        right = bm1.at(x + (dirx - diry - 1) / 2, y + (diry + dirx - 1) / 2)

        if right and not left:
            if (turnpolicy == "right" or
                (turnpolicy == "black" and path.sign == '+') or
                (turnpolicy == "white" and path.sign == '-') or
                (turnpolicy == "majority" and majority(bm1, x, y)) or
                    (turnpolicy == "minority" and (not majority(bm1, x, y)))):

                tmp = dirx
                dirx = -diry
                diry = tmp
            else:
                tmp = dirx
                dirx = diry
                diry = -tmp
        elif right:
            tmp = dirx
            dirx = -diry
            diry = tmp
        elif not left:
            tmp = dirx
            dirx = diry
            diry = -tmp

    return path


def xor_path(bm1, path):
    """
    xor the given pixmap with the interior of the given path.
    Note: the path must be within the dimensions of the pixmap
    :param bm1: bitmap with pixels to flip
    :param path: the path whose interior will have pixels flipped
    """
    y1 = path.pt[0].y
    length = path.len

    for i in range(1, length):
        x = path.pt[i].x
        y = path.pt[i].y

        if y != y1:
            min_y = y1 if y1 < y else y
            max_x = path.max_x
            for j in range(x, max_x):
                bm1.flip(j, min_y)

            y1 = y


def bm_to_path_list(bm, turnpolicy, turdsize):
    """
    Decompose the given bitmap into paths. Returns a list of
    paths with the fields len, pt, area, sign filled
    in.
    :param bm: the bitmap in which to find paths
    :param turnpolicy: the rule for handling turns
    :param turdsize: paths whose area is less than turdsize are
    ignored
    :returns: a list of paths
    """
    bm1 = bm.copy()
    current_point = Point()

    pathlist = []
    while True:
        current_point = find_next(bm1, current_point)
        if current_point is None:
            break

        path = find_path(bm, bm1, turnpolicy, current_point)
        xor_path(bm1, path)

        if path.area > turdsize:
            pathlist.append(path)

    return pathlist


class Quad:
    """
    the type of (affine) quadratic forms, represented as symmetric 3x3
    matrices.  The value of the quadratic form at a vector (x,y) is v^t
    Q v, where v = (x,y,1)^t.
    """
    def __init__(self):
        self.data = new_array(9)

    def at(self, x, y):
        """
        The value of the quadratic form at a vector
        :param x: the x coordinate
        :param y: the y coordinate
        :returns: the value of the quadratic form at a vector
        """
        return self.data[x * 3 + y]


class Sum:
    """
    cache for fast summing
    """
    def __init__(self, x, y, xy, x2, y2):
        self.x = x
        self.y = y
        self.xy = xy
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        fmt_str = "Sum: {{x: {}, y: {}, xy: {}, x2: {}, y2: {}}}"
        return fmt_str.format(self.x, self.y, self.xy, self.x2, self.y2)


def mod(a, n):
    """
    assists with integer arithmetic
    """
    if a >= n:
        return a % n
    elif a >= 0:
        return a
    else:
        return n - 1 - (-1 - a) % n


def xprod(p1, p2):
    """
    calculate p1 x p2
    """
    return p1.x * p2.y - p1.y * p2.x


def cyclic(a, b, c):
    """
    return true if a <= b < c < a, in a cyclic sense (mod n)
    """
    if a <= c:
        return a <= b < c
    else:
        return a <= b or b < c


def sign(i):
    """
    determines sign of i value (negative, 0, positive)
    :param i: the number to process
    :returns: if positive, 1; if negative, -1; if 0, 0
    """
    if i > 0:
        return 1
    elif i < 0:
        return -1
    else:
        return 0


def quadform(q, w):
    """
    Apply quadratic form Q to vector w = (w.x,w.y)
    """
    v = new_array(3)

    v[0] = w.x
    v[1] = w.y
    v[2] = 1
    cur_sum = 0.0

    for i in range(3):
        for j in range(3):
            cur_sum += v[i] * q.at(i, j) * v[j]

    return cur_sum


def interval(funct, a, b):
    """
    range over the straight line segment [a,b] when lambda ranges over [0,1]
    """
    x = a.x + funct * (b.x - a.x)
    y = a.y + funct * (b.y - a.y)
    return Point(x, y)


def dorth_infty(p0, p2):
    """
    return a direction that is 90 degrees counterclockwise from p2-p0,
    but then restricted to one of the major wind directions (n, nw, w, etc)
    :param p0: point 0
    :param p2: point 2
    :returns: point that is 90 degrees counterclockwise from p2 to p0
    """
    y = sign(p2.x - p0.x)
    x = -sign(p2.y - p0.y)
    return Point(x, y)


def ddenom(p0, p2):
    """
    ddenom/dpara have the property that the square of radius 1 centered
    at p1 intersects the line p0p2 iff |dpara(p0,p1,p2)| <= ddenom(p0,p2)
    :param p0: point 0
    :param p2: point 2
    :returns: generated point
    """
    r = dorth_infty(p0, p2)
    return r.y * (p2.x - p0.x) - r.x * (p2.y - p0.y)


def dpara(p0, p1, p2):
    """
    returns the area of the parallelogram
    :param p0: point 0
    :param p1: point 1
    :param p2: point 2
    :returns: (p1-p0)x(p2-p0), the area of the parallelogram
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p2.x - p0.x
    y2 = p2.y - p0.y
    return x1 * y2 - x2 * y1


def cprod(p0, p1, p2, p3):
    """
    calculate (p1-p0)x(p3-p2)
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p3.x - p2.x
    y2 = p3.y - p2.y
    return x1 * y2 - x2 * y1


def iprod(p0, p1, p2):
    """
    calculate (p1-p0)*(p2-p0)
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p2.x - p0.x
    y2 = p2.y - p0.y
    return x1 * x2 + y1 * y2


def iprod1(p0, p1, p2, p3):
    """
    calculate (p1-p0)*(p3-p2)
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p3.x - p2.x
    y2 = p3.y - p2.y
    return x1 * x2 + y1 * y2


def ddist(p, q):
    """
    calculate distance between two points
    """
    return math.sqrt((p.x - q.x) * (p.x - q.x) +
                     (p.y - q.y) * (p.y - q.y))


def bezier(t, p0, p1, p2, p3):
    """
    calculate point of a bezier curve
    """
    s = 1 - t
    x = (s * s * s * p0.x + 3 * (s * s * t) * p1.x +
         3 * (t * t * s) * p2.x + t * t * t * p3.x)
    y = (s * s * s * p0.y + 3 * (s * s * t) * p1.y +
         3 * (t * t * s) * p2.y + t * t * t * p3.y)
    return Point(x, y)


def tangent(p0, p1, p2, p3, q0, q1):
    """
    calculate the point t in [0..1] on the (convex) bezier curve
    (p0,p1,p2,p3) which is tangent to q1-q0. Return -1.0 if there is no
    solution in [0..1].
    """
    a_prod = cprod(p0, p1, q0, q1)
    b_prod = cprod(p1, p2, q0, q1)
    c_prod = cprod(p2, p3, q0, q1)

    a = a_prod - 2 * b_prod + c_prod
    b = -2 * a_prod + 2 * b_prod
    c = a_prod

    d = b * b - 4 * a * c

    if a == 0 or d < 0:
        return -1.0

    s = math.sqrt(d)

    r1 = (-b + s) / (2 * a)
    r2 = (-b - s) / (2 * a)

    if 1 >= r1 >= 0:
        return r1
    elif 1 >= r2 >= 0:
        return r2
    else:
        return -1.0


def calc_sums(path):
    """
    Preparation: fill in the sum fields of a path (used for later
    rapid summing)
    """
    path.x0 = path.pt[0].x
    path.y0 = path.pt[0].y

    path.sums = []
    s = path.sums
    s.append(Sum(0, 0, 0, 0, 0))
    for i in range(path.len):
        x = path.pt[i].x - path.x0
        y = path.pt[i].y - path.y0
        s.append(Sum(s[i].x + x, s[i].y + y, s[i].xy + x * y,
                     s[i].x2 + x * x, s[i].y2 + y * y))


def calc_lon(path):
    """
    determine the straight subpaths.  Fill in the
    "lon" component of a path object (based on pt/len).	For each i,
    lon[i] is the furthest index such that a straight line can be drawn
    from i to lon[i].
    """
    n = path.len
    pt = path.pt
    pivk = new_array(n)
    nc = new_array(n)
    ct = new_array(4)
    path.lon = new_array(n)

    constraint = [Point(), Point()]
    cur = Point()
    off = Point()
    dk = Point()

    k = 0
    for i in range(n - 1, -1, -1):
        if pt[i].x != pt[k].x and pt[i].y != pt[k].y:
            k = i + 1

        nc[i] = k

    for i in range(n - 1, -1, -1):
        ct[0] = 0
        ct[1] = 0
        ct[2] = 0
        ct[3] = 0

        cur_dir = int((3 + 3 * (pt[mod(i + 1, n)].x - pt[i].x) +
                       (pt[mod(i + 1, n)].y - pt[i].y)) / 2)
        ct[cur_dir] += 1

        constraint[0].x = 0
        constraint[0].y = 0
        constraint[1].x = 0
        constraint[1].y = 0

        k = nc[i]
        k1 = i
        while True:
            foundk = 0
            cur_dir = int((3 + 3 * sign(pt[k].x - pt[k1].x) +
                           sign(pt[k].y - pt[k1].y)) / 2)

            ct[cur_dir] += 1

            if ct[0] != 0 and ct[1] != 0 and ct[2] != 0 and ct[3] != 0:
                pivk[i] = k1
                foundk = 1
                break

            cur.x = pt[k].x - pt[i].x
            cur.y = pt[k].y - pt[i].y

            if xprod(constraint[0], cur) < 0 or xprod(constraint[1], cur) > 0:
                break

            if not (abs(cur.x) <= 1 and abs(cur.y) <= 1):
                off.x = cur.x + (1 if (cur.y >= 0 and
                                       (cur.y > 0 or cur.x < 0)) else -1)
                off.y = cur.y + (1 if (cur.x <= 0 and
                                       (cur.x < 0 or cur.y < 0)) else -1)

                if xprod(constraint[0], off) >= 0:
                    constraint[0].x = off.x
                    constraint[0].y = off.y

                off.x = cur.x + (1 if (cur.y <= 0 and
                                       (cur.y < 0 or cur.x < 0)) else -1)

                off.y = cur.y + (1 if (cur.x >= 0 and
                                       (cur.x > 0 or cur.y < 0)) else -1)

                if xprod(constraint[1], off) <= 0:
                    constraint[1].x = off.x
                    constraint[1].y = off.y

            k1 = k
            k = nc[k1]
            if not cyclic(k, i, k1):
                break

        if foundk == 0:
            dk.x = sign(pt[k].x-pt[k1].x)
            dk.y = sign(pt[k].y-pt[k1].y)
            cur.x = pt[k1].x - pt[i].x
            cur.y = pt[k1].y - pt[i].y

            a = xprod(constraint[0], cur)
            b = xprod(constraint[0], dk)
            c = xprod(constraint[1], cur)
            d = xprod(constraint[1], dk)

            j = 10000000
            if b < 0:
                j = math.floor(a / -b)
            if d > 0:
                j = min(j, math.floor(-c / d))

            pivk[i] = mod(k1 + j, n)

    j = pivk[n - 1]
    path.lon[n - 1] = j
    for i in range(n - 2, -1, -1):
        if cyclic(i + 1, pivk[i], j):
            j = pivk[i]

        path.lon[i] = j

    i = n - 1
    while cyclic(mod(i + 1, n), j, path.lon[i]):
        path.lon[i] = j
        i -= 1


def penalty3(path, i, j):
    """
    Auxiliary function: calculate the penalty of an edge from i to j in
    the given path. This needs the "lon" and "sum*" data.
    """
    n = path.len
    pt = path.pt
    sums = path.sums
    r = 0

    if j >= n:
        j -= n
        r = 1

    if r == 0:
        x = sums[j + 1].x - sums[i].x
        y = sums[j + 1].y - sums[i].y
        x2 = sums[j + 1].x2 - sums[i].x2
        xy = sums[j + 1].xy - sums[i].xy
        y2 = sums[j + 1].y2 - sums[i].y2
        k = j + 1 - i
    else:
        x = sums[j + 1].x - sums[i].x + sums[n].x
        y = sums[j + 1].y - sums[i].y + sums[n].y
        x2 = sums[j + 1].x2 - sums[i].x2 + sums[n].x2
        xy = sums[j + 1].xy - sums[i].xy + sums[n].xy
        y2 = sums[j + 1].y2 - sums[i].y2 + sums[n].y2
        k = j + 1 - i + n

    px = (pt[i].x + pt[j].x) / 2.0 - pt[0].x
    py = (pt[i].y + pt[j].y) / 2.0 - pt[0].y
    ey = (pt[j].x - pt[i].x)
    ex = -(pt[j].y - pt[i].y)

    a = ((x2 - 2 * x * px) / k + px * px)
    b = ((xy - x * py - y * px) / k + px * py)
    c = ((y2 - 2 * y * py) / k + py * py)

    s = ex * ex * a + 2 * ex * ey * b + ey * ey * c
    return math.sqrt(s)


def best_polygon(path):
    """
    find the optimal polygon. Fill in the m and po components.
    """
    n = path.len
    pen = new_array(n + 1)
    prev = new_array(n + 1)
    clip0 = new_array(n)
    clip1 = new_array(n + 1)
    seg0 = new_array(n + 1)
    seg1 = new_array(n + 1)

    for i in range(0, n):
        c = mod(path.lon[mod(i - 1, n)] - 1, n)
        if c == i:
            c = mod(i + 1, n)

        if c < i:
            clip0[i] = n
        else:
            clip0[i] = c

    j = 1
    for i in range(0, n):
        while j <= clip0[i]:
            clip1[j] = i
            j += 1

    i = 0
    j = 0
    while i < n:
        seg0[j] = i
        i = clip0[i]
        j += 1

    seg0[j] = n
    m = j

    i = n
    for j in range(m, 0, -1):
        seg1[j] = i
        i = clip1[i]

    seg1[0] = 0

    pen[0] = 0
    for j in range(1, m + 1):
        for i in range(seg1[j], seg0[j] + 1):
            best = -1
            for k in range(seg0[j - 1], clip1[i] - 1, -1):
                thispen = penalty3(path, k, i) + pen[k]
                if best < 0 or thispen < best:
                    prev[i] = k
                    best = thispen

            pen[i] = best

    path.m = m
    path.po = new_array(m)

    j = m - 1
    i = n
    while i > 0:
        i = prev[i]
        path.po[j] = i
        j -= 1


def pointslope(path, i, j, ctr, direct):
    """
    determine the center and slope of the line i..j. Assume i<j. Needs
    "sum" components of p to be set.
    """
    n = path.len
    sums = path.sums
    r = 0

    while j >= n:
        j -= n
        r += 1

    while i >= n:
        i -= n
        r -= 1

    while j < 0:
        j += n
        r -= 1

    while i < 0:
        i += n
        r += 1

    x = sums[j + 1].x - sums[i].x + r * sums[n].x
    y = sums[j + 1].y - sums[i].y + r * sums[n].y
    x2 = sums[j + 1].x2 - sums[i].x2 + r * sums[n].x2
    xy = sums[j + 1].xy - sums[i].xy + r * sums[n].xy
    y2 = sums[j + 1].y2 - sums[i].y2 + r * sums[n].y2
    k = j + 1 - i + r * n

    ctr.x = x / k
    ctr.y = y / k

    a = (x2 - x * x / k) / k
    b = (xy - x * y / k) / k
    c = (y2 - y * y / k) / k

    lambda2 = (a + c + math.sqrt((a - c) * (a - c) + 4 * b * b)) / 2

    a -= lambda2
    c -= lambda2

    if abs(a) >= abs(c):
        length = math.sqrt(a * a + b * b)
        if length != 0:
            direct.x = -b / length
            direct.y = a / length
    else:
        length = math.sqrt(c * c + b * b)
        if length != 0:
            direct.x = -c / length
            direct.y = b / length

    if length == 0:
        direct.x = 0
        direct.y = 0


def adjust_vertices(path):
    """
    Adjust vertices of optimal polygon: calculate the intersection of
    the two "optimal" line segments, then move it into the unit square
    if it lies outside. Return 1 with errno set on error; 0 on
    success.
    """
    m = path.m
    po = path.po
    n = path.len
    pt = path.pt
    x0 = path.x0
    y0 = path.y0
    ctr = new_array(m)
    cur_dir = new_array(m)
    q = new_array(m)
    v = new_array(3)
    s = Point()

    path.curve = Curve(m)
    for i in range(m):
        j = po[mod(i + 1, m)]
        j = mod(j - po[i], n) + po[i]
        ctr[i] = Point()
        cur_dir[i] = Point()
        pointslope(path, po[i], j, ctr[i], cur_dir[i])

    for i in range(m):
        q[i] = Quad()
        d = cur_dir[i].x * cur_dir[i].x + cur_dir[i].y * cur_dir[i].y
        if d == 0.0:
            for j in range(3):
                for k in range(3):
                    q[i].data[j * 3 + k] = 0

        else:
            v[0] = cur_dir[i].y
            v[1] = -cur_dir[i].x
            v[2] = - v[1] * ctr[i].y - v[0] * ctr[i].x
            for l in range(3):
                for k in range(3):
                    q[i].data[l * 3 + k] = v[l] * v[k] / d

    for i in range(m):
        quad = Quad()
        w = Point()

        s.x = pt[po[i]].x - x0
        s.y = pt[po[i]].y - y0

        j = mod(i - 1, m)

        for l in range(3):
            for k in range(3):
                quad.data[l * 3 + k] = q[j].at(l, k) + q[i].at(l, k)

        while True:
            det = quad.at(0, 0) * quad.at(1, 1) - quad.at(0, 1) * quad.at(1, 0)
            if det != 0.0:
                w.x = (-quad.at(0, 2) * quad.at(1, 1) +
                       quad.at(1, 2) * quad.at(0, 1)) / det

                w.y = (quad.at(0, 2) * quad.at(1, 0) -
                       quad.at(1, 2) * quad.at(0, 0)) / det
                break

            if quad.at(0, 0) > quad.at(1, 1):
                v[0] = -quad.at(0, 1)
                v[1] = quad.at(0, 0)
            elif quad.at(1, 1):
                v[0] = -quad.at(1, 1)
                v[1] = quad.at(1, 0)
            else:
                v[0] = 1
                v[1] = 0

            d = v[0] * v[0] + v[1] * v[1]
            v[2] = -v[1] * s.y - v[0] * s.x
            for l in range(3):
                for k in range(3):
                    quad.data[l * 3 + k] += v[l] * v[k] / d

        dx = abs(w.x - s.x)
        dy = abs(w.y - s.y)
        if dx <= 0.5 and dy <= 0.5:
            path.curve.vertex[i] = Point(w.x + x0, w.y + y0)
            continue

        cur_min = quadform(quad, s)
        xmin = s.x
        ymin = s.y

        if quad.at(0, 0) != 0.0:
            for z in range(2):
                w.y = s.y - 0.5 + z
                w.x = -(quad.at(0, 1) * w.y + quad.at(0, 2)) / quad.at(0, 0)
                dx = abs(w.x - s.x)
                cand = quadform(quad, w)
                if dx <= 0.5 and cand < cur_min:
                    cur_min = cand
                    xmin = w.x
                    ymin = w.y

        if quad.at(1, 1) != 0.0:
            for z in range(2):
                w.x = s.x - 0.5 + z
                w.y = -(quad.at(1, 0) * w.x + quad.at(1, 2)) / quad.at(1, 1)
                dy = abs(w.y - s.y)
                cand = quadform(quad, w)
                if dy <= 0.5 and cand < cur_min:
                    cur_min = cand
                    xmin = w.x
                    ymin = w.y

        for l in range(2):
            for k in range(2):
                w.x = s.x - 0.5 + l
                w.y = s.y - 0.5 + k
                cand = quadform(quad, w)
                if cand < cur_min:
                    cur_min = cand
                    xmin = w.x
                    ymin = w.y

        path.curve.vertex[i] = Point(xmin + x0, ymin + y0)


def reverse(path):
    """
    reverse orientation of a path
    """
    curve = path.curve
    m = curve.n
    v = curve.vertex

    i = 0
    j = m - 1
    while i < j:
        tmp = v[i]
        v[i] = v[j]
        v[j] = tmp
        i += 1
        j -= 1


def smooth(path, alphamax):
    """
    determines if path curve is CORNER or CURVE as well as
    pertinent points for each kind
    :param path: the path to smooth
    :param alphamax: tolerance between line segment CORNER
    and CURVE
    """
    m = path.curve.n
    curve = path.curve
    for i in range(m):
        j = mod(i + 1, m)
        k = mod(i + 2, m)
        p4 = interval(1 / 2.0, curve.vertex[k], curve.vertex[j])
        denom = ddenom(curve.vertex[i], curve.vertex[k])
        if denom != 0.0:
            dd = (dpara(curve.vertex[i], curve.vertex[j], curve.vertex[k]) /
                  denom)
            dd = abs(dd)
            alpha = (1 - 1.0 / dd) if dd > 1 else 0
            alpha = alpha / 0.75
        else:
            alpha = 4 / 3.0

        curve.alpha0[j] = alpha

        if alpha >= alphamax:
            curve.tag[j] = "CORNER"
            curve.c[3 * j + 1] = curve.vertex[j]
            curve.c[3 * j + 2] = p4
        else:
            if alpha < 0.55:
                alpha = 0.55
            elif alpha > 1:
                alpha = 1

            p2 = interval(0.5 + 0.5 * alpha, curve.vertex[i], curve.vertex[j])
            p3 = interval(0.5 + 0.5 * alpha, curve.vertex[k], curve.vertex[j])
            curve.tag[j] = "CURVE"
            curve.c[3 * j + 0] = p2
            curve.c[3 * j + 1] = p3
            curve.c[3 * j + 2] = p4

        curve.alpha[j] = alpha
        curve.beta[j] = 0.5

    curve.alphacurve = 1


class Opti:
    """
    a type for the result of opti_penalty
    """
    def __init__(self):
        self.pen = 0
        self.c = [Point(), Point()]
        self.t = 0
        self.s = 0
        self.alpha = 0


def opti_penalty(path, i, j, res, opttolerance, convc, areac):
    """
    calculate best fit from i+.5 to j+.5.  Assume i<j (cyclically).
    Return 0 and set badness and parameters (alpha, beta), if
    possible. Return 1 if impossible.
    """
    m = path.curve.n
    curve = path.curve
    vertex = curve.vertex

    if i == j:
        return 1

    k = i
    i1 = mod(i + 1, m)
    k1 = mod(k + 1, m)
    conv = convc[k1]

    if conv == 0:
        return 1

    d = ddist(vertex[i], vertex[i1])

    k = k1
    while k != j:
        k1 = mod(k + 1, m)
        k2 = mod(k + 2, m)
        if convc[k1] != conv:
            return 1

        if sign(cprod(vertex[i], vertex[i1], vertex[k1], vertex[k2])) != conv:
            return 1

        if (iprod1(vertex[i], vertex[i1], vertex[k1], vertex[k2]) <
                d * ddist(vertex[k1], vertex[k2]) * -0.999847695156):
            return 1

        k = k1

    p0 = curve.c[mod(i, m) * 3 + 2].copy()
    p1 = vertex[mod(i + 1, m)].copy()
    p2 = vertex[mod(j, m)].copy()
    p3 = curve.c[mod(j, m) * 3 + 2].copy()

    area = areac[j] - areac[i]
    area -= dpara(vertex[0], curve.c[i * 3 + 2], curve.c[j * 3 + 2]) / 2
    if i >= j:
        area += areac[m]

    a_1 = dpara(p0, p1, p2)
    a_2 = dpara(p0, p1, p3)
    a_3 = dpara(p0, p2, p3)

    a_4 = a_1 + a_3 - a_2

    if a_2 == a_1:
        return 1

    t = a_3 / (a_3 - a_4)
    s = a_2 / (a_2 - a_1)
    a_0 = a_2 * t / 2.0

    if a_0 == 0.0:
        return 1

    r_0 = area / a_0
    alpha = 2 - math.sqrt(4 - r_0 / 0.3)

    res.c[0] = interval(t * alpha, p0, p1)
    res.c[1] = interval(s * alpha, p3, p2)
    res.alpha = alpha
    res.t = t
    res.s = s

    p1 = res.c[0].copy()
    p2 = res.c[1].copy()

    res.pen = 0

    k = mod(i + 1, m)
    while k != j:
        k1 = mod(k + 1, m)
        t = tangent(p0, p1, p2, p3, vertex[k], vertex[k1])
        if t < -0.5:
            return 1

        pt = bezier(t, p0, p1, p2, p3)
        d = ddist(vertex[k], vertex[k1])
        if d == 0.0:
            return 1

        d1 = dpara(vertex[k], vertex[k1], pt) / d
        if abs(d1) > opttolerance:
            return 1

        if (iprod(vertex[k], vertex[k1], pt) < 0 or
                iprod(vertex[k1], vertex[k], pt) < 0):
            return 1

        res.pen += d1 * d1
        k = k1

    k = i
    while k != j:
        k1 = mod(k + 1, m)
        t = tangent(p0, p1, p2, p3, curve.c[k * 3 + 2], curve.c[k1 * 3 + 2])
        if t < -0.5:
            return 1

        pt = bezier(t, p0, p1, p2, p3)
        d = ddist(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2])
        if d == 0.0:
            return 1

        d1 = dpara(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2], pt) / d
        d2 = dpara(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2], vertex[k1]) / d
        d2 *= 0.75 * curve.alpha[k1]
        if d2 < 0:
            d1 = -d1
            d2 = -d2

        if d1 < d2 - opttolerance:
            return 1
        if d1 < d2:
            res.pen += (d1 - d2) * (d1 - d2)

        k = k1

    return 0


def opti_curve(path, opttolerance):
    """
    optimize the path p, replacing sequences of Bezier segments by a
    single segment when possible.
    """
    curve = path.curve
    m = curve.n
    vert = curve.vertex
    pt = new_array(m + 1)
    pen = new_array(m + 1)
    length = new_array(m + 1)
    opt = new_array(m + 1)
    o = Opti()

    convc = new_array(m)
    areac = new_array(m + 1)

    for i in range(m):
        if curve.tag[i] == "CURVE":
            convc[i] = sign(dpara(vert[mod(i - 1, m)], vert[i],
                                  vert[mod(i + 1, m)]))
        else:
            convc[i] = 0

    area = 0.0
    areac[0] = 0.0
    p0 = curve.vertex[0]
    for i in range(m):
        i1 = mod(i + 1, m)
        if curve.tag[i1] == "CURVE":
            alpha = curve.alpha[i1]
            area += (0.3 * alpha * (4 - alpha) *
                     dpara(curve.c[i * 3 + 2], vert[i1],
                           curve.c[i1 * 3 + 2]) / 2)
            area += dpara(p0, curve.c[i * 3 + 2], curve.c[i1 * 3 + 2]) / 2

        areac[i + 1] = area

    pt[0] = -1
    pen[0] = 0
    length[0] = 0

    for j in range(1, m + 1):
        pt[j] = j - 1
        pen[j] = pen[j - 1]
        length[j] = length[j - 1] + 1

        for i in range(j - 2, -1, -1):
            r = opti_penalty(path, i, mod(j, m), o, opttolerance,
                             convc, areac)
            if r:
                break

            if (length[j] > length[i] + 1 or
                    (length[j] == length[i] + 1 and pen[j] > pen[i] + o.pen)):
                pt[j] = i
                pen[j] = pen[i] + o.pen
                length[j] = length[i] + 1
                opt[j] = o
                o = Opti()

    om = length[m]
    ocurve = Curve(om)
    s = new_array(om)
    t = new_array(om)

    j = m
    for i in range(om - 1, -1, -1):
        if pt[j] == j - 1:
            ocurve.tag[i] = curve.tag[mod(j, m)]
            ocurve.c[i * 3 + 0] = curve.c[mod(j, m) * 3 + 0]
            ocurve.c[i * 3 + 1] = curve.c[mod(j, m) * 3 + 1]
            ocurve.c[i * 3 + 2] = curve.c[mod(j, m) * 3 + 2]
            ocurve.vertex[i] = curve.vertex[mod(j, m)]
            ocurve.alpha[i] = curve.alpha[mod(j, m)]
            ocurve.alpha0[i] = curve.alpha0[mod(j, m)]
            ocurve.beta[i] = curve.beta[mod(j, m)]
            s[i] = t[i] = 1.0
        else:
            ocurve.tag[i] = "CURVE"
            ocurve.c[i * 3 + 0] = opt[j].c[0]
            ocurve.c[i * 3 + 1] = opt[j].c[1]
            ocurve.c[i * 3 + 2] = curve.c[mod(j, m) * 3 + 2]
            ocurve.vertex[i] = interval(opt[j].s, curve.c[mod(j, m) * 3 + 2],
                                        vert[mod(j, m)])
            ocurve.alpha[i] = opt[j].alpha
            ocurve.alpha0[i] = opt[j].alpha
            s[i] = opt[j].s
            t[i] = opt[j].t

        j = pt[j]

    for i in range(om):
        i1 = mod(i + 1, om)
        ocurve.beta[i] = s[i] / (s[i] + t[i1])

    ocurve.alphacurve = 1
    path.curve = ocurve


def process_path(pathlist, alphamax, optcurve, opttolerance):
    """
    In change of processing outlines of objects and converting to
    svg-type paths
    """
    for i in range(len(pathlist)):
        path = pathlist[i]
        calc_sums(path)
        calc_lon(path)
        best_polygon(path)
        adjust_vertices(path)

        if path.sign == "-":
            reverse(path)

        smooth(path, alphamax)

        if optcurve:
            opti_curve(path, opttolerance)

    return pathlist


def process(bm, turnpolicy="minority", turdsize=2,
            alphamax=1, optcurve=True, opttolerance=.2):
    """
    Handles processing bitmap image, converting image to path outlines,
    and converting those outlines into svg-type paths.  Defaults are
    provided
    :param bm: the bitmap to be processed
    :param turnpolicy: how turns are handled;
     either 'right', 'black', 'white', 'majority', 'minority'
    :param turdsize: maximum area of ignored paths found
    :param alphamax: tolerance between curve and line segment corner
    :param optcurve: whether or not to optimize bezier curves combining
    where possible.
    :param opttolerance: how much of a penalty to allow due to curve
    combining
    :returns: path list that can be processed into svg
    see get_svg for how they are handled
    """
    pathlist = bm_to_path_list(bm, turnpolicy, turdsize)
    return process_path(pathlist, alphamax, optcurve, opttolerance)


def get_svg_path(curve, x_offset=0, y_offset=0, multiplier=1):
    """
    converts curve to svg path
    :param curve: the potrace curve
    :param x_offset: the x offset of the item in the canvas
    :param y_offset: the y offset of the item in the canvas
    :param multiplier: how much to multiply the size
    (i.e. 500x500 image can become 1000 x 1000 if multiplier is 2)
    """
    def str3(flt):
        """
        utility method to format float to 3 decimal places
        """
        return '{:.3f}'.format(flt)

    def adj_x(x):
        """
        gets the adjusted x position based on offset and multiplier
        """
        return str3((x + x_offset) * multiplier)

    def adj_y(y):
        """
        gets the adjusted y position based on offset and multiplier
        """
        return str3((y + y_offset) * multiplier)

    def bezier(i):
        """
        retrieves the svg path bezier curve format
        """
        return ('C ' + adj_x(curve.c[i * 3 + 0].x) + ' ' +
                adj_y(curve.c[i * 3 + 0].y) + ',' +
                adj_x(curve.c[i * 3 + 1].x) + ' ' +
                adj_y(curve.c[i * 3 + 1].y) + ',' +
                adj_x(curve.c[i * 3 + 2].x) + ' ' +
                adj_y(curve.c[i * 3 + 2].y) + ' ')

    def segment(i):
        """
        retrieves the svg path line segment format
        """
        return ('L ' + adj_x(curve.c[i * 3 + 1].x) + ' ' +
                adj_y(curve.c[i * 3 + 1].y) + ' ' +
                adj_x(curve.c[i * 3 + 2].x) + ' ' +
                adj_y(curve.c[i * 3 + 2].y) + ' ')

    n = curve.n
    p = ('M ' + adj_x(curve.c[(n - 1) * 3 + 2].x) +
         ' ' + adj_y(curve.c[(n - 1) * 3 + 2].y) + ' ')

    for i in range(n):
        if curve.tag[i] == "CURVE":
            p += bezier(i)
        elif curve.tag[i] == "CORNER":
            p += segment(i)

    p += 'Z'
    return p


def get_svg(pathlist, width, height, size=1, opt_type=None):
    """
    gets the full svg string with all paths included
    :param pathlist: the list of path objects
    :param width: the width of the svg
    :param height: the height of the svg
    :param size: the expansion of the image (i.e. 1 is 100%)
    :param opt_type: how to address the image
    (curve gets stroke; otherwise fill)
    """
    w = width * size
    h = height * size
    length = len(pathlist)

    svg = ('<svg id="svg" version="1.1" width="' + str(int(w)) +
           '" height="' + str(int(h)) +
           '" xmlns="http://www.w3.org/2000/svg">')

    svg += '<path d="'
    for i in range(length):
        c = pathlist[i].curve
        svg += get_svg_path(c, multiplier=size)

    if opt_type == "curve":
        strokec = "black"
        fillc = "none"
        fillrule = ''
    else:
        strokec = "none"
        fillc = "black"
        fillrule = ' fill-rule="evenodd"'

    svg += ('" stroke="' + strokec + '" fill="' + fillc +
            '"' + fillrule + '/></svg>')
    return svg


'''
#for testing purposes
from PIL import Image


def load_bm_from_path(path):
    im = Image.open(open(path, 'rb'))
    width, height = im.size

    data_arr = []
    for y in range(height):
        for x in range(width):
            r, g, b = im.getpixel((x, y))
            if (r < 150 and g < 150 and b < 150):
                data_arr.append(True)
            else:
                data_arr.append(False)

    return Bitmap(width, height, data_arr)

test_bm = Bitmap(5, 5,
                 [False, False, False, False, False,
                  False, True, True, True, False,
                  False, True, True, True, False,
                  False, True, True, True, False,
                  False, False, False, False, False])

test_bm = load_bm_from_path('/home/gdicristofaro/Desktop/yao.jpg')


test_paths = process(test_bm)
svg = get_svg(test_paths, test_bm.w, test_bm.h, size=1)
text_file = open("Output.svg", "w")
text_file.write(svg)
text_file.close()
'''
