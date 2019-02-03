from cquents_core import *
import math
import itertools
import statistics

# http://www.macdevcenter.com/pub/a/python/excerpt/pythonckbk_chap1/index1.html?page=2
def gen_primes():
    D = {}
    for q in itertools.islice(itertools.count(3), 0, None, 2):
        p = D.pop(q, None)
        if p is None:
            D[q * q] = q
            yield q
        else:
            x = p + q
            while x in D or not (x & 1):
                x += p
            D[x] = p

primes = [2]
prime_gen = gen_primes()


def next_prime(n):

    if primes[-1] > n:
        index = len(primes) - 1
        while index >= 0 and primes[index] > n:
            index -= 1

        return primes[index + 1]

    else:
        while True:
            nxt = next(prime_gen)

            primes.append(nxt)
            if nxt > n:
                return nxt


def root(origin_interpreter, parameters):
    if len(parameters) == 1:
        return math.sqrt(origin_interpreter.visit(parameters[0]))
    return origin_interpreter.visit(parameters[0]) ** (1 / origin_interpreter.visit(parameters[1]))


def log(origin_interpreter, parameters):
    if len(parameters) == 1:
        return math.log(origin_interpreter.visit(parameters[0]))
    base = origin_interpreter.visit(parameters[1])
    if base == 10:
        math.log10(origin_interpreter.visit(parameters[0]))
    elif base == 2:
        math.log2(origin_interpreter.visit(parameters[0]))
    return math.log(origin_interpreter.visit(parameters[0]), base)


def concat(x, y):
    # what to do with negatives/floats?
    res = str(x) + str(y)

    # both are ints: return int
    if type(x) is type(y) is int and y > 0:
        return int(res)

    # TODO: one str one other, return str

    # both are strs: return str
    elif type(x) is type(y) is str:
        return res

    # one int, one str: return str
    elif (type(x) is str or type(y) is str) and (type(x) is int or type(y) is int):
        return res

    raise CQTypeError("Error concatenating " + str(type(x)) + ":" + str(x) + " and " + str(type(y)) + ":" + str(y))


def length(origin_interpreter, parameters):
    given = origin_interpreter.visit(parameters[0])
    try:
        return len(given)
    except TypeError:
        str_ = str(given)

        str_ = str_.replace("-", "")
        if len(parameters) == 1:
            str_ = str_.replace(".", "")

        return len(str_)

    # raise CQTypeError("Error getting the length of a " + str(type(parameters[0])))


def fill(origin_interpreter, parameters):
    # ??
    pass


def reverse(origin_interpreter, parameters):
    given = origin_interpreter.visit(parameters[0])
    try:
        return given[::-1]
    except TypeError:
        num = given
        is_negative = num < 0
        type_ = type(num)
        keep_dot_position = len(parameters) == 1

        if type_ == int:
            str_ = str(int(math.fabs(num)))
        else:
            str_ = str(math.fabs(num))

        dot_index = str_.find(".")

        if ~dot_index and keep_dot_position:
            str_ = str_.replace(".", "")

        str_ = str_[::-1]

        if ~dot_index:
            if keep_dot_position:
                str_ = str_[:dot_index] + "." + str_[dot_index:]

            res = float(str_)
        else:
            res = int(str_)

        return -res if is_negative else res

    # raise CQTypeError("Error reversing a " + str(type(parameters[0])))


def rotate(origin_interpreter, parameters):
    return primitive_rotate(origin_interpreter.visit(parameters[0]), origin_interpreter.visit(parameters[1]), len(parameters) <= 2)


# https://stackoverflow.com/a/8458282/7605753
def primitive_rotate(given, rotation, keep_dot_position=True):
    try:
        rotation = rotation % len(given)
        return given[-rotation:] + given[:-rotation]
    except TypeError:
        num = given
        is_negative = num < 0
        type_ = type(num)

        if type_ == int:
            str_ = str(int(math.fabs(num)))
        else:
            str_ = str(math.fabs(num))

        dot_index = str_.find(".")

        if ~dot_index and keep_dot_position:
            str_ = str_.replace(".", "")

        rotation = rotation % len(str_)
        str_ = str_[-rotation:] + str_[:-rotation]

        if ~dot_index:
            if keep_dot_position:
                str_ = str_[:dot_index] + "." + str_[dot_index:]

            res = float(str_)
        else:
            res = int(str_)

        return -res if is_negative else res


def digits(given):
    try:
        return [x for x in given]
    except TypeError:
        return [int(x) for x in str(given).replace(".", "")]


def average(given):
    try:
        return statistics.mean(given)
    except TypeError:
        return statistics.mean(digits(given))


def count_(origin_interpreter, parameters):
    lst = origin_interpreter.visit(parameters[0])
    key = origin_interpreter.visit(parameters[1])

    try:
        return lst.count(key)
    except AttributeError:
        return digits(lst).count(key)
    except TypeError:
        return lst.count(str(key))


def min_(lst):
    try:
        return min(lst)
    except TypeError:
        return min(digits(lst))


def max_(lst):
    try:
        return max(lst)
    except TypeError:
        return max(digits(lst))


def sum_(lst):
    try:
        return sum(lst)
    except TypeError:
        return sum(digits(lst))


def sort(origin_interpreter, parameters):
    lst = origin_interpreter.visit(parameters[0])
    order = len(parameters) > 1

    try:
        return sorted(lst, reverse=order)
    except TypeError:
        return int("".join(str(x) for x in sorted(digits(lst), reverse=order)))


def deduplicate(given):
    if type(given) is list:
        return list(dict.fromkeys(given))
    elif type(given) is str:
        return "".join(list(dict.fromkeys(given)))
    elif type(given) is int:
        return int("".join(list(dict.fromkeys(str(given)))))
    elif type(given) is float:
        return float("".join(list(dict.fromkeys(str(given)))))


def join_(origin_interpreter, parameters):
    lst = origin_interpreter.visit(parameters[0])

    try:
        glue = origin_interpreter.visit(parameters[1])
    except IndexError:
        glue = ""

    joined = glue.join(str(x) for x in lst)
    try:
        return int(joined)
    except ValueError:
        return joined


# https://stackoverflow.com/a/6800586/7605753
def divisors(x):
    result = set()
    for i in range(1, int(math.sqrt(x)) + 1):
        div, mod = divmod(x, i)
        if mod == 0:
            result |= {i, div}
    return sorted(list(result))


def proper_divisors(x):
    return divisors(x)[:-1]


# TODO: speed testing
# https://stackoverflow.com/a/22808285/7605753
def prime_factors(x):
    i = primes[0]
    factors = []

    while i * i <= x:
        if x % i:
            i = next_prime(i)
        else:
            x //= i
            factors.append(i)
    if x > 1:
        factors.append(x)
    return factors


# TODO: optimize
def smallest(lst, n):
    i = 1
    found = 0
    while True:
        if i not in lst:
            found += 1
            if found >= n:
                return i
        i += 1


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/"


def to_base(origin_interpreter, parameters):
    base = 2 if len(parameters) == 1 else origin_interpreter.visit(parameters[1])

    in_num = int(origin_interpreter.visit(parameters[0]))

    out_str = ""
    while in_num:
        out_str = ALPHABET[in_num % base] + out_str
        in_num //= base

    return out_str


def from_base(origin_interpreter, parameters):
    base = 2 if len(parameters) == 1 else origin_interpreter.visit(parameters[1])

    in_num = origin_interpreter.visit(parameters[0])

    try:
        in_num[0]
    except TypeError:
        in_num = str(in_num)

    out_num = 0
    for i, char in enumerate(in_num):
        column = len(in_num) - 1 - i
        out_num += ALPHABET.index(char) * (base ** column)

    return out_num
