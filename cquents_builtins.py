from cquents_core import *
import math
import itertools

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

    # both are strs: return str
    elif type(x) is type(y) is str:
        return res

    # one int, one str: return str
    elif (type(x) is str or type(y) is str) and (type(x) is int or type(y) is int):
        return res

    raise CQTypeError("Error concatenating " + str(type(x)) + ":" + str(x) + " and " + str(type(y)) + ":" + str(y))


def length(origin_interpreter, parameters):
    if isinstance(parameters[0], Literal):
        return len(origin_interpreter.visit(parameters[0]))
    elif isinstance(parameters[0], Number):
        str_ = str(origin_interpreter.visit(parameters[0]))

        str_ = str_.replace("-", "")
        if len(parameters) == 1:
            str_ = str_.replace(".", "")

        return len(str_)

    raise CQTypeError("Error getting the length of a " + str(type(parameters[0])))


def fill(origin_interpreter, parameters):
    # ??
    pass


def reverse(origin_interpreter, parameters):
    if isinstance(parameters[0], Literal):
        return origin_interpreter.visit(parameters[0])[::-1]
    elif isinstance(parameters[0], Number):
        num = origin_interpreter.visit(parameters[0])
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

    raise CQTypeError("Error reversing a " + str(type(parameters[0])))

def rotate(origin_interpreter, parameters):
    return primitive_rotate(origin_interpreter.visit(parameters[0]), origin_interpreter.visit(parameters[1]), len(parameters) <= 2)


#https://stackoverflow.com/a/8458282/7605753
def primitive_rotate(num, rotation, keep_dot_position=True):
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
