from cquents_core import *
import math
import cquents_builtins as builtin_helper
import re
import sys

import oeis

#codepage  = """μηΔ↑≺≻🙸         """
#codepage += """½⅓¼⅒⅟√∛∜∨∧«¬»⨽₊₋"""
#codepage += """ !"#$%&'()*+,-./"""
#codepage += """0123456789:;<=>?"""
#codepage += """@ABCDEFGHIJKLMNO"""
#codepage += """PQRSTUVWXYZ[\\]^_"""
#codepage += """`abcdefghijklmno"""
#codepage += """pqrstuvwxyz{|}~\n"""
#codepage += """⁰¹²³⁴⁵⁶⁷⁸⁹⁻⟨¿⟩÷×"""
#codepage += """₀₁₂₃₄₅₆₇₈₉𝕏ℂ𝕄⦗·⦘"""
#codepage += """∑∏∈∉σφωΩ≤≠≥ẠḄḌẸḤ"""
#codepage += """ỊḲḶṂṆỌṚṢṬỤṾẈỴẒȦḂ"""
#codepage += """ĊḊĖḞĠḢİĿṀṄȮṖṘṠṪẆ"""
#codepage += """ẊẎŻạḅḍẹḥịḳḷṃṇọṛṣ"""
#codepage += """ṭụṿẉỵẓȧḃċḋėḟġḣŀṁ"""
#codepage += """ṅȯṗṙṡṫẇẋẏż…⋯⋮∫Ξ𝔼"""

LITERAL = "LITERAL"
NUMBER = "NUMBER"
EOF = "EOF"
ID = "ID"
BUILTIN = "BUILTIN"
CONSTANT = "CONSTANT"
MODE = "MODE"
PARAM = "PARAM"
OPERATOR = "OPERATOR"
SEPARATOR = "SEPARATOR"
NEWLINE = "NEWLINE"
LCONTAINER = "LCONTAINER"
RCONTAINER = "RCONTAINER"

# modes = []
# params = []
# separators = []
# operators = []
# extra_operators = []
# builtins = []
# extra_builtins = []
# lcontainers = []
# rcontainers = []

SEQUENCE_1 = ":"
SEQUENCE_2 = "::"
SERIES = ";"
CLOSEST = ";;"
QUERY = "?"
NOT_QUERY = "??"
modes = [SEQUENCE_1, SEQUENCE_2, SERIES, CLOSEST, QUERY, NOT_QUERY]

START_INDEX = "$"
START_TERMS = "="
DEFAULT_INPUT = "#"
STRINGED = '"'
params = [START_INDEX, START_TERMS, DEFAULT_INPUT, STRINGED]

TERM_SEPARATOR = ","
META_SEPARATOR = "|"
FUNCTION_SEPARATOR = ";"
separators = [TERM_SEPARATOR, META_SEPARATOR, FUNCTION_SEPARATOR]

LPAREN = "("
LSEQUENCE = "{"
LINDEX = "["
lcontainers = [LPAREN, LSEQUENCE, LINDEX]

RPAREN = ")"
RSEQUENCE = "}"
RINDEX = "]"
rcontainers = [RPAREN, RSEQUENCE, RINDEX]

DECIMAL_POINT = "."

LITERAL_ESCAPE = "@"
LITERAL_QUOTE_1 = "'"
LITERAL_QUOTE_2 = '"'

OEIS_START = "O"

NEWLINE_CHAR = "\n"


""" VARIABLES"""

N = "n"
CURRENT = "$"
K = "k"
TEN = "t"
input_ids = ["A", "B", "C"]
previous_ids = ["Z", "Y", "X"]
variables = [N, CURRENT, K, TEN] + input_ids + previous_ids

def is_variable(x): return x in variables


""" CONSTANTS """

CONSTANTS = "`"     # all have ` appended to beginning of command
constants = {
    CONSTANTS + "2": 1/2,
    CONSTANTS + "3": 1/3,
    CONSTANTS + "4": 1/4,
    CONSTANTS + "0": 1/10,
    CONSTANTS + "c": 0.915965594177219,
    CONSTANTS + "e": math.e,
    CONSTANTS + "g": .5 * (math.sqrt(5) + 1),
    CONSTANTS + "G": 0.128242712910062,
    CONSTANTS + "k": 0.268545200106530,
    CONSTANTS + "p": math.pi,
    CONSTANTS + "y": 0.577215664901532
}

def is_constant(x): return x in constants


""" BUILTINS """

def get_line(origin_interpreter, parameters, index):
    next_parameters = [get_input_item_tree(origin_interpreter.visit(parameter)) for parameter in parameters]

    return origin_interpreter.lines[index].interpreter.interpret(next_parameters)

builtins = {
    "a": lambda inter, node: get_line(inter, node.parameters, 0),
    "b": lambda inter, node: get_line(inter, node.parameters, 1),
    "c": lambda inter, node: get_line(inter, node.parameters, 2),
    "d": lambda inter, node: get_line(inter, node.parameters[1:], int(inter.visit(node.parameters[0]))),
    "D": lambda inter, node: builtin_helper.digits(inter.visit(node.parameters[0])),
    "f": lambda inter, node: math.factorial(inter.visit(node.parameters[0])),
    "F": lambda inter, node: math.floor(inter.visit(node.parameters[0])),
    "h": lambda inter, node: chr(inter.visit(node.parameters[0])),
    "I": lambda inter, node: inter.get_input(inter.visit(node.parameters[0])),
    "l": lambda inter, node: builtin_helper.log(inter, node.parameters),
    "L": lambda inter, node: builtin_helper.length(inter, node.parameters),
    "m": lambda inter, node: builtin_helper.min_(inter.visit(node.parameters[0])),
    "M": lambda inter, node: builtin_helper.max_(inter.visit(node.parameters[0])),
    "o": lambda inter, node: ord(inter.visit(node.parameters[0])),
    "p": lambda inter, node: builtin_helper.next_prime(inter.visit(node.parameters[0])),
    "P": lambda inter, node: inter.get_previous(inter.visit(node.parameters[0])),
    "Q": lambda inter, node: builtin_helper.deduplicate(inter.visit(node.parameters[0])),
    "r": lambda inter, node: builtin_helper.root(inter, node.parameters),
    "R": lambda inter, node: round(inter.visit(node.parameters[0])),
    "s": lambda inter, node: builtin_helper.sort(inter, node.parameters),
    "S": lambda inter, node: str(inter.visit(node.parameters[0])),
    "T": lambda inter, node: math.ceil(inter.visit(node.parameters[0])),
    "u": lambda inter, node: builtin_helper.count_(inter, node.parameters),
    "U": lambda inter, node: builtin_helper.sum_(inter.visit(node.parameters[0])),
    "v": lambda inter, node: abs(inter.visit(node.parameters[0])),
    "V": lambda inter, node: builtin_helper.average(inter.visit(node.parameters[0])),
    "x": lambda inter, node: math.exp(inter.visit(node.parameters[0]))
}

EXTRA_BUILTINS = "\\"   # all have \ appended to beginning of command
extra_builtins = {
    EXTRA_BUILTINS + "c": lambda inter, node: math.cos(inter.visit(node.parameters[0])),
  # EXTRA_BUILTINS + "f": lambda inter, node: builtin_helper.fill(inter, node.parameters),
    EXTRA_BUILTINS + "l": lambda inter, node: math.log10(inter.visit(node.parameters[0])),
    EXTRA_BUILTINS + "r": lambda inter, node: builtin_helper.reverse(inter, node.parameters),
    EXTRA_BUILTINS + "R": lambda inter, node: builtin_helper.rotate(inter, node.parameters),
    EXTRA_BUILTINS + "s": lambda inter, node: math.sin(inter.visit(node.parameters[0])),
    EXTRA_BUILTINS + "t": lambda inter, node: math.tan(inter.visit(node.parameters[0]))
}

def get_OEIS(origin_interpreter, parameters, cq_source):
    next_parameters = [get_input_item_tree(origin_interpreter.visit(parameter)) for parameter in parameters]
    return run(cq_source, next_parameters, is_oeis=True)

#FIXME: Global lines object causes bugs when doing this
builtins.update({sequence: lambda inter, node, sequence=sequence: get_OEIS(inter, node.parameters, oeis.OEIS[sequence]) for sequence in oeis.OEIS})

builtins.update(extra_builtins)

def is_builtin(x): return x in builtins

""" OPERATORS """

unary_ops = {
    "-": lambda x: -x,
    "+": lambda x: +x,
    "~": lambda x: ~x,
    "/": lambda x: 1/x
}
post_unary_ops = {
    "!": lambda x: math.factorial(x)
}

MUL = "*"
binary_ops = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "~": lambda x, y: builtin_helper.concat(x, y),
    MUL: lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "^": lambda x, y: x ** y,
    "%": lambda x, y: x % y,
    "e": lambda x, y: x * (10 ** y)
}

EXTRA_OPS = "_"     # all extra operators have _ appended to beginning of command

extra_unary_ops = {
    EXTRA_OPS + "l": lambda x: builtin_helper.primitive_rotate(x, -1),  # string rotate left
    EXTRA_OPS + "r": lambda x: builtin_helper.primitive_rotate(x, 1),   # string rotate right
    EXTRA_OPS + "+": lambda x: x + 1,
    EXTRA_OPS + "-": lambda x: x - 1
}
extra_binary_ops = {
    EXTRA_OPS + "/": lambda x, y: x // y,
    EXTRA_OPS + "|": lambda x, y: x | y,        # bitwise OR
    EXTRA_OPS + "n": lambda x, y: ~(x | y),     # bitwise NOR
    EXTRA_OPS + "^": lambda x, y: x ^ y,        # bitwise XOR
    EXTRA_OPS + "x": lambda x, y: ~(x ^ y),     # bitwise XNOR
    EXTRA_OPS + "&": lambda x, y: x & y,        # bitwise AND
    EXTRA_OPS + "N": lambda x, y: ~(x & y),     # bitwise NAND
    EXTRA_OPS + "<": lambda x, y: x << y,       # bitwise left shift
    EXTRA_OPS + ">": lambda x, y: x >> y        # bitwise right shift
}

unary_ops.update(extra_unary_ops)
binary_ops.update(extra_binary_ops)

extra_operators = {**extra_unary_ops, **extra_binary_ops}
operators = {**binary_ops, **unary_ops, **post_unary_ops}

def is_operator(x): return x in operators
def is_unary_operator(x): return x in unary_ops or x in extra_unary_ops
def is_binary_operator(x): return x in binary_ops or x in extra_binary_ops

binary_operator_precedence = [
    # last
    ("_|", "_n"),
    ("_^", "_x"),
    ("_&", "_N"),
    ("_<", "_>"),
    ("-", "+", "~"),
    ("*", "/", "_/", "%"),
    ("^", "e")
    # first
]

# print(set([op for line in binary_operator_precedence for op in line]))
# print(set(list(binary_ops) + [EXTRA_OPS + op for op in list(extra_binary_ops)]))

# all operators should have a precedence level
assert set([op for line in binary_operator_precedence for op in line]) == \
    set(list(binary_ops) + list(extra_binary_ops)), \
    "Not all binary operators have a precedence level"


""" START INTERPRETER """

is_one_int = re.compile("^[0-9]$")


class Sequence:

    def __init__(self, interpreter_, node):
        self.interpreter = interpreter_
        self.node = node
        self.current = 0
        self.statement_index = 0
        self.k = 1
        self.sequence = []

    def __getitem__(self, index):
        return self.sequence[index]

    def __iter__(self):
        self.current = 0
        self.statement_index = 0
        self.k = 1
        self.sequence = []
        return self

    def __next__(self):
        if self.current < len(self.node.start):
            cur_val = self.interpreter.visit(self.node.start[self.current])
        else:
            if self.statement_index >= len(self.node.statements):
                self.statement_index = 0
                self.k += 1

            cur_val = self.interpreter.visit(self.node.statements[self.statement_index])
            self.statement_index += 1

        self.sequence.append(cur_val)
        self.current += 1

        # TODO: work with not just ints
        if self.node.is_stringed:
            return int(self.interpreter.join.join([str(x) for x in self.sequence])[self.current - 1])

        return cur_val


def get_input_val(char):
    return ord(char) - 65


class Token:
    def __init__(self, type_, val):
        self.type = type_
        self.val = val

    def can_multiply(self):
        return self.type in (NUMBER, ID, BUILTIN, LCONTAINER, CONSTANT)

    def __str__(self):
        return "<Token: " + self.type + " " + str(self.val) + ">"


class Lexer:

    def __init__(self, text):
        self.text = text
        self.pos = 0

        try:
            self.cur = self.text[self.pos]
        except IndexError:
            self.cur = None

        self.mode_set = False
        self.param_found = False
        self.mode = None

    def read_token(self):
        if self.cur is None:
            return Token(EOF, "")

        # reading literals before first parameter
        if not self.mode_set and not self.param_found:
            if self.cur == LITERAL_ESCAPE:
                self.advance()
                return Token(LITERAL, self.advance())
            elif self.cur in params:
                self.param_found = True
            elif self.cur == META_SEPARATOR:
                return Token(SEPARATOR, self.advance())
            elif self.cur not in modes:
                return Token(LITERAL, self.advance())

        while self.cur == " ":
            self.advance()

            if self.cur is None:
                return Token(EOF, "")

        if not self.mode_set and self.cur in modes:
            self.mode = self.cur
            self.mode_set = True
            self.advance()

            if self.mode + self.cur in modes:
                self.mode += self.advance()

            return Token(MODE, self.mode)

        elif not self.mode_set and self.cur in params:
            return Token(PARAM, self.advance())

        elif self.cur in separators:
            return Token(SEPARATOR, self.advance())

        elif self.cur in lcontainers:
            return Token(LCONTAINER, self.advance())

        elif self.cur in rcontainers:
            return Token(RCONTAINER, self.advance())

        elif self.cur == DECIMAL_POINT or is_one_int.match(self.cur):
            return self.read_number()

        elif self.cur in variables:
            return self.read_id()

        elif self.cur == CONSTANTS and self.cur + self.peek() in constants:
            return Token(CONSTANT, self.advance() + self.advance())

        elif self.cur == OEIS_START:
            self.advance()
            temp = str(self.read_number().val)
            temp = self.cur + "0" * (6 - len(temp)) + temp
            self.advance()
            return Token(BUILTIN, temp)

        elif self.cur in builtins:
            return Token(BUILTIN, self.advance())

        elif self.cur == EXTRA_BUILTINS and self.cur + self.peek() in extra_builtins:
            return Token(BUILTIN, self.advance() + self.advance())

        elif self.cur in operators:
            return Token(OPERATOR, self.advance())

        elif self.cur == EXTRA_OPS and self.cur + self.peek() in extra_operators:
            return Token(OPERATOR, self.advance() + self.advance())

        if self.cur == LITERAL_ESCAPE:
            self.advance()
            return Token(LITERAL, self.advance())
        elif self.cur == LITERAL_QUOTE_1 or self.cur == LITERAL_QUOTE_2:
            quote = self.cur
            self.advance()
            temp = self.read_literal(quote)
            self.advance()
            return temp
        elif self.cur == NEWLINE_CHAR:
            self.advance()
            self.reset()
            return Token(NEWLINE, NEWLINE_CHAR)
        else:
            if self.cur is not None:
                return Token(LITERAL, self.advance())

        # raise CQError("Unknown character found : " + self.cur)

    def advance(self):
        temp = self.cur
        self.pos += 1
        try:
            self.cur = self.text[self.pos]
        except IndexError:
            self.cur = None

        return temp

    # may be longer at some point
    def read_id(self):
        return Token(ID, self.advance())

    def read_number(self):
        result = ""

        while self.cur is not None and is_one_int.match(self.cur):
            result += self.cur
            self.advance()

        if self.cur == DECIMAL_POINT:
            result += DECIMAL_POINT
            self.advance()

            while self.cur is not None and is_one_int.match(self.cur):
                result += self.cur
                self.advance()

            if result == DECIMAL_POINT:
                return Token(LITERAL, DECIMAL_POINT)

            return Token(NUMBER, float(result))

        return Token(NUMBER, int(result))

    def read_literal(self, quote):
        result = ""

        while self.cur is not None and self.cur != quote:
            result += self.cur
            self.advance()

        return Token(LITERAL, result)

    def peek(self):
        if self.pos + 1 >= len(self.text):
            return ""

        return self.text[self.pos + 1]

    def reset(self):
        try:
            self.cur = self.text[self.pos]
        except IndexError:
            self.cur = None

        self.mode_set = False
        self.param_found = False
        self.mode = None


class Parser:
    def __init__(self, lexer_):
        self.lexer = lexer_
        self.token = self.lexer.read_token()

    def eat(self, type_):
        if self.token.type == type_:
            self.token = self.lexer.read_token()
        else:
            if type_ == EOF or (type_ == RCONTAINER and self.token.type in (EOF, NEWLINE, TERM_SEPARATOR)):
                return

            raise CQSyntaxError("Incorrect token found: looking for " + type_ + ", found " + self.token.type + " " + self.token.val + " at " + str(self.lexer.pos))

    def parse(self):
        lines_ = [self.program()]

        while self.token.type == NEWLINE:

            self.eat(NEWLINE)
            lines_.append(self.program())

        self.eat(EOF)
        return lines_

    def program(self):
        parameters = self.params()
        mode = self.mode()
        items = self.items()

        program = Program(parameters, mode, items)
        return program

    def params(self):
        literals = ["", "", ""]

        lit_index = 0

        while (self.token.type == LITERAL and lit_index <= 2) or (self.token.type == SEPARATOR and self.token.val == META_SEPARATOR and lit_index < 2):
            if self.token.type == LITERAL:
                literals[lit_index] += self.token.val
                self.eat(LITERAL)
            elif self.token.val == META_SEPARATOR:
                self.eat(SEPARATOR)
                lit_index += 1

        default_input = [], []
        start = []
        current_start = []
        is_stringed = False

        while self.token.type == PARAM:
            if self.token.val == DEFAULT_INPUT:
                self.eat(PARAM)
                default_input = self.input_list()
            elif self.token.val == START_TERMS:
                self.eat(PARAM)
                start = self.items()
            elif self.token.val == START_INDEX:
                self.eat(PARAM)
                current_start = self.items()
            elif self.token.val == STRINGED:
                self.eat(PARAM)
                is_stringed = True

        parameters = Params(literals, default_input, start, current_start, is_stringed)
        return parameters

    def mode(self):
        temp = self.token
        if self.token.type == MODE:
            self.eat(MODE)
            return temp.val
        return None

    def items(self):
        items = [self.expr()]

        while self.token.val == TERM_SEPARATOR:
            self.eat(SEPARATOR)
            items.append(self.expr())

        return items

    def function_items(self):
        items = [self.expr()]

        while self.token.val == FUNCTION_SEPARATOR:
            self.eat(SEPARATOR)
            items.append(self.expr())

        return items

    def input_list(self):
        items = [], []

        inc = 0

        while inc <= 1:
            if self.token.type == SEPARATOR and self.token.val == META_SEPARATOR and inc == 0:
                self.eat(SEPARATOR)
                inc += 1
            elif inc == 1:
                break

            items[inc].append(self.expr())

            if self.token.type == SEPARATOR and self.token.val == TERM_SEPARATOR:
                self.eat(SEPARATOR)
            elif self.token.val != META_SEPARATOR:
                break

        return items

    def variable(self):
        node = Var(self.token)
        self.eat(ID)
        return node

    # call expr for each layer of precedence, end with call to factor
    def expr(self, cur_precedence=0):
        next_precedence = cur_precedence + 1

        if next_precedence < len(binary_operator_precedence):
            def next_layer(): return self.expr(next_precedence)
        else:
            def next_layer(): return self.factor()

        node = next_layer()

        while self.token.val in binary_operator_precedence[cur_precedence] or \
                (MUL in binary_operator_precedence[cur_precedence] and self.token.can_multiply()):

            tok = self.token

            if tok.can_multiply():
                tok = Token(OPERATOR, MUL)
            else:
                self.eat(tok.type)

            node = BinOp(node, tok, next_layer())

        return node


    def factor(self):
        tok = self.token

        # print(extra_unary_ops)

        if tok.type == OPERATOR and is_unary_operator(tok.val):
            self.eat(OPERATOR)
            node = UnaryOp(tok, self.factor())
        elif tok.type == NUMBER:
            self.eat(NUMBER)
            node = Number(tok)
        elif tok.type == CONSTANT:
            self.eat(CONSTANT)
            node = Constant(tok.val)
        elif tok.type == LCONTAINER:
            self.eat(LCONTAINER)

            if tok.val == LSEQUENCE:
                node = FiniteSequence(self.function_items())
            else:
                node = self.expr()

            self.eat(RCONTAINER)
        elif tok.type == BUILTIN:
            builtin = tok.val
            self.eat(BUILTIN)
            node_list = self.function_items()
            self.eat(RCONTAINER)
            node = Builtin(builtin, node_list)
        elif tok.type == ID:
            node = self.variable()
        elif tok.type == LITERAL:
            self.eat(LITERAL)
            node = Literal(tok)
        else:
            raise CQSyntaxError("Unknown factor : " + (tok.val or tok.type))

        post_tok = self.token

        if post_tok.type == OPERATOR and post_tok.val in post_unary_ops:
            self.eat(OPERATOR)
            node = PostUnaryOp(post_tok, node)

        return node


class NodeVisitor:
    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise CQInternalError("No visit_" + type(node).__name__ + " method")


class Tester(NodeVisitor):
    def __init__(self):
        self.max_input = -1

    def generic_visit(self, node):
        pass

    def visit_Program(self, node):
        for each in node.current_start + node.start + node.input_front + node.input_back + node.statements:
            self.visit(each)

    def visit_FiniteSequence(self, node):
        for term in node.terms:
            self.visit(term)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_Builtin(self, node):
        for parameter in node.parameters:
            self.visit(parameter)

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_PostUnaryOp(self, node):
        self.visit(node.expr)

    def visit_Var(self, node):
        if node.name in input_ids:
            code = get_input_val(node.name)
            if self.max_input < code:
                self.max_input = code


class Interpreter(NodeVisitor):
    def __init__(self, tree_, max_input, lines_):
        self.tree = tree_
        self.input = None
        self.max_input = max_input
        self.lines = lines_

        self.n = None
        self.sequence = None
        self.current = 1
        self.current_inc = 1
        self.program = None
        self.join = TERM_SEPARATOR

    # 0 is last, 1 is second last, etc.
    def get_previous(self, num):
        try:
            temp = self.sequence[-1 - num]
        except IndexError:
            temp = 0
        return temp

    # 0 is first input, 1 is second input, etc.
    def get_input(self, num):
        return self.visit(self.input[num])

    def visit_Program(self, node):
        do_prints = not isinstance(self, HelperInterpreter)

        default_input_length = len(node.input_front) + len(node.input_back)
        user_input_length = len(self.input)

        actual_input_length = default_input_length + user_input_length
        expected_input_length = self.max_input + 1

        # print("Actual: " + str(actual_input_length) + ", Expected: " + str(expected_input_length))

        input_length_difference = actual_input_length - expected_input_length

        if input_length_difference >= 0:
            all_input = []

            all_input.extend(node.input_front)
            all_input.extend(self.input)
            all_input.extend(node.input_back)

            n = None
            if input_length_difference >= 1:
                n = all_input.pop()

            self.input = all_input

            # print(self.input)
        else:
            raise CQInputError("Incorrect input length")

        # TODO: Do different if mode == QUERY

        self.program = node
        self.sequence = Sequence(self, node)
        self.current = self.visit(node.current_start[0]) if len(node.current_start) >= 1 else 1
        self.current_inc = self.visit(node.current_start[1]) if len(node.current_start) >= 2 else 1

        # starting literals
        if do_prints:
            print(node.literals[0], end="", flush=node.is_stringed)

        if n is None:
            query_n = False
        else:
            self.n = n = self.visit(n)
            query_n = n == 0 or n

        if node.is_stringed:
            self.join = node.literals[1] or ""

            # also note that all prints will need to be flushed if is_stringed is true
        else:
            self.join = node.literals[1] or TERM_SEPARATOR

        if node.mode in (SEQUENCE_2, QUERY, NOT_QUERY) and not query_n:
            pass

        else:
            sum_ = 0

            for val in self.sequence:

                if node.mode == SEQUENCE_1:
                    if query_n:
                        if n == self.current:
                            if do_prints:
                                print(val, end="", flush=node.is_stringed)
                            else:
                                return val
                            break

                        # if input n is less than the current index, it will never be in it, so output 0
                        elif n < self.current:
                            if do_prints:
                                print(0, end="", flush=node.is_stringed)
                            else:
                                return val
                            break
                    else:
                        print(val, end=self.join, flush=node.is_stringed)

                elif node.mode == SEQUENCE_2:
                    if query_n:
                        if n == self.current:
                            print(val, end="", flush=node.is_stringed)
                            break
                        else:
                            print(val, end=self.join, flush=node.is_stringed)

                elif node.mode == SERIES:
                    try:
                        sum_ += val
                    except TypeError:
                        if sum_ == 0:
                            sum_ = ""
                    finally:
                        sum_ += val

                    if query_n:
                        if n == self.current:
                            if do_prints:
                                print(sum_, end="", flush=node.is_stringed)
                            else:
                                return sum_
                            break
                    # elif sum_ == previous_sum

                elif node.mode in (QUERY, NOT_QUERY):
                    # TODO: integrate with \# functions when `if` is implemented

                    if_in = "true" if node.mode == QUERY else "false"
                    if_out = "false" if node.mode == QUERY else "true"

                    if query_n:
                        if n == val:
                            print(if_in, end="", flush=node.is_stringed)
                            break
                        # elif previous and cur_val < previous:
                        #     print("false", end="", flush=node.is_stringed)
                        #     done = True

                        # TODO: FIXME
                        elif val > n:
                            print(if_out, end="", flush=node.is_stringed)
                            break

                self.current += self.current_inc

        if do_prints:
            print(node.literals[2], end="", flush=node.is_stringed)

    def visit_FiniteSequence(self, node):
        terms = []
        for term in node.terms:
            terms.append(self.visit(term))

        return terms

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if is_binary_operator(node.op.val):
            return binary_ops[node.op.val](left, right)

    def visit_Constant(self, node):
        if is_constant(node.name):
            return constants[node.name]

    def visit_Builtin(self, node):
        if is_builtin(node.builtin):
            return builtins[node.builtin](self, node)

    def visit_UnaryOp(self, node):
        if is_unary_operator(node.op.val):
            return unary_ops[node.op.val](self.visit(node.expr))

    def visit_PostUnaryOp(self, node):
        if node.op.val in post_unary_ops:
            return post_unary_ops[node.op.val](self.visit(node.expr))

    def visit_Var(self, node):
        if node.name == CURRENT:
            return self.current
        elif node.name == K:
            return self.sequence.k
        elif node.name == N:
            return self.n
        elif node.name == TEN:
            return 10
        elif node.name in previous_ids:
            return self.get_previous(previous_ids.index(node.name))
        elif node.name in input_ids:
            return self.get_input(input_ids.index(node.name))

    def visit_Number(self, node):
        return node.value

    def visit_Literal(self, node):
        return node.value

    def interpret(self, input_):
        self.input = input_

        if self.tree is None:
            return None
        return self.visit(self.tree)


class HelperInterpreter(Interpreter):
    pass


try:
    file = sys.argv[1]
except IndexError:
    file = "source.cq"

# print(file)

source = open(file).read()

# print(source)


def get_input_item_tree(item):
    try:
        return Number(Token(NUMBER, int(item)))
    except ValueError:
        try:
            return Number(Token(NUMBER, float(item)))
        except ValueError:
            return Number(Token(LITERAL, item))


def get_input():
    try:
        input_ = input().split(" ")

        if input_ == ['']:
            input_ = []
    except EOFError:
        input_ = []

    if input_ is not []:
        for input_index in range(len(input_)):
            input_[input_index] = get_input_item_tree(input_[input_index])

    return input_

user_input = get_input()

# print(user_input)


def get_tree(cq_source):
    lexer = Lexer(cq_source)
    parser = Parser(lexer)
    return parser.parse()


def run(cq_source, cq_input, is_oeis=False):
    lines = []

    try:
        programs = get_tree(cq_source)
    except CQSyntaxError:
        # allows for default mode
        programs = get_tree(":" + cq_source)

    first = True
    for program in programs:
        # print(program)

        tester = Tester()
        tester.visit(program)

        if not is_oeis and first:
            first = False

            interpreter = Interpreter(program, tester.max_input, lines)
        else:
            interpreter = HelperInterpreter(program, tester.max_input, lines)

        lines.append(Line(program, interpreter))

    return lines[0].interpreter.interpret(cq_input)

run(source, user_input)