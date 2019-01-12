import cquents_builtins as builtin_helper
import re
import math
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

# MODES = []
# PARAMS = []
# SEPARATORS = []
# OPERATORS = []
# LCONTAINERS = []
# RCONTAINERS = []

SEQUENCE_1 = ":"
SEQUENCE_2 = "::"
SERIES = ";"
CLOSEST = ";;"
QUERY = "?"
NOT_QUERY = "??"
MODES = [SEQUENCE_1, SEQUENCE_2, SERIES, CLOSEST, QUERY, NOT_QUERY]

CURRENT = "$"
START = "="
DEFAULT_INPUT = "#"
STRINGED = '"'
PARAMS = [CURRENT, START, DEFAULT_INPUT, STRINGED]

LITERAL_ESCAPE = "\\"
LITERAL_QUOTE = "'"

PLUS = "+"
MINUS = "-"
CONCAT = "~"
MUL = "*"
DIV = "/"
INT_DIV = "//"
EXPONENT = "^"
MOD = "%"
E = "e"

EXTRA_OPS = "b"
EXTRA_BUILTINS = "\\"
EXTRA_CONSTANTS = "_"

BITWISE_NOT = "b~"
BITWISE_OR = "b|"
BITWISE_NOR = "bn"
BITWISE_XOR = "b^"
BITWISE_XNOR = "bx"
BITWISE_AND = "b&"
BITWISE_NAND = "bN"
BITWISE_LEFT = "b<"
BITWISE_RIGHT = "b>"

STRING_LEFT = "b-"
STRING_RIGHT = "b+"

OEIS_START = "O"

TERM_SEPARATOR = ","
META_SEPARATOR = "|"
FUNCTION_SEPARATOR = ";"
SEPARATORS = [TERM_SEPARATOR, META_SEPARATOR, FUNCTION_SEPARATOR]

LPAREN = "("
LCONTAINERS = [LPAREN]

RPAREN = ")"
RCONTAINERS = [RPAREN]

NEWLINE_CHAR = "\n"

is_one_int = re.compile("^[0-9]$")
is_id = re.compile("^[$nv-zA-E]$")
is_input_id = re.compile("^[A-E]$")
is_previous_id = re.compile("^[v-z]$")


def get_line(origin_interpreter, node, index):
    next_parameters = [get_input_item_tree(origin_interpreter.visit(parameter)) for parameter in node.parameters]

    return origin_interpreter.lines[index].interpreter.interpret(next_parameters)


def get_OEIS(origin_interpreter, parameters, cq_source):
    next_parameters = [get_input_item_tree(origin_interpreter.visit(parameter)) for parameter in parameters]
    return run(cq_source, next_parameters, is_oeis=True)

builtins = {
    "a": lambda inter, node: abs(inter.visit(node.parameters[0])),
    "c": lambda inter, node: math.ceil(inter.visit(node.parameters[0])),
    "f": lambda inter, node: math.factorial(inter.visit(node.parameters[0])),
    "F": lambda inter, node: math.floor(inter.visit(node.parameters[0])),
    "l": lambda inter, node: builtin_helper.log(inter, node.parameters),
    "L": lambda inter, node: builtin_helper.length(inter, node.parameters),
    "p": lambda inter, node: builtin_helper.next_prime(inter.visit(node.parameters[0])),
    "r": lambda inter, node: builtin_helper.root(inter, node.parameters),
    "R": lambda inter, node: round(inter.visit(node.parameters[0])),
    "X": lambda inter, node: math.exp(inter.visit(node.parameters[0])),
    "\\c": lambda inter, node: math.cos(inter.visit(node.parameters[0])),
    #"\\f": lambda inter, node: builtin_helper.fill(inter, node.parameters),
    "\\l": lambda inter, node: math.log10(inter.visit(node.parameters[0])),
    "\\r": lambda inter, node: builtin_helper.reverse(inter, node.parameters),
    "\\R": lambda inter, node: builtin_helper.rotate(inter, node.parameters),
    "\\s": lambda inter, node: math.sin(inter.visit(node.parameters[0])),
    "\\t": lambda inter, node: math.tan(inter.visit(node.parameters[0]))
}

# functions for other lines
builtins.update({EXTRA_BUILTINS + str(line_number): lambda inter, node, line_number=line_number: get_line(inter, node, line_number) for line_number in range(10)})

#FIXME: Global lines object causes bugs when doing this
builtins.update({sequence: lambda inter, node, sequence=sequence: get_OEIS(inter, node.parameters, oeis.OEIS[sequence]) for sequence in oeis.OEIS})

constants = {
    "c": 0.915965594177219,
    "e": math.e,
    "g": .5 * (math.sqrt(5) + 1),
    "G": 0.128242712910062,
    "k": 0.268545200106530,
    "p": math.pi,
    "y": 0.577215664901532
}

unary_ops = {
    "-": lambda x: -x,
    "+": lambda x: +x,
    BITWISE_NOT: lambda x: ~x,
    STRING_LEFT: lambda x: builtin_helper.primitive_rotate(x, -1),
    STRING_RIGHT: lambda x: builtin_helper.primitive_rotate(x, 1)
}

post_unary_ops = {
    
}

binary_ops = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "~": lambda x, y: builtin_helper.concat(x, y),
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    INT_DIV: lambda x, y: x // y,
    "^": lambda x, y: x ** y,
    "%": lambda x, y: x % y,
    "e": lambda x, y: x * (10 ** y),
    BITWISE_OR: lambda x, y: x | y,
    BITWISE_NOR: lambda x, y: ~(x | y),
    BITWISE_XOR: lambda x, y: x ^ y,
    BITWISE_XNOR: lambda x, y: ~(x ^ y),
    BITWISE_AND: lambda x, y: x & y,
    BITWISE_NAND: lambda x, y: ~(x & y),
    BITWISE_LEFT: lambda x, y: x << y,
    BITWISE_RIGHT: lambda x, y: x >> y
}

OPERATORS = list(binary_ops.keys()) + list(unary_ops.keys()) + list(post_unary_ops.keys())


class Sequence:

    def __init__(self, interpreter_, node):
        self.interpreter = interpreter_
        self.node = node
        self.current = 0
        self.statement_index = 0
        self.sequence = []

    def __getitem__(self, index):
        return self.sequence[index]

    def __iter__(self):
        self.current = 0
        self.statement_index = 0
        self.sequence = []
        return self

    def __next__(self):
        if self.current < len(self.node.start):
            cur_val = self.interpreter.visit(self.node.start[self.current])
        else:
            if self.statement_index >= len(self.node.statements):
                self.statement_index = 0

            cur_val = self.interpreter.visit(self.node.statements[self.statement_index])
            self.statement_index += 1

        self.sequence.append(cur_val)
        self.current += 1

        # TODO: work with not just ints
        if self.node.is_stringed:
            return int(self.interpreter.join.join([str(x) for x in self.sequence])[self.current - 1])

        return cur_val


class CQError(Exception):
    pass


class CQSyntaxError(CQError):
    pass


class CQInputError(CQError):
    pass


class CQInternalError(CQError):
    pass


class CQConcatError(CQError):
    pass


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
            elif self.cur in PARAMS:
                self.param_found = True
            elif self.cur == META_SEPARATOR:
                return Token(SEPARATOR, self.advance())
            elif self.cur not in MODES:
                return Token(LITERAL, self.advance())

        while self.cur == " ":
            self.advance()

            if self.cur is None:
                return Token(EOF, "")

        if not self.mode_set and self.cur in MODES:
            self.mode = self.cur
            self.mode_set = True
            self.advance()

            if self.mode + self.cur in MODES:
                self.mode += self.advance()

            return Token(MODE, self.mode)

        elif not self.mode_set and self.cur in PARAMS:
            #self.param_found = True
            #temp = self.cur
            #self.advance()
            return Token(PARAM, self.advance())

        elif self.cur in SEPARATORS:
            #temp = self.cur
            #self.advance()
            return Token(SEPARATOR, self.advance())

        elif self.cur in LCONTAINERS:
            return Token(LCONTAINER, self.advance())

        elif self.cur in RCONTAINERS:
            return Token(RCONTAINER, self.advance())

        elif self.cur == "." or is_one_int.match(self.cur):
            return self.read_number()

        elif is_id.match(self.cur):
            return self.read_id()

        elif self.cur == EXTRA_CONSTANTS:
            #self.advance()
            #temp = self.cur
            #self.advance()
            return Token(CONSTANT, self.advance() + self.advance())

        elif self.cur == OEIS_START:
            self.advance()
            temp = str(self.read_number().val)
            temp = self.cur + "0" * (6 - len(temp)) + temp
            self.advance()
            return Token(BUILTIN, temp)

        elif self.cur in builtins:
            #temp = self.cur
            #self.advance()
            return Token(BUILTIN, self.advance())

        elif self.cur == EXTRA_BUILTINS and self.cur + self.peek() in builtins:
            #temp = self.cur
            #self.advance()
            #temp += self.cur
            #self.advance()
            return Token(BUILTIN, self.advance() + self.advance())

        elif self.cur == EXTRA_OPS and self.cur + self.peek() in OPERATORS or self.cur + self.peek() in OPERATORS:
            #temp = self.cur
            #self.advance()
            #temp += self.cur
            #self.advance()
            return Token(OPERATOR, self.advance() + self.advance())

        elif self.cur in OPERATORS:
            #temp = self.cur
            #self.advance()
            return Token(OPERATOR, self.advance())

        if self.cur == LITERAL_ESCAPE:
            self.advance()
            #temp = self.cur
            #self.advance()
            return Token(LITERAL, self.advance())
        elif self.cur == LITERAL_QUOTE:
            self.advance()
            temp = self.read_literal()
            self.advance()
            return temp
        elif self.cur == NEWLINE_CHAR:
            self.advance()
            self.reset()
            return Token(NEWLINE, NEWLINE_CHAR)
        else:
            if self.cur is not None:
                #temp = self.cur
                #self.advance()
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

    def read_id(self):
        #temp = self.cur
        #self.advance()
        return Token(ID, self.advance())

    def read_number(self):
        result = ""

        while self.cur is not None and is_one_int.match(self.cur):
            result += self.cur
            self.advance()

        if self.cur == ".":
            result += "."
            self.advance()

            while self.cur is not None and is_one_int.match(self.cur):
                result += self.cur
                self.advance()

            if result == ".":
                return Token(LITERAL, ".")

            return Token(NUMBER, float(result))

        return Token(NUMBER, int(result))

    def read_literal(self):
        result = ""

        while self.cur is not None and self.cur != LITERAL_QUOTE:
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

        self.operations = [
            (BITWISE_OR, BITWISE_NOR),
            (BITWISE_XOR, BITWISE_XNOR),
            (BITWISE_AND, BITWISE_NAND),
            (BITWISE_LEFT, BITWISE_RIGHT),
            (MINUS, PLUS, CONCAT),
            (MUL, DIV, INT_DIV, MOD),
            (EXPONENT, E)
        ]

    def eat(self, type_, char=None):
        if self.token.type == type_:
            self.token = self.lexer.read_token()
        else:
            if type_ == EOF or type_ == RCONTAINER and self.token.type in (EOF, NEWLINE):
                return

            raise CQSyntaxError("Incorrect token found: looking for " + type_ + ", found " + self.token.type + " | " + self.token.val + " at " + self.lexer.pos)

    def parse(self):
        lines_ = [self.program()]

        while self.token.type == NEWLINE:

            self.eat(NEWLINE)
            lines_.append(self.program())

        self.eat(EOF)
        return lines_

    def program(self):
        params = self.params()
        mode = self.mode()
        items = self.items()

        program = Program(params, mode, items)
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
            elif self.token.val == START:
                self.eat(PARAM)
                start = self.items()
            elif self.token.val == CURRENT:
                self.eat(PARAM)
                current_start = self.items()
            elif self.token.val == STRINGED:
                self.eat(PARAM)
                is_stringed = True

        params = Params(literals, default_input, start, current_start, is_stringed)
        return params

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

    def expr(self):
        node = self.op_0()

        # valid 3rd priority operations
        while self.token.val in self.operations[0]:
            tok = self.token

            self.eat(tok.type)

            node = BinOp(node, tok, self.op_0())

        return node

    def op_0(self):
        node = self.op_1()

        while self.token.val in self.operations[1]:
            tok = self.token
            self.eat(tok.type)
            node = BinOp(node, tok, self.op_1())

        return node

    def op_1(self):
        node = self.op_2()

        while self.token.val in self.operations[2]:
            tok = self.token
            self.eat(tok.type)
            node = BinOp(node, tok, self.op_2())

        return node

    def op_2(self):
        node = self.op_3()

        while self.token.val in self.operations[3]:
            tok = self.token
            self.eat(tok.type)
            node = BinOp(node, tok, self.op_3())

        return node

    def op_3(self):
        node = self.term()

        while self.token.val in self.operations[-3]:
            tok = self.token
            self.eat(tok.type)
            node = BinOp(node, tok, self.term())

        return node

    def term(self):
        # get first factor
        node = self.short_term()

        # valid 2nd priority operations
        while self.token.val in self.operations[-2] or self.token.can_multiply():
            tok = self.token

            if tok.can_multiply():
                tok = Token(MUL, MUL)
            else:
                self.eat(tok.type)

            node = BinOp(node, tok, self.short_term())

        return node

    def short_term(self):
        # get first factor
        node = self.factor()

        # valid 1st priority operations
        while self.token.val in self.operations[-1]:
            tok = self.token

            self.eat(tok.type)

            node = BinOp(node, tok, self.factor())

        return node

    def factor(self):
        tok = self.token

        if tok.val in unary_ops:
            self.eat(tok.type)
            return UnaryOp(tok, self.factor())
        elif tok.type == NUMBER:
            self.eat(NUMBER)
            return Number(tok)
        elif tok.type == CONSTANT:
            self.eat(CONSTANT)
            return Constant(tok.val)
        elif tok.type == LCONTAINER:
            self.eat(LCONTAINER)
            node = self.expr()
            self.eat(RCONTAINER)
            return node
        elif tok.type == BUILTIN:
            builtin = tok.val
            self.eat(BUILTIN)
            nodes = self.items()
            self.eat(RCONTAINER)
            return Builtin(builtin, nodes)
        elif tok.type == ID:
            return self.variable()

        raise CQSyntaxError("Unknown factor : " + (tok.val or tok.type))


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

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_Builtin(self, node):
        for parameter in node.parameters:
            self.visit(parameter)

    def visit_UnaryOp(self, node):
        if node.op.type in unary_ops:
            self.visit(node.expr)

    def visit_Var(self, node):
        if is_input_id.match(node.name):
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

    def visit_Program(self, node):
        do_prints = not isinstance(self, HelperInterpreter)

        default_input_length = len(node.input_front) + len(node.input_back)
        user_input_length = len(self.input)

        actual_input_length = default_input_length + user_input_length
        expected_input_length = self.max_input + 1

        # print("Actual: " + str(actual_input_length) + ", Expected: " + str(expected_input_length))

        input_length_difference = actual_input_length - expected_input_length

        if input_length_difference == 0 or input_length_difference == 1:
            all_input = []

            all_input.extend(node.input_front)
            all_input.extend(self.input)
            all_input.extend(node.input_back)

            n = None
            if input_length_difference == 1:
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

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if node.op.val in binary_ops:
            return binary_ops[node.op.val](left, right)

    def visit_Constant(self, node):
        # _ is included in node.name but not in constant dict
        return constants[node.name[1:]]

    def visit_Builtin(self, node):
        if node.builtin in builtins:
            return builtins[node.builtin](self, node)

    def visit_UnaryOp(self, node):
        if node.op.val in unary_ops:
            return unary_ops[node.op.val](self.visit(node.expr))

    def visit_Var(self, node):
        if node.name == CURRENT:
            return self.current
        if node.name == "n":
            return self.n
        elif is_previous_id.match(node.name):
            try:
                temp = self.sequence[-1 + ord(node.name) - 122]
            except IndexError:
                temp = 0
            return temp
        elif is_input_id.match(node.name):
            return self.visit(self.input[get_input_val(node.name)])

    def visit_Number(self, node):
        return node.value

    def interpret(self, input_):
        self.input = input_

        if self.tree is None:
            return None
        return self.visit(self.tree)


class HelperInterpreter(Interpreter):
    pass


class Line:
    def __init__(self, tree, interpreter):
        self.tree = tree
        self.interpreter = interpreter


class Params:
    def __init__(self, literals, default_input, start, current_start, is_stringed):
        self.literals = literals
        self.default_input = default_input
        self.start = start
        self.current_start = current_start
        self.is_stringed = is_stringed


class AST:
    pass


class Program(AST):
    def __init__(self, params, mode, statements):
        self.literals = params.literals
        self.input_front = params.default_input[0]
        self.input_back = params.default_input[1]
        self.start = params.start
        self.current_start = params.current_start
        self.mode = mode
        self.statements = statements
        self.is_stringed = params.is_stringed

    def __str__(self):
        return "<Program: " + ",".join([str(x) for x in self.statements]) + ">"


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return "<BinOp: " + str(self.op) + " " + str(self.left) + " " + str(self.right) + ">"


class Constant(AST):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "<Constant: " + self.name + ">"


class Builtin(AST):
    def __init__(self, builtin, parameters):
        self.builtin = builtin
        self.parameters = parameters

    def __str__(self):
        return "<Builtin: " + self.builtin + " " + ",".join([str(x) for x in self.parameters]) + ">"


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

    def __str__(self):
        return "<UnaryOp: " + str(self.op) + " " + str(self.expr) + ">"


class Var(AST):
    def __init__(self, token):
        self.token = token
        self.name = token.val

    def __str__(self):
        return "<Var: " + str(self.token) + " " + self.name + ">"


class Number(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.val

    def __str__(self):
        return "<Number: " + str(self.token) + " " + str(self.value) + ">"

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