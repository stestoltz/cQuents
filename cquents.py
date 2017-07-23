import re
import math
import sys

LITERAL = "LITERAL"
LITERAL_ESCAPE = "@"
LITERAL_QUOTE = "'"

NUMBER = "NUMBER"
EOF = "EOF"
ID = "ID"

BUILTIN = "BUILTIN"
EXTRA_BUILTINS_START = "\\"
OEIS_START = "O"

CONSTANT = "CONSTANT"
CONSTANT_START = "_"

MODE = "MODE"
SEQUENCE_1 = ":"
SEQUENCE_2 = "::"
SERIES = ";"
QUERY = "?"

PARAM = "PARAM"
CURRENT = "$"
START = "="
DEFAULT_INPUT = "#"
STRINGED = '"'

PLUS = "+"
MINUS = "-"
MUL = "*"
DIV = "/"
INT_DIV = "//"
EXPONENT = "^"
MOD = "%"

SEPARATOR = ","
META_SEPARATOR = "|"
LPAREN = "("
RPAREN = ")"
NEWLINE = "\n"

# ~`!  # _{}[]?:;

# &= | \"<>

is_one_int = re.compile("^[0-9]$")
is_id = re.compile("^[$v-zA-E]$")
is_input_id = re.compile("^[A-E]$")
is_previous_id = re.compile("^[v-z]$")


class Builtins:
    def __init__(self):
        self.primes = [2, 3, 5, 7]

    def next_prime(self, n):

        # TODO: better algorithm

        if self.primes[-1] > n:
            index = len(self.primes) - 1
            while index >= 0 and self.primes[index] > n:
                index -= 1

            return self.primes[index + 1]
        else:
            to_check = self.primes[-1] + 2

            while True:
                is_prime = True

                j = 1
                while j < len(self.primes) and self.primes[j] < int(math.ceil(n / 3 + 1)):
                    if to_check % self.primes[j] == 0:
                        is_prime = False
                        break

                    j += 1

                if is_prime:
                    self.primes.append(to_check)
                    if to_check > n:
                        return to_check

                to_check += 2

    def root(self, inter, node):
        if len(node.parameters) == 1:
            return math.sqrt(inter.visit(node.parameters[0]))
        return inter.visit(node.parameters[0]) ** (1 / inter.visit(node.parameters[1]))

    def log(self, inter, node):
        if len(node.parameters) == 1:
            return math.log(inter.visit(node.parameters[0]))
        base = inter.visit(node.parameters[1])
        if base == 10:
            math.log10(inter.visit(node.parameters[0]))
        elif base == 2:
            math.log2(inter.visit(node.parameters[0]))
        return math.log(inter.visit(node.parameters[0]), base)

    def get_line(self, node, lines_, index):
        return lines_[index].interpreter.interpret(node.parameters)


builtin_helper = Builtins()

builtins = {
    "a": lambda inter, node: math.fabs(inter.visit(node.parameters[0])),
    "c": lambda inter, node: math.ceil(inter.visit(node.parameters[0])),
    "f": lambda inter, node: math.factorial(inter.visit(node.parameters[0])),
    "F": lambda inter, node: math.floor(inter.visit(node.parameters[0])),
    "l": lambda inter, node: builtin_helper.log(inter, node),
    "p": lambda inter, node: builtin_helper.next_prime(inter.visit(node.parameters[0])),
    "r": lambda inter, node: builtin_helper.root(inter, node),
    "\\c": lambda inter, node: math.cos(inter.visit(node.parameters[0])),
    "\\l": lambda inter, node: math.log10(inter.visit(node.parameters[0])),
    "\\s": lambda inter, node: math.sin(inter.visit(node.parameters[0])),
    "\\t": lambda inter, node: math.tan(inter.visit(node.parameters[0])),
    # functions for other lines
    "\\0": lambda inter, node: builtin_helper.get_line(node, lines, 0),
    "\\1": lambda inter, node: builtin_helper.get_line(node, lines, 1),
    "\\2": lambda inter, node: builtin_helper.get_line(node, lines, 2),
    "\\3": lambda inter, node: builtin_helper.get_line(node, lines, 3),
    "\\4": lambda inter, node: builtin_helper.get_line(node, lines, 4),
    "\\5": lambda inter, node: builtin_helper.get_line(node, lines, 5),
    "\\6": lambda inter, node: builtin_helper.get_line(node, lines, 6),
    "\\7": lambda inter, node: builtin_helper.get_line(node, lines, 7),
    "\\8": lambda inter, node: builtin_helper.get_line(node, lines, 8),
    "\\9": lambda inter, node: builtin_helper.get_line(node, lines, 9)
}

OEIS = {
    "A000001": lambda inter, node: 1
}

builtins.update(OEIS)

constants = {
    "e": math.e,
    "p": math.pi
}

unary_ops = {
    MINUS: lambda x: -x
}

binary_ops = {
    PLUS: lambda x, y: x + y,
    MINUS: lambda x, y: x - y,
    MUL: lambda x, y: x * y,
    DIV: lambda x, y: x / y,
    INT_DIV: lambda x, y: x // y,
    EXPONENT: lambda x, y: x ** y,
    MOD: lambda x, y: x % y
}


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


def get_input_val(char):
    return ord(char) - 65


class Token:
    def __init__(self, type_, val):
        self.type = type_
        self.val = val

    def can_multiply(self):
        return self.type in (NUMBER, ID, BUILTIN, LPAREN, CONSTANT)


class Lexer:

    def __init__(self, text):
        self.text = text
        self.pos = 0

        try:
            self.cur = self.text[self.pos]
        except IndexError:
            self.cur = None

        self.param_found = False
        self.mode = None

    def read_token(self):
        if self.cur is None:
            return Token(EOF, "")

        if not self.mode:
            if self.cur in (SEQUENCE_1, SERIES, QUERY):
                self.mode = self.cur
                self.param_found = True
                self.advance()

                if self.mode == self.cur == SEQUENCE_1:
                    self.mode += SEQUENCE_1
                    self.advance()

                return Token(MODE, self.mode)
            elif self.cur in (START, DEFAULT_INPUT, CURRENT, STRINGED):
                self.param_found = True
                param = self.cur
                self.advance()
                return Token(PARAM, param)
            elif self.cur == META_SEPARATOR:
                self.advance()
                return Token(META_SEPARATOR, META_SEPARATOR)

        if self.param_found:
            if self.cur == "." or is_one_int.match(self.cur):
                return self.read_number()
            elif is_id.match(self.cur):
                return self.read_id()
            elif self.cur == CONSTANT_START:
                self.advance()
                temp = self.cur
                self.advance()
                return Token(CONSTANT, temp)
            elif self.cur in builtins:
                temp = self.cur
                self.advance()
                return Token(BUILTIN, temp)
            elif self.cur == EXTRA_BUILTINS_START and self.cur + self.peek() in builtins:
                temp = self.cur
                self.advance()
                temp += self.cur
                self.advance()
                return Token(BUILTIN, temp)
            elif self.cur == OEIS_START:
                self.advance()
                temp = str(self.read_number().val)
                temp = self.cur + "0" * (6 - len(temp)) + temp
                self.advance()
                return Token(BUILTIN, temp)
            elif self.cur in (PLUS, MINUS, MUL, DIV, EXPONENT, MOD, LPAREN, RPAREN, SEPARATOR):
                if self.cur == DIV and self.peek() == DIV:
                    self.advance()
                    self.advance()
                    return Token(INT_DIV, INT_DIV)

                temp = self.cur
                self.advance()
                return Token(temp, temp)

        if self.cur == LITERAL_ESCAPE:
            self.advance()
            temp = self.cur
            self.advance()
            return Token(LITERAL, temp)
        elif self.cur == LITERAL_QUOTE:
            self.advance()
            temp = self.read_literal()
            self.advance()
            return temp
        elif self.cur == NEWLINE:
            self.advance()
            return Token(NEWLINE, NEWLINE)
        else:
            if self.cur is not None:
                temp = self.cur
                self.advance()
                return Token(LITERAL, temp)

        # raise SyntaxError("Unknown character found : " + self.cur)

    def advance(self):
        self.pos += 1
        try:
            self.cur = self.text[self.pos]
        except IndexError:
            self.cur = None

    def read_id(self):
        temp = self.cur
        self.advance()
        return Token(ID, temp)

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


class Parser:
    def __init__(self, lexer_):
        self.lexer = lexer_
        self.token = self.lexer.read_token()

        self.operations_1 = (EXPONENT,)
        self.operations_2 = (MUL, DIV, INT_DIV, MOD)
        self.operations_3 = (MINUS, PLUS)

    def eat(self, type_):
        if self.token.type == type_:
            self.token = self.lexer.read_token()
        else:
            if type_ == EOF or type_ == RPAREN and self.token.type in (EOF, NEWLINE):
                return

            raise SyntaxError("Incorrect token found: looking for " + type_ + ", found " + self.token.type + " | " + self.token.val + " at " + self.lexer.pos)

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

        while (self.token.type == LITERAL and lit_index <= 2) or (self.token.type == META_SEPARATOR and lit_index < 2):
            if self.token.type == LITERAL:
                literals[lit_index] += self.token.val
                self.eat(LITERAL)
            elif self.token.type == META_SEPARATOR:
                self.eat(META_SEPARATOR)
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

        while self.token.type == SEPARATOR:
            self.eat(SEPARATOR)
            items.append(self.expr())

        return items

    def input_list(self):
        items = [], []

        inc = 0

        while inc <= 1:
            if self.token.type == META_SEPARATOR and inc == 0:
                self.eat(META_SEPARATOR)
                inc += 1
            elif inc == 1:
                break

            items[inc].append(self.expr())

            if self.token.type == SEPARATOR:
                self.eat(SEPARATOR)
            elif self.token.type != META_SEPARATOR:
                break

        return items

    def variable(self):
        node = Var(self.token)
        self.eat(ID)
        return node

    def expr(self):
        # get first term
        node = self.term()

        # valid 3rd priority operations
        while self.token.type in self.operations_3:
            tok = self.token

            self.eat(tok.type)

            node = BinOp(node, tok, self.term())

        return node

    def term(self):
        # get first factor
        node = self.short_term()

        # valid 2nd priority operations
        while self.token.type in self.operations_2 or self.token.can_multiply():
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
        while self.token.type in self.operations_1:
            tok = self.token

            self.eat(tok.type)

            node = BinOp(node, tok, self.factor())

        return node

    def factor(self):
        tok = self.token

        if tok.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(tok, self.factor())
            return node
        elif tok.type == NUMBER:
            self.eat(NUMBER)
            return Number(tok)
        elif tok.type == CONSTANT:
            self.eat(CONSTANT)
            return Constant(tok.val)
        elif tok.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        elif tok.type == BUILTIN:
            builtin = tok.val
            self.eat(BUILTIN)
            nodes = self.items()
            self.eat(RPAREN)
            return Builtin(builtin, nodes)
        elif tok.type == ID:
            return self.variable()

        raise SyntaxError("Unknown factor : " + (tok.val or tok.type))


class NodeVisitor:
    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise ValueError("No visit_" + type(node).__name__ + " method")


class Tester(NodeVisitor):
    def __init__(self):
        self.max_input = -1
        self.mode = None

    def generic_visit(self, node):
        pass

    def visit_Program(self, node):
        self.mode = node.mode

        for each in node.current_start:
            self.visit(each)
        for each in node.start:
            self.visit(each)
        for each in node.input_front:
            self.visit(each)
        for each in node.input_back:
            self.visit(each)
        for each in node.statements:
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
    def __init__(self, tree_, max_input):
        self.tree = tree_
        self.input = None
        self.max_input = max_input

        self.sequence = None
        self.current = 1
        self.current_inc = 1
        self.program = None
        self.join = SEPARATOR

    def visit_Program(self, node):
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
            raise ValueError("Incorrect input length")

        # TODO: Do different if mode == QUERY

        self.program = node
        self.sequence = Sequence(self, node)
        self.current = self.visit(node.current_start[0]) if len(node.current_start) >= 1 else 1
        self.current_inc = self.visit(node.current_start[1]) if len(node.current_start) >= 2 else 1

        # starting literals
        print(node.literals[0], end="")

        if n is None:
            query_n = False
        else:
            n = self.visit(n)
            query_n = n == 0 or n

        if node.is_stringed:
            self.join = node.literals[1] or ""
        else:
            self.join = node.literals[1] or SEPARATOR

        if node.mode in (SEQUENCE_2, QUERY) and not query_n:
            pass

        else:
            sum_ = 0

            for val in self.sequence:

                if node.mode == SEQUENCE_1:
                    if query_n:
                        if n == self.current:
                            print(val, end="")
                            break
                    else:
                        print(val, end=self.join)

                elif node.mode == SEQUENCE_2:
                    if query_n:
                        if n == self.current:
                            print(val, end="")
                            break
                        else:
                            print(val, end=self.join)

                elif node.mode == SERIES:
                    sum_ += val

                    if query_n:
                        if n == self.current:
                            print(sum_, end="")
                            break
                    # elif sum_ == previous_sum

                elif node.mode == QUERY:
                    if query_n:
                        if n == val:
                            print("true", end="")
                            break
                        # elif previous and cur_val < previous:
                        #     print("false", end="")
                        #     done = True

                        # TODO: FIXME
                        elif val > n:
                            print("false", end="")
                            break

                self.current += self.current_inc

        print(node.literals[2], end="")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if node.op.type in binary_ops:
            return binary_ops[node.op.type](left, right)

    def visit_Constant(self, node):
        return constants[node.name]

    def visit_Builtin(self, node):
        if node.builtin in builtins:
            return builtins[node.builtin](self, node)

    def visit_UnaryOp(self, node):
        if node.op.type in unary_ops:
            return unary_ops[node.op.type](self.visit(node.expr))

    def visit_Var(self, node):
        if node.name == CURRENT:
            return self.current
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


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


class Constant(AST):
    def __init__(self, name):
        self.name = name


class Builtin(AST):
    def __init__(self, builtin, parameters):
        self.builtin = builtin
        self.parameters = parameters


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr


class Var(AST):
    def __init__(self, token):
        self.token = token
        self.name = token.val


class Number(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.val

try:
    file = sys.argv[1]
except IndexError:
    file = "source.cq"

# print(file)

source = open(file).read()

# print(source)


def get_input():
    try:
        input_ = input().split(" ")
    except EOFError:
        input_ = []

    if input_ == ['']:
        input_ = []

    if input_ is not []:
        for input_index in range(len(input_)):
            item = input_[input_index]
            try:
                input_[input_index] = Number(Token(NUMBER, int(item)))
            except ValueError:
                try:
                    input_[input_index] = Number(Token(NUMBER, float(item)))
                except ValueError:
                    input_[input_index] = Number(Token(LITERAL, item))

    return input_

user_input = get_input()

# print(user_input)

lines = []


def run(cq_source, cq_input):
    lexer = Lexer(cq_source)
    parser = Parser(lexer)
    programs = parser.parse()

    first = True
    for program in programs:
        tester = Tester()
        tester.visit(program)

        if first:
            first = False
            if tester.mode is None:
                run(":" + cq_source, cq_input)
                return

        interpreter = Interpreter(program, tester.max_input)

        lines.append(Line(program, interpreter))

    lines[0].interpreter.interpret(cq_input)

run(source, user_input)
