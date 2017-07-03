import re
import math
import sys

LITERAL = "LITERAL"
NUMBER = "NUMBER"
MODE = "MODE"
ID = "ID"
PARAM = "PARAM"
SEPARATOR = "SEPARATOR"
LITERALSEPARATOR = "LITERALSEPARATOR"
BUILTIN = "BUILTIN"
CONSTANT = "CONSTANT"
EOF = "EOF"

PLUS = "+"
MINUS = "-"
MUL = "*"
DIV = "/"
EXPONENT = "^"
MOD = "%"

# ~`!  # _{}[]?:;

# &= | \"<>

LPAREN = "("
RPAREN = ")"

is_one_int = re.compile("^[0-9]$")
is_id = re.compile("^[$q-zA-J]$")
is_input_id = re.compile("^[A-J]$")
is_previous_id = re.compile("^[q-z]$")


class Builtins:
    def __init__(self):
        #self.all = "cfp"
        self.all = "fpR"

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

builtins = Builtins()
constants = "ep"


def get_input_val(char):
    return ord(char) - 65


def get_test_lexer(text):
    test_lexer = Lexer(text)
    token = test_lexer.read_token()
    while token.type != EOF:
        token = test_lexer.read_token()
    return test_lexer


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

        self.max_input = -1

    def read_token(self):
        if self.cur is None:
            return Token(EOF, "")

        if self.mode:
            if self.cur in builtins.all:
                temp = self.cur
                self.advance()
                return Token(BUILTIN, temp)
        else:
            if self.cur in (":", ";", "?"):
                self.mode = self.cur
                self.param_found = True
                self.advance()

                if self.mode == self.cur == ":":
                    self.mode += ":"
                    self.advance()

                return Token(MODE, self.mode)
            elif self.cur == "=":
                self.param_found = True
                self.advance()
                return Token(PARAM, "=")
            elif self.cur == "#":
                self.param_found = True
                self.advance()
                return Token(PARAM, "#")
            elif self.cur == '"':
                self.param_found = True
                self.advance()
                return Token(PARAM, '"')
            elif self.cur == "|":
                self.advance()
                return Token(LITERALSEPARATOR, "|")

        if self.param_found:
            if self.cur == "." or is_one_int.match(self.cur):
                return self.read_number()
            elif is_id.match(self.cur):
                return self.read_id()
            elif self.cur == "_":
                self.advance()
                temp = self.cur
                self.advance()
                return Token(CONSTANT, temp)
            elif self.cur == "+":
                self.advance()
                return Token(PLUS, "+")
            elif self.cur == "-":
                self.advance()
                return Token(MINUS, "-")
            elif self.cur == "*":
                self.advance()
                return Token(MUL, "*")
            elif self.cur == "/":
                self.advance()
                return Token(DIV, "/")
            elif self.cur == "^":
                self.advance()
                return Token(EXPONENT, "^")
            elif self.cur == "%":
                self.advance()
                return Token(MOD, "%")
            elif self.cur == "(":
                self.advance()
                return Token(LPAREN, "(")
            elif self.cur == ")":
                self.advance()
                return Token(RPAREN, ")")
            elif self.cur == ",":
                self.advance()
                return Token(SEPARATOR, ",")

        if self.cur == "@":
            self.advance()
            temp = self.cur
            self.advance()
            return Token(LITERAL, temp)
        elif self.cur == "'":
            self.advance()
            temp = self.read_literal()
            self.advance()
            return temp
        else:
            if self.cur is not None:
                temp = self.cur
                self.advance()
                return Token(LITERAL, temp)

        raise SyntaxError("Unknown character found : " + self.cur)

    def advance(self):
        self.pos += 1
        try:
            self.cur = self.text[self.pos]
        except IndexError:
            self.cur = None

    def read_id(self):
        temp = self.cur
        self.advance()
        if is_input_id.match(temp):
            code = get_input_val(temp)
            if self.max_input < code:
                self.max_input = code
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

        while self.cur is not None and self.cur != "'":
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
        self.test_lexer = get_test_lexer(self.lexer.text)

        self.operations_1 = (EXPONENT,)
        self.operations_2 = (MUL, DIV, MOD)
        self.operations_3 = (MINUS, PLUS)

    def eat(self, type_):
        if self.token.type == type_:
            # print(self.token.type + " " + str(self.token.val))
            self.token = self.lexer.read_token()
        else:
            if type_ == RPAREN or type_ == EOF:
                return

            raise SyntaxError("Incorrect token found: looking for " + type_ + ", found " + self.token.type + " | " + self.token.val + " at " + self.lexer.pos)

    def is_mode_set(self):
        return self.test_lexer.mode

    def parse(self, input_):
        program = self.program(input_)
        self.eat(EOF)
        return program

    def program(self, input_):
        params = self.params()
        mode = self.mode()
        items = self.items()

        default_input_length = len(params.default_input[0]) + len(params.default_input[1])
        user_input_length = len(input_)

        actual_input_length = default_input_length + user_input_length
        expected_input_length = self.test_lexer.max_input + 1

        # print("Actual: " + str(actual_input_length) + ", Expected: " + str(expected_input_length))

        input_length_difference = actual_input_length - expected_input_length

        if input_length_difference == 0 or input_length_difference == 1:
            all_input = []

            all_input.extend(params.default_input[0])
            all_input.extend(input_)
            all_input.extend(params.default_input[1])

            n = None
            if input_length_difference == 1:
                n = int(all_input.pop())

            # print(all_input)

            program = Program(params.literals, all_input, params.start, n, mode, items, params.is_stringed)
            return program

        raise ValueError("Incorrect input length")

    def params(self):
        literals = ["", "", ""]

        lit_index = 0

        while (self.token.type == LITERAL and lit_index <= 2) or (self.token.type == LITERALSEPARATOR and lit_index < 2):
            if self.token.type == LITERAL:
                literals[lit_index] += self.token.val
                self.eat(LITERAL)
            elif self.token.type == LITERALSEPARATOR:
                self.eat(LITERALSEPARATOR)
                lit_index += 1

        default_input = ([], [])
        start = []
        is_stringed = False

        while self.token.type == PARAM:
            if self.token.val == "#":
                self.eat(PARAM)
                default_input = self.num_list()
            elif self.token.val == "=":
                self.eat(PARAM)
                start = self.items()
            elif self.token.val == '"':
                self.eat(PARAM)
                is_stringed = True

        params = Params(literals, default_input, start, is_stringed)
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

    def num_list(self):
        items = ([], [])

        inc = 0

        while inc <= 1:

            if self.token.type == NUMBER:
                items[inc].append(self.token.val)
                self.eat(NUMBER)

            while self.token.type == SEPARATOR:
                self.eat(SEPARATOR)

                if self.token.type == NUMBER:
                    items[inc].append(self.token.val)
                    self.eat(NUMBER)

            if self.token.type == LITERALSEPARATOR and inc == 0:
                self.eat(LITERALSEPARATOR)
                inc += 1
            else:
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

            if tok.type == PLUS:
                self.eat(PLUS)
            elif tok.type == MINUS:
                self.eat(MINUS)

            node = BinOp(node, tok, self.term())

        return node

    def term(self):
        # get first factor
        node = self.short_term()

        # valid 2nd priority operations
        while self.token.type in self.operations_2 or self.token.can_multiply():
            tok = self.token

            if tok.type == MUL:
                self.eat(MUL)
            if tok.type == DIV:
                self.eat(DIV)
            if tok.type == MOD:
                self.eat(MOD)
            if tok.can_multiply():
                tok = Token(MUL, "*")

            node = BinOp(node, tok, self.short_term())

        return node

    def short_term(self):
        # get first factor
        node = self.factor()

        # valid 1st priority operations
        while self.token.type in self.operations_1:
            tok = self.token

            if tok.type == EXPONENT:
                self.eat(EXPONENT)

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


class Interpreter(NodeVisitor):
    def __init__(self, tree):
        self.tree = tree
        self.sequence = []
        self.current = 1
        self.program = None

    def visit_Program(self, node):
        # TODO: Do different if mode == "?"

        self.program = node

        # starting literals
        print(node.literals[0], end="")

        query_n = node.n == 0 or node.n

        if node.is_stringed:
            join = node.literals[1] or ""
        else:
            join = node.literals[1] or ","

        if (node.mode == "::" and not node.n) or (node.mode == "?" and not query_n):
            pass

        else:
            done = False
            sum_ = statement_index = 0
            previous = None

            while not done:
                if self.current <= len(node.start):
                    cur_val = self.visit(node.start[self.current - 1])
                else:
                    if statement_index >= len(node.statements):
                        statement_index = 0

                    cur_val = self.visit(node.statements[statement_index])

                    statement_index += 1

                self.sequence.append(cur_val)

                if node.mode == ":":
                    if node.n:
                        if node.n == self.current:
                            print(cur_val, end="")
                            done = True
                    else:
                        print(cur_val, end=join)
                        # if self.current == 100:
                        #     break

                elif node.mode == "::":
                    if node.n:
                        if node.n == self.current:
                            print(cur_val, end="")
                            done = True
                        else:
                            print(cur_val, end=join)
                    else:
                        pass  # should never be reached

                elif node.mode == ";":
                    sum_ += cur_val

                    if node.n:
                        if node.n == self.current:
                            print(sum_, end="")
                            done = True
                    # elif sum_ == previous_sum

                elif node.mode == "?":
                    if query_n:
                        if node.n == cur_val:
                            print("true", end="")
                            done = True
                        # elif previous and cur_val < previous:
                        #     print("false", end="")
                        #     done = True

                        # TODO: FIXME
                        elif cur_val > node.n:
                            print("false", end="")
                            done = True
                    else:
                        pass  # should never be reached

                if not done:
                    self.current += 1
                    previous = cur_val

        print(node.literals[2], end="")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if node.op.type == PLUS:
            return left + right
        elif node.op.type == MINUS:
            return left - right
        elif node.op.type == MUL:
            return left * right
        elif node.op.type == DIV:
            return left / right
        elif node.op.type == MOD:
            return left % right
        elif node.op.type == EXPONENT:
            return left ** right

    def visit_Constant(self, node):
        if node.name in constants:
            if node.name == "e":
                return math.e
            elif node.name == "p":
                return math.pi

    def visit_Builtin(self, node):
        if node.builtin in builtins.all:
            if node.builtin == "p":
                return builtins.next_prime(self.visit(node.parameters[0]))
            elif node.builtin == "f":
                return math.factorial(self.visit(node.parameters[0]))
            elif node.builtin == "R":
                try:
                    root = self.visit(node.parameters[1])
                except IndexError:
                    return math.sqrt(self.visit(node.parameters[0]))
                return self.visit(node.parameters[0]) ** (1 / root)
            #elif node.builtin == "c":
            #    val = self.visit(node.parameters[0])
            #    try:
            #       return chr(val)
            #    except ValueError:
            #        return val

    def visit_UnaryOp(self, node):
        if node.op.type == MINUS:
            return -self.visit(node.expr)

    def visit_Var(self, node):
        if node.name == "$":
            return self.current
        elif is_previous_id.match(node.name):
            try:
                temp = self.sequence[-1 + ord(node.name) - 122]
            except IndexError:
                temp = 0
            return temp
        elif is_input_id.match(node.name):
            return self.program.input[-1 - get_input_val(node.name)]

    def visit_Number(self, node):
        return node.value

    def interpret(self):
        if self.tree is None:
            return None
        return self.visit(self.tree)


class Params:
    def __init__(self, literals, default_input, start, is_stringed):
        self.literals = literals
        self.default_input = default_input
        self.start = start
        self.is_stringed = is_stringed


class AST:
    pass


class Program(AST):
    def __init__(self, literals, input_, start, n, mode, statements, is_stringed):
        self.literals = literals
        self.input = input_
        self.start = start
        self.n = n
        self.mode = mode
        self.statements = statements
        self.is_stringed = is_stringed


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

try:
    user_input = input().split(" ")
except EOFError:
    user_input = []

for i in range(len(user_input)):
    item = user_input[i]
    try:
        user_input[i] = int(item)
    except ValueError:
        try:
            user_input[i] = float(item)
        except ValueError:
            pass

if user_input == ['']:
    user_input = []

# print(user_input)

lexer = Lexer(source)
parser = Parser(lexer)

if parser.test_lexer.mode is None:
    lexer = Lexer(":" + source)
    parser = Parser(lexer)

tree = parser.parse(user_input)
interpreter = Interpreter(tree)
interpreter.interpret()
