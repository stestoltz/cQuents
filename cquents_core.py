"""
contains core classes for cquents.py and cquents_builtins.py
"""

"""
cQuents errors
"""


class CQError(Exception):
    pass


class CQSyntaxError(CQError):
    pass


class CQInputError(CQError):
    pass


class CQInternalError(CQError):
    pass


class CQTypeError(CQError):
    pass


"""
Line class - used to store the interpreter and the AST tree for a line
"""


class Line:
    def __init__(self, tree, interpreter):
        self.tree = tree
        self.interpreter = interpreter


"""
Params class - used to store all the information about a line's parameters
"""


class Params:
    def __init__(self, literals, default_input, start, current_start, is_stringed):
        self.literals = literals
        self.default_input = default_input
        self.start = start
        self.current_start = current_start
        self.is_stringed = is_stringed


"""
Parser classes - all possible ASTs
"""


class AST:
    pass


class Program(AST):
    def __init__(self, parameters, mode, statements):
        self.literals = parameters.literals
        self.input_front = parameters.default_input[0]
        self.input_back = parameters.default_input[1]
        self.start = parameters.start
        self.current_start = parameters.current_start
        self.mode = mode
        self.statements = statements
        self.is_stringed = parameters.is_stringed

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


class PostUnaryOp(UnaryOp):
    def __str__(self):
        return "<PostUnaryOp: " + str(self.expr) + " " + str(self.op) + ">"


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


class Literal(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.val

    def __str__(self):
        return "<Literal: " + str(self.token) + " " + self.value + ">"