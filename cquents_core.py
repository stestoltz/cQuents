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
        return "<Program: " + ",".join(str(x) for x in self.statements) + ">"


class FiniteSequence(AST):
    def __init__(self, parameters):
        self.parameters = parameters

    def __iter__(self):
        self.index = 0
        return self

    def __getitem__(self, key):
        return self.parameters[key]

    def __next__(self):
        try:
            cur = self.parameters[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return cur

    def __str__(self):
        return "<FiniteSequence: " + ",".join(str(node) for node in self.parameters) + ">"


class Index(AST):
    def __init__(self, base_node, options):
        self.base_node = base_node
        self.options = options

    def __str__(self):
        return "<Index: " + str(self.base_node) + str(self.options) + ">"


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return "<BinOp: " + str(self.left) + " " + str(self.op) + " " + str(self.right) + ">"


class Ternary(AST):
    def __init__(self, condition, true_val, false_val):
        self.condition = condition
        self.true_val = true_val
        self.false_val = false_val

    def __str__(self):
        return "<Ternary: " + str(self.condition) + " ? " + str(self.true_val) + " : " + str(self.false_val) + ">"


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
        return "<Builtin: " + self.builtin + " " + ",".join(str(x) for x in self.parameters) + ">"


class Applier(AST):
    def __init__(self, builtin, parameters):
        self.builtin = builtin
        self.parameters = parameters

    def __str__(self):
        return "<Applier: " + self.builtin + " " + ",".join(str(x) for x in self.parameters) + ">"


class Applied(AST):
    def __init__(self, term):
        self.parameters = (term,)

    def __str__(self):
        return "<Applied: " + self.parameters[0] + ">"


class Conditional(AST):
    def __init__(self, conditional, conditions):
        self.conditional = conditional
        self.conditions = conditions
        self.n = 1

    def __str__(self):
        return "<Conditional: " + self.conditional + " " + ",".join(str(x) for x in self.conditions) + ">"


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
