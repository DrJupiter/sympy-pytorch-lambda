from typing import Any
from sympy.printing.pycode import AbstractPythonCodePrinter
from sympy.external import import_module
from sympy.codegen.cfunctions import Sqrt
import sympy
import functools as ft
from sympy.core.numbers import equal_valued

import torch

def number_symbol_to_torch(symbol: sympy.NumberSymbol, *args: Any) -> torch.Tensor:
    return torch.tensor(float(symbol))


class TorchPrinter(AbstractPythonCodePrinter):
    printmethod = "_torchcode"
    _module = "torch"

    mapping = {
        sympy.Abs: "torch.abs",
        sympy.sign: "torch.sign",
        # XXX May raise error for ints.
        sympy.ceiling: "torch.ceil",
        sympy.floor: "torch.floor",
        sympy.log: "torch.log",
        sympy.exp: "torch.exp",
        Sqrt: "torch.sqrt",
        sympy.cos: "torch.cos",
        sympy.acos: "torch.acos",
        sympy.sin: "torch.sin",
        sympy.asin: "torch.asin",
        sympy.tan: "torch.tan",
        sympy.atan: "torch.atan",
        sympy.atan2: "torch.atan2",
        # XXX Also may give NaN for complex results.
        sympy.cosh: "torch.cosh",
        sympy.acosh: "torch.acosh",
        sympy.sinh: "torch.sinh",
        sympy.asinh: "torch.asinh",
        sympy.tanh: "torch.tanh",
        sympy.atanh: "torch.atanh",
        sympy.Pow: "torch.pow",
        sympy.re: "torch.real",
        sympy.im: "torch.imag",
        sympy.arg: "torch.angle",
        # XXX May raise error for ints and complexes
        sympy.erf: "torch.erf",
        sympy.loggamma: "torch.lgamma",
        sympy.Eq: "torch.eq",
        sympy.Ne: "torch.ne",
        sympy.StrictGreaterThan: "torch.gt",
        sympy.StrictLessThan: "torch.lt",
        sympy.LessThan: "torch.le",
        sympy.GreaterThan: "torch.ge",
        sympy.And: "torch.logical_and",
        sympy.Or: "torch.logical_or",
        sympy.Not: "torch.logical_not",
        sympy.Max: "torch.max",
        sympy.Min: "torch.min",
        # Matrices
        sympy.MatAdd: "torch.add",
        sympy.HadamardProduct: "torch.mul",
        sympy.Trace: "torch.trace",
        # XXX May raise error for integer matrices.
        sympy.Determinant: "torch.det",
    }
    number_symbols = [cls for cls in sympy.NumberSymbol.__subclasses__()]
    mapping.update({s: ft.partial(number_symbol_to_torch, s()) for s in number_symbols})

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings, torch_version=None
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        version = self._settings["torch_version"]
        if version is None and torch:
            version = torch.__version__
        self.torch_version = version

    def _print_Function(self, expr):
        op = self.mapping.get(type(expr), None)
        if op is None:
            return super(TorchPrinter, self)._print_Basic(expr)
        children = [self._print(arg) for arg in expr.args]
        if len(children) == 1:
            return "%s(%s)" % (self._module_format(op), children[0])
        else:
            return self._expand_fold_binary_op(op, children)

    _print_Expr = _print_Function
    _print_Application = _print_Function
    _print_MatrixExpr = _print_Function
    # TODO: a better class structure would avoid this mess:
    _print_Relational = _print_Function
    _print_Not = _print_Function
    _print_And = _print_Function
    _print_Or = _print_Function
    _print_HadamardProduct = _print_Function
    _print_Trace = _print_Function
    _print_Determinant = _print_Function

    def _print_MatMul(self, expr):
        return self._expand_fold_binary_op("torch.mm", expr.args)

    def _print_MatPow(self, expr):
        return self._expand_fold_binary_op("torch.mm", [expr.base] * expr.exp)

    def _print_MatrixBase(self, expr):
        data = (
            "["
            + ", ".join(
                [
                    "torch.stack("
                    + "["
                    + ", ".join([self._print(j) for j in i])
                    + "]"
                    + ")"
                    for i in expr.tolist()
                ]
            )
            + "]"
        )
        return "%s(%s)" % (self._module_format("torch.stack"), str(data))

    def _print_NDimArray(self, expr):
        array = "[" + ", ".join([self._print(i) for i in expr.tolist()]) + "]"
        return f'{self._module_format("torch.stack")}({array})'

    def _print_list(self, expr):
        return f"torch.stack({super()._print_list(expr)})"

    def _print_Float(self, expr):
        return f"torch.tensor({float(expr)})"

    def _print_Integer(self, expr):
        return f"torch.tensor({int(expr)})"

    def _print_Rational(self, expr):
        return f"torch.tensor({float(expr)})"

    def _print_Zero(self, expr):
        return f"torch.tensor(0)"

    def _print_Pow(self, expr):
        """
        Custom print handler for Pow expressions, to correctly handle square roots.
        """
        base, exp = expr.base, expr.exp
        if equal_valued(exp, 1 / 2):
            return "{}.sqrt({})".format(self._module_format("torch"), self._print(base))
        if equal_valued(exp, -1 / 2):
            return "1 / {}.sqrt({})".format(
                self._module_format("torch"), self._print(base)
            )
        else:
            return "{}.pow({}, {})".format(
                self._module_format("torch"), self._print(base), self._print(exp)
            )

    def _print_CodegenArrayTensorProduct(self, expr):
        # array_list = [j for i, arg in enumerate(expr.args) for j in
        #         (self._print(arg), "[%i, %i]" % (2*i, 2*i+1))]
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ",".join(
            ["".join([next(letters) for j in range(i)]) for i in expr.subranks]
        )
        return '%s("%s", [%s])' % (
            self._module_format("torch.einsum"),
            contraction_string,
            ", ".join([self._print(arg) for arg in expr.args]),
        )


    def _print_CodegenArrayPermuteDims(self, expr):
        return "%s.permute(%s)" % (
            self._print(expr.expr),
            ", ".join([self._print(i) for i in expr.permutation.array_form]),
        )

    def _print_CodegenArrayElementwiseAdd(self, expr):
        return self._expand_fold_binary_op("torch.add", expr.args)