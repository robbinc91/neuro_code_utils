"""Convert simple Python custom functions to a C++ header.

This utility supports converting small single-expression Python functions
(using arithmetic, math/np unary functions) into `extern "C"` C++ functions
that accept and return `double`. The generated header can be included in the
C++ LibTorch example and compiled with `-DHAS_CUSTOM_FUNCS` to enable usage.

Limitations: only expressions using numbers, names (variables), binary ops,
unary ops, and calls to a limited set of functions (sin, cos, exp, log, tanh)
are supported. Complex control flow, multiple statements, or NumPy array
operations are not supported.
"""
import ast
import argparse
import textwrap


ALLOWED_FUNCS = {
    'sin': 'std::sin',
    'cos': 'std::cos',
    'exp': 'std::exp',
    'log': 'std::log',
    'tanh': 'std::tanh',
    'abs': 'std::fabs',
}


class SimplePyToCpp(ast.NodeVisitor):
    def __init__(self):
        self.code = ''

    def visit_Module(self, node):
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                self.visit(n)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Only support single-argument functions
        if len(node.args.args) != 1:
            raise ValueError('Only single-argument functions are supported')
        arg_name = node.args.args[0].arg
        # Expect single return statement
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
            raise ValueError('Function must consist of a single return statement')
        expr = node.body[0].value
        cpp_expr = self._expr_to_cpp(expr)
        fname = node.name
        self.code += f'extern "C" double {fname}(double {arg_name})' + ' {\n'
        self.code += f'    return {cpp_expr};\n' + '}\n\n'

    def _expr_to_cpp(self, node):
        if isinstance(node, ast.BinOp):
            left = self._expr_to_cpp(node.left)
            right = self._expr_to_cpp(node.right)
            op = self._op_to_cpp(node.op)
            return f'({left} {op} {right})'
        if isinstance(node, ast.UnaryOp):
            operand = self._expr_to_cpp(node.operand)
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            return operand
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                raise ValueError('Unsupported function call form')
            if name not in ALLOWED_FUNCS:
                raise ValueError(f'Function {name} not allowed')
            args = [self._expr_to_cpp(a) for a in node.args]
            return f"{ALLOWED_FUNCS[name]}({', '.join(args)})"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Num):
            return repr(node.n)
        if isinstance(node, ast.Constant):
            return repr(node.value)
        raise ValueError(f'Unsupported AST node: {type(node)}')

    def _op_to_cpp(self, op):
        if isinstance(op, ast.Add):
            return '+'
        if isinstance(op, ast.Sub):
            return '-'
        if isinstance(op, ast.Mult):
            return '*'
        if isinstance(op, ast.Div):
            return '/'
        if isinstance(op, ast.Pow):
            # use std::pow
            return '/*pow*/'
        raise ValueError(f'Unsupported operator: {type(op)}')


def convert(py_path: str, out_header: str):
    src = open(py_path, 'r', encoding='utf-8').read()
    tree = ast.parse(src)
    converter = SimplePyToCpp()
    converter.visit(tree)
    header = textwrap.dedent(f"""
    #pragma once
    #include <cmath>

    // Generated from {py_path}

    {converter.code}
    """)
    with open(out_header, 'w', encoding='utf-8') as f:
        f.write(header)
    print(f'Wrote C++ header: {out_header}')


def main():
    parser = argparse.ArgumentParser(description='Convert simple Python function(s) to C++ header')
    parser.add_argument('pyfile', help='Python file containing function definitions')
    parser.add_argument('--out', default='custom_funcs.h', help='Output C++ header file')
    args = parser.parse_args()
    convert(args.pyfile, args.out)


if __name__ == '__main__':
    main()
