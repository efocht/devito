from collections import defaultdict

import numpy as np

from devito.data import FULL
from devito.ir import (BlankLine, DummyEq, Expression, LocalExpression, FindNodes,
                       FindSymbols, Transformer)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (DefFunction, MacroArgument, ccode, retrieve_indexed,
                              uxreplace)
from devito.tools import Bunch, DefaultOrderedDict, filter_ordered, flatten, prod
from devito.types import Symbol, FIndexed, Indexed
from devito.types.basic import IndexedData


__all__ = ['linearize']


def linearize(iet, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j]` -> `a[i*n + j]`.
    """
    # Simple data structure to avoid generation of duplicated code
    cache = defaultdict(lambda: Bunch(stmts0=[], stmts1=[], cbk=None))

    linearize_accesses(iet, cache=cache, **kwargs)


@iet_pass
def linearize_accesses(iet, **kwargs):
    """
    Actually implement linearize()
    """
    sregistry = kwargs['sregistry']
    cache = kwargs['cache']

    # Find unique sizes (unique -> minimize necessary registers)
    symbol_names = {i.name for i in FindSymbols('indexeds').visit(iet)}
    functions = [f for f in FindSymbols().visit(iet)
                 if f.is_AbstractFunction and f.name in symbol_names]
    functions = sorted(functions, key=lambda f: len(f.dimensions), reverse=True)
    mapper = DefaultOrderedDict(list)
    for f in functions:
        if f not in cache:
            # NOTE: the outermost dimension is unnecessary
            for d in f.dimensions[1:]:
                # TODO: same grid + same halo => same padding, however this is
                # never asserted throughout the compiler yet... maybe should do
                # it when in debug mode at `prepare_arguments` time, ie right
                # before jumping to C?
                mapper[(d, f._size_halo[d], getattr(f, 'grid', None))].append(f)

    # Build all exprs such as `x_fsz0 = u_vec->size[1]`
    imapper = DefaultOrderedDict(list)
    for (d, halo, _), v in mapper.items():
        name = sregistry.make_name(prefix='%s_fsz' % d.name)
        s = Symbol(name=name, dtype=np.int32, is_const=True)
        try:
            expr = LocalExpression(DummyEq(s, v[0]._C_get_field(FULL, d).size))
        except AttributeError:
            assert v[0].is_Array
            expr = LocalExpression(DummyEq(s, v[0].symbolic_shape[d]))
        for f in v:
            imapper[f].append((d, s))
            cache[f].stmts0.append(expr)

    # Build all exprs such as `y_slc0 = y_fsz0*z_fsz0`
    built = {}
    mapper = DefaultOrderedDict(list)
    for f, v in imapper.items():
        for n, (d, _) in enumerate(v):
            expr = prod(list(zip(*v[n:]))[1])
            try:
                s = built[expr]
            except KeyError:
                name = sregistry.make_name(prefix='%s_slc' % d.name)
                s = built[expr] = Symbol(name=name, dtype=np.int32, is_const=True)
                cache[f].stmts1.append(LocalExpression(DummyEq(s, expr)))
            mapper[f].append(s)
    mapper.update([(f, []) for f in functions if f not in mapper])

    # Build defines. For example:
    # `define uL(t, x, y, z) u[(t)*t_slice_sz + (x)*x_slice_sz + (y)*y_slice_sz + (z)]`
    headers = []
    findexeds = {}
    for f, szs in mapper.items():
        if cache[f].cbk is not None:
            # Perhaps we've already built an access macro for `f` through another efunc
            findexeds[f] = cache[f].cbk
        else:
            assert len(szs) == len(f.dimensions) - 1
            pname = sregistry.make_name(prefix='%sL' % f.name)

            expr = sum([MacroArgument(d.name)*s for d, s in zip(f.dimensions, szs)])
            expr += MacroArgument(f.dimensions[-1].name)
            expr = Indexed(IndexedData(f.name, None, f), expr)
            define = DefFunction(pname, f.dimensions)
            headers.append((ccode(define), ccode(expr)))

            cache[f].cbk = findexeds[f] = lambda i, pname=pname: FIndexed(i, pname)

    # Build "functional" Indexeds. For example:
    # `u[t2, x+8, y+9, z+7] => uL(t2, x+8, y+9, z+7)`
    mapper = {}
    for n in FindNodes(Expression).visit(iet):
        subs = {i: findexeds[i.function](i) for i in retrieve_indexed(n.expr)}
        mapper[n] = n._rebuild(expr=uxreplace(n.expr, subs))

    # Put together all of the necessary exprs for `y_fsz0`, ..., `y_slc0`, ...
    stmts0 = filter_ordered(flatten(cache[f].stmts0 for f in functions))
    if stmts0:
        stmts0.append(BlankLine)
    stmts1 = filter_ordered(flatten(cache[f].stmts1 for f in functions))
    if stmts1:
        stmts1.append(BlankLine)

    iet = Transformer(mapper).visit(iet)
    body = iet.body._rebuild(body=tuple(stmts0) + tuple(stmts1) + iet.body.body)
    iet = iet._rebuild(body=body)

    return iet, {'headers': headers}
