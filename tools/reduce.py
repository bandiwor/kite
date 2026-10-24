#!/usr/bin/env python3
"""Shrinks a failing fuzz program (by seed) while the kite/oracle mismatch persists.

usage: tools/reduce.py SEED [kite-flags...]
"""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import fuzz

BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "kite")
EXPR = {"num", "var", "bin", "mod", "if", "call", "callv", "block", "loopval"}
COND = {"cmp", "and", "or", "not"}
TRIVIAL_EXPR = ("num", 0)
TRIVIAL_COND = ("cmp", "<", ("num", 0), ("num", 1))

def fails(funcs, body, toplevel, flags):
    try:
        want = fuzz.oracle(funcs, body)
    except Exception:
        return False
    src = fuzz.render(funcs, body, toplevel)
    with tempfile.NamedTemporaryFile("w", suffix=".kite", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([BIN, *flags, path], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)
    if p.returncode != 0:
        # a candidate that merely became invalid (bad arity, unknown name, ...) is not a repro
        invalid = any(m in p.stderr for m in ("syntax error", "expects", "undefined variable", "error: "))
        invalid = invalid and "runtime error" not in p.stderr or "expects" in p.stderr or "undefined variable" in p.stderr
        return not invalid
    return p.stdout.splitlines() != want

def candidates(x):
    """Yield simplified copies of the tree x."""
    if isinstance(x, list):
        for i in range(len(x)):
            yield x[:i] + x[i + 1:]
        for i, item in enumerate(x):
            for c in candidates(item):
                yield x[:i] + [c] + x[i + 1:]
    elif isinstance(x, tuple) and x and isinstance(x[0], str):
        if x[0] in EXPR and x != TRIVIAL_EXPR:
            yield TRIVIAL_EXPR
            for ch in x[1:]:
                if isinstance(ch, tuple) and ch and ch[0] in EXPR:
                    yield ch
        if x[0] in COND and x != TRIVIAL_COND:
            yield TRIVIAL_COND
        if x[0] == "ifs":
            yield ("print", ("num", 1))
        for i in range(1, len(x)):
            ch = x[i]
            if isinstance(ch, (tuple, list)):
                for c in candidates(ch):
                    yield x[:i] + (c,) + x[i + 1:]

def main():
    seed = int(sys.argv[1])
    flags = sys.argv[2:]
    funcs, body, toplevel = fuzz.gen_ast(seed)
    assert fails(funcs, body, toplevel, flags), "seed does not fail"
    changed = True
    while changed:
        changed = False
        pairs = [(c, body) for c in candidates(funcs)] + [(funcs, c) for c in candidates(body)]
        for f2, b2 in pairs:
            if fails(f2, b2, toplevel, flags):
                funcs, body = f2, b2
                changed = True
                break
    src = fuzz.render(funcs, body, toplevel)
    print(src)
    print("// oracle:", fuzz.oracle(funcs, body))

main()
