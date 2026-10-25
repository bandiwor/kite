#!/usr/bin/env python3
"""Differential fuzzer for Kite.

Generates random terminating programs, evaluates them with an independent
reference interpreter (pure Python, same AST) and compares the output with the
real `kite` binary in three configurations: default, --no-opt, --gc-stress.

usage: tools/fuzz.py [count] [seed] [path/to/kite]
"""
import math, os, random, subprocess, sys, tempfile

# --------------------------------------------------------------------------- generator

class Gen:
    def __init__(self, rng):
        self.r = rng
        self.uid = 0
        self.funcs = []          # (name, arity) of already defined functions
        self.scopes = [[]]       # stack of lists of visible variable names
        self.closures = []       # names of closure variables (callable with 1 arg)
        self.in_loop = 0
        self.in_func = 0
        self.depth = 0

    def fresh(self, p="v"):
        self.uid += 1
        return f"{p}{self.uid}"

    def visible(self):
        return [n for s in self.scopes for n in s]

    def numeric(self):
        # closures (c*), tables (t*) and struct instances (q*) are not numbers
        return [n for n in self.visible() if n[0] not in "ctq"]

    def tables(self):
        return [n for n in self.visible() if n[0] == "t"]

    def insts(self):
        return [n for n in self.visible() if n[0] == "q"]

    def assignable(self):
        # loop counters (w*, n*) must never be reassigned or the loops might not terminate
        return [n for n in self.numeric() if n[0] not in "wn"]

    # expressions ----------------------------------------------------------
    def expr(self, d=0):
        r = self.r
        vs = self.numeric()
        c = r.random()
        if d > 3 or c < 0.22:
            if vs and r.random() < 0.7:
                return ("var", r.choice(vs))
            return ("num", r.randint(-9, 30))
        if c < 0.50:
            return ("bin", r.choice("+-*"), self.expr(d + 1), self.expr(d + 1))
        if c < 0.58:
            return ("mod", self.expr(d + 1), r.randint(1, 9))
        if c < 0.70:
            return ("if", self.cond(d + 1), self.expr(d + 1), self.expr(d + 1))
        if c < 0.78 and self.funcs:
            name, ar = r.choice(self.funcs)
            return ("call", name, [self.expr(d + 1) for _ in range(ar)])
        tabs, qs = self.tables(), self.insts()
        if c < 0.80 and tabs:
            t = r.choice(tabs)
            if r.random() < 0.25:
                return ("tlen", t)
            return ("tget", t, self.expr(d + 1), self.fresh("k"))
        if c < 0.82 and qs:
            return ("pget", r.choice(qs), r.choice("ab"))
        vis = set(self.visible())
        live = [n for n in self.closures if n in vis]
        if c < 0.84 and live:
            return ("callv", r.choice(live), self.expr(d + 1))
        if c < 0.92 and self.depth < 2:
            self.depth += 1
            self.scopes.append([])
            stmts = self.stmts(r.randint(0, 2))
            e = self.expr(d + 1)
            self.scopes.pop()
            self.depth -= 1
            return ("block", stmts, e)
        if c < 0.97 and self.depth < 2:
            self.depth += 1
            self.scopes.append([])
            k = self.fresh("n")
            self.scopes[-1].append(k)
            saved_loop, self.in_loop = self.in_loop, 0
            stmts = self.stmts(r.randint(0, 2))
            cnd = self.cond(d + 1)
            val = self.expr(d + 1)
            self.in_loop = saved_loop
            self.scopes.pop()
            self.depth -= 1
            return ("loopval", k, r.randint(1, 4), stmts, cnd, val)
        return ("num", r.randint(0, 9))

    def cond(self, d=0):
        r = self.r
        c = r.random()
        qs = self.insts()
        if len(qs) >= 1 and c < 0.12:
            return ("qcmp", r.choice(["<", "<=", ">", ">=", "==", "!="]), r.choice(qs), r.choice(qs))
        if d > 2 or c < 0.6:
            return ("cmp", r.choice(["<", "<=", ">", ">=", "==", "!="]), self.expr(d + 1), self.expr(d + 1))
        if c < 0.75:
            return ("and", self.cond(d + 1), self.cond(d + 1))
        if c < 0.9:
            return ("or", self.cond(d + 1), self.cond(d + 1))
        return ("not", self.cond(d + 1))

    # statements -----------------------------------------------------------
    def table_struct_stmt(self):
        r = self.r
        tabs, qs = self.tables(), self.insts()
        ops = ["tnew", "qnew"]
        if tabs: ops += ["tset", "tset", "taug", "tdel", "tpush", "tsum"]
        if qs: ops += ["qset", "qaug", "qshow", "qop"]
        op = r.choice(ops)
        if op == "tnew":
            name = self.fresh("t")
            items = [self.expr() for _ in range(r.randint(0, 3))] if r.random() < 0.6 else None
            self.scopes[-1].append(name)
            return ("tnew", name, items)
        if op == "qnew":
            name = self.fresh("q")
            e1, e2 = self.expr(), self.expr()
            self.scopes[-1].append(name)
            return ("qnew", name, e1, e2)
        if op in ("tset", "taug", "tdel", "tpush", "tsum"):
            t = r.choice(tabs)
            if op == "tset": return ("tset", t, self.expr(), self.expr())
            if op == "taug": return ("taug", t, self.expr(), r.choice("+-"), self.expr(), self.fresh("k"))
            if op == "tdel": return ("tdel", t, self.expr())
            if op == "tpush": return ("tpush", t, self.expr())
            return ("tsum", t, self.fresh("s"), self.fresh("k"), self.fresh("v"), r.random() < 0.5)
        q = r.choice(qs)
        if op == "qset": return ("qset", q, r.choice("ab"), self.expr())
        if op == "qaug": return ("qaug", q, r.choice("ab"), r.choice("+-"), self.expr())
        if op == "qshow": return ("qshow", q, r.random() < 0.5)
        name = self.fresh("q")
        kind2 = r.choice(["add", "neg"])
        n1, n2 = r.choice(qs), r.choice(qs)
        self.scopes[-1].append(name)
        return ("qop", name, kind2, n1, n2)


    def stmts(self, n):
        out = []
        for _ in range(n):
            out.append(self.stmt())
        return out

    def stmt(self):
        r = self.r
        vs = self.numeric()
        if r.random() < 0.3:
            return self.table_struct_stmt()
        c = r.random()
        if c < 0.18 or not vs:
            # declare (possibly shadowing an outer name inside a nested scope)
            if len(self.scopes) > 1 and vs and r.random() < 0.3:
                name = r.choice(vs)
                if name in self.scopes[-1] or name.startswith("n"):
                    name = self.fresh()
            else:
                name = self.fresh()
            e = self.expr()
            self.scopes[-1].append(name)
            return ("var", name, e)
        av = self.assignable()
        if c < 0.34 and av:
            return ("assign", r.choice(av), self.expr())
        if c < 0.46 and av:
            return ("aug", r.choice(av), r.choice("+-*"), self.expr())
        if c < 0.58:
            return ("print", self.expr())
        if c < 0.70 and self.depth < 3:
            self.depth += 1
            self.scopes.append([]); a = self.stmts(r.randint(0, 3)); self.scopes.pop()
            self.scopes.append([]); b = self.stmts(r.randint(0, 3)); self.scopes.pop()
            self.depth -= 1
            return ("ifs", self.cond(), a, b)
        if c < 0.78 and self.depth < 3:
            self.depth += 1
            i = self.fresh("i")
            self.scopes.append([i]); self.in_loop += 1
            body = self.stmts(r.randint(1, 4))
            self.in_loop -= 1; self.scopes.pop(); self.depth -= 1
            return ("for", i, r.randint(-2, 3), r.randint(0, 6), body)
        if c < 0.84 and self.depth < 3:
            self.depth += 1
            w = self.fresh("w")
            self.scopes.append([w])
            guard = self.cond() if r.random() < 0.5 else None   # generated outside the loop context
            self.in_loop += 1
            body = self.stmts(r.randint(1, 4))
            self.in_loop -= 1; self.scopes.pop(); self.depth -= 1
            return ("while", w, r.randint(1, 6), guard, body)
        if c < 0.90 and self.in_loop:
            return ("brk" if r.random() < 0.5 else "cont", self.cond())
        if c < 0.95:
            p = self.fresh("p")
            self.scopes.append([p])
            saved_loop, self.in_loop = self.in_loop, 0   # `break` cannot cross a function boundary
            e = self.expr()
            self.in_loop = saved_loop
            self.scopes.pop()
            name = self.fresh("c")
            self.closures.append(name)
            self.scopes[-1].append(name)
            return ("closure", name, p, e)
        if self.in_func and r.random() < 0.5:
            return ("ret", self.cond(), self.expr())
        return ("print", self.expr())

    def function(self):
        name = self.fresh("f")
        ar = self.r.randint(0, 3)
        params = [self.fresh("a") for _ in range(ar)]
        saved = (self.scopes, self.closures, self.in_loop, self.in_func, self.depth)
        self.scopes = [list(params)]
        self.closures = []
        self.in_loop = 0
        self.in_func = 1
        self.depth = 0
        body = self.stmts(self.r.randint(1, 5))
        tail = self.expr()
        self.scopes, self.closures, self.in_loop, self.in_func, self.depth = saved
        self.funcs.append((name, ar))
        return ("fn", name, params, body, tail)

# --------------------------------------------------------------------------- Kite printer

def kexpr(e):
    k = e[0]
    if k == "num":
        return f"({e[1]})" if e[1] < 0 else str(e[1])
    if k == "var":
        return e[1]
    if k == "bin":
        return f"(({kexpr(e[2])} {e[1]} {kexpr(e[3])}) % 100003)"
    if k == "mod":
        return f"({kexpr(e[1])} % {e[2]})"
    if k == "if":
        return f"(if {kcond(e[1])} {{ {kexpr(e[2])} }} else {{ {kexpr(e[3])} }})"
    if k == "call":
        return f"{e[1]}({', '.join(kexpr(a) for a in e[2])})"
    if k == "callv":
        return f"{e[1]}({kexpr(e[2])})"
    if k == "tget":
        _, t, ie, kk = e
        return f"do {{ const {kk} = ({kexpr(ie)} % 5); if {t}:has({kk}) {{ {t}[{kk}] }} else {{ 0 }} }}"
    if k == "tlen":
        return f"{e[1]}:len()"
    if k == "pget":
        return f"{e[1]}.{e[2]}"
    if k == "block":
        return "do { " + " ".join(kstmt(s) for s in e[1]) + f" {kexpr(e[2])} }}"
    if k == "loopval":
        _, n, lim, stmts, cnd, val = e
        body = " ".join(kstmt(s) for s in stmts)
        return (f"do {{ var {n} = 0; loop {{ {n} += 1; {body} "
                f"if {kcond(cnd)} || {n} > {lim} {{ break {kexpr(val)}; }} }} }}")
    raise Exception(k)

def kcond(c):
    k = c[0]
    if k == "cmp":
        return f"({kexpr(c[2])} {c[1]} {kexpr(c[3])})"
    if k == "and":
        return f"({kcond(c[1])} && {kcond(c[2])})"
    if k == "or":
        return f"({kcond(c[1])} || {kcond(c[2])})"
    if k == "not":
        return f"(!{kcond(c[1])})"
    if k == "qcmp":
        return f"({c[2]} {c[1]} {c[3]})"
    raise Exception(k)

def kblock(ss):
    return "{ " + " ".join(kstmt(s) for s in ss) + " }"

def kstmt(s):
    k = s[0]
    if k == "var":
        return f"var {s[1]} = {kexpr(s[2])};"
    if k == "assign":
        return f"{s[1]} = {kexpr(s[2])};"
    if k == "aug":
        return f"{s[1]} = (({s[1]} {s[2]} {kexpr(s[3])}) % 100003);"
    if k == "print":
        return f"print({kexpr(s[1])});"
    if k == "ifs":
        return f"if {kcond(s[1])} {kblock(s[2])} else {kblock(s[3])}"
    if k == "for":
        return f"for {s[1]} in {s[2]}..{s[3]} {kblock(s[4])}"
    if k == "while":
        _, w, lim, guard, body = s
        g = f" && {kcond(guard)}" if guard else ""
        return f"do {{ var {w} = 0; while {w} < {lim}{g} {{ {w} += 1; " + " ".join(kstmt(x) for x in body) + " } }"
    if k == "brk":
        return f"if {kcond(s[1])} {{ break; }}"
    if k == "cont":
        return f"if {kcond(s[1])} {{ continue; }}"
    if k == "closure":
        return f"const {s[1]} = |{s[2]}| => {kexpr(s[3])};"
    if k == "ret":
        return f"if {kcond(s[1])} {{ return {kexpr(s[2])}; }}"
    if k == "tnew":
        if s[2] is None:
            return f"var {s[1]} = {{}};"
        return f"var {s[1]} = {{{', '.join(f'{i}: {kexpr(x)}' for i, x in enumerate(s[2]))}}};"
    if k == "qnew":
        return f"var {s[1]} = P({kexpr(s[2])}, {kexpr(s[3])});"
    if k == "tset":
        return f"{s[1]}[({kexpr(s[2])} % 5)] = {kexpr(s[3])};"
    if k == "taug":
        _, t, ie, op, e, kk = s
        return f"do {{ const {kk} = ({kexpr(ie)} % 5); if {t}:has({kk}) {{ {t}[{kk}] {op}= ({kexpr(e)} % 7); }} }}"
    if k == "tdel":
        return f"{s[1]}[({kexpr(s[2])} % 5)] = nil;"
    if k == "tpush":
        # tables have no push: take the first free integer key
        kk = "pk_" + s[1]
        return f"do {{ const {kk}v = {kexpr(s[2])}; var {kk} = 0; while {s[1]}:has({kk}) {{ {kk} += 1; }} {s[1]}[{kk}] = {kk}v; }}"
    if k == "tsum":
        _, t, sv, kk, vv, with_key = s
        if with_key:
            return f"do {{ var {sv} = 0; for {kk}, {vv} in {t} {{ {sv} += {vv} + {kk}; }} print({sv}); }}"
        return f"do {{ var {sv} = 0; for {vv} in {t} {{ {sv} += {vv}; }} print({sv}); }}"
    if k == "qset":
        return f"{s[1]}.{s[2]} = {kexpr(s[3])};"
    if k == "qaug":
        return f"{s[1]}.{s[2]} {s[3]}= ({kexpr(s[4])} % 7);"
    if k == "qshow":
        if s[2]:
            return f"print({s[1]});"
        return f"print({s[1]}.a, {s[1]}.b);"
    if k == "qop":
        _, name, kind, n1, n2 = s
        return f"var {name} = {n1} + {n2};" if kind == "add" else f"var {name} = -{n1};"
    raise Exception(k)

PRELUDE = """struct P { a, b }
impl P {
    fn new(self, a, b) { self.a = a; self.b = b; }
    fn add(x, y) { P((x.a + y.a) % 100003, (x.b + y.b) % 100003) }
    fn neg(x) { P(-x.a, -x.b) }
    fn eq(x, y) { x.a == y.a && x.b == y.b }
    fn lt(x, y) { x.a < y.a || (x.a == y.a && x.b < y.b) }
}
"""

def kfn(f):
    _, name, params, body, tail = f
    return f"fn {name}({', '.join(params)}) {{ " + " ".join(kstmt(s) for s in body) + f" {kexpr(tail)} }}"

# --------------------------------------------------------------------------- reference interpreter

class Break(Exception): pass
class Continue(Exception): pass
class Return(Exception):
    def __init__(self, v): self.v = v

class Env:
    __slots__ = ("vars", "parent")
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent
    def find(self, name):
        e = self
        while e is not None:
            if name in e.vars:
                return e
            e = e.parent
        raise KeyError(name)
    def get(self, name): return self.find(name).vars[name]
    def set(self, name, v): self.find(name).vars[name] = v

def fmt(x):
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return str(int(x))   # negative zero prints as 0
    return repr(x)

class Closure:
    def __init__(self, param, body, env): self.param, self.body, self.env = param, body, env

class Tab:
    """Reference model of a Kite table restricted to integer keys."""
    def __init__(self, items=None):
        self.d = {}
        for x in (items or []): self.push(x)
    def first_free(self):
        n = 0
        while n in self.d: n += 1
        return n
    def push(self, v): self.d[self.first_free()] = v

class Inst:
    def __init__(self, a, b): self.a, self.b = a, b

def p_add(x, y): return Inst(math.fmod(x.a + y.a, 100003.0), math.fmod(x.b + y.b, 100003.0))
def p_lt(x, y): return x.a < y.a or (x.a == y.a and x.b < y.b)
def p_eq(x, y): return x.a == y.a and x.b == y.b

class Ref:
    def __init__(self): self.funcs = {}; self.out = []

def ev(e, env, R):
    k = e[0]
    if k == "num": return float(e[1])
    if k == "var": return env.get(e[1])
    if k == "bin":
        a = ev(e[2], env, R); b = ev(e[3], env, R)
        return math.fmod(a + b if e[1] == "+" else a - b if e[1] == "-" else a * b, 100003.0)
    if k == "mod": return math.fmod(ev(e[1], env, R), float(e[2]))
    if k == "if": return ev(e[2], env, R) if cv(e[1], env, R) else ev(e[3], env, R)
    if k == "call":
        args = [ev(a, env, R) for a in e[2]]
        return call_fn(R.funcs[e[1]], args, R)
    if k == "tget":
        _, t, ie, kk = e
        key = math.fmod(ev(ie, env, R), 5.0)
        key = int(key)
        return env.get(t).d.get(key, 0.0)
    if k == "tlen": return float(len(env.get(e[1]).d))
    if k == "pget": return getattr(env.get(e[1]), e[2])
    if k == "callv":
        c = env.get(e[1]); a = ev(e[2], env, R)
        ne = Env(c.env); ne.vars[c.param] = a
        try:
            return ev(c.body, ne, R)
        except Return as r:
            return r.v
    if k == "block":
        ne = Env(env)
        for s in e[1]: ex(s, ne, R)
        return ev(e[2], ne, R)
    if k == "loopval":
        _, n, lim, stmts, cnd, val = e
        outer = Env(env)               # do { var n = 0; loop {...} }
        outer.vars[n] = 0.0
        while True:
            body = Env(outer)          # fresh scope per iteration
            outer.vars[n] += 1
            try:
                for s in stmts: ex(s, body, R)
            except Break: raise Exception("break/continue escaped loopval")
            except Continue: raise Exception("continue escaped loopval")
            if cv(cnd, body, R) or outer.vars[n] > lim:
                return ev(val, body, R)
    raise Exception(k)

def cv(c, env, R):
    k = c[0]
    if k == "cmp":
        a = ev(c[2], env, R); b = ev(c[3], env, R)
        o = c[1]
        return a < b if o == "<" else a <= b if o == "<=" else a > b if o == ">" else a >= b if o == ">=" else a == b if o == "==" else a != b
    if k == "and": return cv(c[1], env, R) and cv(c[2], env, R)
    if k == "or": return cv(c[1], env, R) or cv(c[2], env, R)
    if k == "not": return not cv(c[1], env, R)
    if k == "qcmp":
        x, y, o = env.get(c[2]), env.get(c[3]), c[1]
        if o == "<": return p_lt(x, y)
        if o == ">": return p_lt(y, x)
        if o == "<=": return not p_lt(y, x)
        if o == ">=": return not p_lt(x, y)
        if o == "==": return p_eq(x, y)
        return not p_eq(x, y)
    raise Exception(k)

def run_block(ss, env, R):
    ne = Env(env)
    for s in ss: ex(s, ne, R)

def ex(s, env, R):
    k = s[0]
    if k == "var": env.vars[s[1]] = ev(s[2], env, R)
    elif k == "assign": env.set(s[1], ev(s[2], env, R))
    elif k == "aug":
        cur = env.get(s[1]); v = ev(s[3], env, R)
        env.set(s[1], math.fmod(cur + v if s[2] == "+" else cur - v if s[2] == "-" else cur * v, 100003.0))
    elif k == "print": R.out.append(fmt(ev(s[1], env, R)))
    elif k == "ifs": run_block(s[2] if cv(s[1], env, R) else s[3], env, R)
    elif k == "for":
        _, i, lo, hi, body = s
        cur = float(lo)
        if cur < hi:
            while True:
                ie = Env(env); ie.vars[i] = cur       # fresh variable per iteration
                try:
                    run_block(body, ie, R)
                except Break: break
                except Continue: pass
                cur += 1
                if not cur < hi: break
    elif k == "while":
        _, w, lim, guard, body = s
        oe = Env(env); oe.vars[w] = 0.0
        while oe.vars[w] < lim and (guard is None or cv(guard, oe, R)):
            oe.vars[w] += 1
            try:
                run_block(body, oe, R)
            except Break: break
            except Continue: pass
    elif k == "brk":
        if cv(s[1], env, R): raise Break()
    elif k == "cont":
        if cv(s[1], env, R): raise Continue()
    elif k == "closure": env.vars[s[1]] = Closure(s[2], s[3], env)
    elif k == "ret":
        if cv(s[1], env, R): raise Return(ev(s[2], env, R))
    elif k == "tnew":
        env.vars[s[1]] = Tab(None if s[2] is None else [ev(x, env, R) for x in s[2]])
    elif k == "qnew":
        a = ev(s[2], env, R); b = ev(s[3], env, R)
        env.vars[s[1]] = Inst(a, b)
    elif k == "tset":
        t = env.get(s[1]); key = int(math.fmod(ev(s[2], env, R), 5.0)); v = ev(s[3], env, R)
        t.d[key] = v
    elif k == "taug":
        _, t, ie, op, e, kk = s
        t = env.get(t); key = int(math.fmod(ev(ie, env, R), 5.0))
        if key in t.d:
            cur = t.d[key]                       # read before the right-hand side runs
            v = math.fmod(ev(e, env, R), 7.0)
            t.d[key] = cur + v if op == "+" else cur - v
    elif k == "tdel":
        t = env.get(s[1]); key = int(math.fmod(ev(s[2], env, R), 5.0))
        t.d.pop(key, None)
    elif k == "tpush":
        t = env.get(s[1]); v = ev(s[2], env, R)
        t.push(v)
    elif k == "tsum":
        t = env.get(s[1]); tot = 0.0
        for key, v in t.d.items(): tot += v + (key if s[2] else 0)
        R.out.append(fmt(tot))
    elif k == "qset":
        q = env.get(s[1]); v = ev(s[3], env, R)
        setattr(q, s[2], v)
    elif k == "qaug":
        q = env.get(s[1])
        cur = getattr(q, s[2])                   # read before the right-hand side runs
        v = math.fmod(ev(s[4], env, R), 7.0)
        setattr(q, s[2], cur + v if s[3] == "+" else cur - v)
    elif k == "qshow":
        q = env.get(s[1])
        if s[2]: R.out.append(f"P{{a: {fmt(q.a)}, b: {fmt(q.b)}}}")
        else: R.out.append(f"{fmt(q.a)} {fmt(q.b)}")
    elif k == "qop":
        _, name, kind, n1, n2 = s
        x = env.get(n1)
        env.vars[name] = p_add(x, env.get(n2)) if kind == "add" else Inst(-x.a, -x.b)
    else: raise Exception(k)

def call_fn(f, args, R):
    _, name, params, body, tail = f
    env = Env(None)
    for p, a in zip(params, args): env.vars[p] = a
    be = Env(env)
    try:
        for s in body: ex(s, be, R)
        return ev(tail, be, R)
    except Return as r:
        return r.v

# --------------------------------------------------------------------------- static name resolution
# The oracle must use *lexical* scoping like Kite: a name refers to the declaration
# that was visible where it is written.  This pass renames every declaration to a
# unique `name@n` and rewrites all references accordingly.

class Resolver:
    def __init__(self):
        self.n = 0
        self.scopes = []

    def declare(self, name):
        self.n += 1
        u = f"{name}@{self.n}"
        self.scopes[-1][name] = u
        return u

    def lookup(self, name):
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        raise KeyError(name)

    def push(self): self.scopes.append({})
    def pop(self): self.scopes.pop()

    def fn(self, f):
        _, name, params, body, tail = f
        self.push()
        ps = [self.declare(p) for p in params]
        self.push()
        b = [self.stmt(x) for x in body]
        t = self.expr(tail)
        self.pop(); self.pop()
        return ("fn", name, ps, b, t)

    def block(self, ss):
        self.push()
        out = [self.stmt(x) for x in ss]
        self.pop()
        return out

    def expr(self, e):
        k = e[0]
        if k == "num": return e
        if k == "var": return ("var", self.lookup(e[1]))
        if k == "bin": return ("bin", e[1], self.expr(e[2]), self.expr(e[3]))
        if k == "mod": return ("mod", self.expr(e[1]), e[2])
        if k == "if": return ("if", self.cond(e[1]), self.expr(e[2]), self.expr(e[3]))
        if k == "call": return ("call", e[1], [self.expr(a) for a in e[2]])
        if k == "callv": return ("callv", self.lookup(e[1]), self.expr(e[2]))
        if k == "tget":
            _, t, ie, kk = e
            ri = self.expr(ie)
            self.push()
            uk = self.declare(kk)
            self.pop()
            return ("tget", self.lookup(t), ri, uk)
        if k == "tlen": return ("tlen", self.lookup(e[1]))
        if k == "pget": return ("pget", self.lookup(e[1]), e[2])
        if k == "block":
            self.push()
            ss = [self.stmt(x) for x in e[1]]
            v = self.expr(e[2])
            self.pop()
            return ("block", ss, v)
        if k == "loopval":
            _, n, lim, stmts, cnd, val = e
            self.push()
            un = self.declare(n)
            self.push()
            ss = [self.stmt(x) for x in stmts]
            c = self.cond(cnd)
            v = self.expr(val)
            self.pop(); self.pop()
            return ("loopval", un, lim, ss, c, v)
        raise Exception(k)

    def cond(self, c):
        k = c[0]
        if k == "cmp": return ("cmp", c[1], self.expr(c[2]), self.expr(c[3]))
        if k in ("and", "or"): return (k, self.cond(c[1]), self.cond(c[2]))
        if k == "not": return ("not", self.cond(c[1]))
        if k == "qcmp": return ("qcmp", c[1], self.lookup(c[2]), self.lookup(c[3]))
        raise Exception(k)

    def stmt(self, s):
        k = s[0]
        if k == "var":
            init = self.expr(s[2])
            return ("var", self.declare(s[1]), init)
        if k == "assign": return ("assign", self.lookup(s[1]), self.expr(s[2]))
        if k == "aug": return ("aug", self.lookup(s[1]), s[2], self.expr(s[3]))
        if k == "print": return ("print", self.expr(s[1]))
        if k == "ifs": return ("ifs", self.cond(s[1]), self.block(s[2]), self.block(s[3]))
        if k == "for":
            _, i, lo, hi, body = s
            self.push()
            ui = self.declare(i)
            b = self.block(body)
            self.pop()
            return ("for", ui, lo, hi, b)
        if k == "while":
            _, w, lim, guard, body = s
            self.push()
            uw = self.declare(w)
            g = self.cond(guard) if guard else None
            b = self.block(body)
            self.pop()
            return ("while", uw, lim, g, b)
        if k in ("brk", "cont"): return (k, self.cond(s[1]))
        if k == "closure":
            _, name, p, e = s
            uname = self.declare(name)
            self.push()
            up = self.declare(p)
            body = self.expr(e)
            self.pop()
            return ("closure", uname, up, body)
        if k == "ret": return ("ret", self.cond(s[1]), self.expr(s[2]))
        if k == "tnew":
            items = None if s[2] is None else [self.expr(x) for x in s[2]]
            return ("tnew", self.declare(s[1]), items)
        if k == "qnew":
            e1, e2 = self.expr(s[2]), self.expr(s[3])
            return ("qnew", self.declare(s[1]), e1, e2)
        if k == "tset": 
            t = self.lookup(s[1]); ie = self.expr(s[2]); e = self.expr(s[3])
            return ("tset", t, ie, e)
        if k == "taug":
            _, t, ie, op, e, kk = s
            t = self.lookup(t); ri = self.expr(ie)
            self.push(); uk = self.declare(kk); re_ = self.expr(e); self.pop()
            return ("taug", t, ri, op, re_, uk)
        if k == "tdel": return ("tdel", self.lookup(s[1]), self.expr(s[2]))
        if k == "tpush": return ("tpush", self.lookup(s[1]), self.expr(s[2]))
        if k == "tsum":
            _, t, sv, kk, vv, with_key = s
            return ("tsum", self.lookup(t), with_key)
        if k == "qset": return ("qset", self.lookup(s[1]), s[2], self.expr(s[3]))
        if k == "qaug": return ("qaug", self.lookup(s[1]), s[2], s[3], self.expr(s[4]))
        if k == "qshow": return ("qshow", self.lookup(s[1]), s[2])
        if k == "qop":
            _, name, kind, n1, n2 = s
            a, b = self.lookup(n1), self.lookup(n2)
            return ("qop", self.declare(name), kind, a, b)
        raise Exception(k)

# --------------------------------------------------------------------------- driver

def gen_ast(seed):
    rng = random.Random(seed)
    g = Gen(rng)
    funcs = [g.function() for _ in range(rng.randint(0, 3))]
    toplevel = rng.random() < 0.5
    g.scopes = [[]]
    g.closures = []
    g.in_func = 0
    g.in_loop = 0
    g.depth = 0
    body = g.stmts(rng.randint(3, 10))
    return funcs, body, toplevel

def render(funcs, body, toplevel):
    src = PRELUDE + "\n".join(kfn(f) for f in funcs) + "\n"
    if toplevel:
        src += "\n".join(kstmt(s) for s in body) + "\n"
    else:
        src += "fn main() { " + " ".join(kstmt(s) for s in body) + " }\nmain();\n"
    return src

def oracle(funcs, body):
    R = Ref()
    rs = Resolver()
    rs.push()
    rfuncs = [rs.fn(f) for f in funcs]
    for f in rfuncs: R.funcs[f[1]] = f
    rs.push()
    rbody = [rs.stmt(x) for x in body]
    genv = Env(None)
    for s in rbody: ex(s, genv, R)
    return R.out

def make_program(seed):
    funcs, body, toplevel = gen_ast(seed)
    return render(funcs, body, toplevel), oracle(funcs, body)

def run_kite(binary, path, flags):
    p = subprocess.run([binary, *flags, path], capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout.splitlines(), p.stderr

def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    binary = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "target", "release", "kite")
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for seed in range(seed0, seed0 + count):
            src, want = make_program(seed)
            path = os.path.join(d, f"p{seed}.kite")
            open(path, "w").write(src)
            for name, flags in (("default", []), ("no-opt", ["--no-opt"]), ("gc-stress", ["--gc-stress"])):
                code, got, err = run_kite(binary, path, flags)
                if code != 0 or got != want:
                    bad += 1
                    keep = f"/tmp/kite_fuzz_fail_{seed}.kite"
                    open(keep, "w").write(src)
                    print(f"FAIL seed={seed} mode={name} exit={code} (program saved to {keep})")
                    if err: print("  stderr:", err.strip().splitlines()[0:3])
                    for i in range(max(len(got), len(want))):
                        a = got[i] if i < len(got) else "<missing>"
                        w = want[i] if i < len(want) else "<missing>"
                        if a != w:
                            print(f"  line {i+1}: expected {w!r}, got {a!r}")
                            break
                    break
    print(f"{count} programs, {bad} failures")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
