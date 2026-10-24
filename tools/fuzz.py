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
        # closures are named c*, they are not numbers
        return [n for n in self.visible() if n[0] != "c"]

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
        if d > 2 or c < 0.6:
            return ("cmp", r.choice(["<", "<=", ">", ">=", "==", "!="]), self.expr(d + 1), self.expr(d + 1))
        if c < 0.75:
            return ("and", self.cond(d + 1), self.cond(d + 1))
        if c < 0.9:
            return ("or", self.cond(d + 1), self.cond(d + 1))
        return ("not", self.cond(d + 1))

    # statements -----------------------------------------------------------
    def stmts(self, n):
        out = []
        for _ in range(n):
            out.append(self.stmt())
        return out

    def stmt(self):
        r = self.r
        vs = self.numeric()
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
    raise Exception(k)

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
        s = str(int(x))
        if x == 0 and math.copysign(1, x) < 0:
            return "-0"
        return s
    return repr(x)

class Closure:
    def __init__(self, param, body, env): self.param, self.body, self.env = param, body, env

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
    src = "\n".join(kfn(f) for f in funcs) + "\n"
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
