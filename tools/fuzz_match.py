#!/usr/bin/env python3
"""Differential fuzzer for `match`: random values x random patterns, checked against a Python oracle.

usage: fuzz_match.py [count] [first_seed] [path/to/kite]
"""
import os, random, subprocess, sys, tempfile

TAGS = {"A": 1, "B": 2, "C": 0}  # variant name -> number of fields


# values: int | None | ("A", [v]) | ("B", [v, v]) | ("C", [])
def gen_value(r, depth):
    k = r.random()
    if depth == 0 or k < 0.3:
        return r.choice([0, 1, 2, 3, -1, None])
    name = r.choice("ABC")
    return (name, [gen_value(r, depth - 1) for _ in range(TAGS[name])])


def show(v):
    if v is None:
        return "nil"
    if isinstance(v, int):
        return str(v)
    name, fs = v
    return name + ("(" + ", ".join(show(f) for f in fs) + ")" if fs else "")


def kval(v):
    if v is None:
        return "nil"
    if isinstance(v, int):
        return f"({v})" if v < 0 else str(v)
    name, fs = v
    return f"U.{name}" + ("(" + ", ".join(kval(f) for f in fs) + ")" if fs else "")


# patterns: ("wild",) ("bind", n) ("lit", int|None) ("var", name, [pats]|None, qualified) ("or", [pats])
class PGen:
    def __init__(self, r):
        self.r = r
        self.n = 0

    def fresh(self):
        self.n += 1
        return f"v{self.n}"

    def pat(self, depth, allow_bind=True):
        r = self.r
        k = r.random()
        if depth == 0 or k < 0.35:
            c = r.random()
            if c < 0.2 or (not allow_bind and c < 0.4):
                return ("wild",)
            if c < 0.6 or not allow_bind:
                return ("lit", r.choice([0, 1, 2, 3, -1, None]))
            return ("bind", self.fresh())
        if k < 0.55 and not allow_bind:
            return ("or", [self.pat(depth - 1, False) for _ in range(r.randint(2, 3))])
        name = r.choice("ABC")
        if r.random() < 0.15:
            args = None
        else:
            args = [self.pat(depth - 1, allow_bind) for _ in range(TAGS[name])]
        return ("var", name, args, r.random() < 0.3)


def kpat(p):
    t = p[0]
    if t == "wild":
        return "_"
    if t == "bind":
        return p[1]
    if t == "lit":
        return "nil" if p[1] is None else str(p[1])
    if t == "var":
        _, name, args, q = p
        return ("U." if q else "") + name + ("" if args is None else ("(" + ", ".join(kpat(a) for a in args) + ")" if args or TAGS[name] else "()"))
    return "(" + " | ".join(kpat(a) for a in p[1]) + ")" if False else " | ".join(kpat(a) for a in p[1])


def binds(p, out):
    t = p[0]
    if t == "bind":
        out.append(p[1])
    elif t == "var" and p[2]:
        for a in p[2]:
            binds(a, out)
    return out


def matches(p, v, env):
    t = p[0]
    if t == "wild":
        return True
    if t == "bind":
        env[p[1]] = v
        return True
    if t == "lit":
        return v == p[1] and (v is None) == (p[1] is None) and not isinstance(v, tuple)
    if t == "or":
        return any(matches(a, v, env) for a in p[1])
    _, name, args, _q = p
    if not isinstance(v, tuple) or v[0] != name:
        return False
    if args is None:
        return True
    if len(args) != len(v[1]):
        raise RuntimeError("arity")
    return all(matches(a, x, env) for a, x in zip(args, v[1]))


def make(seed):
    r = random.Random(seed)
    pg = PGen(r)
    arms = []
    for i in range(r.randint(1, 6)):
        top = r.random()
        if top < 0.25:
            # or-pattern at top level; alternatives without bindings, or the shared-binding form
            if r.random() < 0.5:
                pat = ("or", [pg.pat(2, False) for _ in range(r.randint(2, 3))])
            else:
                n = pg.fresh()
                pat = ("or", [("var", "A", [("bind", n)], False), ("var", "B", [("bind", n), ("wild",)], False)])
        else:
            pat = pg.pat(3)
        names = binds(pat, []) if pat[0] != "or" else (["v%d" % pg.n] if pat[1][0][0] == "var" and pat[1][0][1] == "A" and pat[1][0][2] and pat[1][0][2][0][0] == "bind" else [])
        if pat[0] == "or" and names:
            names = [pat[1][0][2][0][1]]
        guard = None
        if names and r.random() < 0.4:
            guard = (names[0], r.choice([-1, 0, 1, 2]))
        arms.append((pat, names, guard))
    values = [gen_value(r, 3) for _ in range(12)]
    # kite source
    src = ["union U { A(x), B(x, y), C }", "fn f(v) {", "    match v {"]
    for i, (pat, names, guard) in enumerate(arms):
        g = f" if type({guard[0]}) == \"number\" && {guard[0]} > {guard[1]}" if guard else ""
        body = "print(" + ", ".join([str(i)] + names) + ")"
        src.append(f"        {kpat(pat)}{g} => {body},")
    src.append("        _ => print(-1),")
    src.append("    }")
    src.append("}")
    # one call per value: nil cannot live inside an array literal
    for v in values:
        src.append(f"f({kval(v)});")
    # oracle
    want = []
    for v in values:
        res = None
        for i, (pat, names, guard) in enumerate(arms):
            env = {}
            if not matches(pat, v, env):
                continue
            if guard:
                g = env[guard[0]]
                if not (isinstance(g, int) and g > guard[1]):
                    continue
            res = " ".join([str(i)] + [show(env[n]) for n in names])
            break
        want.append(res if res is not None else "-1")
    return "\n".join(src) + "\n", want


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    binary = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "target", "release", "kite")
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for seed in range(seed0, seed0 + count):
            try:
                src, want = make(seed)
            except RuntimeError:
                continue
            path = os.path.join(d, f"m{seed}.kite")
            open(path, "w").write(src)
            for name, flags in (("default", []), ("no-opt", ["--no-opt"]), ("gc-stress", ["--gc-stress"])):
                p = subprocess.run([binary, *flags, path], capture_output=True, text=True, timeout=60)
                got = p.stdout.splitlines()
                if p.returncode != 0 or got != want:
                    # the oracle raises on arity mismatches only for matched shapes; kite reports the same as a runtime error
                    bad += 1
                    keep = f"/tmp/kite_fuzz_match_fail_{seed}.kite"
                    open(keep, "w").write(src)
                    print(f"FAIL seed={seed} mode={name} exit={p.returncode} (saved to {keep})")
                    if p.stderr:
                        print("  stderr:", p.stderr.strip().splitlines()[:3])
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
