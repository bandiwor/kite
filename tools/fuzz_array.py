#!/usr/bin/env python3
"""Differential fuzzer for arrays: random operation sequences checked against Python lists.

Items are numbers or nil (None); some operations are deliberately out of bounds and must panic.

usage: fuzz_array.py [count] [first_seed] [path/to/kite]
"""
import os, random, subprocess, sys, tempfile


def show(v):
    if v is None:
        return "nil"
    if isinstance(v, list):
        return "[" + ", ".join(show(x) for x in v) + "]"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def kv(v):
    return "nil" if v is None else (f"({v})" if v < 0 else str(v))


def clamp(i, n):
    if i < 0:
        i += n
    return max(0, min(i, n))


def make(seed):
    r = random.Random(seed)
    arrs = {}  # name -> python list
    src, want = [], []
    items = lambda: r.choice([None, 0, 1, 2, 3, -1, 7, 42])
    nums = lambda: r.choice([0, 1, 2, 3, 5, 9])

    def emit(code, out):
        src.append(code)
        want.append(out)

    for step in range(r.randint(10, 60)):
        names = list(arrs)
        op = r.choice(["new", "lit", "push", "push", "pop", "get", "set", "insert", "remove", "slice", "concat",
                       "extend", "reverse", "copy", "index_of", "len", "iter", "show", "map", "filter", "getopt"])
        if op in ("new", "lit") or not names:
            name = f"a{len(arrs)}"
            if op == "new" and r.random() < 0.5:
                n, f = r.randint(0, 4), items()
                src.append(f"var {name} = array.new({n}, {kv(f)});")
                arrs[name] = [f] * n
            else:
                xs = [items() for _ in range(r.randint(0, 5))]
                src.append(f"var {name} = [{', '.join(kv(x) for x in xs)}];")
                arrs[name] = xs
            want.append(None)
            continue
        n = r.choice(names)
        a = arrs[n]
        if op == "push":
            xs = [items() for _ in range(r.randint(1, 3))]
            a.extend(xs)
            emit(f"print({n}:push({', '.join(kv(x) for x in xs)}));", str(len(a)))
        elif op == "pop":
            out = "None" if not a else f"Some({show(a.pop())})"
            emit(f"print({n}:pop());", out)
        elif op in ("get", "set"):
            i = r.choice([-1, 0, 1, 2, 3, 4, 5, len(a), len(a) - 1, 0.5])
            ok = isinstance(i, int) and 0 <= i < len(a)
            if op == "get":
                if ok:
                    emit(f"print({n}[{kv(i)}]);", show(a[i]))
                else:
                    emit(f"print(catch_panic(|| => {n}[{kv(i)}]):is_err());", "true")
            else:
                v = items()
                if ok:
                    a[i] = v
                    emit(f"{n}[{kv(i)}] = {kv(v)};", None)
                else:
                    emit(f"print(catch_panic(|| => {{ {n}[{kv(i)}] = {kv(v)}; }}):is_err());", "true")
        elif op == "insert":
            i, v = r.randint(0, len(a) + 1), items()
            if i <= len(a):
                a.insert(i, v)
                emit(f"insert({n}, {i}, {kv(v)});", None)
            else:
                emit(f"print(catch_panic(|| => insert({n}, {i}, {kv(v)})):is_err());", "true")
        elif op == "remove":
            i = r.randint(0, len(a) + 1)
            if i < len(a):
                emit(f"print(remove({n}, {i}));", show(a.pop(i)))
            else:
                emit(f"print(catch_panic(|| => remove({n}, {i})):is_err());", "true")
        elif op == "slice":
            i, j = r.randint(-6, 6), r.randint(-6, 6)
            lo, hi = clamp(i, len(a)), clamp(j, len(a))
            emit(f"print({n}:slice({kv(i)}, {kv(j)}));", show(a[lo:hi] if lo < hi else []))
        elif op == "concat":
            m = r.choice(names)
            emit(f"print({n}:concat({m}));", show(a + arrs[m]))
        elif op == "extend":
            m = r.choice(names)
            a.extend(list(arrs[m]))
            emit(f"{n}:extend({m});", None)
        elif op == "reverse":
            a.reverse()
            emit(f"{n}:reverse();", None)
        elif op == "copy":
            name = f"a{len(arrs)}"
            src.append(f"var {name} = {n}:copy();")
            want.append(None)
            arrs[name] = list(a)
        elif op == "index_of":
            v = items()
            out = f"Some({a.index(v)})" if v in a else "None"
            emit(f"print({n}:index_of({kv(v)}));", out)
        elif op == "len":
            emit(f"print(len({n}));", str(len(a)))
        elif op == "iter":
            emit(f"do {{ var s = \"\"; for i, v in {n} {{ s = s + i + \":\" + v + \" \"; }} print(s); }}",
                 "".join(f"{i}:{show(v)} " for i, v in enumerate(a)))
        elif op == "show":
            emit(f"print({n}, len({n}));", f"{show(a)} {len(a)}")
        elif op == "map":
            k = nums()
            emit(f"print(map({n}, |x, i| => if x == nil {{ i }} else {{ x + {k} }}));",
                 show([i if x is None else x + k for i, x in enumerate(a)]))
        elif op == "filter":
            emit(f"print(filter({n}, |x| => x != nil && x > 0));", show([x for x in a if x is not None and x > 0]))
        elif op == "getopt":
            i = r.randint(-1, len(a) + 1)
            ok = 0 <= i < len(a)
            emit(f"print({n}:get({kv(i)}));", f"Some({show(a[i])})" if ok else "None")
    emit("print(" + ", ".join(f"len({k})" for k in arrs) + ");" if arrs else "print();",
         ", ".join(str(len(v)) for v in arrs.values()).replace(", ", " ") if arrs else "")
    lines = [s for s, w in zip(src, want)]
    expected = [w for w in want if w is not None]
    return "\n".join(lines) + "\n", expected


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    binary = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "target", "release", "kite")
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for seed in range(seed0, seed0 + count):
            src, want = make(seed)
            path = os.path.join(d, f"a{seed}.kite")
            open(path, "w").write(src)
            for name, flags in (("default", []), ("no-opt", ["--no-opt"]), ("gc-stress", ["--gc-stress"])):
                p = subprocess.run([binary, *flags, path], capture_output=True, text=True, timeout=60)
                got = p.stdout.splitlines()
                if p.returncode != 0 or got != want:
                    bad += 1
                    keep = f"/tmp/kite_fuzz_array_fail_{seed}.kite"
                    open(keep, "w").write(src)
                    print(f"FAIL seed={seed} mode={name} exit={p.returncode} (saved to {keep})")
                    if p.stderr:
                        print("  stderr:", p.stderr.strip().splitlines()[:2])
                    for i in range(max(len(got), len(want))):
                        x = got[i] if i < len(got) else "<missing>"
                        w = want[i] if i < len(want) else "<missing>"
                        if x != w:
                            print(f"  line {i+1}: expected {w!r}, got {x!r}")
                            break
                    break
    print(f"{count} programs, {bad} failures")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
