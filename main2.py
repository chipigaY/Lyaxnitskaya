"""
Мои методы:
  1) Двойственный симплекс-метод
  2) Транспортная задача (метод потенциалов)

Файл полностью автономен: свои fmt / fmt_vec / read_frac в том же стиле,
что и у коллеги. Его файл не требуется.
"""

from fractions import Fraction as F


# ---------- вспомогательные функции (стиль как у коллеги) ----------

def fmt(v):
    v = F(v)
    return str(v.numerator) if v.denominator == 1 else \
        f"{v.numerator}/{v.denominator} (≈{float(v):.4g})"


def fmt_vec(v):
    return "(" + ", ".join(fmt(t) for t in v) + ")"


def read_frac(s):
    return F(s.replace(',', '.'))


# ============================================================
#  1. ДВОЙСТВЕННЫЙ СИМПЛЕКС
# ============================================================

def ds_print_table(T, basis, c, names, title):
    n = len(c)
    print(f"\n--- {title} ---")
    header = "  БП  | " + " ".join(f"{nm:>8}" for nm in names) + " |    b"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(T):
        bp = names[basis[i]] if basis[i] < n else "?"
        cells = " ".join(f"{fmt(v):>8}" for v in row[:n])
        print(f" {bp:>4} | {cells} | {fmt(row[n]):>8}")
    deltas = []
    for j in range(n):
        s = c[j] - sum(c[basis[i]] * T[i][j] for i in range(len(T)))
        deltas.append(s)
    print("  Δ   | " + " ".join(f"{fmt(d):>8}" for d in deltas))
    return deltas


def ds_solve(c, A, b, names=None):
    """
    Двойственный симплекс для задачи:
        max Z = c·x,  A x = b,  x >= 0,
    где стартовый базис двойственно допустим (все Δ >= 0),
    но b может содержать отрицательные компоненты.
    Ожидается, что A уже содержит единичный базис в последних m столбцах.
    """
    m, n = len(A), len(c)
    A = [[F(v) for v in row] for row in A]
    b = [F(v) for v in b]
    c = [F(v) for v in c]
    if names is None:
        names = [f"x{j+1}" for j in range(n)]

    T = [A[i][:] + [b[i]] for i in range(m)]
    basis = list(range(n, n + m))

    print("\n=== Двойственный симплекс ===")
    print("max Z =", " + ".join(f"{fmt(c[j])}·{names[j]}" for j in range(n)))

    it = 0
    while True:
        it += 1
        deltas = ds_print_table(T, basis, c, names, f"Итерация {it}")

        if any(d < 0 for d in deltas):
            print("\n[!] Текущее решение не двойственно допустимо (есть Δ < 0) — "
                  "двойственный симплекс неприменим.")
            return {'status': 'not_dual_feasible'}

        r = None
        for i in range(m):
            if T[i][n] < 0 and (r is None or T[i][n] < T[r][n]):
                r = i
        if r is None:
            print("\nВсе b_i >= 0 — прямое решение допустимо и оптимально.")
            break

        enter, best = None, None
        for j in range(n):
            if T[r][j] < 0:
                ratio = abs(deltas[j] / T[r][j])
                if best is None or ratio < best:
                    best, enter = ratio, j
        if enter is None:
            print("\nВ ведущей строке нет отрицательных a_rj — "
                  "задача не имеет допустимых решений.")
            return {'status': 'infeasible'}

        print(f"\nВедущая строка: {r+1} (b = {fmt(T[r][n])}), "
              f"ведущий столбец: {names[enter]} "
              f"(min |Δ/a| = {fmt(best)})")

        p = T[r][enter]
        T[r] = [v / p for v in T[r]]
        for i in range(m):
            if i != r and T[i][enter] != 0:
                k = T[i][enter]
                T[i] = [vi - k * vr for vi, vr in zip(T[i], T[r])]
        basis[r] = enter

    x = [F(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][n]
    obj = sum(c[j] * x[j] for j in range(n))
    print("\nОптимальное решение:", fmt_vec(x))
    print("Z* =", fmt(obj))

    ok = all(v >= 0 for v in x) and all(
        sum(A[i][j] * x[j] for j in range(n)) == b[i] for i in range(m))
    print(f"\n[{'✓' if ok else '✗'}] Прямая допустимость: Ax = b, x >= 0")
    return {'status': 'optimal', 'x': x, 'obj': obj, 'basis': basis}


def main_dual_simplex():
    print("\n" + "=" * 60)
    print("ДВОЙСТВЕННЫЙ СИМПЛЕКС")
    print("max Z = c·x,  A x = b,  x >= 0")
    print("Стартовый базис двойственно допустим (все Δ >= 0),")
    print("b может содержать отрицательные компоненты.")
    print("=" * 60 + "\n")

    n = int(input("Число переменных n (включая базисные): "))
    m = int(input("Число ограничений m: "))
    c = [read_frac(t) for t in
         input(f"Коэффициенты c ({n}): ").split()]
    A, b = [], []
    for i in range(m):
        A.append([read_frac(t) for t in
                  input(f"Строка {i+1} матрицы A ({n}): ").split()])
        b.append(read_frac(input(f"b{i+1}: ")))

    res = ds_solve(c, A, b)
    if res['status'] != 'optimal':
        print("\nСтатус:", res['status'])


# ============================================================
#  2. ТРАНСПОРТНАЯ ЗАДАЧА (метод потенциалов)
# ============================================================

def tp_nw_corner(a, b):
    m, n = len(a), len(b)
    a, b = a[:], b[:]
    X = [[F(0)] * n for _ in range(m)]
    i = j = 0
    while i < m and j < n:
        t = min(a[i], b[j])
        X[i][j] = t
        a[i] -= t
        b[j] -= t
        if a[i] == 0 and i < m - 1:
            i += 1
        elif b[j] == 0 and j < n - 1:
            j += 1
        else:
            i += 1
            j += 1
    return X


def tp_print_plan(X, C, title):
    m, n = len(X), len(X[0])
    print(f"\n--- {title} ---")
    header = "       | " + " ".join(f"B{j+1:>7}" for j in range(n)) + " |  Запас"
    print(header)
    print("-" * len(header))
    for i in range(m):
        cells = " ".join(f"{fmt(X[i][j]):>7}" for j in range(n))
        print(f" A{i+1}    | {cells} | {sum(X[i]):>6}")
    print("-" * len(header))
    print(" Потреб | " + " ".join(
        f"{fmt(sum(X[i][j] for i in range(m))):>7}" for j in range(n)))
    Z = sum(C[i][j] * X[i][j] for i in range(m) for j in range(n))
    print(f" Стоимость Z = {fmt(Z)}")
    return Z


def tp_potentials(C, X):
    m, n = len(X), len(X[0])
    u = [None] * m
    v = [None] * n
    u[0] = F(0)
    changed = True
    while changed:
        changed = False
        for i in range(m):
            for j in range(n):
                if X[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = C[i][j] - u[i]
                        changed = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = C[i][j] - v[j]
                        changed = True
    return u, v


def tp_find_cycle(X, start_i, start_j):
    m, n = len(X), len(X[0])
    base = {(i, j) for i in range(m) for j in range(n) if X[i][j] > 0}
    base.add((start_i, start_j))
    path = [(start_i, start_j)]

    def neighbors(cell, horizontal):
        i, j = cell
        if horizontal:
            return [(i, jj) for jj in range(n) if jj != j and (i, jj) in base]
        return [(ii, j) for ii in range(m) if ii != i and (ii, j) in base]

    def dfs(cell, horizontal, visited):
        for nb in neighbors(cell, horizontal):
            if nb == (start_i, start_j) and len(path) >= 4:
                return True
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                if dfs(nb, not horizontal, visited):
                    return True
                path.pop()
                visited.discard(nb)
        return False

    dfs((start_i, start_j), True, {(start_i, start_j)})
    return path if len(path) >= 4 else None


def tp_solve(a, b, C):
    m, n = len(a), len(b)
    a = [F(v) for v in a]
    b = [F(v) for v in b]
    C = [[F(v) for v in row] for row in C]

    print("\n=== Транспортная задача (метод потенциалов) ===")
    if sum(a) != sum(b):
        print(f"[!] Σa = {fmt(sum(a))} != Σb = {fmt(sum(b))} — "
              "задача несбалансирована.")
        return {'status': 'unbalanced'}

    X = tp_nw_corner(a, b)
    tp_print_plan(X, C, "Начальный план (северо-западный угол)")

    it = 0
    while True:
        it += 1
        u, v = tp_potentials(C, X)
        print(f"\n--- Итерация {it}: потенциалы ---")
        print(" u =", [fmt(t) if t is not None else "None" for t in u])
        print(" v =", [fmt(t) if t is not None else "None" for t in v])

        best, cell = None, None
        print("\n Оценки Δ_ij = c_ij − (u_i + v_j):")
        for i in range(m):
            for j in range(n):
                if X[i][j] == 0:
                    d = C[i][j] - u[i] - v[j]
                    mark = ""
                    if d < 0 and (best is None or d < best):
                        best, cell = d, (i, j)
                        mark = "  <-- min (войдёт в план)"
                    print(f"  Δ[{i+1},{j+1}] = {fmt(d)}{mark}")

        if cell is None:
            print("\nВсе Δ >= 0 — план оптимален.")
            break

        print(f"\nВедущая клетка: A{cell[0]+1}–B{cell[1]+1}, Δ = {fmt(best)}")
        cycle = tp_find_cycle(X, *cell)
        if cycle is None:
            print("[!] Не удалось построить цикл (возможно, вырожденность).")
            return {'status': 'degenerate'}
        print("Цикл пересчёта:", " → ".join(
            f"A{i+1}B{j+1}" for i, j in cycle))

        theta = min(X[cycle[k][0]][cycle[k][1]]
                    for k in range(1, len(cycle), 2))
        print(f"θ = {fmt(theta)}")
        for k, (i, j) in enumerate(cycle):
            if k % 2 == 0:
                X[i][j] += theta
            else:
                X[i][j] -= theta

        tp_print_plan(X, C, f"План после итерации {it}")

    Z = sum(C[i][j] * X[i][j] for i in range(m) for j in range(n))
    print("\nОптимальный план:")
    for i in range(m):
        print(" ", [fmt(v) for v in X[i]])
    print("Z* =", fmt(Z))

    ok = (all(sum(X[i][j] for j in range(n)) == a[i] for i in range(m)) and
          all(sum(X[i][j] for i in range(m)) == b[j] for j in range(n)) and
          all(X[i][j] >= 0 for i in range(m) for j in range(n)))
    print(f"\n[{'✓' if ok else '✗'}] Баланс и допустимость плана")
    return {'status': 'optimal', 'X': X, 'obj': Z}


def main_transport():
    print("\n" + "=" * 60)
    print("ТРАНСПОРТНАЯ ЗАДАЧА (метод потенциалов)")
    print("=" * 60 + "\n")
    m = int(input("Число поставщиков m: "))
    n = int(input("Число потребителей n: "))
    a = [read_frac(t) for t in input(f"Запасы ({m}): ").split()]
    b = [read_frac(t) for t in input(f"Потребности ({n}): ").split()]
    print(f"Матрица стоимостей C ({m}×{n}):")
    C = []
    for i in range(m):
        C.append([read_frac(t) for t in
                  input(f" Строка {i+1} ({n}): ").split()])
    tp_solve(a, b, C)


# ============================================================
#  ТОЧКА ВХОДА
# ============================================================

def my_menu():
    print("\nВыбери метод:")
    print("  1 — двойственный симплекс")
    print("  2 — транспортная задача (метод потенциалов)")
    print("  0 — выход")
    choice = input("Выбор: ").strip()
    if choice == '1':
        main_dual_simplex()
    elif choice == '2':
        main_transport()
    elif choice == '0':
        print("Выход.")
    else:
        print("Неизвестный пункт.")


if __name__ == "__main__":
    my_menu()