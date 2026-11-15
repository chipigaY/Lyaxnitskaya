from fractions import Fraction as F

# ==================== ФОРМАТИРОВАНИЕ ====================

def fmt(v):
    v = F(v)
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator} (≈{float(v):.4g})"

def fmt_vec(v):
    return "(" + ", ".join(fmt(t) for t in v) + ")"

def read_frac(s):
    return F(s.replace(',', '.'))

# ==================== БЕЗОПАСНЫЙ ВВОД ====================

def safe_input(prompt):
    while True:
        s = input(prompt).strip()
        if s:
            return s

def safe_int(prompt, min_val=1):
    while True:
        try:
            n = int(safe_input(prompt))
            if n < min_val:
                print(f"  [!] Значение должно быть >= {min_val}. Повторите.")
                continue
            return n
        except ValueError:
            print("  [!] Нужно ввести целое число. Повторите.")

def safe_frac_list(prompt, count, allow_negative=False, allow_zero=True):
    while True:
        try:
            parts = safe_input(prompt).split()
            if len(parts) != count:
                print(f"  [!] Нужно ровно {count} чисел, а введено {len(parts)}. Повторите.")
                continue
            result = []
            ok = True
            for p in parts:
                try:
                    f = read_frac(p)
                except (ValueError, ZeroDivisionError):
                    print(f"  [!] '{p}' не является числом. Повторите.")
                    ok = False
                    break
                if not allow_negative and f < 0:
                    print(f"  [!] Отрицательные значения запрещены (введено {p}). Повторите.")
                    ok = False
                    break
                if not allow_zero and f == 0:
                    print(f"  [!] Нулевые значения запрещены. Повторите.")
                    ok = False
                    break
                result.append(f)
            if ok:
                return result
        except Exception:
            print("  [!] Ошибка ввода. Повторите.")

def safe_frac_matrix(prompt, rows, cols, allow_negative=True):
    print(prompt)
    while True:
        try:
            matrix = []
            ok = True
            for i in range(rows):
                row_parts = safe_input(f"  Строка {i+1} ({cols} значений): ").split()
                if len(row_parts) != cols:
                    print(f"  [!] В строке {i+1} нужно {cols} чисел, введено {len(row_parts)}.")
                    ok = False
                    break
                row = []
                for p in row_parts:
                    try:
                        f = read_frac(p)
                    except (ValueError, ZeroDivisionError):
                        print(f"  [!] '{p}' — не число.")
                        ok = False
                        break
                    if not allow_negative and f < 0:
                        print(f"  [!] Отрицательные значения запрещены.")
                        ok = False
                        break
                    row.append(f)
                if not ok:
                    break
                matrix.append(row)
            if ok:
                return matrix
        except Exception:
            print("  [!] Ошибка ввода матрицы. Повторите.")

# ==================== ДВОЙСТВЕННЫЙ СИМПЛЕКС ====================

MAX_ITER = 500

def ds_print_table(T, basis, c, names, title):
    total_cols = len(c)
    print(f"\n--- {title} ---")
    header = "  БП  | " + " ".join(f"{nm:>8}" for nm in names) + " |    b"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(T):
        bp = names[basis[i]] if basis[i] < total_cols else "?"
        cells = " ".join(f"{fmt(v):>8}" for v in row[:total_cols])
        print(f" {bp:>4} | {cells} | {fmt(row[total_cols]):>8}")
    
    n = total_cols - len(T)
    deltas = []
    for j in range(total_cols):
        s = c[j] - sum(c[basis[i]] * T[i][j] for i in range(len(T)))
        deltas.append(s)
    
    print("  Δ   | " + " ".join(f"{fmt(d):>8}" for d in deltas))
    return deltas

def ds_solve(c, A, b, names=None):
    m = len(A)
    n = len(A[0]) 
    total_cols = n + m

    A = [[F(v) for v in row] for row in A]
    b = [F(v) for v in b]
    c = [F(v) for v in c]

    if len(c) == n:
        c = c + [F(0)] * m

    for i in range(m):
        for j in range(m):
            A[i].append(F(1) if i == j else F(0))

    if names is None:
        names = [f"x{j+1}" for j in range(total_cols)]

    T = [A[i][:] + [b[i]] for i in range(m)]
    basis = list(range(n, n + m))

    print("\n=== Двойственный симплекс ===")
    print("max Z =", " + ".join(f"{fmt(c[j])}·{names[j]}" for j in range(total_cols)))

    for it in range(1, MAX_ITER + 1):
        deltas = ds_print_table(T, basis, c, names, f"Итерация {it}")

        if any(d < 0 for d in deltas):
            print("\n[!] Текущее решение не двойственно допустимо (есть Δ < 0).")
            return {'status': 'not_dual_feasible'}

        r = None
        for i in range(m):
            if T[i][total_cols] < 0 and (r is None or T[i][total_cols] < T[r][total_cols]):
                r = i
        if r is None:
            print("\nВсе b_i >= 0 — решение оптимально.")
            break

        enter, best = None, None
        for j in range(total_cols):
            if T[r][j] < 0:
                ratio = abs(deltas[j] / T[r][j])
                if best is None or ratio < best:
                    best, enter = ratio, j
        if enter is None:
            print("\n[!] В ведущей строке нет отрицательных a_rj — нет допустимых решений.")
            return {'status': 'infeasible'}

        print(f"\nВедущая строка: {r+1} (b = {fmt(T[r][total_cols])}), "
              f"ведущий столбец: {names[enter]} "
              f"(min |Δ/a| = {fmt(best)})")

        p = T[r][enter]
        T[r] = [v / p for v in T[r]]
        for i in range(m):
            if i != r and T[i][enter] != 0:
                k = T[i][enter]
                T[i] = [vi - k * vr for vi, vr in zip(T[i], T[r])]
        basis[r] = enter
    else:
        print(f"\n[!] Достигнут лимит итераций ({MAX_ITER}) — возможно зацикливание.")
        return {'status': 'iteration_limit'}

    x = [F(0)] * total_cols
    for i in range(m):
        x[basis[i]] = T[i][total_cols]
    
    obj = sum(c[j] * x[j] for j in range(total_cols))

    print("\nОптимальное решение:", fmt_vec(x))
    print("Z* =", fmt(obj))

    ok = (all(v >= 0 for v in x) and
          all(sum(A[i][j] * x[j] for j in range(total_cols)) == b[i] for i in range(m)))
    print(f"\n[{'✓' if ok else '✗'}] Проверка: Ax = b, x >= 0")

    return {'status': 'optimal', 'x': x, 'obj': obj, 'basis': basis}

def main_dual_simplex():
    print("\n" + "=" * 60)
    print("ДВОЙСТВЕННЫЙ СИМПЛЕКС")
    print("max Z = c·x,  A x = b,  x >= 0")
    print("=" * 60 + "\n")

    n = safe_int("Число переменных n: ")
    m = safe_int("Число ограничений m: ")

    c = safe_frac_list(f"Коэффициенты c ({n}): ", n, allow_negative=True)
    A = safe_frac_matrix(f"Матрица A ({m}×{n}):", m, n, allow_negative=True)
    b = safe_frac_list(f"Вектор b ({m}): ", m, allow_negative=True)

    res = ds_solve(c, A, b)
    if res['status'] != 'optimal':
        print("\nСтатус:", res['status'])

# ==================== ТРАНСПОРТНАЯ ЗАДАЧА ====================

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

def _has_cycle(cells):
    cells = list(cells)
    parent = list(range(len(cells)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    for k1 in range(len(cells)):
        for k2 in range(k1 + 1, len(cells)):
            i1, j1 = cells[k1]
            i2, j2 = cells[k2]
            if i1 == i2 or j1 == j2:
                if not union(k1, k2):
                    return True
    return False

def tp_fix_degeneracy(X, a, b):
    m, n = len(X), len(X[0])
    need = m + n - 1
    while True:
        basis = [(i, j) for i in range(m) for j in range(n) if X[i][j] > 0]
        if len(basis) >= need:
            return X
        added = False
        for i in range(m):
            for j in range(n):
                if X[i][j] == 0:
                    test_basis = set(basis) | {(i, j)}
                    if not _has_cycle(test_basis):
                        X[i][j] = F(0)
                        added = True
                        break
            if added:
                break
        if not added:
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
        diff = sum(a) - sum(b)
        print(f"[!] Σa = {fmt(sum(a))} != Σb = {fmt(sum(b))} — задача несбалансирована.")
        print(f"    Разница: {fmt(diff)}")
        return {'status': 'unbalanced'}

    if any(v < 0 for v in a) or any(v < 0 for v in b):
        print("[!] Отрицательные запасы/потребности — недопустимо.")
        return {'status': 'negative_values'}

    X = tp_nw_corner(a, b)
    tp_print_plan(X, C, "Начальный план (северо-западный угол)")
    X = tp_fix_degeneracy(X, a, b)

    for it in range(1, MAX_ITER + 1):
        u, v = tp_potentials(C, X)
        print(f"\n--- Итерация {it}: потенциалы ---")
        print(" u =", [fmt(t) if t is not None else "?" for t in u])
        print(" v =", [fmt(t) if t is not None else "?" for t in v])

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
            print("[!] Не удалось построить цикл (возможно, вырожденность не устранена).")
            return {'status': 'degenerate'}
        
        print("Цикл пересчёта:", " → ".join(f"A{i+1}B{j+1}" for i, j in cycle))

        theta = min(X[cycle[k][0]][cycle[k][1]] for k in range(1, len(cycle), 2))
        print(f"θ = {fmt(theta)}")
        for k, (i, j) in enumerate(cycle):
            if k % 2 == 0:
                X[i][j] += theta
            else:
                X[i][j] -= theta

        tp_print_plan(X, C, f"План после итерации {it}")
    else:
        print(f"\n[!] Достигнут лимит итераций ({MAX_ITER}).")
        return {'status': 'iteration_limit'}

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
    m = safe_int("Число поставщиков m: ")
    n = safe_int("Число потребителей n: ")
    a = safe_frac_list(f"Запасы ({m}): ", m, allow_negative=False)
    b = safe_frac_list(f"Потребности ({n}): ", n, allow_negative=False)
    C = safe_frac_matrix(f"Матрица стоимостей C ({m}×{n}):", m, n, allow_negative=False)
    tp_solve(a, b, C)

# ==================== МЕНЮ ====================

def my_menu():
    while True:
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
            return
        else:
            print("Неизвестный пункт. Повторите.")

if __name__ == "__main__":
    my_menu()