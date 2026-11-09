from fractions import Fraction as F


def simplex(c, A, b, signs):
    m, n = len(A), len(c)
    A = [[F(v) for v in row] for row in A]
    b = [F(v) for v in b]
    c = [F(v) for v in c]
    signs = list(signs)

    flip = [1] * m
    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]
            signs[i] = {'<=': '>=', '>=': '<=', '=': '='}[signs[i]]
            flip[i] = -1

    n_slack = sum(1 for s in signs if s in ('<=', '>='))
    n_art = sum(1 for s in signs if s in ('>=', '='))
    N = n + n_slack + n_art
    T = [[F(0)] * (N + 1) for _ in range(m)]
    basis = [None] * m
    ident_col = [None] * m
    art_cols = []
    sc, ac = n, n + n_slack
    for i in range(m):
        for j in range(n):
            T[i][j] = A[i][j]
        T[i][N] = b[i]
        if signs[i] == '<=':
            T[i][sc] = F(1)
            basis[i] = ident_col[i] = sc
            sc += 1
        elif signs[i] == '>=':
            T[i][sc] = F(-1)
            sc += 1
            T[i][ac] = F(1)
            basis[i] = ident_col[i] = ac
            art_cols.append(ac)
            ac += 1
        else:
            T[i][ac] = F(1)
            basis[i] = ident_col[i] = ac
            art_cols.append(ac)
            ac += 1

    def pivot(r, col):
        p = T[r][col]
        T[r] = [v / p for v in T[r]]
        for i in range(m):
            if i != r and T[i][col] != 0:
                k = T[i][col]
                T[i] = [vi - k * vr for vi, vr in zip(T[i], T[r])]
        basis[r] = col

    def run(cost, allowed):
        while True:
            enter = None
            for j in range(N):
                if j not in allowed or j in basis:
                    continue
                d = cost[j] - sum(cost[basis[i]] * T[i][j] for i in range(m))
                if d > 0:
                    enter = j
                    break
            if enter is None:
                return 'optimal'
            best, row = None, None
            for i in range(m):
                if T[i][enter] > 0:
                    ratio = T[i][N] / T[i][enter]
                    if best is None or ratio < best or \
                            (ratio == best and basis[i] < basis[row]):
                        best, row = ratio, i
            if row is None:
                return 'unbounded'
            pivot(row, enter)

    if art_cols:
        cost1 = [F(0)] * N
        for j in art_cols:
            cost1[j] = F(-1)
        run(cost1, set(range(N)))
        if sum(T[i][N] for i in range(m) if basis[i] in art_cols) != 0:
            return {'status': 'infeasible'}
        for i in range(m):
            if basis[i] in art_cols:
                for j in range(N):
                    if j not in art_cols and T[i][j] != 0:
                        pivot(i, j)
                        break

    cost2 = c + [F(0)] * (N - n)
    allowed = set(range(N)) - set(art_cols)
    if run(cost2, allowed) == 'unbounded':
        return {'status': 'unbounded'}

    x = [F(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][N]
    obj = sum(ci * xi for ci, xi in zip(c, x))
    duals = []
    for i in range(m):
        col = ident_col[i]
        y = sum(cost2[basis[k]] * T[k][col] for k in range(m))
        duals.append(y * flip[i])
    return {'status': 'optimal', 'x': x, 'obj': obj, 'duals': duals}


def solve_dual(c, A, b):
    m, n = len(A), len(c)
    At = [[A[i][j] for i in range(m)] for j in range(n)]
    res = simplex([-v for v in b], At, c, ['>='] * n)
    if res['status'] == 'optimal':
        res['obj'] = -res['obj']
    return res


def check_optimality(c, A, b, x, y):
    m, n = len(A), len(c)
    report = []

    feas_p = all(v >= 0 for v in x) and all(
        sum(A[i][j] * x[j] for j in range(n)) <= b[i] for i in range(m))
    report.append(("Допустимость x* в прямой задаче", feas_p))

    feas_d = all(v >= 0 for v in y) and all(
        sum(A[i][j] * y[i] for i in range(m)) >= c[j] for j in range(n))
    report.append(("Допустимость y* в двойственной задаче", feas_d))

    Z = sum(c[j] * x[j] for j in range(n))
    W = sum(b[i] * y[i] for i in range(m))
    report.append((f"Равенство целевых функций Z = W  ({fmt(Z)} = {fmt(W)})", Z == W))

    cs1 = all(y[i] * (b[i] - sum(A[i][j] * x[j] for j in range(n))) == 0
              for i in range(m))
    cs2 = all(x[j] * (sum(A[i][j] * y[i] for i in range(m)) - c[j]) == 0
              for j in range(n))
    report.append(("Дополняющая нежёсткость: y_i·(b_i − A_i x) = 0", cs1))
    report.append(("Дополняющая нежёсткость: x_j·(A^T_j y − c_j) = 0", cs2))

    return all(r[1] for r in report), report


def fmt(v):
    v = F(v)
    return str(v.numerator) if v.denominator == 1 else \
        f"{v.numerator}/{v.denominator} (≈{float(v):.4g})"


def fmt_vec(v):
    return "(" + ", ".join(fmt(t) for t in v) + ")"


def read_frac(s):
    return F(s.replace(',', '.'))


def read_problem():
    n = int(input("Число переменных n: "))
    m = int(input("Число ограничений m: "))
    c = [read_frac(t) for t in input(f"Коэффициенты целевой функции c ({n} чисел): ").split()]
    A, b = [], []
    for i in range(m):
        row = [read_frac(t) for t in input(f"Строка {i + 1} матрицы A ({n} чисел): ").split()]
        A.append(row)
        b.append(read_frac(input(f"Правая часть b{i + 1}: ")))
    return c, A, b


def main():
    print("Прямая задача:   max Z = c·x,  A x <= b,  x >= 0")
    print("Двойственная:    min W = b·y,  A^T y >= c, y >= 0\n")
    c, A, b = read_problem()

    m, n = len(A), len(c)
    names_x = [f"x{j + 1}" for j in range(n)]
    names_y = [f"y{i + 1}" for i in range(m)]

    print("\n=== Прямая задача ===")
    print("max Z =", " + ".join(f"{fmt(c[j])}·{names_x[j]}" for j in range(n)))
    for i in range(m):
        print("   ", " + ".join(f"{fmt(A[i][j])}·{names_x[j]}" for j in range(n)),
              "<=", fmt(b[i]))

    print("\n=== Двойственная задача ===")
    print("min W =", " + ".join(f"{fmt(b[i])}·{names_y[i]}" for i in range(m)))
    for j in range(n):
        print("   ", " + ".join(f"{fmt(A[i][j])}·{names_y[i]}" for i in range(m)),
              ">=", fmt(c[j]))

    p = simplex(c, A, b, ['<='] * m)
    d = solve_dual(c, A, b)

    print("\n=== Результаты ===")
    for title, r in (("Прямая", p), ("Двойственная", d)):
        if r['status'] != 'optimal':
            print(f"{title}: {r['status']}")
        else:
            print(f"{title}: оптимум = {fmt(r['obj'])}, решение = {fmt_vec(r['x'])}")

    if p['status'] != 'optimal' or d['status'] != 'optimal':
        print("\nПо теории двойственности: если одна задача неограничена, "
              "другая не имеет допустимых решений (и наоборот).")
        return

    x, y = p['x'], d['x']
    print("\nДвойственные оценки из финальной таблицы прямой задачи:",
          fmt_vec(p['duals']))
    print("Решение двойственной задачи, найденное независимо:     ",
          fmt_vec(y),
          "— совпадают" if p['duals'] == y else
          "— отличаются (двойственная задача имеет несколько оптимумов)")

    ok, report = check_optimality(c, A, b, x, y)
    print("\n=== Проверка оптимальности ===")
    for text, res in report:
        print(f"  [{'✓' if res else '✗'}] {text}")
    print("\nВывод:", "пара (x*, y*) ОПТИМАЛЬНА." if ok else "пара (x*, y*) НЕ оптимальна.")


if __name__ == "__main__":
    main()