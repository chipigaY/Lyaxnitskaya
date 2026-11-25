
import numpy as np

TOL = 1e-9


def simplex_core(c, A, b, signs):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(c)
    m = len(b)

    for i in range(m):
        if b[i] < -TOL:
            A[i] *= -1
            b[i] *= -1

            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    extra = 0
    for s in signs:
        if s == "<=":
            extra += 1
        elif s == ">=":
            extra += 2
        else:
            extra += 1

    total = n + extra
    tableau = np.zeros((m + 1, total + 1))
    basis = []
    artificial = []

    current = n

    for i in range(m):
        tableau[i, :n] = A[i]
        tableau[i, -1] = b[i]

        if signs[i] == "<=":
            tableau[i, current] = 1
            basis.append(current)
            current += 1

        elif signs[i] == ">=":
            tableau[i, current] = -1
            current += 1

            tableau[i, current] = 1
            basis.append(current)
            artificial.append(current)
            current += 1

        elif signs[i] == "=":
            tableau[i, current] = 1
            basis.append(current)
            artificial.append(current)
            current += 1

        else:
            raise ValueError("Неизвестный знак ограничения")

    def set_objective(objective):
        tableau[-1, :] = 0
        tableau[-1, :len(objective)] = -objective

        for i in range(len(basis)):
            cb = objective[basis[i]]
            tableau[-1] += cb * tableau[i]

    def run_simplex():
        iterations = 0
        max_iterations = 10000

        while iterations < max_iterations:
            iterations += 1

            entering = None

            for j in range(tableau.shape[1] - 1):
                if tableau[-1, j] < -TOL:
                    entering = j
                    break

            if entering is None:
                return "optimal"

            ratios = []

            for i in range(len(basis)):
                if tableau[i, entering] > TOL:
                    ratios.append(
                        (tableau[i, -1] / tableau[i, entering], i)
                    )

            if not ratios:
                return "unbounded"

            min_ratio = min(r[0] for r in ratios)

            candidates = [
                i for ratio, i in ratios
                if ratio <= min_ratio + TOL
            ]

            leaving = min(candidates, key=lambda i: basis[i])

            pivot = tableau[leaving, entering]

            tableau[leaving] /= pivot

            for i in range(tableau.shape[0]):
                if i != leaving:
                    factor = tableau[i, entering]
                    tableau[i] -= factor * tableau[leaving]

            basis[leaving] = entering

        return "iteration_limit"

    phase1 = np.zeros(total)

    for j in artificial:
        phase1[j] = -1

    set_objective(phase1)

    status = run_simplex()

    if status != "optimal":
        return None, None, "Не удалось выполнить первую фазу"

    if tableau[-1, -1] < -TOL:
        return None, None, "Задача не имеет допустимого решения"

    art_set = set(artificial)

    i = 0
    while i < len(basis):

        if basis[i] in art_set:
            entering = None

            for j in range(total):
                if j not in art_set and abs(tableau[i, j]) > TOL:
                    entering = j
                    break

            if entering is not None:
                pivot = tableau[i, entering]
                tableau[i] /= pivot

                for k in range(tableau.shape[0]):
                    if k != i:
                        factor = tableau[k, entering]
                        tableau[k] -= factor * tableau[i]

                basis[i] = entering
                i += 1

            else:
                if abs(tableau[i, -1]) <= TOL:
                    tableau = np.delete(tableau, i, axis=0)
                    basis.pop(i)
                else:
                    return None, None, "Задача несовместна"

        else:
            i += 1

    keep = [j for j in range(total) if j not in art_set]

    old_to_new = {
        old: new for new, old in enumerate(keep)
    }

    tableau = tableau[:, keep + [total]]
    basis = [old_to_new[j] for j in basis]

    new_c = np.zeros(len(keep))
    new_c[:n] = c

    set_objective(new_c)

    status = run_simplex()

    if status == "unbounded":
        return None, None, "Целевая функция не ограничена"

    if status != "optimal":
        return None, None, "Превышен лимит итераций"

    x = np.zeros(n)

    for i, basic in enumerate(basis):
        if basic < n:
            x[basic] = tableau[i, -1]

    optimum = np.dot(c, x)

    return x, optimum, "optimal"


def solve_lp(c, A, b, signs, direction):
    c = np.array(c, dtype=float)

    if direction == "min":
        x, value, status = simplex_core(
            -c, A, b, signs.copy()
        )

        if status == "optimal":
            value = -value

    else:
        x, value, status = simplex_core(
            c, A, b, signs.copy()
        )

    return x, value, status


def build_dual(c, A, b, signs, direction):
    m, n = A.shape

    dual_signs = []

    for s in signs:
        if direction == "max":
            if s == "<=":
                dual_signs.append("positive")
            elif s == ">=":
                dual_signs.append("negative")
            else:
                dual_signs.append("free")
        else:
            if s == "<=":
                dual_signs.append("negative")
            elif s == ">=":
                dual_signs.append("positive")
            else:
                dual_signs.append("free")

    transformation = []

    for s in dual_signs:
        if s == "positive":
            transformation.append([1])
        elif s == "negative":
            transformation.append([-1])
        else:
            transformation.append([1, -1])

    T = np.zeros((m, sum(len(row) for row in transformation)))

    col = 0

    for i, row in enumerate(transformation):
        for sign in row:
            T[i, col] = sign
            col += 1

    A_dual = A.T @ T
    c_dual = b @ T
    b_dual = np.array(c, dtype=float)

    if direction == "max":
        dual_direction = "min"
        dual_constraint_signs = [">="] * n
    else:
        dual_direction = "max"
        dual_constraint_signs = ["<="] * n

    return (
        c_dual,
        A_dual,
        b_dual,
        dual_constraint_signs,
        dual_direction,
        T
    )


def check_primal(x, A, b, signs):
    if x is None:
        return False

    values = A @ x

    for i in range(len(b)):
        if signs[i] == "<=" and values[i] > b[i] + TOL:
            return False

        if signs[i] == ">=" and values[i] < b[i] - TOL:
            return False

        if signs[i] == "=" and abs(values[i] - b[i]) > TOL:
            return False

    if np.any(x < -TOL):
        return False

    return True


def check_dual(y, A, c, direction):
    if y is None:
        return False

    values = A.T @ y

    if direction == "max":
        return np.all(values >= c - TOL)
    else:
        return np.all(values <= c + TOL)


def check_dual_signs(y, signs, direction):
    if y is None:
        return False

    for i, s in enumerate(signs):
        if direction == "max":
            if s == "<=" and y[i] < -TOL:
                return False

            if s == ">=" and y[i] > TOL:
                return False
        else:
            if s == "<=" and y[i] > TOL:
                return False

            if s == ">=" and y[i] < -TOL:
                return False

    return True


def check_complementary(x, y, A, b, c, direction):
    if x is None or y is None:
        return False

    primal_slack = A @ x - b
    dual_values = A.T @ y

    if direction == "max":
        dual_slack = dual_values - c
    else:
        dual_slack = c - dual_values

    first = np.all(np.abs(y * primal_slack) <= 1e-6)
    second = np.all(np.abs(x * dual_slack) <= 1e-6)

    return first and second


def main():
    print("=" * 55)
    print("РЕШЕНИЕ ПРЯМОЙ И ДВОЙСТВЕННОЙ ЗАДАЧ")
    print("САМОСТОЯТЕЛЬНЫЙ ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД")
    print("=" * 55)

    n = int(input("Введите количество переменных: "))
    m = int(input("Введите количество ограничений: "))

    direction = input(
        "Введите направление (min или max): "
    ).strip().lower()

    if direction not in ("min", "max"):
        print("Ошибка: необходимо указать min или max.")
        return

    print("\nВведите коэффициенты целевой функции:")

    c = np.array([
        float(input(f"c[{i + 1}] = "))
        for i in range(n)
    ])

    A = []
    b = []
    signs = []

    print("\nВведите ограничения:")

    for i in range(m):
        print(f"\nОграничение №{i + 1}")

        row = [
            float(input(f"a[{i + 1},{j + 1}] = "))
            for j in range(n)
        ]

        sign = input(
            "Введите знак (<=, >= или =): "
        ).strip()

        if sign not in ("<=", ">=", "="):
            print("Ошибка: неверный знак ограничения.")
            return

        right = float(input("Правая часть b = "))

        A.append(row)
        signs.append(sign)
        b.append(right)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    print("\n" + "=" * 55)
    print("РЕШЕНИЕ ПРЯМОЙ ЗАДАЧИ")
    print("=" * 55)

    x, L, status = solve_lp(
        c, A, b, signs, direction
    )

    if status != "optimal":
        print("Прямая задача:", status)
        return

    print("\nОптимальный план прямой задачи:")

    for i, value in enumerate(x):
        print(f"x{i + 1} = {value:.8f}")

    print(f"\nЗначение целевой функции = {L:.8f}")

    primal_ok = check_primal(x, A, b, signs)

    print(
        "Проверка ограничений прямой задачи:",
        "выполнена" if primal_ok else "не выполнена"
    )

    print("\n" + "=" * 55)
    print("ПОСТРОЕНИЕ И РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ")
    print("=" * 55)

    (
        c_d,
        A_d,
        b_d,
        signs_d,
        direction_d,
        T
    ) = build_dual(c, A, b, signs, direction)

    z, W, status_d = solve_lp(
        c_d, A_d, b_d, signs_d, direction_d
    )

    if status_d != "optimal":
        print("Двойственная задача:", status_d)
        return

    y = T @ z

    print("\nОптимальный план двойственной задачи:")

    for i, value in enumerate(y):
        print(f"y{i + 1} = {value:.8f}")

    W = np.dot(b, y)

    print(f"\nЗначение двойственной функции = {W:.8f}")

    dual_ok = check_dual(y, A, c, direction)

    print(
        "Проверка ограничений двойственной задачи:",
        "выполнена" if dual_ok else "не выполнена"
    )

    signs_ok = check_dual_signs(y, signs, direction)

    print(
        "Проверка знаков двойственных переменных:",
        "выполнена" if signs_ok else "не выполнена"
    )

    print("\n" + "=" * 55)
    print("ПРОВЕРКА ОПТИМАЛЬНОСТИ")
    print("=" * 55)

    print(f"Значение прямой функции: {L:.8f}")
    print(f"Значение двойственной функции: {W:.8f}")

    equality = np.isclose(L, W, atol=1e-7, rtol=1e-7)

    print(
        "\nРавенство целевых функций:",
        "выполнено" if equality else "не выполнено"
    )

    complementary = check_complementary(
        x, y, A, b, c, direction
    )

    print(
        "Условия дополняющей нежёсткости:",
        "выполнены" if complementary else "не выполнены"
    )

    if (
        primal_ok
        and dual_ok
        and signs_ok
        and equality
        and complementary
    ):
        print("\nОПТИМАЛЬНОСТЬ ПОДТВЕРЖДЕНА!")
        print("Найдены оптимальные решения прямой")
        print("и двойственной задач.")
    else:
        print("\nПолная проверка оптимальности не пройдена.")

    print("\nРабота программы завершена.")


if __name__ == "__main__":
    main()