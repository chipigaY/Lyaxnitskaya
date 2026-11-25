
import numpy as np
from scipy.optimize import linprog

TOL = 1e-7


def parse_numbers(s):
    s = s.replace(';', ' ')
    return [float(p.replace(',', '.')) for p in s.split()]


def read_vector(prompt, size):
    while True:
        try:
            v = np.array(parse_numbers(input(prompt)))
        except ValueError:
            print("Ошибка: вводите только числа.")
            continue

        if len(v) == size:
            return v

        print(f"Нужно {size} чисел. Повторите ввод.")


def read_matrix(rows, cols):
    return np.array([
        read_vector(f"Строка {i + 1}: ", cols)
        for i in range(rows)
    ])


def read_int(prompt):
    while True:
        try:
            v = int(input(prompt))
            if v > 0:
                return v
        except ValueError:
            pass

        print("Введите целое положительное число.")


def status_text(res):
    return {
        1: "лимит итераций",
        2: "задача несовместна",
        3: "целевая функция не ограничена",
        4: "численные трудности"
    }.get(res.status, res.message)


def main():

    print("РЕШЕНИЕ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")

    n = read_int("Число переменных n: ")
    m = read_int("Число ограничений m: ")

    # Ввод целевой функции
    print("\nВведите коэффициенты целевой функции:")
    c = read_vector(f"Вектор c ({n} чисел): ", n)

    while True:
        direction = input("Направление (min/max): ").strip().lower()
        if direction in ("min", "max"):
            break
        print("Введите min или max.")

    # Ввод матрицы
    print("\nВведите матрицу коэффициентов A:")
    A = read_matrix(m, n)

    # Ввод знаков ограничений
    print("\nВведите знаки ограничений (<=, >= или =):")

    signs = []

    for i in range(m):
        while True:
            sign = input(f"Знак ограничения {i+1}: ").strip()

            if sign in ("<=", ">=", "="):
                signs.append(sign)
                break

            print("Допустимые знаки: <=, >=, =")

    # Ввод свободных членов
    print("\nВведите вектор свободных членов:")
    b = read_vector(f"Вектор b ({m} чисел): ", m)

    print("\nУсловия: все переменные неотрицательны (x >= 0).")

    # ==========================
    # ПРЕОБРАЗОВАНИЕ ОГРАНИЧЕНИЙ
    # ==========================

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for i in range(m):

        if signs[i] == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])

        elif signs[i] == ">=":
            A_ub.append(-A[i])
            b_ub.append(-b[i])

        else:
            A_eq.append(A[i])
            b_eq.append(b[i])

    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    # ==========================
    # ПРЯМАЯ ЗАДАЧА
    # ==========================

    print("\n========== ПРЯМАЯ ЗАДАЧА ==========")

    c_solver = c if direction == "min" else -c

    rp = linprog(
        c=c_solver,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * n,
        method='highs'
    )

    if not rp.success:
        print("Решение не найдено:", status_text(rp))
        return

    x = rp.x

    optimum = rp.fun if direction == "min" else -rp.fun

    print(f"Оптимальное значение = {optimum:.6f}")
    print("Оптимальный план x* =", np.round(x, 6))

    # Проверка прямой задачи
    print("\nПроверка ограничений прямой задачи:")

    primal_ok = True

    for i in range(m):
        value = A[i] @ x

        if signs[i] == "<=":
            ok = value <= b[i] + TOL
        elif signs[i] == ">=":
            ok = value >= b[i] - TOL
        else:
            ok = abs(value - b[i]) <= TOL

        primal_ok &= ok

        print(
            f"Ограничение {i+1}: "
            f"{value:.6f} {signs[i]} {b[i]:.6f} "
            f" [{'OK' if ok else 'НЕТ'}]"
        )

    x_ok = np.all(x >= -TOL)

    print("Неотрицательность x:", "OK" if x_ok else "НЕТ")

    primal_ok &= x_ok

    # ==========================
    # ДВОЙСТВЕННАЯ ЗАДАЧА
    # ==========================

    print("\n========== ДВОЙСТВЕННАЯ ЗАДАЧА ==========")

    dual_bounds = []

    for sign in signs:

        if direction == "min":
            if sign == "<=":
                dual_bounds.append((None, 0))
            elif sign == ">=":
                dual_bounds.append((0, None))
            else:
                dual_bounds.append((None, None))

        else:
            if sign == "<=":
                dual_bounds.append((0, None))
            elif sign == ">=":
                dual_bounds.append((None, 0))
            else:
                dual_bounds.append((None, None))

    if direction == "min":

        # min cx -> max by
        # A^T y <= c

        rd = linprog(
            c=-b,
            A_ub=A.T,
            b_ub=c,
            bounds=dual_bounds,
            method='highs'
        )

    else:

        # max cx -> min by
        # A^T y >= c

        rd = linprog(
            c=b,
            A_ub=-A.T,
            b_ub=-c,
            bounds=dual_bounds,
            method='highs'
        )

    if not rd.success:
        print("Решение не найдено:", status_text(rd))
        return

    y = rd.x

    dual_optimum = -rd.fun if direction == "min" else rd.fun

    dual_direction = "max" if direction == "min" else "min"

    print(f"Оптимальное значение W_{dual_direction} = "
          f"{dual_optimum:.6f}")

    print("Оптимальный план y* =", np.round(y, 6))

    # ==========================
    # ПРОВЕРКА ДВОЙСТВЕННОЙ ЗАДАЧИ
    # ==========================

    print("\nПроверка ограничений двойственной задачи:")

    dual_ok = True

    Ay = A.T @ y

    for j in range(n):

        if direction == "min":
            ok = Ay[j] <= c[j] + TOL
            sign = "<="
        else:
            ok = Ay[j] >= c[j] - TOL
            sign = ">="

        dual_ok &= ok

        print(
            f"Ограничение {j+1}: "
            f"{Ay[j]:.6f} {sign} {c[j]:.6f} "
            f"[{'OK' if ok else 'НЕТ'}]"
        )

    # Проверка знаков двойственных переменных

    print("\nПроверка знаков переменных y:")

    for i in range(m):

        if signs[i] == "=":
            ok = True
            description = "свободная переменная"

        elif direction == "min" and signs[i] == "<=":
            ok = y[i] <= TOL
            description = "y <= 0"

        elif direction == "min" and signs[i] == ">=":
            ok = y[i] >= -TOL
            description = "y >= 0"

        elif direction == "max" and signs[i] == "<=":
            ok = y[i] >= -TOL
            description = "y >= 0"

        else:
            ok = y[i] <= TOL
            description = "y <= 0"

        dual_ok &= ok

        print(
            f"Переменная y{i+1}: {y[i]:.6f}, "
            f"{description} [{'OK' if ok else 'НЕТ'}]"
        )

    # ==========================
    # ИТОГ
    # ==========================

    print("\n========== ИТОГОВАЯ ПРОВЕРКА ==========")

    print("Прямая задача допустима:",
          "OK" if primal_ok else "НЕТ")

    print("Двойственная задача допустима:",
          "OK" if dual_ok else "НЕТ")

    print(f"\nЗначение прямой задачи = {optimum:.6f}")
    print(f"Значение двойственной задачи = {dual_optimum:.6f}")

    if primal_ok and dual_ok:
        if np.isclose(optimum, dual_optimum,
                      rtol=1e-7, atol=1e-9):
            print("\nВЫВОД: обе задачи решены оптимально.")
        else:
            print("\nВЫВОД: значения задач не совпадают.")
    else:
        print("\nВЫВОД: обнаружены ошибки допустимости.")


if __name__ == "__main__":
    main()
