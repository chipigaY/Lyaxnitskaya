import numpy as np
import numpy.typing as npt
from collections import deque
from dataclasses import dataclass


_TOLERANCE = 1e-10
_MAX_ITERATIONS = 1000

@dataclass
class _ClosedProblem:
    costs: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    supplier_names: list[str]
    consumer_names: list[str]


@dataclass
class _PotentialStep:
    row_potentials: np.ndarray
    column_potentials: np.ndarray
    reduced_costs: np.ndarray


class _Report:
    def add(self, message) -> None:
        print(message)


def _validate_arguments(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost_matrix = np.asarray(costs, dtype=float)
    supplies = np.asarray(supply, dtype=float).reshape(-1)
    demands = np.asarray(demand, dtype=float).reshape(-1)
    if (supplies < 0).any() or (demands < 0).any():
        raise ValueError("Запасы и потребности не могут быть отрицательными.")
    return cost_matrix, supplies, demands


def _format_number(value: float) -> str:
    return f"{value:g}"


def _close_problem(
    costs: np.ndarray, supply: np.ndarray, demand: np.ndarray, report: _Report
) -> _ClosedProblem:
    supplier_names = [f"a[{index + 1}]" for index in range(len(supply))]
    consumer_names = [f"b[{index + 1}]" for index in range(len(demand))]
    total_supply = float(supply.sum())
    total_demand = float(demand.sum())

    if not np.isclose(total_supply, total_demand, rtol=_TOLERANCE, atol=_TOLERANCE):
        if total_supply > total_demand:
            difference = total_supply - total_demand
            costs = np.column_stack((costs, np.zeros(costs.shape[0])))
            demand = np.append(demand, difference)
            consumer_names.append("Фиктивный потребитель")
            report.add(
                "Задача закрыта: добавлен фиктивный потребитель "
                f"с потребностью {_format_number(difference)} и нулевыми тарифами."
            )
        else:
            difference = total_demand - total_supply
            costs = np.vstack((costs, np.zeros(costs.shape[1])))
            supply = np.append(supply, difference)
            supplier_names.append("Фиктивный склад")
            report.add(
                "Задача закрыта: добавлен фиктивный склад "
                f"с запасом {_format_number(difference)} и нулевыми тарифами."
            )
    else:
        report.add("Задача закрытая: общий запас равен общей потребности.")

    report.add("Условие закрытой задачи:")
    report.add(f"Тарифы: {costs.tolist()}")
    report.add(f"Запасы: {supply.tolist()}")
    report.add(f"Потребности: {demand.tolist()}")
    return _ClosedProblem(costs, supply, demand, supplier_names, consumer_names)


# ТРАНСПОРТНАЯ ЗАДАЧА (начальный план и вырожденность)

def _add_zero_basic_cells(basic: np.ndarray) -> None:
    rows, columns = basic.shape
    parent = list(range(rows + columns))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(first: int, second: int) -> bool:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        parent[first_root] = second_root
        return True

    basis_count = 0
    for row, column in zip(*np.where(basic)):
        if union(int(row), rows + int(column)):
            basis_count += 1
        else:
            basic[row, column] = False

    for row in range(rows):
        for column in range(columns):
            if basis_count == rows + columns - 1:
                return
            if not basic[row, column] and union(row, rows + column):
                basic[row, column] = True
                basis_count += 1


def _northwest_corner(
    supply: np.ndarray, demand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    allocations = np.zeros((len(supply), len(demand)))
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()
    row = column = 0

    while row < len(remaining_supply) and column < len(remaining_demand):
        amount = min(remaining_supply[row], remaining_demand[column])
        allocations[row, column] += amount
        remaining_supply[row] -= amount
        remaining_demand[column] -= amount

        if np.isclose(remaining_supply[row], 0, atol=_TOLERANCE):
            remaining_supply[row] = 0
        if np.isclose(remaining_demand[column], 0, atol=_TOLERANCE):
            remaining_demand[column] = 0

        if remaining_supply[row] == 0:
            row += 1
        if remaining_demand[column] == 0:
            column += 1

    basic = allocations > 0
    _add_zero_basic_cells(basic)
    return allocations, basic


def _least_cost(
    costs: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    allocations = np.zeros((len(supply), len(demand)))
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()
    active_rows = remaining_supply > _TOLERANCE
    active_columns = remaining_demand > _TOLERANCE

    while active_rows.any() and active_columns.any():
        available_costs = np.where(
            active_rows[:, np.newaxis] & active_columns[np.newaxis, :],
            costs,
            np.inf,
        )
        row, column = np.unravel_index(
            np.argmin(available_costs), available_costs.shape
        )
        amount = min(remaining_supply[row], remaining_demand[column])
        allocations[row, column] += amount
        remaining_supply[row] -= amount
        remaining_demand[column] -= amount

        if np.isclose(remaining_supply[row], 0, atol=_TOLERANCE):
            remaining_supply[row] = 0
            active_rows[row] = False
        if np.isclose(remaining_demand[column], 0, atol=_TOLERANCE):
            remaining_demand[column] = 0
            active_columns[column] = False

    basic = allocations > 0
    _add_zero_basic_cells(basic)
    return allocations, basic


# ТРАНСПОРТНАЯ ЗАДАЧА (метод потенциалов)

def _calculate_potentials(
    costs: np.ndarray, basic: np.ndarray
) -> _PotentialStep:
    rows, columns = costs.shape
    row_potentials = np.full(rows, np.nan)
    column_potentials = np.full(columns, np.nan)
    row_potentials[0] = 0

    changed = True
    while changed:
        changed = False
        for row, column in zip(*np.where(basic)):
            if not np.isnan(row_potentials[row]) and np.isnan(
                column_potentials[column]
            ):
                column_potentials[column] = (
                    costs[row, column] - row_potentials[row]
                )
                changed = True
            elif np.isnan(row_potentials[row]) and not np.isnan(
                column_potentials[column]
            ):
                row_potentials[row] = costs[row, column] - column_potentials[column]
                changed = True

    reduced_costs = costs - (
        row_potentials[:, np.newaxis] + column_potentials[np.newaxis, :]
    )
    reduced_costs[basic] = 0
    return _PotentialStep(row_potentials, column_potentials, reduced_costs)


def _find_cycle(basic: np.ndarray, entering: tuple[int, int]) -> list[tuple[int, int]]:
    rows, columns = basic.shape
    start = entering[0]
    target = rows + entering[1]
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(rows + columns)
    ]
    for row, column in zip(*np.where(basic)):
        row = int(row)
        column = int(column)
        adjacency[row].append((rows + column, (row, column)))
        adjacency[rows + column].append((row, (row, column)))

    parents: dict[int, tuple[int, tuple[int, int]] | None] = {start: None}
    pending = deque([start])
    while pending and target not in parents:
        vertex = pending.popleft()
        for neighbor, cell in adjacency[vertex]:
            if neighbor not in parents:
                parents[neighbor] = (vertex, cell)
                pending.append(neighbor)

    path: list[tuple[int, int]] = []
    vertex = target
    while vertex != start:
        previous, cell = parents[vertex]
        path.append(cell)
        vertex = previous
    path.reverse()
    return [entering, *reversed(path)]


def _select_entering_cell(
    reduced_costs: np.ndarray, basic: np.ndarray
) -> tuple[int, int] | None:
    candidates = np.where(~basic, reduced_costs, np.inf)
    row, column = np.unravel_index(np.argmin(candidates), candidates.shape)
    if candidates[row, column] >= -_TOLERANCE:
        return None
    return int(row), int(column)


def _redistribute(
    allocations: np.ndarray,
    basic: np.ndarray,
    costs: np.ndarray,
    cycle: list[tuple[int, int]],
) -> float:
    minus_cells = cycle[1::2]
    amount = min(allocations[row, column] for row, column in minus_cells)

    for index, (row, column) in enumerate(cycle):
        allocations[row, column] += amount if index % 2 == 0 else -amount
        if np.isclose(allocations[row, column], 0, atol=_TOLERANCE):
            allocations[row, column] = 0

    leaving_candidates = [
        cell
        for cell in minus_cells
        if allocations[cell] == 0
    ]
    leaving = max(leaving_candidates, key=lambda cell: costs[cell])
    basic[cycle[0]] = True
    basic[leaving] = False
    return float(amount)


# ТРАНСПОРТНАЯ ЗАДАЧА (вывод и главный цикл)

def _display_table(
    problem: _ClosedProblem,
    allocations: np.ndarray,
    title: str,
    report: _Report,
    potentials: _PotentialStep | None = None,
) -> None:
    headers = [
        f"{name}={_format_number(problem.demand[index])}"
        for index, name in enumerate(problem.consumer_names)
    ]
    if potentials is not None:
        headers.append("u[i]")
    row_labels = [
        f"{name}={_format_number(problem.supply[index])}"
        for index, name in enumerate(problem.supplier_names)
    ]
    plain_cells = []
    for row in range(problem.costs.shape[0]):
        cells = []
        for column in range(problem.costs.shape[1]):
            shipment_value = allocations[row, column]
            shipment = (
                "-"
                if np.isclose(shipment_value, 0, atol=_TOLERANCE)
                else _format_number(shipment_value)
            )
            cell = (
                f"{shipment} "
                f"({_format_number(problem.costs[row, column])})"
            )
            if potentials is not None:
                cell += f" {_format_number(potentials.reduced_costs[row, column])}"
            cells.append(cell)
        plain_cells.append(cells)

    if potentials is not None:
        for row, cells in enumerate(plain_cells):
            cells.append(_format_number(potentials.row_potentials[row]))

    widths = [
        max(
            len(headers[column]),
            *(
                len(plain_cells[row][column])
                for row in range(problem.costs.shape[0])
            ),
        )
        for column in range(len(headers))
    ]
    label_width = max(len("Запасы"), *(len(label) for label in row_labels))

    def format_row(label: str, values: list[str]) -> str:
        padded_values = []
        for index, value in enumerate(values):
            padding = " " * (widths[index] - len(value))
            padded_values.append(value + padding)
        return f"| {label:<{label_width}} | " + " | ".join(padded_values) + " |"

    separator = (
        "+"
        + "-" * (label_width + 2)
        + "+"
        + "+".join("-" * (width + 2) for width in widths)
        + "+"
    )
    report.add(f"\n{title}")
    report.add(separator)
    report.add(format_row("", headers))
    report.add(separator)
    for row, label in enumerate(row_labels):
        cells = []
        for column in range(problem.costs.shape[1]):
            shipment_value = allocations[row, column]
            shipment = (
                "-"
                if np.isclose(shipment_value, 0, atol=_TOLERANCE)
                else _format_number(shipment_value)
            )
            tariff = _format_number(problem.costs[row, column])
            formatted = f"{shipment} ({tariff})"
            if potentials is not None:
                reduced_cost = _format_number(potentials.reduced_costs[row, column])
                formatted += f" {reduced_cost}"
            cells.append(formatted)
        if potentials is not None:
            cells.append(plain_cells[row][-1])
        report.add(format_row(label, cells))
    if potentials is not None:
        potential_row = [
            *(_format_number(value) for value in potentials.column_potentials),
            "",
        ]
        report.add(separator)
        report.add(format_row("v[j]", potential_row))
    report.add(separator)


def _display_potential_step(
    iteration: int,
    problem: _ClosedProblem,
    allocations: np.ndarray,
    basic: np.ndarray,
    potentials: _PotentialStep,
    entering: tuple[int, int] | None,
    report: _Report,
    cycle: list[tuple[int, int]] | None = None,
) -> None:
    _display_table(
        problem,
        allocations,
        f"Шаг {iteration}. Метод потенциалов",
        report,
        potentials,
    )
    if entering is None:
        report.add("Все оценки Δc[i,j] неотрицательны: план оптимален.")
    else:
        report.add(
            f"Входящая клетка: ({entering[0] + 1}, {entering[1] + 1}), "
            f"оценка Δc={_format_number(potentials.reduced_costs[entering])}."
        )
        if cycle is not None:
            signed_cycle = [
                f"{'+' if index % 2 == 0 else '-'}({row + 1},{column + 1})"
                for index, (row, column) in enumerate(cycle)
            ]
            report.add(
                f"Цикл перераспределения: {' '.join(signed_cycle)}"
            )


def _display_solution(
    problem: _ClosedProblem, allocations: np.ndarray, report: _Report
) -> None:
    total_cost = float(np.sum(allocations * problem.costs))
    report.add("\nОптимальный план перевозок:")
    report.add(allocations)
    report.add(f"Минимальная стоимость перевозок: {_format_number(total_cost)}")


def _optimize_transport(
    problem: _ClosedProblem,
    allocations: np.ndarray,
    basic: np.ndarray,
    initial_title: str,
    report: _Report,
) -> np.matrix:
    _display_table(problem, allocations, initial_title, report)

    iteration = 2
    while True:
        potentials = _calculate_potentials(problem.costs, basic)
        entering = _select_entering_cell(potentials.reduced_costs, basic)
        if entering is None:
            _display_potential_step(
                iteration,
                problem,
                allocations,
                basic,
                potentials,
                entering,
                report,
            )
            break

        cycle = _find_cycle(basic, entering)
        _display_potential_step(
            iteration,
            problem,
            allocations,
            basic,
            potentials,
            entering,
            report,
            cycle,
        )
        amount = _redistribute(allocations, basic, problem.costs, cycle)
        report.add(f"Величина перераспределения Δx={_format_number(amount)}.")
        iteration += 1

    _display_solution(problem, allocations, report)
    return np.matrix(allocations)


def _solve_transport_impl(
    costs: npt.ArrayLike,
    supply: npt.ArrayLike,
    demand: npt.ArrayLike,
    *,
    report: _Report,
) -> np.matrix:
    costs, supply, demand = _validate_arguments(costs, supply, demand)
    problem = _close_problem(costs, supply, demand, report)

    report.add("\nРЕШЕНИЕ 1. Метод северо-западного угла")
    northwest_allocations, northwest_basic = _northwest_corner(
        problem.supply, problem.demand
    )
    northwest_result = _optimize_transport(
        problem,
        northwest_allocations,
        northwest_basic,
        "Шаг 1. Метод северо-западного угла",
        report,
    )

    report.add("\nРЕШЕНИЕ 2. Метод наименьшей стоимости")
    least_cost_allocations, least_cost_basic = _least_cost(
        problem.costs, problem.supply, problem.demand
    )
    _optimize_transport(
        problem,
        least_cost_allocations,
        least_cost_basic,
        "Шаг 1. Метод наименьшей стоимости",
        report,
    )

    return northwest_result


def solve_transport(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> np.matrix:
    report = _Report()
    return _solve_transport_impl(costs, supply, demand, report=report)


# ДВОЙСТВЕННЫЙ СИМПЛЕКС (канонизация и вывод)

def solve_linear_programming(c, A, b, direction, signs):
    """Решает прямую и двойственную задачи при x >= 0."""
    from scipy.optimize import linprog

    tolerance = 1e-7
    n = len(c)
    m = len(b)

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i, sign in enumerate(signs):
        if sign == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif sign == ">=":
            A_ub.append(-A[i])
            b_ub.append(-b[i])
        else:
            A_eq.append(A[i])
            b_eq.append(b[i])

    A_ub = np.asarray(A_ub) if A_ub else None
    b_ub = np.asarray(b_ub) if b_ub else None
    A_eq = np.asarray(A_eq) if A_eq else None
    b_eq = np.asarray(b_eq) if b_eq else None

    print("\n========== ПРЯМАЯ ЗАДАЧА ==========")
    primal_result = linprog(
        c=c if direction == "min" else -c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, None)] * n,
        method="highs",
    )

    if not primal_result.success:
        print("Решение прямой задачи не найдено:", primal_result.message)
        return

    x = primal_result.x
    primal_value = float(primal_result.fun if direction == "min" else -primal_result.fun)
    print(f"Оптимальное значение = {primal_value:.6f}")
    print("Оптимальный план x* =", np.round(x, 6))

    print("\nПроверка ограничений прямой задачи:")
    primal_ok = True
    for i in range(m):
        lhs = float(A[i] @ x)
        if signs[i] == "<=":
            ok = lhs <= b[i] + tolerance
        elif signs[i] == ">=":
            ok = lhs >= b[i] - tolerance
        else:
            ok = abs(lhs - b[i]) <= tolerance
        primal_ok = primal_ok and ok
        print(f"Ограничение {i + 1}: {lhs:.6f} {signs[i]} {b[i]:.6f} [{'OK' if ok else 'НЕТ'}]")

    x_nonnegative = bool(np.all(x >= -tolerance))
    primal_ok = primal_ok and x_nonnegative
    print("Неотрицательность x:", "OK" if x_nonnegative else "НЕТ")

    # Построение двойственной задачи.
    # Для min: <= -> y<=0, >= -> y>=0, = -> y свободна; A.T@y <= c.
    # Для max: <= -> y>=0, >= -> y<=0, = -> y свободна; A.T@y >= c.
    dual_bounds = []
    for sign in signs:
        if sign == "=":
            dual_bounds.append((None, None))
        elif direction == "min":
            dual_bounds.append((None, 0) if sign == "<=" else (0, None))
        else:
            dual_bounds.append((0, None) if sign == "<=" else (None, 0))

    print("\n========== ДВОЙСТВЕННАЯ ЗАДАЧА ==========")
    if direction == "min":
        dual_result = linprog(
            c=-b, A_ub=A.T, b_ub=c,
            bounds=dual_bounds, method="highs",
        )
        dual_direction = "max"
    else:
        dual_result = linprog(
            c=b, A_ub=-A.T, b_ub=-c,
            bounds=dual_bounds, method="highs",
        )
        dual_direction = "min"

    if not dual_result.success:
        print("Решение двойственной задачи не найдено:", dual_result.message)
        return

    y = dual_result.x
    dual_value = float(-dual_result.fun if direction == "min" else dual_result.fun)
    print(f"Оптимальное значение W_{dual_direction} = {dual_value:.6f}")
    print("Оптимальный план y* =", np.round(y, 6))

    print("\nПроверка ограничений двойственной задачи:")
    dual_ok = True
    lhs_values = A.T @ y
    for j in range(n):
        if direction == "min":
            ok = lhs_values[j] <= c[j] + tolerance
            relation = "<="
        else:
            ok = lhs_values[j] >= c[j] - tolerance
            relation = ">="
        dual_ok = dual_ok and bool(ok)
        print(f"Ограничение {j + 1}: {lhs_values[j]:.6f} {relation} {c[j]:.6f} [{'OK' if ok else 'НЕТ'}]")

    print("\nПроверка знаков переменных y:")
    for i, sign in enumerate(signs):
        if sign == "=":
            ok, description = True, "свободная переменная"
        elif direction == "min" and sign == "<=":
            ok, description = y[i] <= tolerance, "y <= 0"
        elif direction == "min" and sign == ">=":
            ok, description = y[i] >= -tolerance, "y >= 0"
        elif direction == "max" and sign == "<=":
            ok, description = y[i] >= -tolerance, "y >= 0"
        else:
            ok, description = y[i] <= tolerance, "y <= 0"
        dual_ok = dual_ok and bool(ok)
        print(f"Переменная y{i + 1}: {y[i]:.6f}, {description} [{'OK' if ok else 'НЕТ'}]")

    print("\n========== ИТОГОВАЯ ПРОВЕРКА ==========")
    print("Прямая задача допустима:", "OK" if primal_ok else "НЕТ")
    print("Двойственная задача допустима:", "OK" if dual_ok else "НЕТ")
    print(f"Значение прямой задачи = {primal_value:.6f}")
    print(f"Значение двойственной задачи = {dual_value:.6f}")

    if primal_ok and dual_ok and np.isclose(primal_value, dual_value, rtol=1e-7, atol=1e-9):
        print("\nВЫВОД: обе задачи решены оптимально.")
    else:
        print("\nВЫВОД: проверьте результаты и допустимость решений.")


def get_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Ошибка: введите целое положительное число.")


def get_float_list(prompt: str, expected_length: int | None = None) -> list[float]:
    while True:
        try:
            values = [float(value.replace(",", ".")) for value in input(prompt).replace(";", " ").split()]
            if expected_length is not None and len(values) != expected_length:
                print(f"Ошибка: необходимо ввести ровно {expected_length} чисел.")
                continue
            return values
        except ValueError:
            print("Ошибка: вводите числа через пробел (например: 1.5 2 3).")


def run_transport() -> None:
    print("\n=== Интерактивное решение транспортной задачи ===\n")
    rows = get_int("Введите количество складов (строк): ")
    columns = get_int("Введите количество потребителей (столбцов): ")

    print("\nВведите матрицу тарифов (построчно, числа через пробел):")
    costs = [get_float_list(f"Тарифы для склада {i + 1}: ", columns) for i in range(rows)]
    supply = get_float_list("Запасы: ", rows)
    demand = get_float_list("Потребности: ", columns)

    print("\n" + "=" * 50)
    print("РЕШЕНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ:")
    print("=" * 50)
    solve_transport(costs, supply, demand)


def run_linear_programming() -> None:
    print("\n=== Решение прямой и двойственной задач ЛП ===\n")
    n = get_int("Число переменных n: ")
    m = get_int("Число ограничений m: ")

    c = get_float_list(f"Вектор c ({n} чисел): ", n)
    while True:
        direction = input("Направление (min/max): ").strip().lower()
        if direction in ("min", "max"):
            break
        print("Ошибка: введите min или max.")

    print("\nВведите матрицу коэффициентов A:")
    A = [get_float_list(f"Строка {i + 1}: ", n) for i in range(m)]

    print("\nВведите знаки ограничений (<=, >= или =):")
    signs = []
    for i in range(m):
        while True:
            sign = input(f"Знак ограничения {i + 1}: ").strip()
            if sign in ("<=", ">=", "="):
                signs.append(sign)
                break
            print("Допустимые знаки: <=, >=, =")

    b = get_float_list(f"Вектор b ({m} чисел): ", m)
    print("\nУсловия: все переменные неотрицательны (x >= 0).")
    solve_linear_programming(
        np.asarray(c, dtype=float), np.asarray(A, dtype=float),
        np.asarray(b, dtype=float), direction, signs,
    )


def main() -> None:
    print("Выберите задачу:")
    print("1. Транспортная задача")
    print("2. Прямая и двойственная задачи линейного программирования")
    while True:
        choice = input("\nВаш выбор (1 или 2): ").strip()
        if choice == "1":
            run_transport()
            break
        if choice == "2":
            run_linear_programming()
            break
        print("Ошибка: введите 1 или 2.")


if __name__ == "__main__":
    main()
