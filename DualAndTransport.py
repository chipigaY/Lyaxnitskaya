import numpy as np
import numpy.typing as npt
from collections import deque
from dataclasses import dataclass

# ТРАНСПОРТНАЯ ЗАДАЧА

_TOLERANCE = 1e-10

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

# ДВОЙСТВЕННЫЙ СИМПЛЕКС 

_TOLERANCE = 1e-10
_MAX_ITERATIONS = 1000


def format_equation(row, var_names):
    terms = []
    for coef, name in zip(row, var_names):
        if abs(coef) < 1e-9:
            continue
        if abs(coef - 1.0) < 1e-9:
            terms.append(name)
        elif abs(coef + 1.0) < 1e-9:
            terms.append(f"-{name}")
        else:
            terms.append(f"{coef:g}*{name}")
    if not terms:
        return "0"
    equation = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            equation += f" - {term[1:]}"
        else:
            equation += f" + {term}"
    return equation


def _canonicalize(A, b, c, sense, constraint_signs, variable_constraints):
    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float).reshape(-1)
    c_np = np.asarray(c, dtype=float).reshape(-1)

    m, n = A_np.shape
    signs = ["<="] * m if constraint_signs is None else list(constraint_signs)
    if len(signs) != m or any(sign not in ("<=", ">=", "=") for sign in signs):
        raise ValueError("constraint_signs должны задавать знак '<=', '>=' или '=' для каждого ограничения.")

    if variable_constraints is None:
        variable_constraints = ["nonnegative"] * n
    if len(variable_constraints) != n or any(constraint not in ("nonnegative", "free") for constraint in variable_constraints):
        raise ValueError("variable_constraints должны задавать 'nonnegative' или 'free' для каждой переменной.")

    var_names = [f"x{i + 1}" for i in range(n)]
    output = ["Приведение к канонической форме:"]

    if sense == "min":
        output.append("1) Умножаем целевую функцию на -1, чтобы перейти к максимизации:")
        c_np = -c_np
        output.append(f"   f(x) = {format_equation(c_np, var_names)} -> max")
    else:
        output.append(f"1) Задача уже на максимум: f(x) = {format_equation(c_np, var_names)} -> max")

    free_vars = [index for index, constraint in enumerate(variable_constraints) if constraint == "free"]
    if free_vars:
        output.append("2) Замена свободных переменных (x = x' - x''):")
        for index in reversed(free_vars):
            A_np = np.insert(A_np, index + 1, -A_np[:, index], axis=1)
            c_np = np.insert(c_np, index + 1, -c_np[index])
            var_names[index] = f"x'{index + 1}"
            var_names.insert(index + 1, f"x''{index + 1}")
            output.append(f"   x{index + 1} -> x'{index + 1} - x''{index + 1}")
    else:
        output.append("2) Свободных переменных нет.")

    output.append("3) Преобразование ограничений в уравнения:")
    extra_columns = []
    extra_names = []
    variable_count = n

    for index, sign in enumerate(signs):
        row_str = format_equation(A_np[index], var_names)
        value = b_np[index]

        if sign == "<=":
            variable_count += 1
            column = np.zeros(m)
            column[index] = 1.0
            extra_columns.append(column)
            extra_names.append(f"x{variable_count}")
            output.append(f"   {row_str} <= {value:g}  =>  {row_str} + x{variable_count} = {value:g}")
        elif sign == ">=":
            variable_count += 1
            column = np.zeros(m)
            column[index] = -1.0
            extra_columns.append(column)
            extra_names.append(f"x{variable_count}")
            output.append(f"   {row_str} >= {value:g}  =>  {row_str} - x{variable_count} = {value:g}")
        else:
            output.append(f"   {row_str} = {value:g}   (уже уравнение)")

    decision_count = A_np.shape[1]
    if extra_columns:
        A_final = np.column_stack([A_np, *extra_columns])
        c_final = np.pad(c_np, (0, len(extra_columns)))
        var_names.extend(extra_names)
    else:
        A_final = A_np
        c_final = c_np

    output.append("4) Все переменные неотрицательны.")
    output.append("\nИТОГОВАЯ КАНОНИЧЕСКАЯ ФОРМА:")
    output.append(f"f(x) = {format_equation(c_final, var_names)} -> max")
    for index in range(m):
        output.append(f"  {format_equation(A_final[index], var_names)} = {b_np[index]:g}")
    output.append(f"  {', '.join(var_names)} >= 0")

    return (A_final, b_np, c_final, decision_count), var_names, free_vars, output


def _print_table(tableau, basis, names, title, pivot_row=None, pivot_column=None):
    print(f"\n{title}")
    rows, cols = tableau.shape
    variable_count = cols - 1

    non_basic = [j for j in range(variable_count) if j not in basis]
    
    headers = ["Базис"] + [names[j] for j in non_basic] + ["b"]
    
    data_rows = []
    
    for i in range(rows - 1):
        b_name = names[basis[i]] if i < len(basis) else ""
        row_vals = [b_name]
        for j in non_basic:
            val_str = f"{tableau[i, j]:.4g}"
            if pivot_row == i and pivot_column == j:
                val_str += "*"
            row_vals.append(val_str)
        row_vals.append(f"{tableau[i, -1]:.4g}")
        data_rows.append(row_vals)
        
    f_vals = ["Δ"]
    for j in non_basic:
        val_str = f"{tableau[-1, j]:.4g}"
        if pivot_row is None and pivot_column == j:
            val_str += "*"
        f_vals.append(val_str)
    f_vals.append(f"{tableau[-1, -1]:.4g}")
    data_rows.append(f_vals)

    widths = []
    for col_idx in range(len(headers)):
        col_vals = [headers[col_idx]] + [row[col_idx] for row in data_rows]
        widths.append(max(len(str(v)) for v in col_vals))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    
    h_str = "| "
    for i, h in enumerate(headers):
        h_str += f"{h:<{widths[i]}} | "
    print(h_str)
    print(sep)
    
    for row in data_rows:
        r_str = "| "
        for i, val in enumerate(row):
            r_str += f"{val:<{widths[i]}} | "
        print(r_str)
        print(sep)


def solve_dual_simplex(A, b, c, sense="max", constraint_signs=None, variable_constraints=None):
    table, names, free_vars, output = _canonicalize(A, b, c, sense, constraint_signs, variable_constraints)
    canonical_matrix, canonical_rhs, objective, decision_count = table
    
    for line in output:
        print(line)
        
    constraint_count, variable_count = canonical_matrix.shape
    original_objective = np.asarray(c, dtype=float).reshape(-1)

    variable_mapping = []
    transformed_index = 0
    for index in range(len(original_objective)):
        if index in free_vars:
            variable_mapping.append((transformed_index, transformed_index + 1))
            transformed_index += 2
        else:
            variable_mapping.append((transformed_index, None))
            transformed_index += 1

    column_support = np.abs(canonical_matrix) > _TOLERANCE
    basis = []
    for row in range(constraint_count):
        candidates = np.flatnonzero(
            column_support[row] & (column_support.sum(axis=0) == 1)
        )
        if not len(candidates):
            raise ValueError("Нет базиса. Используйте МИБ.")
        supplemental = candidates[candidates >= decision_count]
        basis.append(int(supplemental[0] if len(supplemental) else candidates[0]))

    for row, basic_variable in enumerate(basis):
        coefficient = canonical_matrix[row, basic_variable]
        canonical_matrix[row, :] /= coefficient
        canonical_rhs[row] /= coefficient

    row_count = constraint_count

    dual_candidates = np.flatnonzero(canonical_rhs < -_TOLERANCE)
    if not len(dual_candidates):
        print("\nНет нужды в двойственном симплексе. Используйте обычный симплекс-метод.")
        return

    print("\nПосле нормализации базисных переменных найдены отрицательные b; применяем двойственный симплекс-метод.")
    
    tableau = np.zeros((row_count + 1, variable_count + 1))
    tableau[:row_count, :variable_count] = canonical_matrix
    tableau[:row_count, -1] = canonical_rhs
    tableau[-1, :variable_count] = -objective
    
    for row, basic_variable in enumerate(basis):
        tableau[-1, :] += objective[basic_variable] * tableau[row, :]
    tableau[np.abs(tableau) < _TOLERANCE] = 0

    def pivot(row, column):
        element = tableau[row, column]
        tableau[row, :] /= element
        for other_row in range(row_count + 1):
            if other_row == row:
                continue
            factor = tableau[other_row, column]
            if abs(factor) > _TOLERANCE:
                tableau[other_row, :] -= factor * tableau[row, :]
        basis[row] = column
        tableau[np.abs(tableau) < _TOLERANCE] = 0

    for iteration in range(1, _MAX_ITERATIONS + 1):
        print("\nБазисные переменные: " + ", ".join(names[index] for index in basis))
        
        negative_rhs = np.flatnonzero(tableau[:row_count, -1] < -_TOLERANCE)
        negative_delta = np.flatnonzero(tableau[-1, :-1] < -_TOLERANCE)
        
        if not len(negative_rhs) and not len(negative_delta):
            _print_table(tableau, basis, names, f"Симплекс-таблица, итерация {iteration}")
            break

        if len(negative_rhs):
            leaving = int(negative_rhs[np.argmin(tableau[negative_rhs, -1])])
            candidates = [
                column
                for column in range(variable_count)
                if tableau[leaving, column] < -_TOLERANCE
            ]
            if not candidates:
                _print_table(tableau, basis, names, f"Симплекс-таблица, итерация {iteration}", pivot_row=leaving)
                raise ValueError("Задача несовместна: в ведущей строке нет подходящего ведущего столбца.")
            
            ratios = {
                column: tableau[-1, column] / tableau[leaving, column]
                for column in candidates
            }
            entering = max(candidates, key=ratios.__getitem__)
            print(f"Ведущая строка: {leaving + 1}; ведущий столбец: {names[entering]}; Δ/a = {ratios[entering]:g}.")
        else:
            entering = int(negative_delta[np.argmin(tableau[-1, negative_delta])])
            candidates = [
                row
                for row in range(row_count)
                if tableau[row, entering] > _TOLERANCE
            ]
            if not candidates:
                _print_table(tableau, basis, names, f"Симплекс-таблица, итерация {iteration}", pivot_column=entering)
                raise ValueError("Целевая функция не ограничена на допустимой области.")
                
            leaving = min(
                candidates,
                key=lambda row: tableau[row, -1] / tableau[row, entering],
            )
            print(f"Все b неотрицательны; завершаем оптимизацию обычным симплекс-шагом. Входит {names[entering]}.")

        _print_table(
            tableau, basis, names,
            f"Симплекс-таблица, итерация {iteration}",
            pivot_column=entering, pivot_row=leaving,
        )
        pivot(leaving, entering)
    else:
        raise ValueError("Превышено максимальное количество итераций.")

    transformed_solution = np.zeros(variable_count)
    for row, variable in enumerate(basis):
        if variable < variable_count:
            transformed_solution[variable] = tableau[row, -1]
    
    transformed_solution = transformed_solution[:decision_count]
    
    original_solution = np.asarray([
        transformed_solution[first]
        if second is None
        else transformed_solution[first] - transformed_solution[second]
        for first, second in variable_mapping
    ])
    
    objective_value = float(np.dot(original_objective, original_solution))
    
    print("\nРешение исходной задачи: x* = (" + ", ".join(f"{value:g}" for value in original_solution) + f")")
    print(f"Значение целевой функции: {objective_value:g}")

# ИНТЕРФЕЙС

def get_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Ошибка: введите целое число.")


def get_float_list(prompt: str, expected_length: int | None = None) -> list[float]:
    while True:
        try:
            raw = input(prompt).split()
            values = [float(x) for x in raw]
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
    costs = []
    for i in range(rows):
        row = get_float_list(f"Тарифы для склада {i + 1}: ", columns)
        costs.append(row)

    print("\nВведите запасы для каждого склада (через пробел):")
    supply = get_float_list("Запасы: ", rows)

    print("\nВведите потребности для каждого потребителя (через пробел):")
    demand = get_float_list("Потребности: ", columns)

    print("\n" + "=" * 50)
    print("РЕШЕНИЕ ЗАДАЧИ:")
    print("=" * 50)
    
    solve_transport(costs, supply, demand)


def run_dual_simplex() -> None:
    print("\n=== Интерактивное решение задачи двойственным симплекс-методом ===\n")
    
    print("1. Введите коэффициенты целевой функции (через пробел):")
    c = get_float_list("c: ")
    
    print("\n2. Введите тип оптимизации (max или min):")
    while True:
        sense = input("Sense: ").strip().lower()
        if sense in ("max", "min"):
            break
        print("Ошибка: введите 'max' или 'min'.")

    print(f"\n3. Количество переменных определено по целевой функции: {len(c)}")
    n = len(c)
    m = get_int("   Введите количество ограничений (m): ")

    print("\n4. Введите матрицу ограничений A (построчно, числа через пробел):")
    A = []
    for i in range(m):
        row = get_float_list(f"Ограничение {i + 1}: ", n)
        A.append(row)

    print("\n5. Введите правую часть ограничений b (через пробел):")
    b = get_float_list("b: ", m)

    print("\n6. Введите знаки ограничений через пробел (например: <= >= =):")
    while True:
        signs = input("Знаки: ").split()
        if len(signs) == m and all(s in ("<=", ">=", "=") for s in signs):
            break
        print(f"Ошибка: необходимо ввести ровно {m} знаков (<=, >=, =).")

    print("\n7. Введите ограничения на переменные (>=0 или any для каждой):")
    while True:
        vc_input = input("Ограничения: ").split()
        if len(vc_input) == n and all(s in (">=0", "any") for s in vc_input):
            break
        print(f"Ошибка: необходимо ввести ровно {n} условий (>=0 или any).")
    
    variable_constraints = ["nonnegative" if s == ">=0" else "free" for s in vc_input]

    print("\n" + "=" * 50)
    print("РЕШЕНИЕ ЗАДАЧИ:")
    print("=" * 50)
    
    try:
        solve_dual_simplex(A, b, c, sense, signs, variable_constraints)
    except Exception as e:
        print(f"\nОшибка при решении: {e}")


def main() -> None:
    print("Выберите метод решения:")
    print("1. Транспортная задача")
    print("2. Двойственный симплекс-метод")
    
    while True:
        choice = input("\nВаш выбор (1 или 2): ").strip()
        if choice == "1":
            run_transport()
            break
        elif choice == "2":
            run_dual_simplex()
            break
        else:
            print("Ошибка: введите 1 или 2.")


if __name__ == "__main__":
    main()