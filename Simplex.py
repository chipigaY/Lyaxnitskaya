import numpy as np
from fractions import Fraction


class SimplexSolver:
    def __init__(self):
        c = list(map(float, input('Введите коэффициенты целевой функции: ').split()))
        A = []
        b = list(map(float, input('Введите правые части ограничений: ').split()))
        for i in range(len(b)):
            A.append(list(map(float, input('Введите коэффициенты матрицы ограничений: ').split())))
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.signs = input("Введите знаки ограничений: ").split()
        self.maximize = input("maximize/minimize: ").strip().lower() == "maximize"
        self.verbose = input("verbose/not verbose: ").strip().lower() == "verbose"
        self.n_orig = len(c)
        self.m = len(b)

    def _to_canonical(self):
        A = []
        b = []
        extra = 0
        for i in range(self.m):
            s = self.signs[i]
            if s == '<=':
                extra += 1
            elif s == '>=':
                extra += 2
            elif s == '=':
                extra += 1
        n_total = self.n_orig + extra

        art_indices = []
        basis = []
        col = self.n_orig

        for i in range(self.m):
            row_full = [0.0] * n_total
            for j in range(self.n_orig):
                row_full[j] = self.A_orig[i][j]
            s = self.signs[i]
            rhs = self.b_orig[i]

            if rhs < 0:
                row_full = [-x for x in row_full]
                rhs = -rhs
                s = {'<=': '>=', '>=': '<=', '=': '='}[s]

            if s == '<=':
                row_full[col] = 1.0
                basis.append(col)
                col += 1
            elif s == '>=':
                row_full[col] = -1.0
                col += 1
                row_full[col] = 1.0
                art_indices.append(col)
                basis.append(col)
                col += 1
            elif s == '=':
                row_full[col] = 1.0
                art_indices.append(col)
                basis.append(col)
                col += 1

            A.append(row_full)
            b.append(rhs)

        c_full = [0.0] * n_total
        for j in range(self.n_orig):
            c_full[j] = self.c_orig[j] if self.maximize else -self.c_orig[j]

        return np.array(A), np.array(b), np.array(c_full), basis, art_indices

    @staticmethod
    def _print_table(T, basis, n_total, title=""):
        print("\n" + "=" * 70)
        if title:
            print(title)
        header = "Базис | " + " ".join(f"x{j+1}      " for j in range(n_total)) + " |   b"
        print(header)
        print("-" * len(header))

        m = len(basis)
        for i in range(m):
            row = T[i]
            bname = f"x{basis[i]+1}"
            vals = " ".join(f"{v:7.3f}" for v in row[:-1])
            print(f"{bname:>5} | {vals} | {row[-1]:7.3f}")

        print("-" * len(header))
        f_row = T[-1]
        vals = " ".join(f"{v:7.3f}" for v in f_row[:-1])
        print(f"{'f':>5} | {vals} | {f_row[-1]:7.3f}")

    def _simplex_step(self, T, basis, allow_artificial_out=True):
        """Выполняет один шаг симплекс-метода. Возвращает True, если шаг сделан."""
        f_row = T[-1, :-1]
        pivot_col = int(np.argmax(f_row))
        if f_row[pivot_col] <= 1e-9:
            return False       

        ratios = []
        for i in range(len(basis)):
            a = T[i, pivot_col]
            if a > 1e-9:
                ratios.append((T[i, -1] / a, i))
        if not ratios:
            raise ValueError("Задача неограничена (нет ведущей строки).")
        pivot_row = min(ratios)[1]

        pivot = T[pivot_row, pivot_col]
        T[pivot_row] /= pivot
        for i in range(T.shape[0]):
            if i != pivot_row:
                T[i] -= T[i, pivot_col] * T[pivot_row]
        basis[pivot_row] = pivot_col
        return True

    def solve(self):
        A, b, c, basis, art_indices = self._to_canonical()
        n_total = A.shape[1]
        m = A.shape[0]

        T = np.zeros((m + 1, n_total + 1))
        T[:m, :n_total] = A
        T[:m, -1] = b
        T[-1, :n_total] = c
        T[-1, -1] = 0.0

        if self.verbose:
            self._print_table(T, basis, n_total, "Начальная таблица")

        if art_indices:

            T[-1, :] = 0.0                        
            for j in art_indices:
                T[-1, j] = 1.0                      
            for i in range(m):
                if basis[i] in art_indices:
                    T[-1] -= T[i]                   
            T[-1, :n_total] *= -1                   

            if self.verbose:
                self._print_table(T, basis, n_total, "Фаза 1 (искусственный базис)")

            while self._simplex_step(T, basis):
                if self.verbose:
                    self._print_table(T, basis, n_total)

            art_sum = sum(T[i, -1] for i in range(m) if basis[i] in art_indices)
            if abs(art_sum) > 1e-6:
                print("Задача несовместна (искусственные не вышли из базиса).")
                return None
            
            keep = [j for j in range(n_total) if j not in art_indices]
            for i in range(m):
                if basis[i] in art_indices:
                    for j in keep:
                        if abs(T[i, j]) > 1e-9:
                            pivot = T[i, j]
                            T[i] /= pivot
                            for k in range(T.shape[0]):
                                if k != i:
                                    T[k] -= T[k, j] * T[i]
                            basis[i] = j
                            break
                    else:
                        T = np.delete(T, i, axis=0)
                        del basis[i]
                        m -= 1

            T = T[:, keep + [n_total]] if False else np.hstack(
                [T[:, keep], T[:, -1:]]
            )
            n_total = len(keep)

            c_new = c[keep]
            T[-1, :] = 0.0
            for j in range(n_total):
                T[-1, j] = c_new[j]
            for i in range(m):
                cb = c_new[basis[i]] if basis[i] < n_total else 0.0
                T[-1] -= cb * T[i]

            if self.verbose:
                self._print_table(T, basis, n_total, "Фаза 2 (исходная цель)")

            while self._simplex_step(T, basis):
                if self.verbose:
                    self._print_table(T, basis, n_total)
        else:
            while self._simplex_step(T, basis):
                if self.verbose:
                    self._print_table(T, basis, n_total)

        x = np.zeros(n_total)
        for i, bj in enumerate(basis):
            if bj < n_total:
                x[bj] = T[i, -1]

        f_val = T[-1, -1]
        if not self.maximize:
            f_val = -f_val

        f_val = T[-1, -1]
        if not self.maximize:
            f_val = -f_val

        x_orig = x[: self.n_orig]
        return {
            "x": x_orig,
            "f": f_val,
            "status": "optimal",
        }

if __name__ == "__main__":
    print("=" * 70)
    print("ПРИМЕР 1. Обычный симплекс-метод")
    print("max  f = 3x1 + 2x2")
    print("при  x1 + x2 <= 4")
    print("      x1 + 3x2 <= 6")
    print("      x1, x2 >= 0")
    print("=" * 70)

    solver = SimplexSolver()
    res = solver.solve()
    print(f"\nОтвет: x = {res['x']}, f = {res['f']:.3f}, статус = {res['status']}")

    print("\n\n" + "=" * 70)
    print("ПРИМЕР 2. Метод искусственного базиса")
    print("max  f = 2x1 + 3x2")
    print("при  x1 + x2 >= 3")
    print("      x1 + 2x2 <= 8")
    print("      x1, x2 >= 0")
    print("=" * 70)

    solver2 = SimplexSolver()
    res2 = solver2.solve()
    print(f"\nОтвет: x = {res['x']}, f = {res['f']:.3f}, статус = {res['status']}")