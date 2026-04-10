# Модель: Модель балки Ейлера-Бернуллі (5 семестр)
# Автори: Калкатін Владислав та Апостолов Микити, група АІ-235

import numpy as np
import matplotlib.pyplot as plt

class BridgeLoadModel:
    def __init__(self, length=20, q=1e4, E=2e11, I=0.0054, n=200):
        """
        Ініціалізація моделі балки мосту
        length – довжина балки (м)
        q – рівномірно розподілене навантаження (Н/м)
        E – модуль пружності (Па)
        I – момент інерції (м⁴)
        n – кількість вузлів дискретизації
        """
        self.L = length
        self.q = q
        self.E = E
        self.I = I
        self.n = n
        self.dx = self.L / (self.n - 1)
        self.x = np.linspace(0, self.L, self.n)
        self.w = np.zeros(self.n)  # прогин
        self.M = np.zeros(self.n)  # момент
        self.sigma = np.zeros(self.n)  # напруга

    def solve_beam_deflection(self):
        """
        Розв'язання рівняння E*I*w'''' = q методом скінченних різниць
        Використовується однорідна балка з шарнірним закріпленням
        """
        A = np.zeros((self.n, self.n))
        b = np.full(self.n, self.q / (self.E * self.I))

        # Четверта похідна: центральна різниця
        for i in range(2, self.n - 2):
            A[i, i - 2] = 1
            A[i, i - 1] = -4
            A[i, i] = 6
            A[i, i + 1] = -4
            A[i, i + 2] = 1
        A = A / (self.dx ** 4)

        # Граничні умови: w(0)=0, w(L)=0, M(0)=0, M(L)=0
        A[0, 0] = 1
        A[1, 0:3] = [1, -2, 1]  # M(0)=0
        A[-2, -3:] = [1, -2, 1]  # M(L)=0
        A[-1, -1] = 1

        b[0] = 0
        b[1] = 0
        b[-2] = 0
        b[-1] = 0

        # Розв'язання системи
        self.w = np.linalg.solve(A, b)

    def calculate_moment_and_stress(self, y_max=0.3):
        """
        Обчислення згинального моменту M(x) і максимальної напруги σ(x)
        """
        # M(x) = -E * I * w''
        d2w = np.gradient(np.gradient(self.w, self.dx), self.dx)
        self.M = -self.E * self.I * d2w

        # σ = M*y_max / I
        self.sigma = self.M * y_max / self.I

    def visualize_results(self):
        """
        Побудова графіків прогину, моменту і напруги
        """
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(self.x, self.w, 'b')
        plt.title("Розподіл прогину балки")
        plt.ylabel("w(x), м")

        plt.subplot(3, 1, 2)
        plt.plot(self.x, self.M, 'r')
        plt.title("Згинальний момент уздовж балки")
        plt.ylabel("M(x), Н·м")

        plt.subplot(3, 1, 3)
        plt.plot(self.x, self.sigma / 1e6, 'g')
        plt.title("Напруга у волокнах балки")
        plt.xlabel("x, м")
        plt.ylabel("σ(x), МПа")

        plt.tight_layout()
        plt.show()

    def print_summary(self):
        print("ПІДСУМКОВІ РЕЗУЛЬТАТИ")
        print(f"Максимальний прогин w_max = {np.min(self.w):.6f} м")
        print(f"Максимальний момент M_max = {np.max(np.abs(self.M)):.2e} Н·м")
        print(f"Максимальна напруга σ_max = {np.max(np.abs(self.sigma)) / 1e6:.2f} МПа")

# Приклад використання
model = BridgeLoadModel()
model.solve_beam_deflection()
model.calculate_moment_and_stress()
model.visualize_results()
model.print_summary()
