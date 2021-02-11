import numpy as np

class Neuralna:
    def __init__(self):

        self.tezine = np.random.random((3, 1)) - 1

    def sigmoidalna(self, x):
        return 1 / (1 + np.exp(-x))

    def izvod_sigmoidalne(self, x):
        return (1 - x) * x

    def treniraj(self, ulazi, izlazi_nn, iteracije):
        for i in range(iteracije):
            izlaz = self.misli(ulazi)
            greska = izlazi_nn - izlaz
            ispravka = np.dot(ulazi.T, greska * self.izvod_sigmoidalne(izlaz))
            self.tezine += ispravka

    def misli(self,ulazi):
        ulazi = ulazi.astype(float)
        izlaz = self.sigmoidalna(np.dot(ulazi, self.tezine))
        return izlaz


nm = Neuralna()
print("Nasumicno izabrane tezine: ")
print(nm.tezine)

ulazi = np.array([[1, 1, 1],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 0, 0]
                  ])
izlazi_nn = np.array([[1, 0, 1, 0]]).T

nm.treniraj(ulazi, izlazi_nn, 10000)
print("Tezine nakon treniranja: ")
print(nm.tezine)

print("Novi ulazi: ")
ulaz1 = str(input("Unesi ulaz 1: "))
ulaz2 = str(input("Unesi ulaz 2: "))
ulaz3 = str(input("Unesi ulaz 3: "))
print(ulaz1, ulaz2, ulaz3)
print("Novi izlaz: ")

novi = nm.misli(np.array([ulaz1, ulaz2, ulaz3]))
print(novi)
