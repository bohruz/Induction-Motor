import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cmath import polar


class InductionMotor():
    """Induction Motor Class"""

    def __init__(self, v, s, r1, x1, r2, x2, xm, rc=0, polos=4, frequency=60, perdasNucleo=0, perdasAtritVent=0, perdasSuplementares=0):
        """v: Tensão de linha
           s: Escorregamento
           r1: Resistência do Estator
           x1: Indutância do Estator
           r2: Resistência do Rotor
           x2: Indutância do Rotor
           xm: Indutância de magnetização
           rc: Resistência do nucleo
           polos: numero de polos do motor - default 4
           frequency: Frequência da rede - default 4
           perdasNucleo: Perdas no núcleo - default 0
           perdasAtritVent: Perdas por atrito e ventilação - default 0
           perdasSuplementares: Perdas Suplementares - default 0
           """

        self.setEscorregamento(s)
        self.setVoltage(v)
        self.setNumPolos(polos)
        self.setFrequency(frequency)
        self.__r2 = r2
        self.__z1 = r1 + 1j * x1
        self.__z2 = (r2/self.__getEscorregamento()) + 1j * x2
        self.__xm = 1j * xm
        self.__rc = rc
        self.__perdasNucleo = perdasNucleo
        self.__perdasAtritVent = perdasAtritVent
        self.__perdasSuplementares = perdasSuplementares

    # getters and setters properties
    def setVoltage(self, v):
        self.__v = v/np.sqrt(3)

    def __getVoltage(self):
        return self.__v

    def setEscorregamento(self, s):
        self.__escorregamento = s

    def __getEscorregamento(self):
        return self.__escorregamento

    def setNumPolos(self, numeroPolos):
        self.__polos = numeroPolos

    def __getNumPolos(self):
        return self.__polos

    def setFrequency(self, frequency):
        self.__frequency = frequency

    def __getFrequency(self):
        return self.__frequency

    # Função privada para calcular a impedância em paralelo
    def __getParallel(self, z1, z2):
        return (z1 * z2)/(z1 + z2)

    # Calcula a Frenquência do Rotor
    def getRotorFrequency(self):
        return self.__getEscorregamento() * self.__getFrequency()

    # Velocidade sincrona
    def WSinc(self, rad=False):
        return (4 * np.pi * self.__getFrequency())/self.__getNumPolos() \
            if rad else (120 * self.__getFrequency())/self.__getNumPolos()

    # Velocidade mecânica
    def WMec(self, rad=False):
        return (1 - self.__getEscorregamento()) * self.WSinc(rad)

    # Velocidade mecânica para o torque máximo
    def WMecMaxTorque(self, rad=False):
        return (1 - self.getSMaxTorque()) * self.WSinc(rad)

    # Impedância Equivalente
    def getZeq(self):

        if(self.__rc == 0):
            return self.__getParallel(self.__xm, self.__z2) + self.__z1
        else:
            xmrc = self.__getParallel(self.__rc, self.__xm)
            return self.__getParallel(xmrc, self.__z2) + self.__z1

    # Impedância Equivalente de Thevenin
    def getZThevenin(self):

        if(self.__rc == 0):
            return self.__getParallel(self.__z1, self.__xm)

        else:
            xmrc = self.__getParallel(self.__rc, self.__xm)

            return self.__getParallel(self.__z1, xmrc)

    # Tensão de Thevenin
    def getVThevenin(self):
        return self.__getVoltage() * (self.__xm/(self.__xm + self.__z1))

    # Corrente de Linha

    def correnteEntrada(self):
        return self.__getVoltage()/self.getZeq()

    # Fator de potência
    def getPowerFactor(self):
        return np.cos(np.angle(self.correnteEntrada()))

    # Potência de Entrada
    def potenciaEntrada(self):

        return 3 * self.__getVoltage() * np.abs(np.abs(self.correnteEntrada())) * self.getPowerFactor()

    # Calcula as perdas no cobre
    def perdasCobreEstator(self):
        return 3 * np.abs(self.correnteEntrada())**2 * np.real(self.__z1)

    # Calcula a Potência do Entreferro
    def potenciaEntreFerro(self):
        return self.potenciaEntrada() - self.perdasCobreEstator() - self.__perdasNucleo

    # Calcula a Potência convertida
    def potenciaConvertida(self):
        return (1 - self.__getEscorregamento()) * self.potenciaEntreFerro()

    def potenciaSaida(self):
        return self.potenciaConvertida() - self.__perdasAtritVent - self.__perdasSuplementares

    # Caucula a Eficiencia do Motor
    def eficienciaMotor(self):
        return (self.potenciaSaida()/self.potenciaEntrada()) * 100

    # Conjugado Induzido

    # def conjugadoInduzido(self):
    #     return self.potenciaConvertida()/self.WMec(rad=True)

    # Conjugado de Carga
    def conjugadoCarga(self):
        return self.potenciaSaida()/self.WMec(rad=True)

    # Função privada para calcular o Torque
    def __torque(self, s):
        Vth = np.abs(self.getVThevenin())
        Zth = self.getZThevenin()

        newR2 = self.__r2/s
        newZ2 = newR2 + np.imag(self.__z2)
        Z = ((newR2 + np.real(Zth))**2 + (np.imag(self.__z2) + np.imag(Zth))**2)

        return (3 * Vth ** 2 * newR2) / (self.WSinc(rad=True) * Z)

    # Calcula o torque induzido
    def torqueInduzido(self):
        return self.__torque(self.__getEscorregamento())

    # Calcula o valor de S para o Torque Máximo
    def getSMaxTorque(self):
        Zth = self.getZThevenin()
        Rth = np.real(Zth)
        Xth = np.imag(Zth)
        X2 = np.imag(self.__z2)
        return (self.__r2)/np.sqrt((Rth)**2 + (Xth + X2)**2)

    # Calcula o torque Máximo
    def torqueMaximo(self):
        return self.__torque(self.getSMaxTorque())

    # Função privada para plotar
    def __plot(self, x, y, title, xlabel, ylabel, color, save=None):

        plt.style.use("seaborn")

        fig, ax = plt.subplots()

        ax.plot(x, y, c=color)
        ax.set_xlim(left=0, right=self.WSinc() + 1)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        if(save != None):
            plt.savefig(save)

        return ax

    # Função privada para calcular as velocidades mecânica para plotar

    def __getVelocidadesMecanica(self, rad=False):
        velocidades = []
        for s in np.arange(0.0001, 1, 0.001):
            velocidades.append((1-s) * self.WSinc(rad))

        return np.array(velocidades)

    # Função privada para calcular os torques para plotar
    def __getTorques(self):
        torques = []
        for s in np.arange(0.0001, 1, 0.001):
            torques.append(self.__torque(s))

        return np.array(torques)

    # Função privada para calcular as potências de saída para plotar
    def __getPotenciasSaida(self):
        return self.__getVelocidadesMecanica(True) * self.__getTorques()

    # Plota o gráfico de torque pela velocidade mecânica
    def plotTorque(self, save=None):

        self.__plot(self.__getVelocidadesMecanica(), self.__getTorques(), "Conjugado Induzido",
                    r"$\omega_{mec}$ (rpm)", r"$\tau_{ind}$ (N.m)", "dodgerblue", save)

        plt.show()

    # Plota o gráfico da Potencia de saída pela velocidade mecânica

    def plotPotenciaSaida(self, save=None):

        self.__plot(self.__getVelocidadesMecanica(), self.__getPotenciasSaida(), "Potência de Saída",
                    r"$\omega_{mec}$ (rpm)", r"$P_{saída}$ (kW)", "mediumturquoise", save)

        plt.show()

    # Função para plotar os 2 gráficos
    def plot(self):

        self.__plot(self.__getVelocidadesMecanica(), self.__getTorques(), "Conjugado Induzido",
                    r"$\omega_{mec}$ (rpm)", r"$\tau_{ind}$ (N.m)", "dodgerblue")

        self.__plot(self.__getVelocidadesMecanica(), self.__getPotenciasSaida(), "Potência de Saída",
                    r"$\omega_{mec}$ (rpm)", r"$P_{saída}$ (kW)", "mediumturquoise")

        plt.tight_layout()
        plt.show()

    def table(self):

        corrente = np.abs(self.correnteEntrada())
        pin = self.potenciaEntrada()/1000
        pce = self.perdasCobreEstator()/1000
        pef = self.potenciaEntreFerro()/1000
        pconv = self.potenciaConvertida()/1000
        pout = self.potenciaSaida()/1000
        conjugadoCarga = self.conjugadoCarga()
        eficiencia = self.eficienciaMotor()

        motor = {"Caracteristicas do Motor": {"Ws [rpm]": self.WSinc(),
                                              "Wm [rpm]": self.WMec(),
                                              "Corrente [A]": corrente,
                                              "FP": self.getPowerFactor(),
                                              "Pin [kW]": pin,
                                              "Pce [kW]": pce,
                                              "Pef [kW]": pef,
                                              "Pconv [kW]": pconv,
                                              "Pout [kW]": pout,
                                              "Conjugado de Carga [N.m]": conjugadoCarga,
                                              "Eficiência [%]": eficiencia}}

        return pd.DataFrame(motor).round(3)
