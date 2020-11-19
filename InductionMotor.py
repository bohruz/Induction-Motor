import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cmath import polar


class InductionMotor():
    """Induction Motor Class"""

    def __init__(self, v, s, r1, x1, r2, x2, xm, potenciaPlenaCarga=25000, rc=0, polos=4, frequency=60, perdasNucleo=0, perdasAtritVent=0, perdasSuplementares=0):
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
        self.__potenciaPlenaCarga = potenciaPlenaCarga
        self.__perdasNucleo = perdasNucleo
        self.__perdasAtritVent = perdasAtritVent
        self.__perdasSuplementares = perdasSuplementares

    #################### Private Methods #########################
    # Função para calcular a impedância em paralelo
    def __getParallel(self, z1, z2):
        return (z1 * z2)/(z1 + z2)

    # Velocidade mecânica
    def __WMec(self, s, rad=False):
        return (1 - s) * self.WSinc(rad)

    # Impedância Equivalente
    def __Zeq(self, s):
        Z2 = self.__r2/s + 1j * np.imag(self.__z2)
        if(self.__rc == 0):
            return self.__getParallel(self.__xm, Z2) + self.__z1
        else:
            xmrc = self.__getParallel(self.__rc, self.__xm)
            return self.__getParallel(xmrc, Z2) + self.__z1

    # Corrente de Linha
    def __correnteEntrada(self, s):
        return self.__getVoltage()/self.__Zeq(s)

    # Fator de potência
    def __powerFactor(self, s):
        return np.cos(np.angle(self.__correnteEntrada(s)))

    # Potência de Entrada
    def __potenciaEntrada(self, s):
        return (3 * self.__getVoltage() * np.abs(self.__correnteEntrada(s)) * self.__powerFactor(s))

    # Perdas cobre Estator
    def __perdasCobreEstator(self, s):
        return 3 * np.abs(self.__correnteEntrada(s))**2 * np.real(self.__z1)

    # Potência EntreFerro
    def __potenciaEntreFerro(self, s):
        return self.__potenciaEntrada(s) - self.__perdasCobreEstator(s) - self.__perdasNucleo

    # Potência Convertida
    def __potenciaConvertida(self, s):
        return (1 - s) * self.__potenciaEntreFerro(s)

    # Potência Saída
    def __potenciaSaida(self, s):
        return self.__potenciaConvertida(s) - self.__perdasAtritVent - self.__perdasSuplementares

    # Caucula a Eficiencia
    def __eficienciaMotor(self, s):
        return (self.__potenciaSaida(s)/self.__potenciaEntrada(s)) * 100

    # Função para calcular o Torque
    def __torque(self, s):
        Vth = np.abs(self.VThevenin())
        Zth = self.Zthevenin()

        newR2 = self.__r2/s
        newZ2 = newR2 + np.imag(self.__z2)
        Z = ((newR2 + np.real(Zth))**2 + (np.imag(self.__z2) + np.imag(Zth))**2)

        return (3 * Vth ** 2 * newR2) / (self.WSinc(rad=True) * Z)

     # Função para calcular as velocidades mecânica para plotar
    def __getVelocidadesMecanica(self, rad=False):
        velocidades = []
        for s in np.arange(0.0001, 1, 0.001):
            velocidades.append((1-s) * self.WSinc(rad))

        return np.array(velocidades)

    # Função para calcular os torques para plotar
    def __getTorques(self):
        torques = []
        for s in np.arange(0.0001, 1, 0.001):
            torques.append(self.__torque(s))

        return np.array(torques)

    # Função para calcular as potências de saída para plotar
    def __getPotenciasSaida(self):
        return (self.__getVelocidadesMecanica(rad=True) * self.__getTorques())/1000

    # Função para calcular as potências de entrada para plotar
    def __getPotenciasEntrada(self):
        potencias = []
        for s in np.arange(0.0001, 1, 0.001):
            potencias.append(self.__potenciaEntrada(s)/1000)

        return np.array(potencias)

    # Função para calcular as enficiências e potencias de saida para plotar
    def __getPotenciasSaidaEficiencias(self, nm_0, nm_1):
        velocidades = np.arange(nm_0, nm_1, -0.05)
        potenciasSaida = []
        eficiencias = []

        for velocidade in velocidades:
            s = (self.WSinc() - velocidade)/self.WSinc()

            potenciasSaida.append(self.__potenciaSaida(s)/1000)
            eficiencias.append(self.__eficienciaMotor(s))

        return potenciasSaida, eficiencias

    def __findVelocidade(self, potencia):
        error = 10.
        for s in np.arange(1e-4, 1, 1e-5):
            Pout = np.round(self.__potenciaSaida(s))
            if(Pout >= potencia - error and Pout <= potencia + error):
                return self.__WMec(s)

    # Função para plotar gráficos

    def __plot(self, x, y, title, xlabel, ylabel, color, nameToSave=None):

        plt.style.use("seaborn")

        fig, ax = plt.subplots()

        ax.plot(x, y, c=color)
        ax.set_xlim(left=0, right=np.max(x))
        ax.set_ylim(bottom=np.min(y))
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        if(nameToSave != None):
            plt.savefig(f"{nameToSave}.png")

        return ax
    ##############################################################
    ################# getters and setters properties #############

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

    # Calcula a Frenquência do Rotor

    def RotorFrequency(self):
        return self.__getEscorregamento() * self.__getFrequency()

    # Velocidade sincrona
    def WSinc(self, rad=False):
        return (4 * np.pi * self.__getFrequency())/self.__getNumPolos() \
            if rad else (120 * self.__getFrequency())/self.__getNumPolos()

    # Velocidade mecânica
    def WMec(self, rad=False):
        return self.__WMec(self.__getEscorregamento())

    def velocidadeVazio(self):
        return self.__findVelocidade(self.__perdasAtritVent)

    def velocidadePlenaCarga(self):
        return self.__findVelocidade(self.__potenciaPlenaCarga)

    # Velocidade mecânica para o torque máximo
    def WMecMaxTorque(self, rad=False):
        return (1 - self.SMaxTorque()) * self.WSinc(rad)

    def Zeq(self):
        return self.__Zeq(self.__getEscorregamento())

    # Impedância Equivalente de Thevenin
    def Zthevenin(self):

        if(self.__rc == 0):
            return self.__getParallel(self.__z1, self.__xm)

        else:
            xmrc = self.__getParallel(self.__rc, self.__xm)

            return self.__getParallel(self.__z1, xmrc)

    # Tensão de Thevenin
    def VThevenin(self):
        return self.__getVoltage() * (self.__xm/(self.__xm + self.__z1))

    # Corrente do Terminal
    def correnteEntrada(self):
        return self.__correnteEntrada(self.__getEscorregamento())

    # Fator de Potência
    def PowerFactor(self):
        return self.__powerFactor(self.__getEscorregamento())

    # Potência de entrada
    def potenciaEntrada(self):
        return self.__potenciaEntrada(self.__getEscorregamento())

    # Calcula as perdas no cobre
    def perdasCobreEstator(self):
        return self.__perdasCobreEstator(self.__getEscorregamento())

    # Calcula a Potência do Entreferro
    def potenciaEntreFerro(self):
        return self.__potenciaEntreFerro(self.__getEscorregamento())

    # Calcula a Potência convertida
    def potenciaConvertida(self):
        return self.__potenciaConvertida(self.__getEscorregamento())

    def potenciaSaida(self):
        return self.__potenciaSaida(self.__getEscorregamento())

    # Caucula a Eficiencia do Motor
    def eficienciaMotor(self):
        return self.__eficienciaMotor(self.__getEscorregamento())

    # Conjugado de Carga
    def conjugadoCarga(self):

        return self.potenciaSaida()/self.WMec(rad=True)

    # Calcula o torque induzido

    def torqueInduzido(self):
        return self.__torque(self.__getEscorregamento())

    # Calcula o valor de S para o Torque Máximo
    def SMaxTorque(self):
        Zth = self.Zthevenin()
        Rth = np.real(Zth)
        Xth = np.imag(Zth)
        X2 = np.imag(self.__z2)
        return (self.__r2)/np.sqrt((Rth)**2 + (Xth + X2)**2)

    # Calcula o torque Máximo
    def torqueMaximo(self):
        return self.__torque(self.SMaxTorque())

    # Plota o gráfico de torque pela velocidade mecânica

    def plotEficiencia(self, nameToSave=None):
        self.__plot(*self.__getPotenciasSaidaEficiencias(self.velocidadeVazio(),
                                                         self.velocidadePlenaCarga()), "Eficiencia",
                    r"$P_{saída}$ (kW)", r"$\eta$ (%)", "deepskyblue", nameToSave)

        plt.show()

    def plotTorque(self, nameToSave=None):

        self.__plot(self.__getVelocidadesMecanica(), self.__getTorques(), "Conjugado Induzido",
                    r"$\omega_{mec}$ (rpm)", r"$\tau_{ind}$ (N.m)", "dodgerblue", nameToSave)

        plt.show()

    # Plota o gráfico da Potencia de saída pela velocidade mecânica

    def plotPotenciaSaida(self, nameToSave=None):

        self.__plot(self.__getVelocidadesMecanica(), self.__getPotenciasSaida(), "Potência de Saída",
                    r"$\omega_{mec}$ (rpm)", r"$P_{saída}$ (kW)", "mediumturquoise", nameToSave)

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
                                              "FP": self.PowerFactor(),
                                              "Pin [kW]": pin,
                                              "Pce [kW]": pce,
                                              "Pef [kW]": pef,
                                              "Pconv [kW]": pconv,
                                              "Pout [kW]": pout,
                                              "Conjugado de Carga [N.m]": conjugadoCarga,
                                              "Eficiência [%]": eficiencia}}

        return pd.DataFrame(motor).round(3)
