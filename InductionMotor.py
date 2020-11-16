import numpy as np
import matplotlib.pyplot as plt


class InductionMotor():
    """Induction Motor Class"""

    def __init__(self, v, s, r1, x1, r2, x2, xm, rc=0, polos=4, frequency=60):
        """v: Tensão de linha
           s: Escorregamento
           r1: Resistência do Estator
           x1: Indutância do Estator
           r2: Resistência do Rotor
           x2: Indutância do Rotor
           xm: Indutância de magnetização
           rc: Resistência do nucleo
           polos: numero de polos do motor - default 4
           frequency: Frequência da rede - default 4"""

        self.setEscorregamento(s)
        self.setVoltage(v)
        self.setNumPolos(polos)
        self.setFrequency(frequency)
        self.__r2 = r2
        self.__z1 = r1 + 1j * x1
        self.__z2 = (r2/self.__getEscorregamento()) + 1j * x2
        self.__xm = 1j * xm
        self.__rc = rc

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

    # Velocidade sincrona
    def WSinc(self, rad=False):
        return (4 * np.pi * self.__getFrequency())/self.__getNumPolos() if rad else (120 * self.__getFrequency())/self.__getNumPolos()

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
    def getCurrent(self):
        return self.__getVoltage()/self.getZeq()

    # Fator de potência
    def getPowerFactor(self):
        return np.cos(np.angle(self.getCurrent()))

    # Potência de Entrada
    def getPowerIn(self):

        return 3 * self.__getVoltage() * np.abs(np.abs(self.getCurrent())) * self.getPowerFactor()

    # Função privada para calcular o Torque
    def __torque(self, s):
        Vth = np.abs(self.getVThevenin())
        Zth = self.getZThevenin()

        newR2 = self.__r2/s
        newZ2 = newR2 + np.imag(self.__z2)
        Z = ((newR2 + np.real(Zth))**2 + (np.imag(self.__z2) + np.imag(Zth))**2)

        return (3 * Vth ** 2 * newR2) / (self.WSinc(rad=True) * Z)

    # Calcula o torque induzido
    def getTorqueInduzido(self):
        return self.__torque(self.__getEscorregamento())

    # Calcula o valor de S para o Torque Máximo
    def getSMaxTorque(self):
        Zth = self.getZThevenin()
        Rth = np.real(Zth)
        Xth = np.imag(Zth)
        X2 = np.imag(self.__z2)
        return (self.__r2)/np.sqrt((Rth)**2 + (Xth + X2)**2)

    # Calcula o torque Máximo
    def getMaxTorque(self):
        return self.__torque(self.getSMaxTorque())

    # Plota o gráfico de torque pela velocidade mecânica
    def plotTorque(self):
        torqueVector = []
        velocity = []
        for s in np.arange(0.0001, 1, 0.001):
            torqueVector.append(self.__torque(s))
            velocity.append((1-s) * self.WSinc())

        plt.plot(velocity, torqueVector, c="dodgerblue")
        plt.xlim(left=0, right=self.WSinc() + 1)
        plt.ylim(bottom=0)
        plt.title("Conjugado Induzido", fontsize=20)
        plt.xlabel(r"$\omega_{mec}$ (rpm)", fontsize=14)
        plt.ylabel(r"$\tau_{ind}$ (N.m)", fontsize=14)
        plt.show()
