import numpy as np
import matplotlib.pyplot as plt


class InductionMotor():
    def __init__(self, v, s, r1, x1, r2, x2, xm, rc=0, polos=4, frequency=60):
        self.slip = s
        self.r2 = r2
        self.v = v/np.sqrt(3)
        self.z1 = r1 + 1j * x1
        self.z2 = (r2/self.slip) + 1j * x2
        self.xm = 1j * xm
        self.rc = rc
        self.polos = polos
        self.frequency = frequency

    def __getParallel(self, z1, z2):
        return (z1 * z2)/(z1 + z2)

    def WSinc(self, rad=False):
        return (4 * np.pi * self.frequency)/self.polos if rad else (120 * self.frequency)/self.polos

    def WMec(self, rad=False):
        return (1 - self.slip) * self.WSinc(rad)

    def WMecMaxTorque(self, rad=False):
        return (1 - self.getSMaxTorque()) * self.WSinc(rad)

    def getZeq(self):

        if(self.rc == 0):
            return self.__getParallel(self.xm, self.z2) + self.z1
        else:
            xmrc = self.__getParallel(self.rc, self.xm)
            return self.__getParallel(xmrc, self.z2) + self.z1

    def getCurrent(self):
        return self.v/self.getZeq()

    def getPowerFactor(self):
        return np.cos(np.angle(self.getCurrent()))

    def getPowerIn(self):

        return 3 * self.v * np.abs(np.abs(self.getCurrent())) * self.getPowerFactor()

    def getZThevenin(self):

        return self.__getParallel(self.z1, self.xm)

    def getVThevenin(self):
        return self.v * (self.xm/(self.xm + self.z1))

    def __torque(self, s):
        Vth = np.abs(self.getVThevenin())
        Zth = self.getZThevenin()

        newR2 = self.r2/s
        newZ2 = newR2 + np.imag(self.z2)
        Z = ((newR2 + np.real(Zth))**2 + (np.imag(self.z2) + np.imag(Zth))**2)

        return (3 * Vth ** 2 * newR2) / (self.WSinc(rad=True) * Z)

    def getTorqueInduzido(self):
        return self.__torque(self.slip)

    def getSMaxTorque(self):
        Zth = self.getZThevenin()
        Rth = np.real(Zth)
        Xth = np.imag(Zth)
        X2 = np.imag(self.z2)
        return (self.r2)/np.sqrt((Rth)**2 + (Xth + X2)**2)

    def getMaxTorque(self):
        return self.__torque(self.getSMaxTorque())

    def setVoltage(self, v):
        self.v = v

    def getVoltage(self):
        return self.v

    def setSlip(self, slip):
        self.slip = slip

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
        plt.ylabel(r"$\tau$ (N.m)", fontsize=14)
        plt.show()
