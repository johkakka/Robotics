import matplotlib.pyplot as plt
import matplotlib.animation as anm


class World:
    def __init__(self, debug=False):
        self.objects = []
        self.debug = debug

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        elems = []

        if self.debug:
            for i in range(1000):
                self.step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.step, fargs=(elems, ax),
                                         frames=10, interval=100, repeat=False)
            plt.show()

    def step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        elems.append(ax.text(-4.4, 4.5, "t = " + str(i), fontsize=10))