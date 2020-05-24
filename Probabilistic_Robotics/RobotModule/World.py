import matplotlib.pyplot as plt
import matplotlib.animation as anm


class World:
    def __init__(self, time_span=10, time_interval=10, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.interval = time_interval

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
                                         frames=int(self.time_span/self.interval)+1, interval=self.interval*1000, repeat=False)
            plt.show()

    def step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        elems.append(ax.text(-4.4, 4.5, 't= {:.2f}[s]'.format(i*self.interval), fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "step"):
                obj.step(self.interval)