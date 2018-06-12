import numpy as np
import matplotlib.pyplot as plt


class Runner(object):
    """
    Runs algorithms and reports their performance in terms of number of iterations until convergence.
    """

    def __init__(self, algs, env, run_count=20):
        self.algs = algs
        self.env = env
        self.run_count = run_count

    def run(self):
        template = "{0:50}|{1}"  # column widths: 8, 10, 15, 7, 10
        print(template.format("Shortest path", "Cost"))  # header
        report_data = np.zeros((len(self.algs), self.run_count))
        for i in range(self.run_count):
            start, goal = self.env.reset()
            try:
                j = 0
                for alg in self.algs:
                    try:
                        result = alg.find_shortest_path(start, goal, self.env)
                        report_data[j, i] = result[2]
                        j += 1
                    except Exception as e:
                        pass
            # Sometimes happens when no path between start and goal exists
            except Exception:
                continue

            result = list(result[:2])
            result[0] = "".join(str(result[0]))
            print(template.format(*result))
            if i % (self.run_count // 5) == 0:
                self.env.show()

        plt.xlabel("Gridworld samples")
        plt.ylabel("Number of iterations")
        for i in range(report_data.shape[0]):
            plt.plot(np.arange(self.run_count), report_data[i, :], label=self.algs[i].name)
        plt.legend()
        plt.show()
