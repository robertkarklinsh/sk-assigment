import numpy as np
import matplotlib.pyplot as plt
import sys


class Runner(object):
    """
    Runs algorithms and reports their performance in terms of number of iterations until convergence.
    """

    def __init__(self, algs, env, run_count=50, interactive=False):
        self.algs = algs
        self.env = env
        self.run_count = run_count
        self.interactive = interactive

    def run(self):

        template = "{0:50}|{1:10}|{2}"
        print(template.format("Shortest path", "Cost", "Number of iterations"))

        if self.interactive:

            try:
                while True:
                    start, _ = self.env.reset()
                    self.env.goal = None
                    self.env.show(block=False)
                    while True:
                        try:
                            user_input = input("Enter goal position as 'x_position y_position': ")
                            goal = [int(i) for i in user_input.split()]
                            goal = np.ravel_multi_index(goal, self.env.dims, order='F')
                            self.env.goal = goal
                        except Exception:
                            print("The entered value is invalid, try again")
                        else:
                            break
                    plt.close()

                    for alg in self.algs:
                        try:
                            result = alg.find_shortest_path(start, goal, self.env)
                            result = list(result)
                            result[0] = "".join(str(result[0]))
                            result[1] = "%.2f" % result[1]
                            result[2] = str(result[2]) + " (%s)" % alg.name
                            print(template.format(*result))
                        except Exception as e:
                            print(str(e))
                    self.env.show()

            except KeyboardInterrupt:
                sys.exit(0)

        report_data = np.zeros((len(self.algs), self.run_count), dtype=np.int16)
        for i in range(self.run_count):
            start, goal = self.env.reset()
            j = 0
            for alg in self.algs:
                try:
                    result = alg.find_shortest_path(start, goal, self.env)
                    report_data[j, i] = result[2]

                    result = list(result)
                    result[0] = "".join(str(result[0]))
                    result[1] = "%.2f" % result[1]
                    result[2] = str(result[2]) + " (%s)" % alg.name
                    print(template.format(*result))
                    j += 1
                except Exception as e:
                    print(str(e))
                    self.env.show()
            if i % (self.run_count // 5) == 0:
                self.env.show()

        plt.xlabel("Gridworld samples")
        plt.ylabel("Number of iterations")
        plt.xticks(np.linspace(1, self.run_count, int(self.run_count / 5), dtype=int))
        for i in range(report_data.shape[0]):
            plt.plot(np.arange(1, self.run_count + 1), report_data[i, :], label=self.algs[i].name)
        plt.legend()
        plt.show()
