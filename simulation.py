import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from evolution import Evolution, CoEvolution, StaticStrategyEvolution
from population import Population, DynamicPopulation
import game
import rule
import adapter

# 绘图参数
colors = 'bgrcmykw'
markers = '.,ov^v<>1234sp*hH+xDd|-'
lines = ['-', '--', '-.', ':']
fmt = ['bd-', 'ro--', 'g^-.', 'c+:', 'mx--', 'y*-.']

class Simulation:
    def __init__(self):
        self.G = nx.barabasi_albert_graph(1000, 3)  # 生成网络
        self.p = Population(self.G)  # 网络结构
        self.g = game.PDG(b=5)  # 博弈类型
        self.u = rule.DeathBirth()  # 学习策略
        self.a = adapter.Preference(3)  # 连接策略

    def once(self):
        e = Evolution(has_mut=True)
        e.set_population(self.p).set_game(self.g).set_rule(self.u)
        e.evolve(1000)
        e.show()

    def lattice(self):
        l = 100
        def observe(s, f):
            plt.imshow(s.reshape((l, l)), interpolation='sinc', cmap='bwr')
            plt.show()
        G = nx.grid_2d_graph(l, l)
        p = Population(G)
        e = Evolution()
        e.set_population(p).set_game(self.g).set_rule(self.u)
        observe(p.strategy, p.fitness)

    def cora(self):
        G = nx.barabasi_albert_graph(1000, 3)
        p = Population(G)
        e = Evolution()
        g = game.PDG(2)
        e.set_population(p).set_game(g)
        f, axs = plt.subplots(3, 4)
        axs = axs.reshape(12)
        for r in range(11):
            if r > 5:
                r_ = 10 - r
                p.strategy = np.zeros(len(p), dtype=np.int)
                s = 1
            else:
                r_ = r
                p.strategy = np.ones(len(p), dtype=np.int)
                s = 0
            n = int(round(len(p) / 10.0 * r_))
            selected_list = np.random.choice(range(len(p)), n)
            p.strategy[selected_list] = s
            g.strategy = p.strategy
            g.play()
            p.show_degree(axs[r])
        plt.show()

    def once_co(self):
        dp = DynamicPopulation(self.G)
        c = CoEvolution(has_mut=False)
        c.set_population(dp).set_game(self.g).set_rule(self.u)
        c.set_adapter(self.a)
        c.evolve(50000)
        c.show()

    def repeat2d(self):
        e = Evolution()
        bs = np.linspace(1, 10, 3)
        colors = 'brgcmykwa'
        symbs = '.ox+*sdph-'
        for i in range(1, 10):
            G = nx.random_regular_graph(i + 1, 1000)
            p = Population(G)
            a = [0] * len(bs)
            for j, b in enumerate(bs):
                g = game.PDG(b)
                e.set_population(p).set_game(g).set_rule(self.u)
                e.evolve(10000)
                a[j] = e.cooperate[-1]
                plt.plot(bs, a, colors[j] + symbs[j], label='b=%f' % b)
            break
        plt.show()

    def repeat_k(self):
        e = Evolution(has_mut=False)
        k = 2
        for i in range(k):
            G = nx.random_regular_graph(i * 2 + 2, 1000)
            p = Population(G)
            e.set_population(p).set_game(self.g).set_rule(self.u)
            print('Control Variable k: %d' % (i * 2 + 2))
            e.evolve(100000, restart=True, quiet=True)
            e.show(label="k=%d" % (i * 2 + 2))
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig('outputpk.png')

    def repeat_b(self):
        e = Evolution(has_mut=False)
        G = nx.random_regular_graph(4, 1000)
        p = Population(G)
        b = 5
        for i in range(b):
            g = game.PDG(i * 2 + 2)
            e.set_population(p).set_game(g).set_rule(self.u)
            print('Control Variable b: %d' % (i * 2 + 2))
            e.evolve(100000, restart=True, quiet=True, autostop=False)
            e.show(label="b=%d" % (i * 2 + 2))
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig('outputb.png')

    def repeat_start_pc(self):
        G = nx.barabasi_albert_graph(1000, 3)
        p = Population(G)
        g = game.PDG(b=10)
        e = Evolution(has_mut=False)
        e.set_population(p).set_game(g).set_rule(self.u)
        for i in range(5):
            pc = (2 * i + 1) / 10.0
            p.init_strategies(g, [pc, 1 - pc])
            print('Initial P(C) is %.2f' % pc)
            e.evolve(100000, restart=True, quiet=True, autostop=False)
            e.show(label=r'start $P_C$=%.2f' % pc)
        plt.legend(loc='lower right')
        plt.title(r'Evolution under Different Start $P_C$')
        plt.xlabel('Number of generations')
        plt.ylabel(r'Fraction of cooperations, $\rho_c$')
        plt.show()
        plt.savefig('outputpc.png')

    def repeat_ss_rewire(self):
        from network import LatticeWithLongTie
        p = LatticeWithLongTie(30)
        g = game.PDG(b=8)
        u = rule.Fermi(0.1)
        a = adapter.Preference(3)
        e = StaticStrategyEvolution()
        e.set_game(g).set_rule(u).set_adapter(a)
        e.set_population(p)
        e.bind_process()
        p.init_longtie()
        p_copy1 = p.copy()
        plt.figure()
        for i in range(6):
            pc = i / 5.0
            p_copy = p.copy()
            e.set_population(p)
            a.dynamic = p.dynamics
            a.bind(p)
            p.init_strategies(g, [pc, 1 - pc])
            p = p_copy
            print('static P(C) is %.2f' % pc)
            e.evolve(100, restart=True, quiet=True, autostop=False)
            e.show(label=r'static $P_C$=%.2f ' % pc)
        plt.legend(loc='upper left')
        plt.title('Static Strategy Evolution')
        plt.xlabel('Number of generations')
        plt.ylabel('Rewire Strategies')
        plt.ylim(0, len(p))
        plt.show()

    def repeat_ll_rewire(self):
        from network import LatticeWithLongTie
        p = LatticeWithLongTie(30)
        g = game.PDG(b=8)
        u = rule.DeathBirth()
        a = adapter.Preference(3)
        e = CoEvolution()
        e.set_game(g).set_rule(u).set_adapter(a)
        e.set_population(p)
        e.bind_process()
        p.degree_distribution()
        print('start evolving')
        e.evolve(10000, restart=True, quiet=True, autostop=False)
        plt.figure()
        e.show(label='LLN Co-Evolution')
        plt.legend(loc='upper left')
        plt.title('LLN Co-Evolution')
        plt.xlabel('Number of generations')
        plt.ylabel('Rewire Strategies')
        plt.show()
        p.degree_distribution()

if __name__ == "__main__":
    sim = Simulation()
    sim.repeat_b()
