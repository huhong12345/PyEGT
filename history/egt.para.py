# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from evolution import Evolution, CoEvolution, StaticStrategyEvolution
from population import Population, DynamicPopulation
import game
import rule
import adapter
import dask
from dask import delayed, compute

# 生成网络
G = nx.barabasi_albert_graph(1000, 3)

# 网络结构
p = Population(G)

# 博弈类型
g = game.PDG(b=5)

# 学习策略
u = rule.DeathBirth()

# 连接策略
a = adapter.Preference(3)

# 绘图参数
colors = 'bgrcmykw'
markers = '.,ov^v<>1234sp*hH+xDd|-'
lines = ['-', '--', '-.', ':']
fmt = ['bd-', 'ro--', 'g^-.', 'c+:', 'mx--', 'y*-.']

def once():
    e = Evolution(has_mut=True)
    e.set_population(p).set_game(g).set_rule(u)
    e.evolve(1000)
    e.show()

def repeat_k():
    # 网络平均度不同，合作率曲线
    e = Evolution(has_mut=False)
    k = 5
    results = []

    @delayed
    def evolve_for_k(i):
        G = nx.random_regular_graph(i * 2 + 2, 1000)
        p = Population(G)
        e.set_population(p).set_game(g).set_rule(u)
        print('Control Variable k: %d' % (i * 2 + 2))
        e.evolve(100000, restart=True, quiet=True)
        return e.cooperate[-1]  # 返回合作率

    # 创建所有任务
    tasks = [evolve_for_k(i) for i in range(k)]
    
    # 计算所有任务
    results = compute(*tasks)

    for i, result in enumerate(results):
        e.show(fmt[i], label="k=%d" % (i * 2 + 2))

    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('outputk_para.png')

def repeat_b():
    # 博弈收益参数不同，合作率曲线
    e = Evolution(has_mut=False)
    G = nx.random_regular_graph(4, 1000)
    p = Population(G)
    b = 5
    results = []

    @delayed
    def evolve_for_b(i):
        g = game.PDG(i * 2 + 2)
        e.set_population(p).set_game(g).set_rule(u)
        print('Control Variable b: %d' % (i * 2 + 2))
        e.evolve(100000, restart=True, quiet=True, autostop=False)
        return e.cooperate[-1]  # 返回合作率

    # 创建所有任务
    tasks = [evolve_for_b(i) for i in range(b)]
    
    # 计算所有任务
    results = compute(*tasks)

    for i, result in enumerate(results):
        e.show(fmt[i], label="b=%d" % (i * 2 + 2))

    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('outputb_para.png')

# 调用函数
repeat_k()
repeat_b()
