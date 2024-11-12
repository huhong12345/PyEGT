# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-
# %%
import networkx as nx
import numpy as np
from evolution import Evolution, CoEvolution, StaticStrategyEvolution
from population import Population, DynamicPopulation
import game
import rule
import adapter
from simulation import Simulation

# 生成网络
# G = ba(N, 5, 3, 1)
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(4, 1000)
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(100, 100, periodic=True))
# G = nx.star_graph(10)
# G = nx.watts_strogatz_graph(1000, 5, 0.2)
G = nx.barabasi_albert_graph(1000, 3)
# G = nx.powerlaw_cluster_graph(1000, 10, 0.2)
# G = nx.convert_node_labels_to_integers(nx.davis_southern_women_graph())
# G = ["/../../DataSet/ASU/Douban-dataset/data/edges.csv", ',']
# G = {"path":"/../wechat/barabasi_albert_graph(5000,100)_adj.txt", "fmt":"adj"}
# G = "/../wechat/facebook.txt"

# 网络结构
p = Population(G)
#print(nx.info(p))
#p.degree_distribution()
# 获取基本信息
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
is_directed = G.is_directed()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Is directed: {is_directed}")

# # 博弈类型
#    # 自定义收益矩阵（4x4）
# custom_payoff_matrix = [
#         [(3, 3), (0, 5), (1, 2), (2, 1)],
#         [(5, 0), (1, 1), (2, 3), (0, 4)],
#         [(2, 1), (3, 2), (4, 4), (1, 0)],
#         [(1, 2), (4, 0), (0, 3), (2, 2)]
#     ]
#
# g = game.CustomGame(payoff_matrix=custom_payoff_matrix).bind(p)
#
#     # 进行博弈
# g.play()


#g = game.PDG(b=5)
# g = game.PGG(3)
# g = game.PGG2(3)

# 学习策略
u = rule.BirthDeath()
# u = rule.DeathBirth()
# u = rule.Fermi()
# u = rule.HeteroFermi(g.delta)

# 连接策略
a = adapter.Preference(3)

# 重复实验
sim = Simulation()
sim.repeat_k()

# %%
# lattice()
# cora()
#once()
# once_co()
# repeat2d()
#repeat_k()
#repeat_b()
#repeat_start_pc()
#repeat_ss_rewire()
#repeat_ll_rewire()

# import cProfile
# import pstats

# cProfile.run("repeat_ss_rewire()", "timeit")
# p = pstats.Stats('timeit')
# p.sort_stats('time')
# p.print_stats(20)