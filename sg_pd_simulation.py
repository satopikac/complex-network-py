# 引入一些关键的库
# 网络构建 数学运算 图像绘制
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 加速运算
from numba import njit

import csv
from typing import List, Tuple
import time

import argparse


def adj_list_to_adj_array(adj_list: List[List[int]]):
    """把create_network函数里创建的网络邻接表转化为numpy可以处理的形式

    Args:
        adj_list (List[List[int]]): 邻接表
    """
    n = len(adj_list)
    # 计算出每个节点的邻居数,degrees是每个节点的邻居数
    degrees = np.array([len(neighbors) for neighbors in adj_list], dtype=np.int32)

    max_degrees = np.max(degrees)
    neighbors_array = np.full((n, max_degrees), -1, dtype=np.int32)
    # 创建邻居表（二维数组） 全部填充-1 表示暂时无连接

    for i in range(n):
        neighbors = adj_list[i]
        # 获取到邻居
        for j, neighbor in enumerate(neighbors):
            neighbors_array[i, j] = neighbor
    # neighbors_array[i,j] 表示 节点i的第j个邻居是neighbor

    return neighbors_array, degrees


@njit(cache=False)
def update_strategies(neighbors, degrees, strategies, payoffs, b):
    """根据比例更新原则更新一轮博弈后各个节点的策略

    Args:
        neighbors (np.ndarray): 邻接表
        degrees (np.ndarray): 度数表
        strategies (np.ndarray): 每个节点当前策略
        payoffs (np.ndarray): 每个节点当前总收益
        b (float): b值（收益矩阵参数）
    """

    n = len(strategies)  # 节点数
    new_strategies = strategies.copy()

    for i in range(n):
        degree = degrees[i]
        if degree == 0:
            continue

        j_idx = np.random.randint(0, degree)  # 随机选一个邻居

        j = neighbors[i, j_idx]  # 这个邻居的序号j

        if j == -1:
            continue

        if payoffs[j] > payoffs[i]:

            D = b
            k_i = degree
            k_j = degrees[j]
            k = max(k_i, k_j)
            W = (payoffs[j] - payoffs[i]) / (D * k)
            # W为模仿概率

            if np.random.random() < W:
                new_strategies[i] = strategies[j]  # 模仿

    return new_strategies


@njit(cache=False)
def simulation(
    neighbors: np.ndarray,
    degrees: np.ndarray,
    b: float,
    generations: int,
    transient: int,
    game_type: str,
) -> float:
    """使用Numba加速整个模拟过程，支持PD和SG博弈"""
    n = len(degrees)
    strategies = np.random.randint(0, 2, n)  # 0=背叛者，1=合作者
    avg_cooperation = 0.0
    count = 0

    for g in range(generations):
        # 计算收益
        payoffs = np.zeros(n)
        for i in range(n):
            degree = degrees[i]
            for j_idx in range(degree):
                j = neighbors[i, j_idx]
                if j == -1 or j > i:
                    continue

                if game_type == "PD":
                    # 囚徒困境博弈(PD)
                    if strategies[i] == 1 and strategies[j] == 1:
                        payoff_i = payoff_j = 1.0
                    elif strategies[i] == 1 and strategies[j] == 0:
                        payoff_i = 0.0
                        payoff_j = b
                    elif strategies[i] == 0 and strategies[j] == 1:
                        payoff_i = b
                        payoff_j = 0.0
                    else:
                        payoff_i = payoff_j = 0.0
                else:  # SG
                    # 雪堆博弈(SG)
                    beta = (1 / b + 1) * 0.5
                    if strategies[i] == 1 and strategies[j] == 1:
                        payoff_i = payoff_j = beta - 0.5
                    elif strategies[i] == 1 and strategies[j] == 0:
                        payoff_i = beta - 1
                        payoff_j = beta
                    elif strategies[i] == 0 and strategies[j] == 1:
                        payoff_i = beta
                        payoff_j = beta - 1
                    else:
                        payoff_i = payoff_j = 0.0

                payoffs[i] += payoff_i
                payoffs[j] += payoff_j

        # 随机挑选邻居比较收益 进行策略更新
        strategies = update_strategies(neighbors, degrees, strategies, payoffs, b)

        if g >= transient:
            avg_cooperation += np.mean(strategies)
            count += 1

    return float(avg_cooperation / count) if count > 0 else 0.0


def create_network(n: int, k: int, network_type: str):
    """创建对应类型的复杂网络，转化为邻接表

    Args:
        n (int): 顶点数
        k (int): 平均度数
        network_type (str): 网络类型
    """
    # 创建对应类型的复杂网络
    # 包括ER随机网络和无标度网络

    if network_type == "er":
        # ER网络里任意两个顶点存在连边的概率
        # 为 k/n-1
        p = k / (n - 1)
        G = nx.erdos_renyi_graph(n, p, directed=False)  # 无向图
    else:
        G = nx.barabasi_albert_graph(n, k // 2)  # 向下取整

    adj_list = [list(G.neighbors(i)) for i in range(n)]
    # 转化为列表组成的列表

    # print(adj_list)
    # print(adj_list_to_adj_array(adj_list))
    return adj_list_to_adj_array(adj_list)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="演化博弈模拟程序")
    parser.add_argument(
        "--game",
        "-g",
        default="SG",
        choices=["PD", "SG"],
        help="博弈类型：PD(囚徒困境)或SG(雪堆博弈)，默认SG",
    )
    parser.add_argument(
        "--k", "-k", type=int, default=4, help="网络的平均度，默认值为4"
    )
    args = parser.parse_args()
    game_type = args.game

    # 模拟参数
    n = 400  # 节点数
    k = args.k  # 平均度
    if game_type == "SG":
        b_values = np.linspace(0.0, 1.0, 21)[1:]  # 调整PD的b范围
    else:
        b_values = np.linspace(1.0, 2.0, 21)[1:]
    generations = 5000
    transient = 3000
    repeats = 80
    network_types = ["er", "scale_free"]
    title = " (PD)" if game_type == "PD" else " (SG)"

    # 预热Numba
    # print(f"JIT编译中，运行{game_type}博弈模拟...")
    dummy_neighbors = np.array(
        [[1, 2, -1], [0, 3, -1], [0, 3, -1], [1, 2, -1]], dtype=np.int32
    )
    dummy_degrees = np.array([2, 2, 2, 2], dtype=np.int32)
    simulation(dummy_neighbors, dummy_degrees, 0.5, 10, 5, game_type)

    # 存储结果
    results = {nt: {b: 0.0 for b in b_values} for nt in network_types}

    # 执行模拟
    for nt in network_types:
        print(f"begin run simulation of  {nt} network...")
        for b in b_values:
            start_time = time.time()
            avg_coop = 0.0

            for _ in range(repeats):
                neighbors, degrees = create_network(n, k, nt)
                avg_coop += simulation(
                    neighbors, degrees, b, generations, transient, game_type
                )

            results[nt][b] = avg_coop / repeats
            elapsed = time.time() - start_time
            print(
                f"b={b:.2f}, frequency of cooperation={results[nt][b]:.4f}, time={elapsed:.2f} seconds"
            )
    # 保存结果到CSV
    csv_filename = f"{game_type.lower()}_results__n={n}_k={k}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["b", "ER", "Scale-Free"])
        for b in b_values:
            writer.writerow([b, results["er"][b], results["scale_free"][b]])

    # 绘制结果图
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(
        list(results["er"].keys()),
        list(results["er"].values()),
        "o-",
        label="ER Network",
    )
    plt.plot(
        list(results["scale_free"].keys()),
        list(results["scale_free"].values()),
        "s-",
        label="Scale-Free Network",
    )
    plt.xlabel("b values")
    plt.ylabel("cooperation frequency")
    plt.title(f"simulation result(z={k}) {title} n={n}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"cooperation_results_{game_type.lower()}__n={n}_k={k}.pdf")
    # plt.savefig(f'cooperation_results_{game_type.lower()}.pdf')


# plt.show()


if __name__ == "__main__":
    main()
