import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
from numba import njit
from typing import List, Tuple
import time

def adj_list_to_adj_array(adj_list: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """将邻接表转换为Numba支持的NumPy数组格式"""
    n = len(adj_list)
    
    # 计算每个节点的邻居数量
    degrees = np.array([len(neighbors) for neighbors in adj_list], dtype=np.int32)
    
    # 构建邻居数组和偏移量数组
    max_degree = np.max(degrees)
    neighbors_array = np.full((n, max_degree), -1, dtype=np.int32)  # -1表示无效邻居
    
    for i in range(n):
        neighbors = adj_list[i]
        for j, neighbor in enumerate(neighbors):
            neighbors_array[i, j] = neighbor
    
    return neighbors_array, degrees

@njit(cache=True)
def update_strategies_fast(neighbors: np.ndarray, degrees: np.ndarray, 
                           strategies: np.ndarray, payoffs: np.ndarray, b: float) -> np.ndarray:
    """使用Numba加速策略更新过程"""
    n = len(strategies)
    new_strategies = strategies.copy()
    
    for i in range(n):
        degree = degrees[i]
        if degree == 0:
            continue
            
        j_idx = np.random.randint(0, degree)
        j = neighbors[i, j_idx]
        
        if j == -1:  # 无效邻居
            continue
            
        if payoffs[j] > payoffs[i]:
            D = b
            k_i = degree
            k_j = degrees[j]
            k = max(k_i, k_j)
            W = (payoffs[j] - payoffs[i]) / (D * k)
            
            if np.random.random() < W:
                new_strategies[i] = strategies[j]
    
    return new_strategies

@njit(cache=True)
def run_simulation_fast(neighbors: np.ndarray, degrees: np.ndarray, b: float, 
                        generations: int, transient: int) -> float:
    """使用Numba加速整个模拟过程"""
    n = len(degrees)
    strategies = np.random.randint(0, 2, n)
    avg_cooperation = 0.0
    count = 0
    
    for g in range(generations):
        # 计算收益
        payoffs = np.zeros(n)
        for i in range(n):
            degree = degrees[i]
            for j_idx in range(degree):
                j = neighbors[i, j_idx]
                if j == -1 or j > i:  # 跳过无效邻居和重复计算
                    continue
                
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
                    payoff_i = payoff_j = 0
                
                payoffs[i] += payoff_i
                payoffs[j] += payoff_j
        
        # 更新策略
        strategies = update_strategies_fast(neighbors, degrees, strategies, payoffs, b)
        
        # 记录稳定期的合作水平
        if g >= transient:
            avg_cooperation += np.mean(strategies)
            count += 1
    
    return float(avg_cooperation / count) if count > 0 else 0.0

def create_network(n: int, k: int, network_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """创建网络并转换为Numba友好的邻接表格式"""
    if network_type == 'er':
        # 根据ER模型创建网络
        p = k / (n - 1)
        G = nx.erdos_renyi_graph(n, p, directed=False)
    else:  # scale-free
        G = nx.barabasi_albert_graph(n, k//2)
    
    # 转换为邻接表列表
    adj_list = [list(G.neighbors(i)) for i in range(n)]
    
    # 转换为NumPy数组
    return adj_list_to_adj_array(adj_list)

def main():
    # 模拟参数
    n = 400           # 节点数
    k = 4             # 平均度
    b_values = np.linspace(0.0, 1.0, 21)[1:]  # 跳过b=0
    generations = 5000
    transient = 3000
    repeats = 50
    network_types = ['er', 'scale_free']

    # 预热Numba
    print("JIT编译中...")
    dummy_neighbors = np.array([[1, 2, -1], [0, 3, -1], [0, 3, -1], [1, 2, -1]], dtype=np.int32)
    dummy_degrees = np.array([2, 2, 2, 2], dtype=np.int32)
    run_simulation_fast(dummy_neighbors, dummy_degrees, 0.5, 10, 5)
    
    # 存储结果
    results = {nt: {b: 0.0 for b in b_values} for nt in network_types}

    # 执行模拟
    for nt in network_types:
        print(f"模拟 {nt} 网络...")
        for b in b_values:
            start_time = time.time()
            avg_coop = 0.0
            
            for _ in range(repeats):
                neighbors, degrees = create_network(n, k, nt)
                avg_coop += run_simulation_fast(neighbors, degrees, b, generations, transient)
            
            results[nt][b] = avg_coop / repeats
            elapsed = time.time() - start_time
            print(f"b={b:.2f}, 合作水平={results[nt][b]:.4f}, 耗时={elapsed:.2f}秒")

    # 保存结果到CSV
    with open('sg_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['b', 'ER', 'Scale-Free'])
        for b in b_values:
            writer.writerow([b, results['er'][b], results['scale_free'][b]])

    # 绘制结果图
    plt.figure(figsize=(10, 6))
    plt.plot(list(results['er'].keys()), list(results['er'].values()), 'o-', label='ER Network')
    plt.plot(list(results['scale_free'].keys()), list(results['scale_free'].values()), 's-', label='Scale-Free Network')
    plt.xlabel('para b')
    plt.ylabel('cooperation')
    plt.title('simulation results(z=4)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cooperation_results.png')
    plt.show()

if __name__ == "__main__":
    main()