import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse


def read_results(filename):
    """读取CSV文件中的模拟结果"""
    b_values = []
    er_values = []
    sf_values = []

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行

        for row in reader:
            b_values.append(float(row[0]))
            er_values.append(float(row[1]))
            sf_values.append(float(row[2]))

    return np.array(b_values), np.array(er_values), np.array(sf_values)


def plot_results(b_values, er_values, sf_values, output_file=None, title=None):
    """绘制符合期刊论文风格的模拟结果图"""
    # 设置期刊风格参数
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 1.0
    
    # 定义专业配色方案
    colors = {
        "er": "#1f77b4",       # 深蓝色
        "scale_free": "#2ca02c"  # 深绿色
    }
    
    # 绘制数据曲线（强化标记与线条）
    plt.plot(
        b_values, er_values, "o-", 
        label="ER Network",
        color=colors["er"],
        linewidth=2.0,
        markersize=8,
        markeredgecolor="white",  # 白色描边增强对比度
        markeredgewidth=0.5,
        markerfacecolor=colors["er"]
    )
    
    plt.plot(
        b_values, sf_values, "s-", 
        label="Scale-Free Network",
        color=colors["scale_free"],
        linewidth=2.0,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markerfacecolor=colors["scale_free"]
    )
    
    # 优化坐标轴与标签
    plt.xlabel("value of $b$", fontsize=12)
    #plt.xlabel("value of $r$", fontsize=12)  #SG博弈
    plt.ylabel("cooperation frequency", fontsize=12)
    
    if title:
        plt.title(title, fontsize=14, pad=10)
    else:
        plt.title("Simulation Results", fontsize=14, pad=10)
    
    # 图例设置
    plt.legend(fontsize=10, frameon=False)
    
    # 背景网格优化
    plt.grid(True, which='major', linestyle='-', color='#f0f0f0', linewidth=1.0)  # 主要网格线
    plt.grid(True, which='minor', linestyle=':', color='#f0f0f0', alpha=1.0)      # 次要网格线
    
    # 次要刻度设置
    plt.minorticks_on()
    
    # 去除多余边框，强化坐标轴
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.gca().spines[["bottom", "left"]].set_linewidth(1.2)
    
    # 增强刻度可读性
    plt.xticks(rotation=0, ha="center", fontsize=10)
    
    # 保存高清图
    if output_file:
        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=600  # 提高分辨率确保印刷清晰
        )
        print(f"Figure saved to {output_file}")

    


def main():
    parser = argparse.ArgumentParser(description="绘制演化博弈模拟结果")
    parser.add_argument(
        "--input", "-i", default="sg_results.csv", help="输入CSV文件路径"
    )
    parser.add_argument("--output", "-o", default=None, help="输出图像文件路径")
    parser.add_argument("--title", "-t", default=None, help="图表标题")

    args = parser.parse_args()

    # 读取数据
    b_values, er_values, sf_values = read_results(args.input)
    if args.title == None:
        args.title = args.input
    if args.output == None:
        args.output = args.input.replace(".csv", ".svg")
    # 绘制图表
    plot_results(b_values, er_values, sf_values, args.output, args.title)


if __name__ == "__main__":
    main()
