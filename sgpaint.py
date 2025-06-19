import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse


def read_results(filename):
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
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(b_values, er_values, "o-", label="ER Network")
    plt.plot(b_values, sf_values, "s-", label="Scale-Free Network")

    plt.xlabel("value of b")
    #plt.xlabel("value of r")  #SG博弈
    plt.ylabel("cooperation frequency")

    if title:
        plt.title(title)
    else:
        plt.title("simulation result")

    plt.legend()
    plt.grid(True)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"figure saved to {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制演化博弈模拟结果")
    parser.add_argument(
        "--input", "-i", default="sg_results.csv", help="输入CSV文件路径"
    )
    parser.add_argument("--output", "-o", default=None, help="输出图像文件路径")
    parser.add_argument("--title", "-t", default=None, help="图表标题")

    args = parser.parse_args()

   
    b_values, er_values, sf_values = read_results(args.input)
    if args.title == None:
        args.title = args.input
    if args.output == None:
        args.output = args.input.replace(".csv", ".png")
  
    plot_results(b_values, er_values, sf_values, args.output, args.title)


if __name__ == "__main__":
    main()
