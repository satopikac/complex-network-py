# 模拟
针对北京大学工学院开设的群体智能课程，2025年春季学期。--QY Mei
本项目实现了复杂网络（包括ER随机网络和SF无标度网络）上博弈中合作的演化模拟，计算出稳定时针对不同的收益矩阵参数$b$合作所占比例。

## 环境要求

程序使用numba库加速计算

运行模拟程序，python版本$\ge$ 3.9，创建环境后运行

```bash
pip install -r requirements.txt
```

安装所需的依赖包。

依赖包要求版本

```
  "matplotlib>=3.9.4",
  "networkx>=3.2.1",
  "numba>=0.60.0",
  "numpy>=2.0.2",
```

## SG_PD

文件sg_pd_simulation.py给出了在两种网络下，SG、PD各自运行结果，

**运行 SG 博弈（默认）** ：

```bash
   python sg_pd_simulation.py
```

**运行 PD 博弈** ：

```bash
   python sg_pd_simulation.py --game PD
```

**指定平均度** ：

```bash
   python sg_pd_simulation.py --game PD --k 8
```

**查看帮助** ：

```bash
   python sg_pd_simulation.py --help
```

## SG

文件sg_simulation.py给出了网络上仅仅运行sg博弈的结果，仅针对$k=4$情形，根据要求可以修改。

```bash
python sg_simulation.py
```

# 绘图

生成的数据使用sgpaint.py绘制，可能需要修改数据来源的文件名。
```bash
python sgpaint.py -i input.csv -o output.png -t result-title
```
需要使用参数指定输入数据的位置，输出图像的名称以及图像中间的标题。如不指定后两者，默认使用输入数据的名称。
sgpaint-pretty.py的功能和用法和上面一样，调整了绘图风格更加美观。
图像生成效果如下

![1749100333617](/demo.svg)
