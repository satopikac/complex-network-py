# 模拟

程序使用numba库加速计算

运行模拟程序，python版本$\ge$ 3.9，创建环境后运行

```bash
pip install -r requirements.txt
```

安装所需的依赖包。

要求版本

```
  "matplotlib>=3.9.4",
    "networkx>=3.2.1",
    "numba>=0.60.0",
    "numpy>=2.0.2",
```

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
   python script.py --help
```

# 绘图

生成的数据使用sgpaint.py绘制，可能需要数据修改文件名。
