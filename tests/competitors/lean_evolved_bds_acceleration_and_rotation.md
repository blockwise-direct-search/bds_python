# Lean Evolved BDS 的加速与抗旋转机制

本文档记录一个工作性理解：`Lean Evolved BDS` 为什么能明显加速原始 `BDS`，以及为什么在 `linearly_transformed / rotated` 问题上没有像普通坐标搜索那样明显崩掉。这里先不写成论文定稿，而是作为后续写入 `bds_cvx_general.tex` 的机制草稿。

核心判断可以概括为：

> Lean Evolved BDS converts successful coordinate-wise progress into reusable non-coordinate search directions.

也就是说，它不放弃 BDS 的 coordinate/block-coordinate polling 骨架，而是在一轮 sweep 之后，把已经观察到的有效位移整理成新的、可复用的非坐标方向。

## 1. 原始 BDS 的限制

原始 `BDS` 的主要搜索信息来自 coordinate directions。对于当前点 \(x_k\)，它围绕 coordinate blocks 尝试类似

```text
+e_i, -e_i
```

这样的方向。如果某个方向产生下降，BDS 会更新当前点，并通过 `cycle direction / cycle block` 让近期成功方向更早被尝试。

这个机制简单、便宜、可靠；当目标函数结构和坐标轴大致 aligned 时，它很有效。但它也天然 `coordinate-bound`：如果真实下降方向是 oblique direction，或者问题经过旋转后主要下降方向不再 aligned with coordinate axes，原始 BDS 往往只能用一串坐标步近似斜向运动，从而产生 zig-zag 和较慢进展。

## 2. 当前 Lean Evolved BDS 多了什么

`Lean Evolved BDS` 保留原始 BDS 的 coordinate/block-coordinate polling 和 ordinary cycling，但增加三个 lightweight mechanisms：

- `sweep-level pattern direction`
- `momentum extrapolation`
- `productive direction memory`

它们的共同点是：从坐标轮询已经产生的成功位移中提取 non-coordinate direction information，而不是让搜索方向永远停留在 \(\pm e_i\) 上。

可以把当前 solver 看成：

```text
coordinate polling + history-dependent aggregate directions
```

这里的 `history-dependent` 不是统计学习意义上的模型训练，而是 direct search 过程中从历史成功步里形成的经验方向。

## 3. 三个机制分别是什么

### 3.1 `sweep-level pattern direction`

一轮 coordinate sweep 从 \(x_k\) 出发，经过若干成功的 coordinate moves，最后到达 \(x_{k+1}\)。这轮 sweep 的总位移是

```text
s_k = x_{k+1} - x_k.
```

如果这一轮确实带来下降，Lean Evolved BDS 会把

```text
s_k / ||s_k||
```

看成一个新的 `pattern direction`。它不是某个单独坐标方向，而是多个成功 coordinate moves 的合成方向。

它回答的问题是：

> What aggregate direction did this successful sweep reveal?

### 3.2 `momentum extrapolation`

`momentum extrapolation` 记录最近成功 sweep directions 的平滑趋势。一个直观形式是

```text
m_k = beta m_{k-1} + (1 - beta) normalized(s_k).
```

这里的 `momentum` 不等同于 gradient method 里的动量；这里没有梯度，也没有精确 line search。它更像 derivative-free 的 `sweep-direction smoothing`：如果连续几轮成功 sweep 的方向相近，就沿这个平滑后的趋势方向做额外试探。

它回答的问题是：

> Do recent successful sweeps share a persistent trend?

### 3.3 `productive direction memory`

`productive direction memory` 是一个短期 memory list。只要某个非坐标方向带来下降，算法就把这个方向保存下来。后续 sweep 之前，算法会优先尝试这些历史上 productive 的 directions。

这和原始 BDS 的 `cycle direction / cycle block` 不同：

- `cycle direction / cycle block` 主要是在已有 coordinate directions 里调整尝试顺序；
- `productive direction memory` 保存的是由实际成功位移发现的 oblique directions。

简洁地说：

> Cycling reorders coordinate directions; productive memory reuses discovered oblique directions.

它回答的问题是：

> Can a previously useful oblique direction remain useful nearby?

## 4. 为什么会加速

原始 BDS 中，一轮 sweep 的总位移 \(s_k\) 只是“结果”；下一轮算法仍主要回到坐标方向上继续轮询。

Lean Evolved BDS 的关键变化是：如果一轮 sweep 成功下降，那么 \(s_k\) 本身就被看作一个有意义的 `aggregate descent direction`。虽然每个小步都是坐标方向，但它们的合成方向可能更接近局部真正有效的下降方向。

核心机制是：

> A successful sweep is not only progress; it is information about a useful non-coordinate direction.

因此 Lean Evolved BDS 会尝试沿 sweep displacement 做 `pattern extrapolation`。如果这个方向继续有效，算法就可能用很少的额外函数评价跨过多个 coordinate zig-zag steps，直接沿组合方向前进。

`momentum extrapolation` 进一步减少单轮 sweep displacement 的偶然性，把连续几轮成功 sweep 中共同出现的趋势提取出来：

```text
recent successful sweep directions -> smoothed trend direction
```

`productive direction memory` 则让算法在邻近区域继续复用已验证有效的 oblique directions，而不是每一轮都从坐标轴重新开始。

## 5. 为什么更抗旋转

原始 BDS 的旋转敏感性来自固定坐标方向。如果问题被旋转，原本沿坐标轴清晰的下降方向会变成斜向方向。此时只用 \(\pm e_i\) 搜索会更慢。

Lean Evolved BDS 并没有变成真正的 rotation-invariant 方法。它的基本 poll 仍然来自坐标方向。然而，成功 sweep 的总位移 \(s_k\) 是多个坐标成功步的组合。这个组合方向不必和任何单个坐标轴对齐；对于 rotated problems，\(s_k\) 可能自然接近旋转后的 oblique descent direction。

随后：

- `pattern extrapolation` 会沿 \(s_k\) 继续试探；
- `momentum extrapolation` 会平滑连续 sweep 的共同趋势；
- `productive direction memory` 会在后续轮次复用这些 discovered oblique directions。

因此 Lean Evolved BDS 的抗旋转能力来自一个实际机制：它用 coordinate polling 生成局部信息，再把局部信息合成为 non-coordinate directions。

核心 claim 是：

> The method does not make BDS rotation-invariant, but it makes BDS less coordinate-bound.

## 6. 和 NOMAD / MADS 的关系

`NOMAD` 基于 `MADS` 思想。MADS 类方法通常比原始 coordinate search 更抗旋转，因为它们不会长期只依赖固定坐标方向。MADS 的 poll directions 会随 mesh 变化；理论上，normalized poll directions 可以在极限意义下变得 dense，从而覆盖更一般的方向。

Lean Evolved BDS 与 NOMAD 的区别是：

- NOMAD/MADS 通过更丰富的 poll direction 机制获得 rotation robustness；
- Lean Evolved BDS 通过 BDS sweep 产生的历史位移学习 useful non-coordinate directions；
- Lean 的优势是机制很轻，额外评价成本低，并且保持了 BDS 的简单结构。

可以这样概括：

> NOMAD explores non-coordinate directions by design; Lean Evolved BDS discovers them from successful coordinate progress.

## 7. 和 Nelder-Mead 的关系

`Nelder-Mead` 通过 simplex 的几何变形来搜索，不依赖固定坐标方向。因此它表面上也不像原始 BDS 那样 coordinate-bound。

但在中高维、固定预算为 \(200n\) 的情况下，Nelder-Mead 的 simplex 需要维护 \(n+1\) 个点，几何结构容易退化，且在复杂非凸或狭长谷问题上可能用很多评价来调整 simplex shape。随着维度升高，这种代价更明显。

Lean Evolved BDS 的优势在于：

- 基本 coordinate polling 很便宜；
- 每轮 sweep 的成功位移可以马上变成 pattern direction；
- memory 和 momentum 只维护少量方向信息；
- 不需要维护完整 simplex geometry。

所以在 \(6\)--\(50\) 维、\(200n\) 预算下，Lean Evolved BDS 相对 Nelder-Mead 更适合这种 fixed-budget direct-search comparison。

## 8. 实验证据

以下结果来自 S2MPJ unconstrained problems，维度范围 \(6\)--\(50\)，评价预算为 \(200n\)。这些实验均为 clean run：没有 abnormal termination，也没有 output fallback。

| Setting | Solvers | Scores |
| --- | --- | --- |
| `plain` | BDS vs Lean Evolved BDS | BDS: 0.6113, Lean Evolved BDS: 1.0000 |
| `plain` | Lean Evolved BDS vs NOMAD | Lean Evolved BDS: 1.0000, NOMAD: 0.7216 |
| `linearly_transformed / rotated` | Lean Evolved BDS vs NOMAD | Lean Evolved BDS: 0.9116, NOMAD: 0.9385 |
| `plain` | Lean Evolved BDS vs Nelder-Mead | Lean Evolved BDS: 1.0000, Nelder-Mead: 0.3484 |
| `linearly_transformed / rotated`, `n_runs = 1` | Lean Evolved BDS vs Nelder-Mead | Lean Evolved BDS: 1.0000, Nelder-Mead: 0.4929 |

这些结果支持一个比较清晰的解释：

- 相比原始 BDS，Lean Evolved BDS 的主要提升来自 non-coordinate direction exploitation；
- 相比 NOMAD，Lean 在 plain problems 上非常高效，但在 rotated problems 上仍不如 MADS 类方向机制天然稳健；
- 相比 Nelder-Mead，Lean 在中维 fixed-budget setting 下明显更可靠。

## 9. 对理论写作的启发

后续写入 `bds_cvx_general.tex` 时，可以考虑把 Lean Evolved BDS 的机制抽象成下面几个概念。

第一，BDS sweep 不只是产生新点，也产生方向信息。若

```text
s_k = x_{k+1} - x_k
```

来自一轮 successful sweep，则 \(s_k\) 可以被解释为一个 empirical aggregate direction。

第二，Lean Evolved BDS 的方向集合不是固定的 coordinate set，而是

```text
coordinate directions + history-dependent aggregate directions.
```

第三，这种 history-dependent direction set 可以解释实际加速，但理论分析要小心。因为 memory 和 momentum 让算法不再是单纯的固定方向 BDS，方向选择依赖历史成功模式。

一个适合后续论文叙事的表述是：

> Lean Evolved BDS preserves the reliability of coordinate polling while adding a lightweight mechanism for exploiting aggregate descent directions discovered during previous sweeps.

这句话可以作为之后从实验现象过渡到理论讨论的桥梁。
