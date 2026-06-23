# Evolved BDS 策略有效性调查

这份笔记回答的问题是：

> Evolved BDS 比 baseline BDS 多了一些策略。哪些策略真的奏效了？为什么会奏效？

本调查基于这次 OptiProfiler 结果：

```text
/Users/lihaitian/Downloads/BDS_Evolved_BDS_u_6_50_plain_s2mpj_20260623_100711
```

## Executive Summary

当前最合理的判断是：

> Evolved BDS 的 `history-based` 优势主要来自 **non-coordinate direction exploitation**，特别是 **sweep-level pattern / momentum extrapolation** 和 **explicit productive displacement memory beyond cycling**。

更具体地说：

- 真正重要的不是 `cycle direction / cycle block` 这类 polling order 调整。
- 更重要的是把 coordinate polling 得到的成功位移组合、外推、记忆成可复用的 **non-coordinate direction**。
- Evolved BDS 的优势主要出现在中后期，即 **mid-to-late-stage exploitation**。

策略重要性的大致排序：

1. **Sweep-level pattern / momentum extrapolation**
2. **Explicit productive displacement memory beyond cycling**
3. **Diagonal probing**，主要对低维 `non-separable problems` 有帮助
4. **Per-coordinate immediate extension**，有帮助但更容易过度 exploitation
5. **Ordering / cycling-type strategies**，即 `cycle direction` 和 `cycle / reorder block`，效果 mixed，不像主因

一句话版本：

> The useful part of Evolved BDS is direction exploitation, not direction ordering.

## Scope and Caveat

这次 `6--50` 维实验不是一次干净的 **output-based comparison**。

OptiProfiler 的报告显示：

- Evolved BDS 在 122 个问题上全部 **abnormal termination**。
- Evolved BDS 的 output 全部 fallback 到初始点 `x0`。
- 但 Evolved BDS 的 **evaluation history** 被保留下来了，所以 `history-based profile` 仍然有参考价值。

目前看，异常的主要原因是 **budget mismatch**：

- OptiProfiler 脚本中设置的是 `max_eval_factor = 200`。
- competitor solver 内部写的是 `maxfun = 500 * n`。
- Evolved BDS 的额外策略会消耗更多 function evaluations。
- 因此它经常超过 OptiProfiler 外层预算，被外层强制中断。

所以本文所有结论都应该理解为：

> Evolved BDS 在 `history-based / best-so-far trajectory` 上更强；但当前实现还需要先修复 budget 和 termination，才能做严肃的 `output-based` 最终解比较。

## Strategy Taxonomy: Ordering vs Direction Exploitation

为了避免把所有“记忆”都混在一起，可以把 Evolved BDS 的策略分成两类。

### Ordering / Cycling-Type Strategies

```text
Sign preference      ~= cycle direction
Coordinate ordering  ~= cycle / reorder block
```

这类策略改变的是 polling order：

- `Sign preference` 决定在一个 coordinate block 里先试 `+e_i` 还是 `-e_i`。
- `Coordinate ordering` 决定哪些 coordinate blocks 先被 poll。

它们本质上仍然是在已有 coordinate directions / coordinate blocks 之间调整顺序，并不创造新的 trial directions。因此这类机制和 MATLAB BDS 里的 `cycling` 思想非常接近。

baseline MATLAB BDS 已经有 **cycling / direction ordering memory**：在 opportunistic polling 中，如果某个已有 polling direction 成功，MATLAB BDS 会通过 `cycling` 调整该 block 内的 `direction_indices`，让成功方向下次更早被尝试。

### Non-Coordinate Direction Exploitation Strategies

```text
Sweep-level pattern / momentum extrapolation
Explicit productive displacement memory beyond cycling
Diagonal probing
```

这类策略更像是 Evolved BDS 的真正新增点。它们会创造、学习或复用 **non-coordinate directions**：

- `pattern / momentum` 从最近成功轨迹中提取组合方向；
- `explicit displacement memory` 显式保存 successful displacement vectors；
- `diagonal probing` 在 stagnation 时补充 hand-crafted diagonal directions。

其中 **explicit productive displacement memory beyond cycling** 需要特别说明：它不是只重排已有 polling directions，而是把成功位移本身作为向量存下来，后续把这些 remembered productive directions 作为额外 trial directions 主动尝试。

```text
MATLAB BDS cycling:
    remembers the order of existing polling directions
    e.g., try the previously successful sign earlier

Evolved productive displacement memory:
    stores successful displacement vectors explicitly
    can reuse coordinate, diagonal, pattern, or other non-coordinate directions
    can try them before the next coordinate sweep and extrapolate along them
```

## Evidence

### Full 6--50 History Profile

如果只看 `history-based best-so-far`，Evolved BDS 最终更强：

```text
BDS history wins:         36
Evolved BDS history wins: 63
ties:                     23
```

但更重要的是 Evolved BDS 不是一开始就强。按预算比例比较 best-so-far：

| Budget fraction | BDS wins | Evolved wins | Ties |
|---:|---:|---:|---:|
| 5% | 59 | 47 | 16 |
| 10% | 62 | 35 | 25 |
| 25% | 60 | 39 | 23 |
| 50% | 46 | 53 | 23 |
| 75% | 42 | 57 | 23 |
| 100% | 36 | 63 | 23 |

这个模式说明：

- 前 5%--25% 预算里，BDS 更常领先。
- 到 50% 预算以后，Evolved BDS 开始反超。
- 到 100% 预算时，Evolved BDS 明显占优。

如果主要有效的是 `cycle direction / cycle block` 这类 **ordering / cycling-type strategies**，我们应该更容易看到 early-stage advantage。实际看到的是中后期反超，所以更像是 **late exploitation advantage**。

在 63 个 Evolved BDS 最终 history wins 里：

- 26 个问题在 25% 预算时还没有领先。
- 13 个问题在 50% 预算时仍然没有领先。

### Dimension Breakdown

按问题维度分组：

| Problem group | BDS wins | Evolved wins | Ties |
|---|---:|---:|---:|
| `n <= 10` | 31 | 44 | 6 |
| `n > 10` | 5 | 19 | 17 |
| `n >= 20` | 2 | 13 | 1 |

**Diagonal probing** 只在 `2 <= n <= 10` 时启用，但 Evolved BDS 在 `n > 10` 和 `n >= 20` 上也很强。因此 diagonal probing 不是全局优势的主要解释。高维优势更可能来自 **pattern moves**、**explicit displacement memory** 和其他 exploitation 机制。

### Local Ablation Probe

为了更直接判断哪个策略重要，我做了一个小规模本地 **ablation probe**。

这不是完整 benchmark，而是 targeted investigation：

- 固定预算：`200n` objective evaluations
- 指标：`history best-so-far`
- 问题：选了一些 Evolved-favored examples 和 BDS-favored controls

Evolved-favored examples：

```text
BIGGS6, EIGENBLS, GENROSE, HEART6LS, FLETCHBV, FLETCBV3,
MOREBV, TRIGON2, BROYDNBDLS, CURLY10, THURBERLS
```

BDS-favored controls：

```text
PALMER1C, PALMER2C, LANCZOS1LS, INDEF
```

在 11 个 Evolved-favored examples 上：

| Ablation | Worse than full Evolved | Better | Tie | Interpretation |
|---|---:|---:|---:|---|
| Remove sweep pattern / momentum | 10 | 0 | 1 | 最强证据，最可能是主贡献 |
| Remove all exploitation extras | 11 | 0 | 0 | exploitation 机制整体非常重要 |
| Remove explicit displacement memory | 9 | 1 | 1 | 通常有帮助 |
| Remove diagonal probing | 4 | 2 | 5 | 只对部分低维问题重要 |
| Remove sign/order adaptation | 4 | 4 | 3 | 效果 mixed，不像主因 |

这支持一个清晰判断：

> ordering / cycling-type strategies 效果 mixed；真正驱动 Evolved BDS `history-based` 优势的是 non-coordinate direction exploitation。

### Best-So-Far Update Sources

在选取的 15 个 probe 问题中，Full Evolved 的 best-so-far 更新来源大致如下：

| Source | Count |
|---|---:|
| Coordinate poll | 3440 |
| Coordinate extension | 2247 |
| Explicit displacement memory | 536 |
| Sweep pattern | 536 |
| Memory extrapolation | 422 |
| Momentum pattern | 294 |
| Diagonal probing | 18 |

这里有一个细节：最终 best value 经常仍然是在 **coordinate poll** 中出现的。这并不说明额外策略没用。更合理的解释是：

1. pattern move / explicit displacement memory 先把 base point 推到更好的区域；
2. 普通 coordinate poll 在这个新区域里找到最终 best point；
3. 因此策略之间存在明显 interaction。

## Strategy-Level Interpretation

### 1. Sweep-Level Pattern / Momentum Extrapolation

**What it does.**  
每完成一轮 coordinate sweep 后，计算这一轮的总位移 `displacement`。如果 sweep 有改进，就沿这个总位移方向继续尝试。与此同时，维护一个 `momentum` direction，用近期成功方向做平滑。

**Why it helps.**  
Baseline BDS 的 coordinate polling 是 axis-aligned 的。但很多问题的真实下降方向不是坐标轴方向，而是多个变量组合出来的方向，例如：

- curved valley
- long narrow valley
- non-separable objective
- coupled variables
- ill-conditioned local geometry

一轮 sweep 中多个 coordinate moves 的净位移，往往比任何一个单坐标方向更接近真实下降方向。**Sweep-level pattern move** 实际上是在做：

> 把一组 coordinate-wise improvements 合成为一个 approximate valley direction。

这也解释了为什么它主要贡献中后期优势：前期 easy coordinate improvements 很多，BDS 也能下降；后期 coordinate progress 变慢，Evolved BDS 可以利用 sweep displacement 和 momentum 继续沿组合方向推进。

**Evidence / caveat.**  
在 ablation probe 中，去掉 `sweep pattern / momentum` 后，11 个 Evolved-favored examples 中有 10 个变差，1 个基本持平。这是目前证据最强的策略。

### 2. Explicit Productive Displacement Memory Beyond Cycling

**What it does.**  
每当某个 displacement 成功，就把它 normalize 后存入 memory。后续迭代优先尝试这些 remembered productive directions；如果再次成功，还可以做 memory extrapolation。

**Why it helps.**  
Baseline BDS 的 `cycling` 能让已有方向的顺序更聪明，但如果有效方向是多个坐标组合出来的 **non-coordinate direction**，baseline BDS 仍然要通过 coordinate polling 一步步重新“拼出”这个方向。Evolved BDS 直接记住这个 displacement direction，下次优先试。

这带来两个好处：

1. 减少重复探索。
2. 让算法可以沿 non-coordinate direction 连续推进。

**Evidence / caveat.**  
去掉 explicit displacement memory 后，11 个 Evolved-favored examples 中有 9 个变差。它通常有帮助，但影响弱于 sweep-level pattern / momentum。

我的判断：

> **Explicit productive displacement memory beyond cycling** 是重要辅助策略。它保留了 cycling 的“成功方向优先”思想，但把记忆对象从 direction ordering 扩展到了 successful displacement vectors。

### 3. Diagonal Probing

**What it does.**  
当 coordinate sweep 和 pattern move 都没有改进时，Evolved BDS 认为可能出现 stagnation，于是尝试少量 hand-crafted diagonal directions。它只在 `2 <= n <= 10` 时启用。

典型方向包括：

```text
[1, 1, ..., 1] / sqrt(n)
[1, -1, 1, -1, ...] / norm
[n, n-1, ..., 1] / norm
```

以及它们的反方向。

**Why it helps.**  
如果问题是 `non-separable` 的，真实下降方向可能是斜的。单纯沿坐标轴试探会很慢，甚至会反复缩步。Diagonal probing 给算法提供了一些便宜的 **non-axis-aligned directions**。

**Evidence / caveat.**  
在 probe 中，diagonal 对 `GENROSE`、`BROYDNBDLS` 等问题比较敏感。但它在 `n > 10` 时不开启，因此不能解释高维问题上的整体优势。

### 4. Per-Coordinate Immediate Extension

**What it does.**  
当某个 coordinate direction 成功后，立即沿同一 displacement 多试一两步。它可以理解成一个非常短的 **opportunistic coordinate line search**。

**Why it helps.**  
如果某个坐标方向确实持续下降，那么一次成功 poll 后马上继续走，可以用很少的额外 evaluations 获得更多下降。它适合：

- locally separable regions
- nearly separable regions
- 某个单坐标方向连续有效的阶段

**Evidence / caveat.**  
这个策略有帮助，但也更容易 over-exploit。因为它比较 greedy，如果某个 coordinate move 只是局部偶然有效，继续沿它走可能会浪费预算，甚至把搜索带偏。

在一些 Evolved-favored problems 上它有帮助；但在 `PALMER`、`LANCZOS`、`INDEF` 等 BDS-favored controls 上，去掉它有时反而更好。

### 5. Ordering / Cycling-Type Strategies

**What it does.**  
这一类包括 `sign preference` 和 `coordinate ordering`：

```text
Sign preference      ~= cycle direction
Coordinate ordering  ~= cycle / reorder block
```

`Sign preference` 决定某个 coordinate block 里先试 `+e_i` 还是 `-e_i`。它和 MATLAB BDS 的 `cycling_inner` 很接近。

`Coordinate ordering` 根据 recent success rate 决定哪个 coordinate block 先被访问。它不是严格的 cyclic permutation，但概念上属于 **cycle / reorder block**。

**Why it helps.**  
它们可能减少一些无效 poll，让最近成功的 direction 或 block 更早被尝试。

**Evidence / caveat.**  
数据不支持它是主因：

- 在 ablation probe 中，去掉 sign/order adaptation 后，11 个 Evolved-favored problems 里 4 个变差、4 个更好、3 个持平。
- 在完整 6--50 history 数据中，Evolved BDS 没有 early-stage advantage。

我的判断：

> **Ordering / cycling-type strategies** 可能有小帮助，但不是 Evolved BDS 胜出的主要原因。它们对应的主要是 cycle direction 和 cycle / reorder block，而不是新的 direction exploitation。

## Mechanism: Why Evolved Wins Later

Evolved BDS 的有效机制可以理解成下面这个 loop：

1. **Coordinate polling** 找到一些局部成功位移。
2. **Explicit productive displacement memory beyond cycling** 记录这些成功 displacement directions。
3. **Sweep-level pattern move** 把一轮 sweep 的多个小位移合成为组合方向。
4. **Momentum** 平滑近期成功方向，避免每次只看当前 sweep。
5. **Diagonal probing** 在低维 stagnation 时提供额外 non-axis directions。
6. 后续 coordinate polling 在更好的 base point 附近继续细化。

因此，Evolved BDS 相比 baseline BDS，不只是多试几个方向，而是有了一种弱形式的方向学习：

> It learns approximate productive directions from successful coordinate moves.

这就是它能在中后期反超 BDS 的原因。

## Limitations and Next Steps

Evolved 的额外策略本质上偏 exploitation。exploitation 太强时，会出现副作用：

- 过早相信某个方向。
- 在错误方向上多花 evaluations。
- 对某些 least-squares 或 ill-conditioned problems 反而不如 baseline BDS 稳。

这能解释为什么 BDS 仍然在 36/122 个问题上有更好的 final history best。probe 中的 `PALMER`、`LANCZOS`、`INDEF` controls 也显示，去掉 explicit displacement memory 或 coordinate extension 后，有些 BDS-favored problems 反而更好。

下一步最值得做的是：

1. 修复 Evolved BDS 的 budget / termination 行为，让它在 `max_eval_factor = 200` 下正常返回。
2. 重新跑一次 clean `6--50` comparison，同时看 `history-based` 和 `output-based` profiles。
3. 做正式 full ablation：
   - Full Evolved
   - No sweep pattern / momentum
   - No explicit productive displacement memory
   - No diagonal probing
   - No coordinate extension
   - No sign/order adaptation
4. 分维度、分问题族分析结果。

我的预期是：

- 去掉 **sweep pattern / momentum** 会造成最大退化。
- 去掉 **explicit productive displacement memory beyond cycling** 会造成中等退化。
- 去掉 **diagonal probing** 主要影响低维 non-separable problems。
- 去掉 **ordering / cycling-type strategies** 的影响会较小且 mixed。

## Final Takeaway

目前最合理的结论是：

> Evolved BDS 的 `history-based` 优势主要来自 **non-coordinate direction exploitation**，而不是 **ordering / cycling-type strategies**。

展开说：

- `Sign preference ~= cycle direction`
- `Coordinate ordering ~= cycle / reorder block`
- 这些策略改变 polling order，但不创造新方向，效果 mixed。
- `Sweep-level pattern / momentum extrapolation` 和 `explicit productive displacement memory beyond cycling` 会创造或复用 productive non-coordinate directions，是当前最可信的主贡献。
- `Diagonal probing` 是低维 non-separable problems 的有效补丁，但不是全局主因。

