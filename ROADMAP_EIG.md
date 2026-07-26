# Nx.LinAlg.eig — LAPACK DGEEV 移植路线图

## 目标

将 LAPACK 的 `DGEEV`（通用实矩阵特征值分解）完整、忠实地移植到 Elixir/Nx 中，精度与 LAPACK 一致。

---

## 阶段 0：BLAS 原语

### LAPACK 中的位置

DGEEV 依赖以下 BLAS 子程序。它们不是 DGEEV 的一部分，但被所有后续阶段调用。

| 函数 | LAPACK 路径 | 行数 | 用途 |
|:-----|:------------|:----:|:------|
| `DAXPY` | `BLAS/SRC/daxpy.f` | 153 | `y = a*x + y` |
| `DCOPY` | `BLAS/SRC/dcopy.f` | 147 | 向量复制 |
| `DSCAL` | `BLAS/SRC/dscal.f` | 140 | `x = a*x` |
| `DSWAP` | `BLAS/SRC/dswap.f` | 154 | 向量交换 |
| `DROT` | `BLAS/SRC/drot.f` | 143 | Givens 旋转应用 |
| `DNRM2` | BLAS 标准 | ~50 | `||x||₂` |
| `IDAMAX` | `BLAS/SRC/idamax.f` | 127 | `max(|xᵢ|)` 的索引 |
| `DGEMM` | `BLAS/SRC/dgemm.f` | 408 | `C = a*op(A)*op(B) + b*C` |
| `DGEMV` | `BLAS/SRC/dgemv.f` | 330 | `y = a*A*x + b*y` |
| `DGER` | `BLAS/SRC/dger.f` | 225 | `A = a*x*y' + A` |
| `DTRMM` | `BLAS/SRC/dtrmm.f` | 402 | 三角矩阵乘法 |
| `DTRMV` | `BLAS/SRC/dtrmv.f` | 330 | 三角矩阵×向量 |
| `DLARTG` | BLAS 标准 | ~60 | 生成 Givens 旋转 |

### Nx 可用函数分析

| BLAS 函数 | Nx 等价操作 | 是否已有 | 备注 |
|:----------|:------------|:--------:|:-----|
| `DAXPY` | `Nx.add(Nx.multiply(a, x), y)` | ✅ | 直接组合 |
| `DCOPY` | 直接赋值 | ✅ | 无需 Nx 函数 |
| `DSCAL` | `Nx.multiply(a, x)` | ✅ | |
| `DSWAP` | `{x, y} = {y, x}` + `Nx.put_slice` | ✅ | 选择性交换需 `put_slice` |
| `DROT` | `Nx.add(Nx.multiply(c, x), Nx.multiply(s, y))` | ✅ | 组合算术 |
| `DNRM2` | `Nx.LinAlg.norm(vector)` | ✅ | 默认 2-范数 |
| `IDAMAX` | `Nx.argmax(Nx.abs(x))` | ✅ | |
| `DGEMM` | `Nx.dot(a, b)` + `Nx.add/2` + `Nx.multiply/2` | ✅ | `Nx.dot` 支持批量 + 转置 |
| `DGEMV` | `Nx.dot(matrix, vector)` | ✅ | `{m,n} × {n} → {m}` |
| `DGER` | `Nx.add(A, Nx.multiply(a, Nx.outer(x, y)))` | ✅ | `Nx.outer/2` 存在 |
| `DTRMM` | `Nx.dot(Nx.triu(A), B)` | ⚠️ 需三角掩码 | 无专用 DTRMM |
| `DTRMV` | `Nx.dot(Nx.triu(A), v)` | ⚠️ 需三角掩码 | 同 DTRMM |
| `DLARTG` | 需自定义实现 | ❌ | 由 `Nx.sqrt`/`Nx.abs`/`Nx.select` 组合 |

**关键决策**：BLAS 层用 Nx 包装，不自己实现矩阵乘法。`DGEMM` 是性能瓶颈——Nx 在 BinaryBackend 上用 O(n³) 的 Elixir 实现，在 EXLA/EMLX 上使用优化的 GPU/CPU 实现。

**`DLARTG` 自定义实现**（约 30 行）：

```elixir
def dlartg(f, g) do
  # 标准 Givens 旋转：计算 c, s, r 使得
  # [c  s; -s  c]^T * [f; g] = [r; 0]
  # 参考 LAPACK DLARTG 算法（避免上溢/下溢）
  ...
end
```

### 测试方法

对每个 BLAS 函数，用已知输入验证：
- `DAXPY`：给定 `a, x, y`，验证 `result[i] = a*x[i] + y[i]`
- `DGEMM`：给定随机 10×10 矩阵，验证 `C = A*B` 与逐元素计算结果一致
- `DROT`：验证旋转后 `||x,y||` 不变
- 精度基准：`assert_all_close(atol=1e-15)`

---

## 阶段 1：工具函数

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DLAMCH` | `INSTALL/dlamch.f` | 194 | 机器精度常数（ε、safe min 等） |
| `ILAENV` | `SRC/ilaenv.f` | 762 | 分块大小/算法选择参数 |
| `XERBLA` | `SRC/xerbla.f` | 95 | 错误报告 |
| `LSAME` | `INSTALL/lsame.f` | 120 | 字符比较（大小写不敏感） |
| `DISNAN` | `SRC/disnan.f` | 76 | NaN 检测 |
| `DLAPY2` | `SRC/dlapy2.f` | 117 | `sqrt(x²+y²)` 无溢出 |
| `DLANGE` | `SRC/dlange.f` | 208 | 矩阵范数（1-范数、∞-范数、F-范数） |

### Nx 可用函数分析

| LAPACK 函数 | Nx 等价操作 | 是否已有 | 备注 |
|:------------|:------------|:--------:|:-----|
| `DLAMCH` | 硬编码常数 | ❌（自行实现） | IEEE 754 常量：`eps=2.22e-16`, `sfmin=2.23e-308` |
| `ILAENV` | 固定参数 | ❌（自行实现） | `NB=64`, `NBMIN=2`, `NX=32` 等 |
| `XERBLA` | `raise ArgumentError` | ✅ | 直接替换 |
| `LSAME` | `String.downcase ==` | ✅ | 简单字符串比较 |
| `DISNAN` | `Nx.is_nan/1` | ✅ | |
| `DLAPY2` | **无内置** | ❌（自行实现） | `max*√(1+(min/max)²)` 避免溢出 |
| `DLANGE` | `Nx.reduce_max(Nx.sum(Nx.abs(A), axes=[0]))` | ✅ | 1-范数/∞-范数皆可 |

**`DLAMCH`**：返回 IEEE 754 浮点常数（须与硬件一致）。Erlang 的 `:math` 模块提供 `:math.sqrt/1` 等，但机器常数需用 Erlang 的 `:erlang.system_info` 或直接硬编码为 IEEE 754 标准值（`eps=2⁻⁵² ≈ 2.22e-16`，`sfmin=2⁻¹⁰²² ≈ 2.23e-308`）。
**`ILAENV`**：返回块大小参数。LAPACK 中有硬编码建议值（`NB=64`，`NBMIN=2`，`NX=32` 等）。初始实现可用固定值，后续再优化。
**`DLANGE`**：`||A||₁ = maxⱼ Σᵢ|aᵢⱼ|`，用 `Nx.sum(Nx.abs(A), axes=[0])` + `Nx.reduce_max`。
**`DLAPY2`**：`sqrt(max(|x|,|y|)) * sqrt(1 + (min(|x|,|y|)/max(|x|,|y|))²)` 避免溢出。

### 测试方法

- `DLAMCH`：验证 `eps * 1.0 + 1.0 > 1.0` 且 `eps/2 * 1.0 + 1.0 == 1.0`
- `DLANGE`：对已知矩阵手动计算范数，与函数返回值对比
- `DLAPY2`：对大数值 `(1e200, 1e200)` 验证返回 `~1.414e200`（不会溢出）

---

## 阶段 2：矩阵平衡 + 反变换

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DGEBAL` | `SRC/dgebal.f` | 417 | 矩阵平衡（置换 + 缩放） |
| `DGEBAK` | `SRC/dgebak.f` | 266 | 特征向量反变换 |

### Nx 可用函数分析

| 需调用的 BLAS | Nx 等价操作 | 是否已有 |
|:--------------|:------------|:--------:|
| `DSCAL` | `Nx.multiply(a, x)` | ✅ |
| `DSWAP` | `{x, y} = {y, x}` / `Nx.put_slice` | ✅ |
| `DNRM2` | `Nx.LinAlg.norm(vector)` | ✅ |
| `IDAMAX` | `Nx.argmax(Nx.abs(x))` | ✅ |
| `DISNAN` | `Nx.is_nan/1` | ✅ |
| `DLANGE` | `Nx.reduce_max(Nx.sum(Nx.abs(A), axes=[0]))` | ✅ |
| `DLAMCH` | 硬编码常数 | ❌ 需自定义 |

DGEBAL/DGEBAK 主要使用标量循环和 BLAS-1 操作，所有 BLAS 依赖都可用。
核心算法是 GOTO/DO WHILE 循环的递归翻译，不需要额外的 Nx 功能。

### 算法原理

**第一步：置换（isolate）。** 扫描矩阵的行/列，寻找可以孤立的特征值：

```
DO WHILE 有可交换的行/列
  IF 行 i 只有一个非零非对角元素 THEN
    将该行与最后未处理的行交换 → 该特征值被孤立
  END IF
  IF 列 i 只有一个非零非对角元素 THEN  
    将该列与最后未处理的列交换
  END IF
END DO
```

结果：矩阵被排列为 `P * A * P'`，其中中间的子矩阵 `K..L` 需要缩放，对角块已被分离为三角子块。

**第二步：缩放（scale）。** 对于子矩阵 `K..L`：

```
DO 最多 5 次扫描
  FOR i = K TO L
    c = ||row(i)||₁ - |aᵢᵢ|  (非对角行范数)
    r = ||col(i)||₁ - |aᵢᵢ|  (非对角列范数)
    IF c ≠ 0 AND r ≠ 0 THEN
      s = sqrt(r/c)  (目标缩放因子)
      s = clamp(s, SCLFAC, 1/SCLFAC)  (SCLFAC = 2)
      IF s 显著偏离 1 THEN
        行 i 乘以 1/s
        列 i 乘以 s
      END IF
    END IF
  END FOR
  如果没有任何行被缩放 → 退出
END DO
```

**DGEBAK：** 逆操作：`V = P⁻¹ * D⁻¹ * V`（对右特征向量）或 `V = D * P⁻¹ * V`（对左特征向量）。

### Fortran → Elixir 的关键转换

**Fortran 的 DO WHILE 循环**：
```fortran
DO WHILE (N > 0)
  IF (...) GO TO 20
  IF (...) EXIT
  ...
END DO
```

**Elixir 等价的递归模式**：
```elixir
defp balance_permute(a, n, ilo, ihi) do
  {result, ilo_new, ihi_new} = balance_scan(a, n, ilo, ihi)
  if ilo_new != ilo or ihi_new != ihi do
    balance_permute(result, n, ilo_new, ihi_new)
  else
    {result, ilo_new, ihi_new, scale}
  end
end
```

### 测试方法

1. **平衡保特征值**：对随机矩阵 A，计算 `eig(A)` 和 `eig(P * D * A * D⁻¹ * P⁻¹)`，特征值应相同
2. **平衡质量**：对平衡后的矩阵，验证每行的范数与对应列的范数相近（`max(row_norm / col_norm) < sqrt(SCLFAC)`）
3. **反变换**：`A * V ≈ V * Λ`

---

## 阶段 3：Householder 反射器

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DLARFG` | `SRC/dlarfg.f` | 192 | 生成 Householder 反射器 `H = I - tau*v*v'` |
| `DLARF1F` | `SRC/dlarf1f.f` | 296 | 应用反射器到矩阵（右侧变体） |

### Nx 可用函数分析

| 需调用的 BLAS | Nx 等价操作 | 是否已有 |
|:--------------|:------------|:--------:|
| `DSCAL` | `Nx.multiply(a, x)` | ✅ |
| `DNRM2` | `Nx.LinAlg.norm(vector)` | ✅ |
| `DGEMM` 等 BLAS-3 | `Nx.dot(A, B)` | ✅ |

DLARFG 和 DLARF1F 主要使用标量计算和 BLAS-1/2 操作，所有依赖都可用。
这两个函数的实现已在我们当前的 `block_eig.ex` 中完成。

### 算法原理

**DLARFG：** 给定向量 `x`，计算 `v` 和 `tau` 使得：

```
H*x = [beta, 0, ..., 0]'，其中 beta = -sign(x₁)*||x||
```

核心公式：

```
sigma = ||x(2:n)||²
IF sigma == 0: beta = x₁, tau = 0, v = [1, 0, ...]
norm = sqrt(x₁² + sigma)
alpha = x₁
IF alpha ≤ 0:
    beta = alpha - norm    (beta 与 alpha 同号)
ELSE:
    beta = -sigma / (alpha + norm)  (beta = alpha - norm 的数值稳定形式)
v(1) = 1
v(2:n) = x(2:n) / (alpha - beta)   
tau = (beta - alpha) / beta   (然后 tau = 2 * tau / 实际上就是 2/(1 + ||v(2:n)||²))
```

**DLARF1F：** 应用 `H = I - tau*v*v'` 到矩阵：
- 左乘 `H*A`：`A - tau * v * (v' * A)`
- 右乘 `A*H`：`A - tau * (A * v) * v'`

### 测试方法

1. **反射验证**：对随机向量 x，验证 `H*x` 的最后 n-1 个元素为零
2. **正交性**：验证 `H' * H == I`
3. **恒等验证**：`beta` 的绝对值应为 `||x||`（符号取 x₁ 的相反数）

---

## 阶段 4：块反射器

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DLARFT` | `SRC/dlarft.f` | 636 | 计算块反射器的三角因子 T |
| `DLARFB` | `SRC/dlarfb.f` | 738 | 应用块反射器 |

### Nx 可用函数分析

| 需调用的 BLAS | Nx 等价操作 | 是否已有 |
|:--------------|:------------|:--------:|
| `DGEMM` | `Nx.dot(A, B)` | ✅ |
| `DTRMM` | `Nx.dot(Nx.triu(T), B)` | ⚠️ 需三角掩码 |
| `DCOPY` | 直接赋值 | ✅ |
| `DGEMV` | `Nx.dot(A, x)` | ✅ |

DLARFT 和 DLARFB 主要依赖 BLAS-3 操作（DGEMM、DTRMM），Nx 都已支持。

### 算法原理

块反射器将多个 Householder 反射器 `H₁, H₂, ..., H_k` 合并为紧凑形式：

```
H₁ * H₂ * ... * H_k = I - V * T * V'
```

其中 `V = [v₁ v₂ ... v_k]`，`T` 是 k×k 上三角矩阵。

**DLARFT：** 给定 `V`，计算 `T`：

```
T 的递推计算：
T(1,1) = tau₁
FOR j = 2 TO k
    z = -tauⱼ * T(1:j-1, 1:j-1) * V' * vⱼ
    T(1:j-1, j) = z
    T(j, j) = tauⱼ
END FOR
```

**DLARFB：** 应用 `I - V*T*V'` 到矩阵：
- 左乘：`A - V * (T * (V' * A))`
- 右乘：`A - (A * V) * T * V'`

用 `DGEMM` 和 `DTRMM` 实现，是块算法的关键性能提升。

### 测试方法

1. **等价性**：验证 `I - V*T*V'` 在 `k` 个 Householder 反射器上的效果与逐个应用 `H₁*H₂*...*H_k` 一致
2. **正交性**：验证 `(I - V*T*V') * (I - V*T*V')' == I`

---

## 阶段 5：Hessenberg 约化

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DGEHRD` | `SRC/dgehrd.f` | 360 | 块 Hessenberg 约化（顶层） |
| `DLAHR2` | `SRC/dlahr2.f` | 324 | 块更新辅助 |
| `DGEHD2` | `SRC/dgehd2.f` | 213 | 无块 Hessenberg 约化（回退） |

### Nx 可用函数分析

| 需调用的 LAPACK/BLAS | Nx 等价操作 | 是否已有 |
|:---------------------|:------------|:--------:|
| `DLARFG`、`DLARF1F` | 阶段 3 实现 | ✅（本阶段前已完成） |
| `DLARFT`、`DLARFB` | 阶段 4 实现 | ✅（本阶段前已完成） |
| `DGEMM` | `Nx.dot(A, B)` | ✅ |
| `DTRMM` | `Nx.dot(Nx.triu(T), B)` | ⚠️ |
| `DGEMV` | `Nx.dot(A, x)` | ✅ |
| `DAXPY` | `Nx.add(Nx.multiply(a, x), y)` | ✅ |
| `DCOPY` | 直接赋值 | ✅ |
| `ILAENV` | 固定参数 | ❌ 需自定义 |

所有依赖在阶段 3-4 完成后可用。DLAHR2 是块算法的核心技术，需要组合 DGEMV、DGEMM、DTRMM 来更新 A、V、T、Y。

### 算法原理

Hessenberg 约化将一般矩阵 A 转化为 `Q * H * Q'` 的上 Hessenberg 形式 H（次对角线以下为零）。

**块算法流程：**

```
FOR i = 1 TO N-NX STEP NB
    DLAHR2(A, i, NB) → 计算 V, T, Y = A*V*T
    A(i+NB:N, i:N) = A(i+NB:N, i:N) - V * Y
    A(i:N, i+N) = A(i:N, i+N) - ... (右乘更新)
END FOR
DGEHD2(A)  {处理剩余列}
```

**DLAHR2** 计算三个关键矩阵：
- `V`：Householder 反射器的累积（块反射器）
- `T`：块反射器的三角因子
- `Y = A*V*T`：用于左乘更新的预计算

**DGEHD2**（无块回退）：标准的一列一列 Householder 约化：

```
FOR i = 1 TO N-2
    x = A(i+1:N, i)
    调用 DLARFG(x) → v, tau, beta
    A(i+1:N, i) = [beta, 0, ..., 0]'
    应用 DLARF1F 到 A(i+1:N, i+1:N)（左乘）
    应用 DLARF1F 到 A(1:N, i+1:N)（右乘）
END FOR
```

### 测试方法

1. **Hessenberg 结构**：验证 `H[i,j]` 当 `i > j+1` 时为零（`abs < 1e-12`）
2. **重构**：验证 `A ≈ Q * H * Q'`（`max_err < 1e-10`）
3. **正交 Q**：验证 `Q' * Q ≈ I`

---

## 阶段 6：生成 Q 矩阵

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DORGHR` | `SRC/dorghr.f` | 237 | 从 Hessenberg 反射器生成 Q |
| `DORGQR` | `SRC/dorgqr.f` | 288 | 从 QR 分解的反射器生成 Q |
| `DORG2R` | `SRC/dorg2r.f` | 195 | 生成 Q（无块回退） |

### Nx 可用函数分析

| 需调用的 LAPACK/BLAS | Nx 等价操作 | 是否已有 |
|:---------------------|:------------|:--------:|
| `DORGHR` | 封装调用 DORGQR | — |
| `DORGQR` | 块算法：DLARFT + DLARFB | ✅（阶段 4） |
| `DORG2R` | 无块回退：DLARF1F 循环 | ✅（阶段 3） |
| `DLARFT`、`DLARFB` | 阶段 4 实现 | ✅ |
| `DGEMM` | `Nx.dot(A, B)` | ✅ |

DORGHR 本身只是一个封装，真正的工作在 DORGQR（块生成）和 DORG2R（无块回退）中。

### 算法原理

Hessenberg 约化中 Householder 反射器被存储在矩阵 A 的下三角中。`DORGHR` 把这些反射器重构为完整的正交矩阵 Q。

**流程：**

```
1. 将 Hessenberg 向量右移一列（留出对角线空间）
2. 初始化右下部分为单位矩阵
3. 调用 DORGQR 从 QR 分解生成 Q
```

**DORGQR（块算法）：**

```
FOR k = N-(1+NB) TO 1 STEP -NB
    用 DLARFT 计算 T
    用 DLARFB 应用 T → 更新剩余部分
END FOR
DORG2R  {处理最后的无块部分}
```

**测试方法：**

1. 验证 `Q' * Q ≈ I`
2. 验证 `Q' * A * Q` 是上 Hessenberg 形式

---

## 阶段 7：小型 QR 算法（DLAHQR）

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DLAHQR` | `SRC/dlahqr.f` | 620 | 小型矩阵的 QR 算法 |
| `DLANV2` | `SRC/dlanv2.f` | 311 | 2×2 块的舒尔形式 |

### Nx 可用函数分析

| 需调用的 BLAS/LAPACK | Nx 等价操作 | 是否已有 |
|:---------------------|:------------|:--------:|
| `DLANV2` | 2×2 舒尔形式 | ❌ 需自定义（~300 行） |
| `DLARFG` | 阶段 3 实现 | ✅ |
| `DCOPY` | 直接赋值 | ✅ |
| `DROT` | `c*x + s*y` | ✅ |
| `DLAPY2` | 自定义实现 | ❌ 需自定义 |
| `DLAMCH` | 硬编码常数 | ❌ 需自定义 |
| `DSCAL` | `Nx.multiply(a, x)` | ✅ |

DLAHQR 的核心是控制流（循环移位、deflation、例外移位），计算上依赖 BLAS-1 操作和 DLANV2。DLANV2 是纯标量算术，需要从 LAPACK 逐行翻译。

### 算法原理

**DLAHQR** 实现经典的 Francis 双移位 QR 算法，针对 `N ≤ 75` 的 Hessenberg 矩阵。

**主循环：**

```
FOR i = N DOWN TO 2
    初始化 J = i  (当前活跃矩阵的大小)
    iter = 0
    
    WHILE J > 1
        IF subdiag(J-1, J-2) 很小 → deflation
            J = J - 1
            iter = 0
        END IF
        
        IF iter == 10 → 采用单移位（例外移位）
        IF iter == 20 → 采用双移位（例外移位）
        
        计算 Wilkinson 移位（从 2×2 右下角块的特征值）
        IF 移位为实数
            单移位：一个 Householder 反射追赶 bulge
        ELSE
            双移位：隐式双移位（Francis 1961）
        END IF
        iter = iter + 1
    END WHILE
END FOR
```

**Wilkinson 移位**：计算右下角 2×2 块的特征值，选更接近 `H[N,N]` 的那个。

**Francis 双移位**：不显式形成 `(H - μ₁I)(H - μ₂I)`，而是通过三个 Householder 反射器（3×3 初始变换）来隐式应用双移位，然后追赶 bulge。

**DLANV2**：计算 2×2 矩阵的舒尔形式 `[a b; c d]` → 特征值或复共轭对。

### 测试方法

1. 对已知特征值的 10×10 Hessenberg 矩阵运行 `DLAHQR`，验证 `A*V ≈ V*Λ`
2. 与 scipy 的 `scipy.linalg.schur` 对比
3. 验证 deflation 正确收敛：次对角线元素趋近于 0

---

## 阶段 8：多重移位 QR 算法（核心）

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DLAQR0` | `SRC/dlaqr0.f` | 739 | 多重移位 QR（顶层） |
| `DLAQR3` | `SRC/dlaqr3.f` | 700 | 激进提前 deflation |
| `DLAQR4` | `SRC/dlaqr4.f` | 738 | 递归位移 QR |
| `DLAQR2` | `SRC/dlaqr2.f` | 689 | 激进提前 deflation（非递归） |
| `DLAQR5` | `SRC/dlaqr5.f` | 834 | 单次多位移扫掠 |
| `DLAQR1` | `SRC/dlaqr1.f` | 182 | 计算 2/3×3 Householder 反射器 |
| `DTREXC` | `SRC/dtrexc.f` | 431 | 舒尔形式的特征值重排 |

### Nx 可用函数分析

| 需调用的 BLAS/LAPACK | Nx 等价操作 | 是否已有 |
|:---------------------|:------------|:--------:|
| `DGEMM` | `Nx.dot(A, B)` | ✅ |
| `DLACPY` | `Nx.tensor(data)` + 切片 | ✅ |
| `DLASET` | `Nx.broadcast(0.0, shape)` | ✅ |
| `DLARFG` | 阶段 3 实现 | ✅ |
| `DLAQR1` | 3×3 反射器向量 | ❌ 需自定义（~180 行） |
| `DLANV2` | 阶段 7 实现 | ✅（阶段 7 完成后） |
| `DLAPY2` | 自定义实现 | ❌ |
| `DLAMCH` | 硬编码常数 | ❌ |
| `DCOPY` | 直接赋值 | ✅ |
| `DROT` | `c*x + s*y` | ✅ |
| `DSCAL` | `Nx.multiply(a, x)` | ✅ |

这一阶段是所有阶段中 BLAS-3 调用量最大的——DLAQR5 中大量使用 DGEMM 进行批处理更新。Nx 的 `Nx.dot/2` 完全可用。

### 算法原理

这是 DGEEV 中最复杂的部分。`DLAQR0` 对 `N > 75` 的 Hessenberg 矩阵使用多位移 QR。对于 `N ≤ 75` 直接调用 DLAHQR。

**DLAQR0 主循环：**

```
iter = 0
WHILE KBOT > KTOP
    {扫描找到 deflation}
    FOR i = KBOT-1 DOWN TO KTOP+1
        IF subdiag(i) 很小 → deflation
            KBOT = i
            iter = 0
    END FOR
    IF KBOT <= KTOP → 收敛
    
    {选择 NW = deflation 窗口大小}
    NW = 根据 KBOT-KTOP+1 自适应选择
    
    {激进提前 deflation：DLAQR3}
    复制右下角 NW×NW 窗口到 W
    调用 DLAQR4/W 对 W 做 QR
    检查窗口上方的"spike" → 若有可 deflate 的特征值
    用 DTREXC 重排
    恢复 Hessenberg 形式
    
    {如果 deflation 不够多，准备移位}
    从尾部 2×(NS/2) 块计算 NS 个移位
    每 KEXSH=6 迭代用一次例外移位
    
    {多位移扫掠：DLAQR5}
    创建 NBMPS = NS/2 个 bulge
    用 DLAQR1 + DLARFG 生成 3×3 Householder
    追赶 bulge 沿对角线向下
    用 DGEMM 批量更新远离对角线的部分
    iter = iter + 1
END WHILE
```

**激进提前 deflation（DLAQR3）**——LAPACK 的标志性优化：

```
1. 拷贝尾部 NW×NW 窗口到 W
2. 对 W 调用 DLAHRQ4 或 DLAHQR（小递归）
3. 计算 W 中 deflated 特征值的个数
4. 检测 spike（窗口上方的 subdiag 元素）
5. IF spike 很小（比较 |spike| 与 deflated 特征值的模）
   THEN 这些特征值被 deflate → 降低 KBOT
       用 DTREXC 重排
       用 DGEHRD + DORMHR 恢复 Hessenberg 形式
   ELSE 不做 deflation
```

**多位移扫掠（DLAQR5）**：

```
FOR INCOL = KTOP TO KBOT-NBMPS*2 STEP 3
    创建第一个 3×3 Householder → H(INCOL:INCOL+2, INCOL-1:INCOL+1)
    用 DLAQR1 计算反射器向量 v
    用 DLARFG 归一化
    追赶 bulge 向右下角
    FOR bulge = 1 TO NBMPS
        应用左乘：H(INCOL+bulge*2:INCOL+bulge*4, ...)
        应用右乘：H(..., INCOL+bulge*2:INCOL+bulge*4)
        如果 bulge 坍塌 → 尝试恢复 / vigilant deflation
    END FOR
    用 DGEMM 从 U 更新远离对角线的部分
END FOR
```

### Fortran → Elixir 的关键转换

DLAQR5 中的 GOTO 控制流最复杂。Fortran 代码使用了大量 `GO TO n` 语句（约 30+ 个标签），需要转换为 Elixir 的嵌套递归 + `if/else` 链。

**模式示例**（Fortran → Elixir）：
```fortran
   90 CONTINUE
      ...
      IF (S1.EQ.ZERO) GO TO 100
      ...
  100 CONTINUE
```
变为：
```elixir
defp bulge_chase(state, opts) do
  state = try_apply_bulge(state)
  if state.bulge_collapsed do
    bulge_recover(state, opts)
  else
    bulge_chase(state, opts)
  end
end
```

### 测试方法

1. **收敛性**：对随机 100×100 矩阵运行 QR，验证所有次对角线元素最终趋近于 0
2. **精度**：与 scipy 对比特征值（`max_err < 1e-10`）
3. **deflation 正确性**：验证 deflated 特征值的几何重数正确
4. **例外移位路径**：构造需要例外移位的矩阵（如具有聚类特征值的矩阵）

---

## 阶段 9：特征向量

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DTREVC3` | `SRC/dtrevc3.f` | 1327 | 从舒尔形式计算特征向量 |
| `DLALN2` | `SRC/dlaln2.f` | 607 | 解 1×1/2×2 线性系统（含缩放） |

### Nx 可用函数分析

| 需调用的 BLAS/LAPACK | Nx 等价操作 | 是否已有 |
|:---------------------|:------------|:--------:|
| `DAXPY` | `Nx.add(Nx.multiply(a, x), y)` | ✅ |
| `DCOPY` | 直接赋值 | ✅ |
| `DSCAL` | `Nx.multiply(a, x)` | ✅ |
| `DGEMV` | `Nx.dot(A, x)` | ✅ |
| `DGEMM` | `Nx.dot(A, B)` | ✅ |
| `DLACPY` | 切片赋值 | ✅ |
| `DLASET` | `Nx.broadcast(0.0, shape)` | ✅ |
| `DLALN2` | 1×1/2×2 条件控制求解 | ❌ 需自定义（~500 行） |
| `IDAMAX` | `Nx.argmax(Nx.abs(x))` | ✅ |
| `DLAMCH` | 硬编码常数 | ❌ |

DLALN2 是核心——它解决了 `(T - λI)x = b` 的条件控制求解，包含溢出保护。这是特征向量精度的关键。

### 算法原理

从舒尔形式 `T = Q'*A*Q` 计算特征向量。T 是准三角矩阵（对角线是 1×1 或 2×2 块）。

**对单个实特征值 λ（1×1 块）：**

解 `(T - λI) * v_i = 0`，通过对三角矩阵从下到上的回代：

```
FOR k = n-1 DOWN TO 1
    IF T[k+1,k] ≠ 0 (2×2 块一部分)
        IF k > 1 AND T[k,k-1] ≠ 0 (上层 2×2)
            跳过（由上一层处理）
        ELSE
            v[i] = 1, v[i+1] = (λ-T[i,i])/T[i,i+1]
        END IF
    ELSE (1×1 块)
        v_k = -Σ_{j>k} T[k,j] * v_j / (T[k,k] - λ)
    END IF
END FOR
```

**DLALN2** 解决 `(T - λI) * x = b` 的条件控制和缩放：
- 估计解的大小，如果可能溢出则缩放 b 并重试
- 处理 2×2 系统的复特征值（实部和虚部拆分）

**DTREVC3 块算法：**

```
FOR 特征值块 = 1 TO N STEP NB
    在 NB×NB 窗口内计算特征向量
    FOR 窗口中的每个特征值
        调用 DLALN2 求解
        回代到窗口顶部
    END FOR
    用 DGEMM 将窗口中的特征向量扩展到完整矩阵
END FOR
```

### 测试方法

1. **定义验证**：`A * v_k ≈ λ_k * v_k`（`max_err < 1e-10`）
2. **正交性**：特征向量矩阵的条件数应该在预期范围内
3. **与 scipy 对比**：`scipy.linalg.eig(A)[1]` 的列应与 Nx 结果同构（允许相位差异）

---

## 阶段 10：DGEEV 驱动器

### LAPACK 中的位置

| 函数 | 路径 | 行数 | 用途 |
|:-----|:-----|:----:|:------|
| `DGEEV` | `SRC/dgeev.f` | 543 | 顶层驱动器 |

### Nx 可用函数分析

| 需调用的 LAPACK | Nx 等价操作 | 是否已有 |
|:----------------|:------------|:--------:|
| `DGEBAL` | 阶段 2 实现 | ✅ |
| `DGEBAK` | 阶段 2 实现 | ✅ |
| `DGEHRD` | 阶段 5 实现 | ✅ |
| `DORGHR` | 阶段 6 实现 | ✅ |
| `DHSEQR` | 阶段 7/8 实现 | ✅ |
| `DTREVC3` | 阶段 9 实现 | ✅ |
| `DSCAL` | `Nx.multiply(a, x)` | ✅ |
| `DLARTG` | 自定义实现 | ❌ |
| `DROT` | `c*x + s*y` | ✅ |
| `DLASCL` | `Nx.multiply(a, A)` | ✅ |

DGEEV 本身不做任何计算——它只是按顺序调用所有前置阶段。所有依赖在阶段 2-9 完成后可用。

### 算法原理

DGEEV 的完整调用序列：

```
DGEEV(JOBVL, JOBVR, N, A, ...)
  │
  ├─ 1. 工作区间查询（确定 LWORK）
  │
  ├─ 2. 缩放矩阵到安全范围（A → A_scaled）
  │
  ├─ 3. DGEBAL(A_scaled) → A_bal, ILO, IHI, SCALE
  │    置换 + 平衡缩放 → 数值稳定性
  │
  ├─ 4. DGEHRD(A_bal) → A_bal 中写入 H
  │    Hessenberg 约化 + Tau 向量
  │
  ├─ 5. DORGHR(A_bal, Tau) → Q
  │    从 Hessenberg 反射器生成正交 Q
  │    (如果不需要特征向量，可跳过此步)
  │
  ├─ 6. DHSEQR(H) → Schur 形式 S + WR/WI
  │    如果请求特征向量，也累积 Q → Z
  │    内部：N ≤ 75 → DLAHQR
  │           N > 75 → DLAQR0 (多位移 QR)
  │
  ├─ 7. DTREVC3(T, Z) → 右特征向量 VR
  │    从准三角 Schur 回代
  │
  ├─ 8. DGEBAK(VR, SCALE, ILO, IHI)
  │    反变换（撤销平衡的置换和缩放）
  │
  ├─ 9. 特征向量归一化
  │    使最大分量为实 + 正
  │
  └─10. 如果 A 被缩放，恢复原始特征值
```

### 关键输出

- `WR[N]`, `WI[N]`：特征值的实部和虚部
- `VR[N][N]`：右特征向量（`A * VR[;,k] = λ_k * VR[;,k]`）
- 对于复特征值 `(k, k+1)`：`λ_k = WR[k] + i*WI[k]`，`λ_{k+1} = conj(λ_k)`
  特征向量 `VR[:,k]` 的实部和 `VR[:,k+1]` 的虚部构成复特征向量的实部和虚部

### 测试方法

1. **完整验证**：对矩阵 A：
   - 计算 `{WR, WI, VR} = DGEEV(A)`
   - 对每个特征值 `k`（实数值），验证 `A * VR[:,k] ≈ WR[k] * VR[:,k]`
   - 对复特征值对，验证 `A * (VR[:,k] + i*VR[:,k+1]) ≈ λ * (VR[:,k] + i*VR[:,k+1])`
2. **批量基准**：8 个已有测试矩阵全部通过
3. **随机基准**：50 个随机矩阵（5×5 到 50×50），与 scipy 结果对比（`max_err < 1e-10`）

---

## 附录：验证工具

所有阶段使用统一的精度度量：

```elixir
def assert_all_close_scipy(a, b, tol \\ 1.0e-10) do
  # scipy 参考数据由 Python 生成，保存为 JSON
  # 在与 scipy 相同的排序约定下排序特征值
  # 计算 max |a_i - b_i|
end
```

**测试行为金字塔：**

```
        /\
       /  \        单元测试（每个 BLAS 函数）
      /    \
     /      \      集成测试（每个 LAPACK 阶段）
    /        \
   /          \    系统测试（完整 DGEEV vs scipy）
  /            \
 /   回归测试   \  随机矩阵 + 已知矩阵
```

每个阶段完成后，其输出必须通过该阶段的测试，并保持之前所有阶段的测试通过。
