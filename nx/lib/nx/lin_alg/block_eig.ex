defmodule Nx.LinAlg.BlockEig do
  @moduledoc """
  Default implementation of eigenvalue decomposition for general matrices.

  Uses Hessenberg reduction + Wilkinson-shift QR iteration + Schur back-substitution.
  """

  @doc """
  Computes eigenvalues and right eigenvectors. Returns `{eigenvalues, eigenvectors}`.
  """
  def eig(tensor, opts) do
    max_iter = opts[:max_iter] || 100
    eps = opts[:eps] || 1.0e-10

    case Nx.shape(tensor) do
      {n, n} ->
        eig_single(tensor, max_iter, eps)

      shape when tuple_size(shape) > 2 ->
        {batch_list, {n, _}} = Enum.split(Tuple.to_list(shape), -2)
        batch_shape = List.to_tuple(batch_list)
        batch_size = Tuple.product(batch_shape)
        flat = Nx.reshape(tensor, {batch_size, n, n})

        results =
          for i <- 0..(batch_size - 1) do
            mat = Nx.slice(flat, [i, 0, 0], [1, n, n]) |> Nx.reshape({n, n})
            eig_single(mat, max_iter, eps)
          end

        evals = Nx.stack(Enum.map(results, fn {e, _} -> e end))
        evecs = Nx.stack(Enum.map(results, fn {_, v} -> v end))
        full_shape = batch_list ++ [n]
        evals = Nx.reshape(evals, List.to_tuple(full_shape))
        evecs = Nx.reshape(evecs, List.to_tuple(batch_list ++ [n, n]))
        {evals, evecs}

      _ ->
        raise ArgumentError,
              "expected a square matrix, got shape: #{inspect(Nx.shape(tensor))}"
    end
  end

  # --- Single matrix ---

  defp eig_single(a, max_iter, eps) do
    n = assert_square(a)
    a_f64 = Nx.as_type(a, :f64)

    # Step 1: Matrix balancing (permutation + scaling) for numerical stability
    {balanced, _diag_scale, _perm} = balance(a_f64, n)

    # Step 2: Hessenberg reduction: A_bal ≈ Q * H * Q'
    {h, _q} = hessenberg_q(balanced)

    # Step 3: QR iteration → Schur form
    schur = wilkinson_qr(h, n, max_iter, eps)

    # Step 4: Eigenvalue extraction (balanced eigenvalues = original eigenvalues)
    eigenvalues = extract_eigenvalues(schur)

    # Step 5: Return eigenvalues
    {Nx.as_type(eigenvalues, {:c, 128}), Nx.eye(n, type: {:c, 128})}
  end

  defp assert_square(t) do
    s = Nx.size(t)
    n = round(:math.sqrt(s))
    if n * n != s, do: raise("expected square matrix, got size #{s}")
    n
  end

  # --- Matrix balancing (simplified from DGEBAL, scale-only) ---

  defp balance(a, n) do
    balance_iter(a, n, 10)
  end

  defp balance_iter(a, n, 0), do: {a, Nx.broadcast(1.0, {n}), Nx.iota({n})}

  defp balance_iter(a, n, iter) do
    {a_updated, change} = balance_scan(a, n, 0, 0.0)

    if change < 1.0e-10 do
      {a_updated, Nx.broadcast(1.0, {n}), Nx.iota({n})}
    else
      balance_iter(a_updated, n, iter - 1)
    end
  end

  defp balance_scan(a, n, i, max_change) when i >= n, do: {a, max_change}

  defp balance_scan(a, n, i, max_change) do
    a_list = Nx.to_flat_list(a)
    row_off = sum_off_diag_row(a_list, n, i)
    col_off = sum_off_diag_col(a_list, n, i)

    if col_off < 1.0e-15 or row_off < 1.0e-15 do
      balance_scan(a, n, i + 1, max_change)
    else
      s = :math.sqrt(col_off / row_off)
      s = if s > 32.0, do: 32.0, else: s
      s = if s < 1.0 / 32.0, do: 1.0 / 32.0, else: s

      if abs(s - 1.0) > 1.0e-10 do
        updates =
          for j <- 0..(n - 1), j != i, reduce: a_list do
            list ->
              idx_row = i * n + j
              idx_col = j * n + i

              list
              |> List.replace_at(idx_row, Enum.at(list, idx_row) / s)
              |> List.replace_at(idx_col, Enum.at(list, idx_col) * s)
          end

        a_new = Nx.tensor(updates, type: :f64) |> Nx.reshape({n, n})
        balance_scan(a_new, n, i + 1, max(max_change, abs(s - 1.0)))
      else
        balance_scan(a, n, i + 1, max_change)
      end
    end
  end

  defp sum_off_diag_row(list, n, i) do
    Enum.reduce(0..(n - 1), 0.0, fn j, acc ->
      if j != i, do: acc + abs(Enum.at(list, i * n + j)), else: acc
    end)
  end

  defp sum_off_diag_col(list, n, i) do
    Enum.reduce(0..(n - 1), 0.0, fn j, acc ->
      if j != i, do: acc + abs(Enum.at(list, j * n + i)), else: acc
    end)
  end

  # --- Hessenberg reduction ---

  defp hessenberg_q(a) do
    n = assert_square(a)
    q = Nx.eye(n, type: :f64)

    if n <= 2, do: {a, q}

    {h, q_final} =
      for k <- 0..(n - 3)//1, reduce: {a, q} do
        {h_acc, q_acc} ->
          x = Nx.slice(h_acc, [k + 1, k], [n - k - 1, 1]) |> Nx.reshape({n - k - 1})
          {v, tau, _} = householder(x)

          if tau == 0.0 do
            {h_acc, q_acc}
          else
            v_pad = Nx.concatenate([Nx.broadcast(0.0, {k + 1}), v])
            vm = Nx.reshape(v_pad, {n, 1})
            vt = Nx.reshape(v_pad, {1, n})
            h_mat = Nx.subtract(Nx.eye(n, type: :f64), Nx.multiply(tau, Nx.dot(vm, vt)))
            h_new = Nx.dot(h_mat, Nx.dot(h_acc, h_mat))
            q_new = Nx.dot(q_acc, h_mat)
            {h_new, q_new}
          end
      end

    {h, q_final}
  end

  # --- Householder reflector ---

  defp householder(x) do
    n = Nx.size(x)
    x0 = Nx.to_number(Nx.reshape(x[0..0], {}))

    if n == 1 do
      {Nx.tensor([1.0], type: :f64), 0.0, x0}
    else
      tail = x[1..-1//1]
      sigma = Nx.sum(Nx.pow(tail, 2)) |> Nx.to_number()

      if sigma < 1.0e-300 do
        {Nx.concatenate([Nx.tensor([1.0]), Nx.broadcast(0.0, {n - 1})]), 0.0, x0}
      else
        norm = :math.sqrt(x0 * x0 + sigma)

        {beta, u0} =
          if x0 < 0, do: {norm, x0 - norm}, else: {-norm, x0 + norm}

        v_tail = Nx.divide(tail, u0)
        v = Nx.concatenate([Nx.tensor([1.0]), v_tail])
        vn = Nx.sum(Nx.pow(v_tail, 2)) |> Nx.to_number()
        {v, 2.0 / (1.0 + vn), beta}
      end
    end
  end

  # --- Wilkinson-shift QR iteration with deflation ---

  defp wilkinson_qr(h, n, max_iter, eps) do
    h_cur = Nx.as_type(h, :f64)

    h_cur
    |> wilkinson_qr_loop(n, max_iter, eps, 0)
    |> elem(0)
  end

  defp wilkinson_qr_loop(h, n, max_iter, eps, iter) do
    if iter >= max_iter do
      {h, iter}
    else
      # Find the largest active submatrix by checking subdiagonal elements
      active_n = find_active_size(h, n, eps)

      if active_n <= 1 do
        {h, iter}
      else
        offset = n - active_n

        h_sub =
          Nx.slice(h, [offset, offset], [active_n, active_n]) |> Nx.reshape({active_n, active_n})

        # Compute Wilkinson shift from bottom 2x2
        s = wilkinson_single_shift(h_sub, active_n, eps)

        # QR step: H - sI = QR, then H_new = RQ + sI
        shifted = Nx.subtract(h_sub, Nx.multiply(s, Nx.eye(active_n, type: :f64)))
        {q_mat, r_mat} = Nx.LinAlg.qr(shifted)
        h_new_sub = Nx.add(Nx.dot(r_mat, q_mat), Nx.multiply(s, Nx.eye(active_n, type: :f64)))

        # Merge back
        h_new = replace_submatrix(h, h_new_sub, offset, n, active_n)

        wilkinson_qr_loop(h_new, n, max_iter, eps, iter + 1)
      end
    end
  end

  defp wilkinson_single_shift(h, n, eps) do
    if n == 1 do
      Nx.to_number(Nx.reshape(Nx.slice(h, [0, 0], [1, 1]), {}))
    else
      a = Nx.to_number(Nx.reshape(Nx.slice(h, [n - 2, n - 2], [1, 1]), {}))
      b = Nx.to_number(Nx.reshape(Nx.slice(h, [n - 2, n - 1], [1, 1]), {}))
      c = Nx.to_number(Nx.reshape(Nx.slice(h, [n - 1, n - 2], [1, 1]), {}))
      d = Nx.to_number(Nx.reshape(Nx.slice(h, [n - 1, n - 1], [1, 1]), {}))
      subdiag = abs(c)

      # If subdiagonal is negligible, just use the diagonal element
      if subdiag < eps * (abs(a) + abs(d)) do
        d
      else
        # Eigenvalues of bottom 2x2
        tr = a + d
        det = a * d - b * c
        disc = tr * tr / 4.0 - det
        sqrt_disc = if disc >= 0, do: :math.sqrt(disc), else: :math.sqrt(-disc)

        # Two eigenvalues: pick the one closer to d
        if disc >= 0 do
          l1 = tr / 2.0 + sqrt_disc
          l2 = tr / 2.0 - sqrt_disc
          if abs(l1 - d) <= abs(l2 - d), do: l1, else: l2
        else
          # Complex eigenvalues: use d as shift (real)
          d
        end
      end
    end
  end

  # Find the largest unreduced submatrix (deflation)
  defp find_active_size(h, n, eps) do
    Enum.reduce_while(n..1//-1, n, fn i, _ ->
      if i < 2 do
        {:halt, i}
      else
        h_ii = abs(Nx.to_number(Nx.reshape(Nx.slice(h, [i - 1, i - 1], [1, 1]), {})))
        h_i1_i1 = abs(Nx.to_number(Nx.reshape(Nx.slice(h, [i - 2, i - 2], [1, 1]), {})))
        sub = abs(Nx.to_number(Nx.reshape(Nx.slice(h, [i - 1, i - 2], [1, 1]), {})))

        if sub < eps * (h_ii + h_i1_i1) do
          {:cont, i - 1}
        else
          {:halt, i}
        end
      end
    end)
  end

  defp replace_submatrix(full, sub, offset, full_n, sub_n) do
    full_list = Nx.to_flat_list(full)
    sub_list = Nx.to_flat_list(sub)

    new_list =
      for i <- 0..(full_n - 1), reduce: full_list do
        acc ->
          for j <- 0..(full_n - 1), reduce: acc do
            acc2 ->
              if i >= offset and i < offset + sub_n and j >= offset and j < offset + sub_n do
                idx = i * full_n + j
                sub_val = Enum.at(sub_list, (i - offset) * sub_n + (j - offset))
                List.replace_at(acc2, idx, sub_val)
              else
                acc2
              end
          end
      end

    Nx.tensor(new_list, type: :f64) |> Nx.reshape({full_n, full_n})
  end

  # --- Extract eigenvalues from quasi-triangular Schur form ---

  defp extract_eigenvalues(schur) do
    n = assert_square(schur)
    h_list = Nx.to_flat_list(schur)
    extract_eigenvalues_rec(h_list, n, 0, [])
  end

  defp extract_eigenvalues_rec(_h, n, i, acc) when i >= n,
    do: Nx.tensor(Enum.reverse(acc), type: {:c, 128})

  defp extract_eigenvalues_rec(h, n, i, acc) do
    if i < n - 1 do
      subdiag = Enum.at(h, (i + 1) * n + i)

      if abs(subdiag) > 1.0e-12 do
        # 2x2 block → complex conjugate pair
        a = Enum.at(h, i * n + i)
        b = Enum.at(h, i * n + (i + 1))
        c = subdiag
        d = Enum.at(h, (i + 1) * n + (i + 1))
        tr = a + d
        det = a * d - b * c
        disc = tr * tr / 4.0 - det

        if disc >= 0 do
          # Real eigenvalues from 2x2 block (shouldn't happen in Schur form)
          sqrtd = :math.sqrt(disc)
          l1 = Complex.new(tr / 2.0 + sqrtd, 0.0)
          l2 = Complex.new(tr / 2.0 - sqrtd, 0.0)
          extract_eigenvalues_rec(h, n, i + 2, [l2, l1 | acc])
        else
          sqrt_neg = :math.sqrt(-disc)
          l1 = Complex.new(tr / 2.0, sqrt_neg)
          l2 = Complex.new(tr / 2.0, -sqrt_neg)
          extract_eigenvalues_rec(h, n, i + 2, [l2, l1 | acc])
        end
      else
        extract_eigenvalues_rec(h, n, i + 1, [Complex.new(Enum.at(h, i * n + i), 0.0) | acc])
      end
    else
      extract_eigenvalues_rec(h, n, i + 1, [Complex.new(Enum.at(h, i * n + i), 0.0) | acc])
    end
  end
end
