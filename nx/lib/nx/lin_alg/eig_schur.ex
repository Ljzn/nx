defmodule Nx.LinAlg.EigSchur do
  @moduledoc """
  Small-matrix QR eigenvalue solver and 2x2 Schur factorization.

  Ported from LAPACK DLAHQR and DLANV2.
  """

  # ──────────────────────────────────
  #  DLANV2: 2x2 Schur factorization
  # ──────────────────────────────────

  @doc """
  DLANV2: Reduce a real 2x2 matrix [A B; C D] to standard Schur form.

  Returns {a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn}.
  """
  def dlanv2(a, b, c, d) do
    safmin = Nx.LinAlg.EigUtil.dlamch("S")
    eps = Nx.LinAlg.EigUtil.dlamch("P")
    base = Nx.LinAlg.EigUtil.dlamch("B")
    safmn2 = :math.pow(base, trunc(:math.log(safmin / eps) / :math.log(base) / 2.0))
    safmx2 = 1.0 / safmn2
    multipl = 4.0

    {a_out, b_out, c_out, d_out, cs, sn} =
      cond do
        c == 0.0 ->
          {a, b, c, d, 1.0, 0.0}

        b == 0.0 ->
          {d, -c, 0.0, a, 0.0, 1.0}

        a - d == 0.0 and lapack_sign(1.0, b) != lapack_sign(1.0, c) ->
          {a, b, c, d, 1.0, 0.0}

        true ->
          dlanv2_rotation(a, b, c, d, safmin, eps, safmn2, safmx2, multipl)
      end

    # Compute eigenvalues from the Schur form (same for all branches)
    rt1r = a_out
    rt2r = d_out

    {rt1i, rt2i} =
      if c_out == 0.0 do
        {0.0, 0.0}
      else
        img = :math.sqrt(abs(b_out)) * :math.sqrt(abs(c_out))
        {img, -img}
      end

    {a_out, b_out, c_out, d_out, rt1r, rt1i, rt2r, rt2i, cs, sn}
  end

  defp dlanv2_rotation(a, b, c, d, safmin, eps, safmn2, safmx2, multipl) do
    temp = a - d
    p = 0.5 * temp
    bcmax = max(abs(b), abs(c))
    bcmis_val = min(abs(b), abs(c)) * lapack_sign(1.0, b) * lapack_sign(1.0, c)
    scale = max(abs(p), bcmax)
    z = p / scale * p + bcmax / scale * bcmis_val

    if z >= multipl * eps do
      # Real eigenvalues
      sqrt_scale = :math.sqrt(scale)
      sqrt_z = :math.sqrt(z)
      z_sign = p + lapack_sign(sqrt_scale * sqrt_z, p)
      a_new = d + z_sign
      d_new = d - bcmax / z_sign * bcmis_val
      tau = Nx.LinAlg.EigUtil.dlapy2(c, z_sign)
      cs = z_sign / tau
      sn = c / tau
      {a_new, b - c, 0.0, d_new, cs, sn}
    else
      # Complex eigenvalues or almost-equal real eigenvalues
      sigma = b + c
      {sigma_sc, temp_sc} = scale_loop(temp, sigma, safmn2, safmx2, 0)

      p2 = 0.5 * temp_sc
      tau = Nx.LinAlg.EigUtil.dlapy2(sigma_sc, temp_sc)
      cs = :math.sqrt(0.5 * (1.0 + abs(sigma_sc) / tau))
      sn = -(p2 / (tau * cs)) * lapack_sign(1.0, sigma_sc)

      # [AA BB; CC DD] = [A B; C D] * [CS -SN; SN CS]
      aa = a * cs + b * sn
      bb = -a * sn + b * cs
      cc = c * cs + d * sn
      dd = -c * sn + d * cs

      # [A B; C D] = [CS SN; -SN CS] * [AA BB; CC DD]
      a_out = aa * cs + cc * sn
      b_out = bb * cs + dd * sn
      c_out = -(aa * sn) + cc * cs
      d_out = -bb * sn + dd * cs

      temp_mid = 0.5 * (a_out + d_out)
      a_final = temp_mid
      d_final = temp_mid

      if c_out != 0.0 do
        if b_out != 0.0 do
          if lapack_sign(1.0, b_out) == lapack_sign(1.0, c_out) do
            sab = :math.sqrt(abs(b_out))
            sac = :math.sqrt(abs(c_out))
            p_val = lapack_sign(sab * sac, c_out)
            tau_rt = 1.0 / :math.sqrt(abs(b_out + c_out))
            a_tri = temp_mid + p_val
            d_tri = temp_mid - p_val
            b_tri = b_out - c_out
            cs1 = sab * tau_rt
            sn1 = sac * tau_rt
            {a_tri, b_tri, 0.0, d_tri, cs * cs1 - sn * sn1, cs * sn1 + sn * cs1}
          else
            {a_final, b_out, c_out, d_final, cs, sn}
          end
        else
          {a_final, -c_out, 0.0, d_final, -sn, cs}
        end
      else
        {a_final, b_out, c_out, d_final, cs, sn}
      end
    end
  end

  defp scale_loop(temp, sigma, _safmn2, _safmx2, count) when count >= 20, do: {sigma, temp}

  defp scale_loop(temp, sigma, safmn2, safmx2, count) do
    scale = max(abs(temp), abs(sigma))

    cond do
      scale >= safmx2 ->
        scale_loop(temp * safmn2, sigma * safmn2, safmn2, safmx2, count + 1)

      scale <= safmn2 ->
        scale_loop(temp * safmx2, sigma * safmx2, safmn2, safmx2, count + 1)

      true ->
        {sigma, temp}
    end
  end

  @doc false
  def lapack_sign(x, y) do
    if y >= 0, do: abs(x), else: -abs(x)
  end

  # ──────────────────────────────────
  #  DLAHQR: small-matrix QR algorithm
  # ──────────────────────────────────

  @kexsh 10

  @doc """
  DLAHQR: QR algorithm for Hessenberg matrices (N <= 75).

  Options:
    :wantt - accumulate Schur form in H (default false)
    :wantz - accumulate Schur vectors in Z (default false)
    :ilo, :ihi - active submatrix bounds (0-indexed, inclusive)
    :iloz, :ihiz - Z update bounds

  Returns {h, wr, wi, z, info}.
  """
  def dlahqr(h, opts \\ []) do
    wantt = opts[:wantt] || false
    wantz = opts[:wantz] || false
    n = assert_square(h)
    ilo = opts[:ilo] || 0
    ihi = opts[:ihi] || n - 1
    iloz = opts[:iloz] || 0
    ihiz = opts[:ihiz] || n - 1

    h_list = Nx.to_flat_list(h)
    wr = List.duplicate(0.0, n)
    wi = List.duplicate(0.0, n)
    z_init = if wantz, do: Nx.to_flat_list(Nx.eye(n, type: :f64)), else: []

    safmin = Nx.LinAlg.EigUtil.dlamch("S")
    ulp = Nx.LinAlg.EigUtil.dlamch("P")
    safmax = 1.0 / safmin
    nh = ihi - ilo + 1
    smlnum = safmin * (nh / ulp)
    itmax = 30 * max(10, nh)

    {h_final, wr_final, wi_final, z_final, info} =
      dlahqr_main(
        h_list,
        wr,
        wi,
        z_init,
        n,
        ihi,
        ihi,
        ilo,
        wantt,
        wantz,
        iloz,
        ihiz,
        safmin,
        safmax,
        ulp,
        smlnum,
        itmax
      )

    h_out = Nx.tensor(h_final, type: :f64) |> Nx.reshape({n, n})
    wr_t = Nx.tensor(wr_final, type: :f64)
    wi_t = Nx.tensor(wi_final, type: :f64)

    z_out =
      if wantz,
        do: Nx.tensor(z_final, type: :f64) |> Nx.reshape({n, n}),
        else: Nx.eye(n, type: :f64)

    {h_out, wr_t, wi_t, z_out, info}
  end

  defp dlahqr_main(
         h,
         wr,
         wi,
         z,
         n,
         i,
         ihi,
         ilo,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         safmax,
         ulp,
         smlnum,
         itmax
       )
       when i < ilo do
    {h, wr, wi, z, 0}
  end

  defp dlahqr_main(
         h,
         wr,
         wi,
         z,
         n,
         i,
         ihi,
         ilo,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         safmax,
         ulp,
         smlnum,
         itmax
       ) do
    {h, wr, wi, z, l, its_fail} =
      qr_loop(
        h,
        wr,
        wi,
        z,
        n,
        i,
        ihi,
        ilo,
        wantt,
        wantz,
        iloz,
        ihiz,
        safmin,
        safmax,
        ulp,
        smlnum,
        itmax,
        0
      )

    if its_fail do
      {h, wr, wi, z, i}
    else
      {h2, wr2, wi2} =
        if l == i do
          {h, List.replace_at(wr, i, Enum.at(h, i * n + i)), List.replace_at(wi, i, 0.0)}
        else
          a11 = Enum.at(h, (i - 1) * n + (i - 1))
          a12 = Enum.at(h, (i - 1) * n + i)
          a21 = Enum.at(h, i * n + (i - 1))
          a22 = Enum.at(h, i * n + i)

          {aa, bb, cc, dd, rt1r, rt1i, rt2r, rt2i, cs, sn} = dlanv2(a11, a12, a21, a22)

          h1 =
            h
            |> List.replace_at((i - 1) * n + (i - 1), aa)
            |> List.replace_at((i - 1) * n + i, bb)
            |> List.replace_at(i * n + (i - 1), cc)
            |> List.replace_at(i * n + i, dd)

          w1 = wr |> List.replace_at(i - 1, rt1r) |> List.replace_at(i, rt2r)
          w2 = wi |> List.replace_at(i - 1, rt1i) |> List.replace_at(i, rt2i)

          h2 =
            if wantt do
              h1_tmp =
                if n - 1 > i, do: drot_cols(h1, n, i - 1, i, i + 1, n - 1, cs, sn), else: h1

              drot_rows(h1_tmp, n, 0, i - 2, i - 1, i, cs, sn)
            else
              h1
            end

          h3 = if wantz, do: drot_z(h2, n, iloz, ihiz, i - 1, i, cs, sn), else: h2

          {h3, w1, w2}
        end

      dlahqr_main(
        h2,
        wr2,
        wi2,
        z,
        n,
        l - 1,
        ihi,
        ilo,
        wantt,
        wantz,
        iloz,
        ihiz,
        safmin,
        safmax,
        ulp,
        smlnum,
        itmax
      )
    end
  end

  defp qr_loop(
         h,
         wr,
         wi,
         z,
         n,
         i,
         ihi,
         ilo,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         safmax,
         ulp,
         smlnum,
         itmax,
         its
       )
       when its > itmax do
    {h, wr, wi, z, i, true}
  end

  defp qr_loop(
         h,
         wr,
         wi,
         z,
         n,
         i,
         ihi,
         ilo,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         safmax,
         ulp,
         smlnum,
         itmax,
         its
       ) do
    # Scan for small subdiagonal (deflation)
    k = do_deflation_scan(h, n, i, ilo, smlnum, ulp, i)
    l = k

    h = if l > ilo, do: List.replace_at(h, l * n + (l - 1), 0.0), else: h

    if l >= i - 1 do
      {h, wr, wi, z, l, false}
    else
      # LAPACK KDEFL: iterations since last deflation, incremented before shift
      kdef = its + 1
      {h11, h12, h21, h22} = shift_window(h, n, i, l, kdef)
      {rt1r, rt1i, rt2r, rt2i} = wilkinson_shift(h11, h12, h21, h22)

      # Find bulge start
      m = bulge_start(h, n, i, l, rt1r, rt1i, rt2r, rt2i, ulp)

      # Bulge chase with l and i passed as context
      {h, z} =
        bulge_chase_main(h, z, n, m, i, l, wantt, wantz, iloz, ihiz, rt1r, rt1i, rt2r, rt2i)

      qr_loop(
        h,
        wr,
        wi,
        z,
        n,
        i,
        ihi,
        ilo,
        wantt,
        wantz,
        iloz,
        ihiz,
        safmin,
        safmax,
        ulp,
        smlnum,
        itmax,
        its + 1
      )
    end
  end

  # ── Deflation scan ──

  defp do_deflation_scan(_h, _n, k, l, _smlnum, _ulp, _i) when k <= l, do: k

  defp do_deflation_scan(h, n, k, l, smlnum, ulp, i) when k <= l, do: k

  defp do_deflation_scan(h, n, k, l, smlnum, ulp, i) do
    h_kk_m1 = abs(Enum.at(h, k * n + (k - 1)))

    if h_kk_m1 <= smlnum do
      k
    else
      h_km1_km1 = Enum.at(h, (k - 1) * n + (k - 1))
      h_kk = Enum.at(h, k * n + k)
      tst = abs(h_km1_km1) + abs(h_kk)

      tst =
        if tst == 0.0 do
          tst = if k - 2 >= l, do: tst + abs(Enum.at(h, (k - 1) * n + (k - 2))), else: tst
          if k + 1 <= i, do: tst + abs(Enum.at(h, (k + 1) * n + k)), else: tst
        else
          tst
        end

      if h_kk_m1 <= ulp * tst do
        ab = max(h_kk_m1, abs(Enum.at(h, (k - 1) * n + k)))
        ba = min(h_kk_m1, abs(Enum.at(h, (k - 1) * n + k)))
        aa = max(abs(h_kk), abs(h_km1_km1 - h_kk))
        bb_val = min(abs(h_kk), abs(h_km1_km1 - h_kk))
        s_val = aa + ab

        if ba * (ab / s_val) <= max(smlnum, ulp * (bb_val * (aa / s_val))) do
          k
        else
          do_deflation_scan(h, n, k - 1, l, smlnum, ulp, i)
        end
      else
        do_deflation_scan(h, n, k - 1, l, smlnum, ulp, i)
      end
    end
  end

  # ── Shift computation ──

  defp shift_window(h, n, i, l, kdef) do
    if rem(kdef, 20) == 0 do
      s = abs(Enum.at(h, i * n + (i - 1))) + abs(Enum.at(h, (i - 1) * n + (i - 2)))
      {0.75 * s + Enum.at(h, i * n + i), -0.4375 * s, s, 0.75 * s + Enum.at(h, i * n + i)}
    else
      if rem(kdef, 10) == 0 do
        s = abs(Enum.at(h, (l + 1) * n + l)) + abs(Enum.at(h, (l + 2) * n + (l + 1)))
        {0.75 * s + Enum.at(h, l * n + l), -0.4375 * s, s, 0.75 * s + Enum.at(h, l * n + l)}
      else
        {Enum.at(h, (i - 1) * n + (i - 1)), Enum.at(h, (i - 1) * n + i),
         Enum.at(h, i * n + (i - 1)), Enum.at(h, i * n + i)}
      end
    end
  end

  defp wilkinson_shift(h11, h12, h21, h22) do
    s = abs(h11) + abs(h12) + abs(h21) + abs(h22)

    if s == 0.0 do
      {0.0, 0.0, 0.0, 0.0}
    else
      h11s = h11 / s
      h21s = h21 / s
      h12s = h12 / s
      h22s = h22 / s
      tr = (h11s + h22s) / 2.0
      det = (h11s - tr) * (h22s - tr) - h12s * h21s
      rtdisc = :math.sqrt(abs(det))

      if det >= 0 do
        {tr * s, rtdisc * s, tr * s, -rtdisc * s}
      else
        rt = tr + rtdisc

        if abs(rt - h22s) <= abs(tr - rtdisc - h22s) do
          {rt * s, 0.0, rt * s, 0.0}
        else
          rt2 = (tr - rtdisc) * s
          {rt2, 0.0, rt2, 0.0}
        end
      end
    end
  end

  # ── Bulge start ──
  # Returns {m, v} where m is the start column and v is the initial bulge vector

  defp bulge_start(h, n, i, l, rt1r, rt1i, rt2r, rt2i, ulp) do
    do_bulge_start(h, n, i - 2, l, i, rt1r, rt1i, rt2r, rt2i, ulp)
  end

  defp do_bulge_start(_h, _n, m, l, _i, _r1r, _r1i, _r2r, _r2i, _ulp) when m < l, do: m

  defp do_bulge_start(h, n, m, l, i, rt1r, rt1i, rt2r, rt2i, ulp) do
    h_m_m = Enum.at(h, m * n + m)
    h_mp1_m = Enum.at(h, (m + 1) * n + m)
    h_m_mp1 = Enum.at(h, m * n + (m + 1))
    h_mp1_mp1 = Enum.at(h, (m + 1) * n + (m + 1))
    h_mp2_mp1 = if m + 2 <= i, do: Enum.at(h, (m + 2) * n + (m + 1)), else: 0.0

    h21s = h_mp1_m
    s = abs(h_m_m - rt2r) + abs(rt2i) + abs(h21s)
    h21s_s = h21s / s

    v1 =
      h21s_s * h_m_mp1 +
        (h_m_m - rt1r) * ((h_m_m - rt2r) / s) - rt1i * (rt2i / s)

    v2 = h21s_s * (h_m_m + h_mp1_mp1 - rt1r - rt2r)
    v3 = h21s_s * h_mp2_mp1
    s_norm = abs(v1) + abs(v2) + abs(v3)
    v1s = v1 / s_norm

    if m == l do
      m
    else
      h_mm_m1 = abs(Enum.at(h, m * n + (m - 1)))
      v_norm_test = (abs(v2) + abs(v3)) / s_norm
      h_mp1_mp1_abs = abs(h_mp1_mp1)
      h_m2_m2_abs = if m + 2 <= i, do: abs(h_mp2_mp1), else: 0.0
      h_m1_m1 = abs(Enum.at(h, (m - 1) * n + (m - 1)))

      if h_mm_m1 * v_norm_test <= ulp * abs(v1s) * (h_m1_m1 + h_mp1_mp1_abs + h_m2_m2_abs) do
        m
      else
        do_bulge_start(h, n, m - 1, l, i, rt1r, rt1i, rt2r, rt2i, ulp)
      end
    end
  end

  # ── Bulge chase ──

  defp bulge_chase_main(h, z, n, m, i, l, wantt, wantz, iloz, ihiz, rt1r, rt1i, rt2r, rt2i) do
    # Compute initial bulge vector from shift (LAPACK DO 60 before bulge chase)
    h_mp1_m = Enum.at(h, (m + 1) * n + m)
    s = abs(Enum.at(h, m * n + m) - rt2r) + abs(rt2i) + abs(h_mp1_m)
    h21s_s = h_mp1_m / s

    v1 =
      h21s_s * Enum.at(h, m * n + (m + 1)) +
        (Enum.at(h, m * n + m) - rt1r) * ((Enum.at(h, m * n + m) - rt2r) / s) -
        rt1i * (rt2i / s)

    v2 = h21s_s * (Enum.at(h, m * n + m) + Enum.at(h, (m + 1) * n + (m + 1)) - rt1r - rt2r)
    v3 = if m + 2 <= i, do: h21s_s * Enum.at(h, (m + 2) * n + (m + 1)), else: 0.0
    s_norm = abs(v1) + abs(v2) + abs(v3)
    v_norm = [v1 / s_norm, v2 / s_norm, v3 / s_norm]

    bulge_chase_k(h, z, n, m, i, l, wantt, wantz, iloz, ihiz, v_norm, m)
  end

  defp bulge_chase_k(h, z, n, k, i, l, wantt, wantz, iloz, ihiz, v, _orig_m) when k >= i do
    {h, z}
  end

  defp bulge_chase_k(h, z, n, k, i, l, wantt, wantz, iloz, ihiz, v, orig_m) do
    nr = min(3, i - k + 1)

    v_t =
      if nr >= 3,
        do: Nx.tensor([v |> Enum.at(0), v |> Enum.at(1), v |> Enum.at(2)], type: :f64),
        else: Nx.tensor([v |> Enum.at(0), v |> Enum.at(1)], type: :f64)

    {v_ref, t1, _beta} = Nx.LinAlg.EigHouseholder.dlarfg(v_t)
    v_f = Nx.to_flat_list(v_ref)
    v1 = Enum.at(v_f, 0)

    # Store reflector back (must propagate h)
    # LAPACK: IF K > M → store v; ELSE IF M > L → scale H(K, K-1) by (1-t1)
    h1 =
      if k > orig_m do
        h
        |> List.replace_at(k * n + (k - 1), v1)
        |> List.replace_at((k + 1) * n + (k - 1), 0.0)
        |> then(fn acc ->
          if nr == 3 and k < i - 1 do
            List.replace_at(acc, (k + 2) * n + (k - 1), 0.0)
          else
            acc
          end
        end)
      else
        # k == orig_m: scale only when orig_m > l (not when M == L)
        if orig_m > l and k * n + (k - 1) >= 0 do
          List.replace_at(h, k * n + (k - 1), Enum.at(h, k * n + (k - 1)) * (1.0 - t1))
        else
          h
        end
      end

    v2 = if nr >= 2, do: Enum.at(v_f, 1), else: 0.0
    t2 = t1 * v2

    {h2, z2} =
      if nr == 3 do
        v3 = Enum.at(v_f, 2)
        t3 = t1 * v3

        h_l =
          Enum.reduce(k..i, h1, fn j, acc ->
            sum =
              Enum.at(acc, k * n + j) + v2 * Enum.at(acc, (k + 1) * n + j) +
                v3 * Enum.at(acc, (k + 2) * n + j)

            acc
            |> List.replace_at(k * n + j, Enum.at(acc, k * n + j) - sum * t1)
            |> List.replace_at((k + 1) * n + j, Enum.at(acc, (k + 1) * n + j) - sum * t2)
            |> List.replace_at((k + 2) * n + j, Enum.at(acc, (k + 2) * n + j) - sum * t3)
          end)

        j_hi = min(k + 3, i)

        h_r =
          Enum.reduce(0..j_hi, h_l, fn j, acc ->
            sum =
              Enum.at(acc, j * n + k) + v2 * Enum.at(acc, j * n + (k + 1)) +
                v3 * Enum.at(acc, j * n + (k + 2))

            acc
            |> List.replace_at(j * n + k, Enum.at(acc, j * n + k) - sum * t1)
            |> List.replace_at(j * n + (k + 1), Enum.at(acc, j * n + (k + 1)) - sum * t2)
            |> List.replace_at(j * n + (k + 2), Enum.at(acc, j * n + (k + 2)) - sum * t3)
          end)

        z_out =
          if wantz do
            Enum.reduce(iloz..ihiz, z, fn j, acc ->
              sum =
                Enum.at(acc, j * n + k) + v2 * Enum.at(acc, j * n + (k + 1)) +
                  v3 * Enum.at(acc, j * n + (k + 2))

              acc
              |> List.replace_at(j * n + k, Enum.at(acc, j * n + k) - sum * t1)
              |> List.replace_at(j * n + (k + 1), Enum.at(acc, j * n + (k + 1)) - sum * t2)
              |> List.replace_at(j * n + (k + 2), Enum.at(acc, j * n + (k + 2)) - sum * t3)
            end)
          else
            z
          end

        {h_r, z_out}
      else
        # NR = 2
        h_l =
          Enum.reduce(k..i, h1, fn j, acc ->
            sum = Enum.at(acc, k * n + j) + v2 * Enum.at(acc, (k + 1) * n + j)

            acc
            |> List.replace_at(k * n + j, Enum.at(acc, k * n + j) - sum * t1)
            |> List.replace_at((k + 1) * n + j, Enum.at(acc, (k + 1) * n + j) - sum * t2)
          end)

        h_r =
          Enum.reduce(0..i, h_l, fn j, acc ->
            sum = Enum.at(acc, j * n + k) + v2 * Enum.at(acc, j * n + (k + 1))

            acc
            |> List.replace_at(j * n + k, Enum.at(acc, j * n + k) - sum * t1)
            |> List.replace_at(j * n + (k + 1), Enum.at(acc, j * n + (k + 1)) - sum * t2)
          end)

        z_out =
          if wantz do
            Enum.reduce(iloz..ihiz, z, fn j, acc ->
              sum = Enum.at(acc, j * n + k) + v2 * Enum.at(acc, j * n + (k + 1))

              acc
              |> List.replace_at(j * n + k, Enum.at(acc, j * n + k) - sum * t1)
              |> List.replace_at(j * n + (k + 1), Enum.at(acc, j * n + (k + 1)) - sum * t2)
            end)
          else
            z
          end

        {h_r, z_out}
      end

    next_nr = min(3, i - (k + 1) + 1)

    next_v =
      cond do
        next_nr == 3 ->
          [
            Enum.at(h2, (k + 1) * n + k),
            Enum.at(h2, (k + 2) * n + k),
            Enum.at(h2, (k + 3) * n + k)
          ]

        next_nr == 2 ->
          [Enum.at(h2, (k + 1) * n + k), Enum.at(h2, (k + 2) * n + k)]

        true ->
          [Enum.at(h2, (k + 1) * n + k)]
      end

    bulge_chase_k(h2, z2, n, k + 1, i, l, wantt, wantz, iloz, ihiz, next_v, orig_m)
  end

  # ── Givens rotation helpers ──

  defp drot_cols(h, n, i, j, c_start, c_end, cs, sn) do
    Enum.reduce(c_start..c_end, h, fn c, acc ->
      hi = Enum.at(acc, i * n + c)
      hj = Enum.at(acc, j * n + c)

      acc
      |> List.replace_at(i * n + c, cs * hi + sn * hj)
      |> List.replace_at(j * n + c, -sn * hi + cs * hj)
    end)
  end

  defp drot_rows(h, n, r_start, r_end, i, j, cs, sn) do
    Enum.reduce(r_start..r_end, h, fn r, acc ->
      hi = Enum.at(acc, r * n + i)
      hj = Enum.at(acc, r * n + j)

      acc
      |> List.replace_at(r * n + i, cs * hi + sn * hj)
      |> List.replace_at(r * n + j, -sn * hi + cs * hj)
    end)
  end

  defp drot_z(z, n, iloz, ihiz, i, j, cs, sn) do
    Enum.reduce(iloz..ihiz, z, fn r, acc ->
      zi = Enum.at(acc, r * n + i)
      zj = Enum.at(acc, r * n + j)

      acc
      |> List.replace_at(r * n + i, cs * zi + sn * zj)
      |> List.replace_at(r * n + j, -sn * zi + cs * zj)
    end)
  end

  defp assert_square(t) do
    s = Nx.size(t)
    n = round(:math.sqrt(s))
    if n * n != s, do: raise("expected square matrix")
    n
  end
end
