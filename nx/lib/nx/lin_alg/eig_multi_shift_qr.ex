defmodule Nx.LinAlg.EigMultiShiftQR do
  @moduledoc """
  Multi-shift QR algorithm for eigenvalue decomposition.

  Ported from LAPACK DLAQR0-5.
  """

  # ── Fortran-indexed array helpers ──
  # Fortran uses 1-based indexing: A(i,j) = flat_list[(i-1)*n + (j-1)]

  defp f_get(h, n, i, j), do: Enum.at(h, (i - 1) * n + (j - 1))
  defp f_set(h, n, i, j, v), do: List.replace_at(h, (i - 1) * n + (j - 1), v)
  defp f_inc(h, n, i, j, dv), do: List.update_at(h, (i - 1) * n + (j - 1), &(&1 + dv))

  @doc """
  DLAQR1: Compute scaling of first column of
    (H - (sr1 + i*si1)*I) * (H - (sr2 + i*si2)*I)
  for 2x2 or 3x3 H. Returns v as list of length N.
  """
  def dlaqr1(n, h, sr1, si1, sr2, si2) do
    cond do
      n == 2 ->
        h00 = h |> Enum.at(0) |> Enum.at(0)
        h01 = h |> Enum.at(0) |> Enum.at(1)
        h10 = h |> Enum.at(1) |> Enum.at(0)

        s = abs(h00 - sr2) + abs(si2) + abs(h10)

        if s == 0.0 do
          [0.0, 0.0]
        else
          h21s = h10 / s
          v1 = h21s * h01 + (h00 - sr1) * ((h00 - sr2) / s) - si1 * (si2 / s)
          v2 = h21s * (h00 + (h |> Enum.at(1) |> Enum.at(1)) - sr1 - sr2)
          [v1, v2]
        end

      n == 3 ->
        h00 = h |> Enum.at(0) |> Enum.at(0)
        h01 = h |> Enum.at(0) |> Enum.at(1)
        h02 = h |> Enum.at(0) |> Enum.at(2)
        h10 = h |> Enum.at(1) |> Enum.at(0)
        h11 = h |> Enum.at(1) |> Enum.at(1)
        h12 = h |> Enum.at(1) |> Enum.at(2)
        h20 = h |> Enum.at(2) |> Enum.at(0)
        h21 = h |> Enum.at(2) |> Enum.at(1)
        h22 = h |> Enum.at(2) |> Enum.at(2)

        s = abs(h00 - sr2) + abs(si2) + abs(h10) + abs(h20)

        if s == 0.0 do
          [0.0, 0.0, 0.0]
        else
          h21s = h10 / s
          h31s = h20 / s
          v1 = (h00 - sr1) * ((h00 - sr2) / s) - si1 * (si2 / s) + h01 * h21s + h02 * h31s
          v2 = h21s * (h00 + h11 - sr1 - sr2) + h12 * h31s
          v3 = h31s * (h00 + h22 - sr1 - sr2) + h21s * h21
          [v1, v2, v3]
        end

      true ->
        []
    end
  end

  # ──────────────────────────────────────
  #  DLAQR5: multi-shift bulge sweep
  # ──────────────────────────────────────

  @doc """
  DLAQR5: single multi-shift QR sweep (compact non-ACCUM version).

  Ported from LAPACK DLAQR5, simplified to non-accumulating mode.
  Applies NSHFTS shifts to H(KTOP:KBOT,:).
  """
  def dlaqr5(h, z, n, ktop, kbot, nshfts, sr, si, opts \\ []) do
    wantt = opts[:wantt] || false
    wantz = opts[:wantz] || false
    iloz = opts[:iloz] || 1
    ihiz = opts[:ihiz] || n

    if nshfts < 2 or ktop >= kbot do
      {h, z}
    else
      safmin = Nx.LinAlg.EigUtil.dlamch("S")
      ulp = Nx.LinAlg.EigUtil.dlamch("P")
      smlnum = safmin * (n / ulp)

      # Shuffle shifts into pairs
      {sr, si} = shuffle_shifts(sr, si, nshfts)
      ns = nshfts - rem(nshfts, 2)
      nbmpbs = div(ns, 2)

      # V(3, nbmpbs) — reflector storage
      v = List.duplicate([0.0, 0.0, 0.0], nbmpbs)

      # Main INCOL sweep
      incol0 = ktop - 2 * nbmpbs + 1

      {h, z} =
        dlaqr5_sweep(
          h,
          z,
          n,
          ktop,
          kbot,
          sr,
          si,
          nbmpbs,
          wantt,
          wantz,
          iloz,
          ihiz,
          safmin,
          smlnum,
          ulp,
          v,
          incol0
        )

      {h, z}
    end
  end

  defp shuffle_shifts(sr, si, nsft) do
    sr_list = if is_list(sr), do: sr, else: Nx.to_flat_list(sr)
    si_list = if is_list(si), do: si, else: Nx.to_flat_list(si)

    Enum.reduce(1..(nsft - 2)//2, {sr_list, si_list}, fn i, {sr_a, si_a} ->
      if Enum.at(si_a, i - 1) != -Enum.at(si_a, i) do
        sr2 =
          sr_a
          |> List.replace_at(i - 1, Enum.at(sr_a, i))
          |> List.replace_at(i, Enum.at(sr_a, i + 1 - 1))
          |> List.replace_at(i + 1 - 1, Enum.at(sr_a, i - 1))

        si2 =
          si_a
          |> List.replace_at(i - 1, Enum.at(si_a, i))
          |> List.replace_at(i, Enum.at(si_a, i + 1 - 1))
          |> List.replace_at(i + 1 - 1, Enum.at(si_a, i - 1))

        {sr2, si2}
      else
        {sr_a, si_a}
      end
    end)
  end

  defp dlaqr5_sweep(
         h,
         z,
         n,
         ktop,
         kbot,
         sr,
         si,
         nbmpbs,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         smlnum,
         ulp,
         v,
         incol
       )
       when incol > kbot - 2 do
    {h, z}
  end

  defp dlaqr5_sweep(
         h,
         z,
         n,
         ktop,
         kbot,
         sr,
         si,
         nbmpbs,
         wantt,
         wantz,
         iloz,
         ihiz,
         safmin,
         smlnum,
         ulp,
         v,
         incol
       ) do
    jtop = if wantt, do: 1, else: ktop

    krcol_max = min(incol + 2 * nbmpbs - 1, kbot - 2)

    {h, z, v} =
      dlaqr5_krcol_loop(
        h,
        z,
        n,
        ktop,
        kbot,
        sr,
        si,
        nbmpbs,
        wantt,
        wantz,
        iloz,
        ihiz,
        smlnum,
        ulp,
        v,
        incol,
        jtop,
        incol,
        krcol_max
      )

    dlaqr5_sweep(
      h,
      z,
      n,
      ktop,
      kbot,
      sr,
      si,
      nbmpbs,
      wantt,
      wantz,
      iloz,
      ihiz,
      safmin,
      smlnum,
      ulp,
      v,
      incol + 2 * nbmpbs
    )
  end

  # ── KRCOL loop ──

  defp dlaqr5_krcol_loop(
         h,
         z,
         _n,
         _ktop,
         _kbot,
         _sr,
         _si,
         _nbmpbs,
         _wantt,
         _wantz,
         _iloz,
         _ihiz,
         _smlnum,
         _ulp,
         v,
         _incol,
         _jtop,
         krcol,
         krcol_max
       )
       when krcol > krcol_max do
    {h, z, v}
  end

  defp dlaqr5_krcol_loop(
         h,
         z,
         n,
         ktop,
         kbot,
         sr,
         si,
         nbmpbs,
         wantt,
         wantz,
         iloz,
         ihiz,
         smlnum,
         ulp,
         v,
         incol,
         jtop,
         krcol,
         krcol_max
       ) do
    mtop = max(1, div(ktop - krcol, 2) + 1)
    mbot = min(nbmpbs, div(kbot - krcol - 1, 2))
    m22 = mbot + 1
    bmp22 = mbot < nbmpbs and krcol + 2 * (m22 - 1) == kbot - 2

    # 2x2 special case
    {h, z, v} =
      if bmp22 do
        k2 = krcol + 2 * (m22 - 1)

        if k2 == ktop - 1 do
          # Create new 2x2 reflector from shifts
          r0 = k2 + 1

          h2 = [
            [f_get(h, n, r0, r0), f_get(h, n, r0, r0 + 1)],
            [f_get(h, n, r0 + 1, r0), f_get(h, n, r0 + 1, r0 + 1)]
          ]

          v_new =
            dlaqr1(
              2,
              h2,
              Enum.at(sr, 2 * m22 - 2),
              Enum.at(si, 2 * m22 - 2),
              Enum.at(sr, 2 * m22 - 1),
              Enum.at(si, 2 * m22 - 1)
            )

          {href, _tau, _beta} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(v_new, type: :f64))
          vf = Nx.to_flat_list(href)
          t1 = Enum.at(vf, 0)
          t2 = t1 * Enum.at(vf, 1)

          # Right update
          {h, _} =
            Enum.reduce(jtop..min(kbot, k2 + 3), {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(acc, n, j, k2 + 2)

              {acc
               |> f_set(n, j, k2 + 1, f_get(acc, n, j, k2 + 1) - refsum * t1)
               |> f_set(n, j, k2 + 2, f_get(acc, n, j, k2 + 2) - refsum * t2), nil}
            end)

          # Left update
          jbl = if wantt, do: n, else: kbot

          {h, _} =
            Enum.reduce((k2 + 1)..jbl, {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, k2 + 1, j) + Enum.at(vf, 1) * f_get(acc, n, k2 + 2, j)

              {acc
               |> f_set(n, k2 + 1, j, f_get(acc, n, k2 + 1, j) - refsum * t1)
               |> f_set(n, k2 + 2, j, f_get(acc, n, k2 + 2, j) - refsum * t2), nil}
            end)

          # Deflation
          h = deflation_2x2(h, n, k2, ktop, kbot, smlnum, ulp)
          # Accumulate Z
          {h, z} =
            if wantz and not false do
              Enum.reduce(iloz..ihiz, {h, z}, fn j, {h_acc, z_acc} ->
                refsum = f_get(z_acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(z_acc, n, j, k2 + 2)

                {h_acc,
                 z_acc
                 |> f_set(n, j, k2 + 1, f_get(z_acc, n, j, k2 + 1) - refsum * t1)
                 |> f_set(n, j, k2 + 2, f_get(z_acc, n, j, k2 + 2) - refsum * t2)}
              end)
            else
              {h, z}
            end

          v = List.replace_at(v, m22 - 1, vf)
          {h, z, v}
        else
          # Existing 2x2 reflector
          beta = f_get(h, n, k2 + 1, k2)
          vt = [beta, f_get(h, n, k2 + 2, k2)]
          {href, _tau, _beta2} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(vt, type: :f64))
          vf = Nx.to_flat_list(href)
          t1 = Enum.at(vf, 0)
          t2 = t1 * Enum.at(vf, 1)
          h = h |> f_set(n, k2 + 1, k2, beta) |> f_set(n, k2 + 2, k2, 0.0)

          {h, _} =
            Enum.reduce(jtop..min(kbot, k2 + 3), {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(acc, n, j, k2 + 2)

              {acc
               |> f_set(n, j, k2 + 1, f_get(acc, n, j, k2 + 1) - refsum * t1)
               |> f_set(n, j, k2 + 2, f_get(acc, n, j, k2 + 2) - refsum * t2), nil}
            end)

          jbl2 = if wantt, do: n, else: kbot

          {h, _} =
            Enum.reduce((k2 + 1)..jbl2, {h, nil}, fn j, {acc, _} ->
              refsum = f_get(acc, n, k2 + 1, j) + Enum.at(vf, 1) * f_get(acc, n, k2 + 2, j)

              {acc
               |> f_set(n, k2 + 1, j, f_get(acc, n, k2 + 1, j) - refsum * t1)
               |> f_set(n, k2 + 2, j, f_get(acc, n, k2 + 2, j) - refsum * t2), nil}
            end)

          h = deflation_2x2(h, n, k2, ktop, kbot, smlnum, ulp)

          {h, z} =
            if wantz do
              Enum.reduce(iloz..ihiz, {h, z}, fn j, {h_acc, z_acc} ->
                refsum = f_get(z_acc, n, j, k2 + 1) + Enum.at(vf, 1) * f_get(z_acc, n, j, k2 + 2)

                {h_acc,
                 z_acc
                 |> f_set(n, j, k2 + 1, f_get(z_acc, n, j, k2 + 1) - refsum * t1)
                 |> f_set(n, j, k2 + 2, f_get(z_acc, n, j, k2 + 2) - refsum * t2)}
              end)
            else
              {h, z}
            end

          v = List.replace_at(v, m22 - 1, vf)
          {h, z, v}
        end
      else
        {h, z, v}
      end

    # 3x3 reflections (from MBOT down to MTOP)
    Enum.reduce(mbot..mtop//-1, {h, z, v}, fn m, {h_acc, z_acc, v_acc} ->
      k = krcol + 2 * (m - 1)
      v_m = Enum.at(v_acc, m - 1)

      {h_mid, vf} =
        if k == ktop - 1 do
          h3 =
            for i <- 0..2 do
              for j <- 0..2, do: f_get(h_acc, n, ktop + i, ktop + j)
            end

          v_init =
            dlaqr1(
              3,
              h3,
              Enum.at(sr, 2 * m - 2),
              Enum.at(si, 2 * m - 2),
              Enum.at(sr, 2 * m - 1),
              Enum.at(si, 2 * m - 1)
            )

          {href, _tau, _beta} = Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(v_init, type: :f64))
          vf_new = Nx.to_flat_list(href)
          {h_acc, vf_new}
        else
          # Delayed transformation
          t1o = Enum.at(v_m, 0)
          t2o = t1o * Enum.at(v_m, 1)
          t3o = t1o * Enum.at(v_m, 2)
          refsum_d = Enum.at(v_m, 2) * f_get(h_acc, n, k + 3, k + 2)

          h_acc =
            h_acc
            |> f_set(n, k + 3, k, -refsum_d * t1o)
            |> f_set(n, k + 3, k + 1, -refsum_d * t2o)
            |> f_set(n, k + 3, k + 2, f_get(h_acc, n, k + 3, k + 2) - refsum_d * t3o)

          beta = f_get(h_acc, n, k + 1, k)
          v2_in = f_get(h_acc, n, k + 2, k)
          v3_in = f_get(h_acc, n, k + 3, k)

          {href_n, _tau_n, _beta_n} =
            Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor([beta, v2_in, v3_in], type: :f64))

          vf_n = Nx.to_flat_list(href_n)

          # Bulge collapse check
          hk3k = f_get(h_acc, n, k + 3, k)
          hk3k1 = f_get(h_acc, n, k + 3, k + 1)
          hk3k2 = f_get(h_acc, n, k + 3, k + 2)

          if hk3k != 0.0 or hk3k1 != 0.0 or hk3k2 == 0.0 do
            {h_acc
             |> f_set(n, k + 1, k, beta)
             |> f_set(n, k + 2, k, 0.0)
             |> f_set(n, k + 3, k, 0.0), vf_n}
          else
            h3_sub = for(i <- 0..2, do: for(j <- 0..2, do: f_get(h_acc, n, k + 1 + i, k + 1 + j)))

            vt_init =
              dlaqr1(
                3,
                h3_sub,
                Enum.at(sr, 2 * m - 2),
                Enum.at(si, 2 * m - 2),
                Enum.at(sr, 2 * m - 1),
                Enum.at(si, 2 * m - 1)
              )

            {vt_ref, _tau_vt, _beta_vt} =
              Nx.LinAlg.EigHouseholder.dlarfg(Nx.tensor(vt_init, type: :f64))

            vt_f = Nx.to_flat_list(vt_ref)
            t1n = Enum.at(vt_f, 0)
            t2n = t1n * Enum.at(vt_f, 1)
            t3n = t1n * Enum.at(vt_f, 2)
            refsum_n = f_get(h_acc, n, k + 1, k) + Enum.at(vt_f, 1) * f_get(h_acc, n, k + 2, k)
            fill_tst = abs(f_get(h_acc, n, k + 2, k) - refsum_n * t2n) + abs(refsum_n * t3n)

            diag_sum =
              abs(f_get(h_acc, n, k, k)) + abs(f_get(h_acc, n, k + 1, k + 1)) +
                abs(f_get(h_acc, n, k + 2, k + 2))

            if fill_tst > ulp * diag_sum do
              {h_acc
               |> f_set(n, k + 1, k, beta)
               |> f_set(n, k + 2, k, 0.0)
               |> f_set(n, k + 3, k, 0.0), vf_n}
            else
              {h_acc
               |> f_set(n, k + 1, k, f_get(h_acc, n, k + 1, k) - refsum_n * t1n)
               |> f_set(n, k + 2, k, 0.0)
               |> f_set(n, k + 3, k, 0.0), vt_f}
            end
          end
        end

      # Apply reflector from right
      t1 = Enum.at(vf, 0)
      t2 = t1 * Enum.at(vf, 1)
      t3 = t1 * Enum.at(vf, 2)

      {h_mid, _} =
        Enum.reduce(jtop..min(kbot, k + 3), {h_mid, nil}, fn j, {acc, _} ->
          refsum =
            f_get(acc, n, j, k + 1) + Enum.at(vf, 1) * f_get(acc, n, j, k + 2) +
              Enum.at(vf, 2) * f_get(acc, n, j, k + 3)

          {acc
           |> f_set(n, j, k + 1, f_get(acc, n, j, k + 1) - refsum * t1)
           |> f_set(n, j, k + 2, f_get(acc, n, j, k + 2) - refsum * t2)
           |> f_set(n, j, k + 3, f_get(acc, n, j, k + 3) - refsum * t3), nil}
        end)

      # First column of left update
      refsum_c =
        f_get(h_mid, n, k + 1, k + 1) + Enum.at(vf, 1) * f_get(h_mid, n, k + 2, k + 1) +
          Enum.at(vf, 2) * f_get(h_mid, n, k + 3, k + 1)

      h_mid =
        h_mid
        |> f_set(n, k + 1, k + 1, f_get(h_mid, n, k + 1, k + 1) - refsum_c * t1)
        |> f_set(n, k + 2, k + 1, f_get(h_mid, n, k + 2, k + 1) - refsum_c * t2)
        |> f_set(n, k + 3, k + 1, f_get(h_mid, n, k + 3, k + 1) - refsum_c * t3)

      # Convergence test
      h_mid = if k >= ktop, do: deflation_3x3(h_mid, n, k, ktop, kbot, smlnum, ulp), else: h_mid

      # Accumulate Z
      {h_mid, z_acc} =
        if wantz do
          Enum.reduce(iloz..ihiz, {h_mid, z_acc}, fn j, {h2, z2} ->
            refsum_z =
              f_get(z2, n, j, k + 1) + Enum.at(vf, 1) * f_get(z2, n, j, k + 2) +
                Enum.at(vf, 2) * f_get(z2, n, j, k + 3)

            {h2,
             z2
             |> f_set(n, j, k + 1, f_get(z2, n, j, k + 1) - refsum_z * t1)
             |> f_set(n, j, k + 2, f_get(z2, n, j, k + 2) - refsum_z * t2)
             |> f_set(n, j, k + 3, f_get(z2, n, j, k + 3) - refsum_z * t3)}
          end)
        else
          {h_mid, z_acc}
        end

      v_acc = List.replace_at(v_acc, m - 1, vf)
      {h_mid, z_acc, v_acc}
    end)
    |> then(fn {h_out, z_out, v_out} ->
      # Left updates for subsequent columns
      jbl3 = if wantt, do: n, else: kbot

      h_out =
        Enum.reduce(mbot..mtop//-1, h_out, fn m, acc ->
          k = krcol + 2 * (m - 1)
          v_m2 = Enum.at(v_out, m - 1)
          t1b = Enum.at(v_m2, 0)
          t2b = t1b * Enum.at(v_m2, 1)
          t3b = t1b * Enum.at(v_m2, 2)

          Enum.reduce(max(ktop, krcol + 2 * m)..jbl3, acc, fn j, acc2 ->
            refsum_l =
              f_get(acc2, n, k + 1, j) + Enum.at(v_m2, 1) * f_get(acc2, n, k + 2, j) +
                Enum.at(v_m2, 2) * f_get(acc2, n, k + 3, j)

            acc2
            |> f_set(n, k + 1, j, f_get(acc2, n, k + 1, j) - refsum_l * t1b)
            |> f_set(n, k + 2, j, f_get(acc2, n, k + 2, j) - refsum_l * t2b)
            |> f_set(n, k + 3, j, f_get(acc2, n, k + 3, j) - refsum_l * t3b)
          end)
        end)

      dlaqr5_krcol_loop(
        h_out,
        z_out,
        n,
        ktop,
        kbot,
        sr,
        si,
        nbmpbs,
        wantt,
        wantz,
        iloz,
        ihiz,
        smlnum,
        ulp,
        v_out,
        incol,
        jtop,
        krcol + 1,
        krcol_max
      )
    end)
  end

  # ── Deflation helpers ──

  defp deflation_2x2(h, n, k, ktop, kbot, smlnum, ulp) do
    if k >= ktop do
      hk1k = f_get(h, n, k + 1, k)

      if hk1k != 0.0 do
        tst1 = abs(f_get(h, n, k, k)) + abs(f_get(h, n, k + 1, k + 1))

        tst1 =
          if tst1 == 0.0 do
            if(k >= ktop + 1, do: abs(f_get(h, n, k, k - 1)), else: 0.0) +
              if k <= kbot - 2, do: abs(f_get(h, n, k + 2, k + 1)), else: 0.0
          else
            tst1
          end

        if abs(hk1k) <= max(smlnum, ulp * tst1) do
          h12d = max(abs(hk1k), abs(f_get(h, n, k, k + 1)))
          h21d = min(abs(hk1k), abs(f_get(h, n, k, k + 1)))

          h11d =
            max(
              abs(f_get(h, n, k + 1, k + 1)),
              abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1))
            )

          h22d =
            min(
              abs(f_get(h, n, k + 1, k + 1)),
              abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1))
            )

          scld = h11d + h12d
          tst2 = h22d * (h11d / scld)

          if tst2 == 0.0 or h21d * (h12d / scld) <= max(smlnum, ulp * tst2) do
            f_set(h, n, k + 1, k, 0.0)
          else
            h
          end
        else
          h
        end
      else
        h
      end
    else
      h
    end
  end

  defp deflation_3x3(h, n, k, ktop, kbot, smlnum, ulp) do
    hk1k = f_get(h, n, k + 1, k)

    if hk1k != 0.0 do
      tst1 = abs(f_get(h, n, k, k)) + abs(f_get(h, n, k + 1, k + 1))

      tst1 =
        if tst1 == 0.0 do
          if(k >= ktop + 1, do: abs(f_get(h, n, k, k - 1)), else: 0.0) +
            if(k >= ktop + 2, do: abs(f_get(h, n, k, k - 2)), else: 0.0) +
            if(k >= ktop + 3, do: abs(f_get(h, n, k, k - 3)), else: 0.0) +
            if(k <= kbot - 2, do: abs(f_get(h, n, k + 2, k + 1)), else: 0.0) +
            if(k <= kbot - 3, do: abs(f_get(h, n, k + 3, k + 1)), else: 0.0) +
            if k <= kbot - 4, do: abs(f_get(h, n, k + 4, k + 1)), else: 0.0
        else
          tst1
        end

      if abs(hk1k) <= max(smlnum, ulp * tst1) do
        h12d = max(abs(hk1k), abs(f_get(h, n, k, k + 1)))
        h21d = min(abs(hk1k), abs(f_get(h, n, k, k + 1)))

        h11d =
          max(abs(f_get(h, n, k + 1, k + 1)), abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1)))

        h22d =
          min(abs(f_get(h, n, k + 1, k + 1)), abs(f_get(h, n, k, k) - f_get(h, n, k + 1, k + 1)))

        scld = h11d + h12d
        tst2 = h22d * (h11d / scld)

        if tst2 == 0.0 or h21d * (h12d / scld) <= max(smlnum, ulp * tst2) do
          f_set(h, n, k + 1, k, 0.0)
        else
          h
        end
      else
        h
      end
    else
      h
    end
  end

  # ──────────────────────────────────────
  #  DLAQR0: top-level multi-shift QR
  # ──────────────────────────────────────

  @doc """
  DLAQR0: multi-shift QR top-level scheduler.
  Delegates to DLAHQR for matrices N ≤ 75.
  """
  def dlaqr0(h, opts \\ []) do
    wantt = opts[:wantt] || false
    wantz = opts[:wantz] || false
    n = elem(Nx.shape(h), 0)
    ilo = opts[:ilo] || 0
    ihi = opts[:ihi] || n - 1
    iloz = opts[:iloz] || 0
    ihiz = opts[:ihiz] || n - 1

    {h_out, wr, wi, z_out, info} =
      Nx.LinAlg.EigSchur.dlahqr(h,
        wantt: wantt,
        wantz: wantz,
        ilo: ilo,
        ihi: ihi,
        iloz: iloz,
        ihiz: ihiz
      )

    z = if wantz, do: z_out, else: Nx.eye(n, type: :f64)
    {h_out, wr, wi, z, info}
  end
end
