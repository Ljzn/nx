defmodule Nx.LinAlg.EigBlockReflector do
  @moduledoc """
  Block reflector operations (DLARFT + DLARFB).
  """

  @doc """
  DLARFT: Compute T from V (n×k, columns include v[0]=1) and tau (k).
  Returns k×k upper triangular T.
  """
  def dlarft(v, tau) do
    n = elem(Nx.shape(v), 0)
    k = elem(Nx.shape(v), 1)

    if k <= 1 do
      Nx.tensor([Nx.to_number(Nx.reshape(tau[0], {}))], type: :f64) |> Nx.reshape({1, 1})
    else
      # Build T as a list of rows in flat list representation
      t_size = k * k
      t_list = List.duplicate(0.0, t_size)

      t_list =
        for i <- 0..(k - 1), reduce: t_list do
          t_acc ->
            tau_i = Nx.to_number(Nx.reshape(tau[i], {}))

            if tau_i == 0.0 do
              t_acc
            else
              # Set diagonal
              t_acc = List.replace_at(t_acc, i * k + i, tau_i)

              if i > 0 do
                # Extract V as flat list for w computation
                v_list = Nx.to_flat_list(v)

                # W = V[:, 0:i-1]' * V[:, i] (size i)
                w = for j <- 0..(i - 1) do
                  Enum.reduce(0..(n - 1), 0.0, fn r, acc ->
                    acc + Enum.at(v_list, r * k + j) * Enum.at(v_list, r * k + i)
                  end)
                end

                # Extract T[0:i-1, 0:i-1] 
                tw = for j <- 0..(i - 1) do
                  Enum.reduce(0..(i - 1), 0.0, fn r, acc ->
                    acc + Enum.at(t_acc, r * k + j) * Enum.at(w, r)
                  end)
                end

                # T[0:i-1, i] = -tau_i * tw
                tw_scaled = Enum.map(tw, fn val -> -tau_i * val end)
                Enum.reduce(0..(i - 1), t_acc, fn j, acc ->
                  List.replace_at(acc, j * k + i, Enum.at(tw_scaled, j))
                end)
              else
                t_acc
              end
            end
        end

      Nx.tensor(t_list, type: :f64) |> Nx.reshape({k, k})
    end
  end

  @doc """
  DLARFB: Apply H = I - V * T * V' to matrix C.
  """
  def dlarfb(v, t, c, side \\ :left)

  def dlarfb(v, t, c, :left) do
    vt_c = Nx.dot(Nx.transpose(v), c)
    t_vtc = Nx.dot(t, vt_c)
    correction = Nx.dot(v, t_vtc)
    Nx.subtract(c, correction)
  end

  def dlarfb(v, t, c, :right) do
    c_v = Nx.dot(c, v)
    cv_t = Nx.dot(c_v, t)
    correction = Nx.dot(cv_t, Nx.transpose(v))
    Nx.subtract(c, correction)
  end
end
