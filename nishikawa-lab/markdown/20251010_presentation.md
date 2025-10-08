---
marp: true
title: "最適化手法まとめ"
paginate: true
---

# 最適化手法まとめ
2025-10-10 / Nishikawa Lab

---

# 本日のゴール
- 勾配降下法に基づく更新則を整理する
- 自動的に学習率を調整する代表的手法のポイントを把握する
- 近年の改良手法と SAM の位置づけを俯瞰する

---

# 基本式と勾配
- パラメータ更新の基本形
  $$
  \mathbf{w}_{t+1} = \mathbf{w}_t - \epsilon \nabla E_t 
  $$
- 勾配ベクトルの定義
  $$
  \mathbf{g}_t = \nabla_{\mathbf{w}} E(\mathbf{w}_t)
  = \begin{pmatrix}
  \frac{\partial E}{\partial w_1} & \cdots & \frac{\partial E}{\partial w_d}
  \end{pmatrix}^{\!\top}
  $$

---

# 学習率の重要性
- 学習の安定性と収束速度は学習率 $\epsilon$ に敏感
- 手動調整の負担を減らすため、更新幅を自動調整する手法が多数提案

---

# 自動調整系の代表例
- AdaGrad（以降の手法のベース）
- RMSProp / AdaDelta
- Adam
- RAdam（Rectified Adam）
- AdamW

---

# AdaGrad
- よく更新された成分には小さな学習率、あまり更新されなかった成分には大きな学習率を割り当てる
- 更新則
  $$
  \Delta w_{t, i} = - \frac{\epsilon}{\sqrt{\sum_{t'=1}^t g^{2}_{t', i} + \epsilon}} g_{t, i}
  $$
- $\sum_{t'=1}^t g^{2}_{t', i}$ が単調増加するため$\Delta w_{t, i} \rightarrow 0$となり更新が止まるのが課題

---

# RMSProp
- 累積和の代わりに移動平均を用いて過度な減衰を防ぐ
  $$
  v_t = \beta v_{t-1} + (1-\beta) g_t^2
  $$
  $$
  w_{t+1} = w_t - \frac{\epsilon}{\sqrt{v_t + \epsilon}} g_t
  $$

---

# AdaDelta
- RMSProp の統計量を利用し、更新量の次元を合わせる
  $$
  \Delta w_t = - \frac{\sqrt{E[\Delta w^2]_{t-1} + \epsilon}}
                   {\sqrt{E[g^2]_t + \epsilon}}\, g_t
  $$
- 本当は $\Delta w_t$ を直接使いたいが得られないため、直前の $\Delta w_{t-1}$ で代用
---

# Adam のモーメント推定
- 勾配の一次・二次モーメントを指数移動平均で推定
  $$
  m_{t, i} = \beta_1\,m_{t-1, i} + (1-\beta_1)g_{t, i},\qquad
  v_{t, i} = \beta_2 v_{t-1, i} + (1-\beta_2) g_{t, i} g^2_{t, i}
  $$
- 初期段階の過小評価を補正
  $$
  \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t},\qquad
  \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
  $$

---

# Adam の更新則と特徴
- 最終的な更新式
  $$
  \Delta w_{t, i} = - \frac{\epsilon}{\sqrt{\hat{v}_{t, i}} + \epsilon} \hat{m}_{t, i}
  $$
- 学習率の微調整をしなくても安定した学習が得やすい
- ラディカルな調整をせずとも広く使えるデフォルトとして普及

---

# CNN での報告例
- Momentum SGD に丁寧な学習率調整を施すと最終精度で優れるケースがある
- 自動調整系と手動チューニングのトレードオフを理解する必要

---

# Adam 系の改善案
- 極端な更新幅を抑える下限を導入する AdaBound
- 初期モーメント推定の不安定さを補正する RAdam
- 学習初期のウォームアップで性能向上するケース
- 正則化を更新式に統合する AdamW（勾配に $-\lambda \mathbf{w}$ を加算）

---

# 学習率スケジューリング
- Adam でもスケジューラを組み合わせると性能が向上する場合が多い
- コサイン減衰やステップ状の減衰などは依然有効な設計要素

---

# SAM（Sharpness-Aware Minimization）
- 平坦な極小点を探索し、汎化性能を高めることを狙う
- 目的関数
  $$
  \min_{\mathbf{w}} \max_{\lVert \boldsymbol{\epsilon} \rVert_p \le \rho}
  E(\mathbf{w} + \boldsymbol{\epsilon}) + \lambda \lVert \mathbf{w} \rVert_2^2
  $$
- 内側最大化で最悪摂動を想定しつつパラメータを更新

---

# まとめ
- 勾配降下の基本形は更新幅の設計で多様な派生が生まれる
- 主要な自動調整手法は過去勾配の統計量で学習率を制御
- 実運用ではスケジューラや正則化、SAM などと組み合わせて性能最適化
