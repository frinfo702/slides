---
marp: true
title: "SGDの改良"
paginate: true
theme: figma_like
---

# SGDの改良
- 教科書p45~
- 3.5章
2025-10-10 / Kenichiro Goto

---

# 扱うテーマ
- 勾配降下法に基づく更新則を整理する
- 自動的に学習率を調整する代表的手法のポイントを把握する
- 近年の改良手法と SAM の位置づけを俯瞰する

---

# 基本式と勾配
- パラメータ更新の基本形は以下のようであった
  $$
  \mathbf{w}_{t+1} = \mathbf{w}_t - \varepsilon \nabla E_t 
  $$
- 以下のように定義することで重みの更新量を扱うことを明示的にする
  - 重みの更新量
    $$
    \Delta\mathbf{w}_t = \mathbf{w}_{t+1} - \mathbf{w}_t
    $$
  - 勾配ベクトルの定義
    $$
    \mathbf{g}_t = \nabla_{\mathbf{w}} E(\mathbf{w}_t)
    = \begin{pmatrix}
    \frac{\partial E}{\partial w_1} & \cdots & \frac{\partial E}{\partial w_d}
    \end{pmatrix}^{\!\top}
    $$
- 以上を用いて次のように更新量をおく($\mathbf{g}_t$の$i$成分を$g_{t, i}$とする)
$$
\Delta w_{t, i} = - \varepsilon g_{t, i}
$$
    

---

# 学習率の選定
- SGDでの学習の安定性と収束速度は学習率 $\varepsilon$ に左右される
- SGDでの選定の重要度を下げるため、更新幅を自動調整する手法が多数提案

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
  \Delta w_{t, i} = - \frac{\epsilon}{\sqrt{\sum_{t'=1}^t g^{2}_{t', i} + \varepsilon}} g_{t, i}
  $$
- $\sum_{t'=1}^t g^{2}_{t', i}$ が単調増加するため$\Delta w_{t, i} \rightarrow 0$となり更新が止まるのが課題
- $\varepsilon$はゼロ除算を避けるため

---

## AdaGradのグラフイメージ
ここで同じ方向に同じペースで更新し続けるより別成分の更新も試した方がいいという図を載せたい


---

# RMSProp
- 累積和の代わり移動平均を用いて更新量$\Delta w_{t, i}$の過度な減衰を防ぐ
- ~~直近のデータのみ使うので~~ 累積和のように大きくなり続けることはない
  $$
  \langle g_i^2 \rangle_t = \gamma \langle g_i^2\rangle_{t-1} + (1-\gamma) g^2_{t, i}
  $$
  を用いて以下のように更新量を決める
  $$
  \Delta w_{t,i} = - \frac{\epsilon}{\sqrt{\langle g_i^2 \rangle _t + \varepsilon}}g_{t, i}
  $$

---

# AdaDelta
  $$
  \langle \Delta w_i^2 \rangle_t = \gamma\langle\Delta w_i^2 \rangle_{t-1} + (1-\gamma)(\Delta w_{t, i})^2
  $$
  を用いて
  $$
  \Delta w_{t, i} = - \frac{\sqrt{\langle\Delta w_i^2 \rangle_{t-1} + \varepsilon}}{\sqrt{\langle g_i^2 \rangle_t + \varepsilon}}g_{t, i}
  $$
### 式の意図
- $\epsilon$を置き換えたのは物理量としての単位を合わせるため
- $\langle \Delta w_t \rangle$ の方が自然に思えるが、未知なので直前の $\langle \Delta w_{t-1} \rangle$ を近似値として代用

---

# Adam のモーメント推定
- 勾配の一次・二次モーメントを指数移動平均で推定
  $$
  m_{t, i} = \beta_1\,m_{t-1, i} + (1-\beta_1)g_{t, i},\qquad
  v_{t, i} = \beta_2 v_{t-1, i} + (1-\beta_2) g^2_{t, i}
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

---

# CNN での報告例
- Momentum SGD に丁寧な学習率調整を施すと最終精度で優れるケースがある
- 自動調整系と手動チューニングのトレードオフを理解する必要

---

# Adam 系の改善案
- 更新幅$\frac{\epsilon}{\sqrt{\hat{v}_{t, i}}}$を抑える上下限を導入する AdaBound
- 学習初期(10ステップほど)にモーメントの移動平均の不安定さを補正する RAdam
- 学習初期の一定期間だけ$\epsilon$を小さめにする「ウォームアップ」
- "重みの減衰"(教科書式3.9)を更新式に統合する AdamW ($-\epsilon\lambda \mathbf{w}$ を加算)
  - 重み$\mathbb{w}_t$自身の大きさに応じて減衰量が大きくなる
  - 元の減衰項はL2正則化由来だけどこれはただ「重みの減衰」のために足した
  - ゆえに正則化には一致しない

---

# 学習率スケジューリング
(これをしないのが強みみたいな話だったけど...)
- Adamでも$\epsilon$に対しスケジューラを組み合わせると性能が向上する場合が多い
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
