# 最適化手法まとめ（ノート原文の整理）

## 1. 基本式と勾配

パラメータ更新の基本形は次式で表される。
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,\mathbf{g}_t
$$
ここで学習率を $\eta$、損失関数 $E$ の勾配を $\mathbf{g}_t$ とすると、
$$
\mathbf{g}_t = \nabla_{\mathbf{w}} E(\mathbf{w}_t) = \left(\frac{\partial E}{\partial w_1}, \dots, \frac{\partial E}{\partial w_d}\right)^\top
$$
と書ける。

---

## 2. 学習率の重要性

学習の成否は、適切な学習率を設定できるかどうかに大きく依存する。
学習率選択の負担を軽減するために、重みの更新幅を自動的に調整する手法が多数提案されている。

---

## 3. 主な手法の一覧

- AdaGrad（後続手法の基礎）
- RMSProp / AdaDelta
- Adam
- RAdam（Rectified Adam）
- AdamW

---

## 4. AdaGrad

AdaGrad の狙いは、これまで大きく更新されてきた成分の更新量を抑え、あまり更新されなかった成分の更新量を増やすことにある。
二次元ベクトルを例にすると、$x$ 方向の更新を小さくし $y$ 方向を大きくすることで勾配方向のバランスを取ろうとする直感に基づく。

更新則は以下の通り。
$$
G_t = \sum_{\tau=1}^{t} \mathbf{g}_\tau \odot \mathbf{g}_\tau, \quad
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot \mathbf{g}_t
$$
ただし $\odot$ は要素ごとの積。
$G_t$ が単調増加するため、学習率は実効的に $\eta_t \to 0$ と縮んでしまう点が課題。

---

## 5. RMSProp と AdaDelta

累積和 $\sum \mathbf{g}_t$ の代わりに指数移動平均を用いると、減衰し過ぎる問題を緩和できる。

RMSProp の更新則：
$$
\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + (1-\beta)\, \mathbf{g}_t \odot \mathbf{g}_t, \quad
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t
$$

AdaDelta は、RMSProp の統計量を用いて更新量の次元を揃えることを目的とし、
$$
\mathbf{s}_t = \beta\,\mathbf{s}_{t-1} + (1-\beta)\, \Delta\mathbf{w}_t \odot \Delta\mathbf{w}_t,
$$
$$
\Delta\mathbf{w}_t = - \frac{\sqrt{\mathbf{s}_{t-1} + \epsilon}}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t, \quad
\mathbf{w}_{t+1} = \mathbf{w}_t + \Delta\mathbf{w}_t
$$
のように更新する。

---

## 6. Adam

Adam は勾配の一次・二次モーメントを指数移動平均で推定する。
$$
\mathbf{m}_t = \beta_1\,\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t, \quad
\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t \odot \mathbf{g}_t
$$
これらは初期段階で過小評価されるため、バイアス補正を行う。
$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
$$
最終的な更新式は次式。
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t
$$
学習率の細かな調整をしなくても比較的安定した性能が得られる。

---

## 7. CNN における比較

CNN では、学習率を丁寧に調整した Momentum SGD のほうが最終精度で優れるケースが報告されている。

---

## 8. Adam の改良

1. 更新幅が極端に大きく振れる → 下限を設ける AdaBound。
2. 初期段階でモーメント推定が不安定 → バイアス補正を強化する RAdam。
3. 学習初期に一定期間 $\eta$ を抑えるウォームアップで性能向上することがある。
4. 重み減衰を Adam に直接組み込む AdamW。減衰項 $-\lambda \mathbf{w}$ を勾配に加えるため、L2 正則化そのものではないが実践的に有効。

---

## 9. 学習率スケジューリング

Adam は「学習率を選ばなくてもよい」とされるものの、学習率スケジューリングを組み合わせるとさらに良い結果が得られる場合が多い。

---

## SGD に関する他の改善案

損失関数の地形と汎化性能には関連があり、平坦な極小点を探すことが望ましいとされる。Sharpness-Aware Minimization (SAM) はこの点で有望な最適化法。
SAM の目的関数は次式で与えられる。
$$
\min_{\mathbf{w}} \max_{\lVert \boldsymbol{\epsilon} \rVert_p \le \rho} E(\mathbf{w} + \boldsymbol{\epsilon}) + \lambda \lVert \mathbf{w} \rVert_2^2
$$
