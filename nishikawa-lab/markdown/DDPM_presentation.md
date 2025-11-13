---
marp: true
theme: academic
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Denoising Diffusion Probabilistic Models (DDPM)

Ho et al., 2020

<br>

Kenichiro Goto
西川研究室 B3
2025-11-14

---

<!-- _header: Agenda -->

1. 導入
   1. 背景
   2. DDPMの位置付け
2. メカニズムの理解
   1. Forward Process (拡散過程)
   2. Reverse Process (逆拡散過程)
3. 理論と定式化
   1. 損失関数の導出
   2. 損失関数の簡略化 ($L_\mathrm{simple}$, $\boldsymbol{\epsilon}$-prediction)
   3. Score Matching との関連性
4. 実験とアーキテクチャ
   1. アーキテクチャ (U-Net)
   2. サンプリング（生成）
   3. 生成結果
5. 議論とまとめ
   1. 本研究の貢献
   2. 課題
   3. 影響

---

<!-- _class : lead -->

# 1. 導入

---

<!-- _header: 1.1. 背景 -->

従来の生成モデルには以下のような弱点があった

**GAN**

* Pros: 高品質な生成
* Cons: モード崩壊, 学習が不安定（ D, Gのバランスが難しい）

**VAE**

- Pros: 尤度ベースで学習が安定
- Cons: ELBOという下界を最大化するため生成がぼやけやすい

**Flow-based Model**

- Pros: 尤度を厳密に計算可能
- Cons: 可逆変換の制約がありアーキテクチャ設計が難しい

---

<!-- _header: 1.2. DDPMの位置付け -->

DDPMは高品質かつ安定した学習（尤度ベース）の両立を目標とする

2つのプロセスで構成される
1. Forward Process (拡散過程) : データを徐々にノイズにしていく (固定プロセスでアルゴリズム的)
2. Reverse Process (逆拡散過程): 学習対象。ノイズから画像空間上のデータへ復元する

![w:600 center](../images/generative-overview.png "Overview of different types of generative models.")

---

<!-- _class: lead -->

# 2. メカニズムの理解

// TODO: 基本的なアイデアをかく

---

<!-- _header: 2.1. Forward Process (拡散過程) -->

$q(\mathbf{x}_t|\mathbf{x}_{t-1})$ : データ→ノイズ方向への変換

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) \tag{1}
$$

- $\beta_t$ は分散スケジュール（ハイパーパラメータ）  
- $t$ が大きくなるにつれて $q$ はガウシアンノイズ $\mathcal{N}(\mathbf{0}, \mathbf{I})$ に漸近する

---

<!-- _header: 2.1. Forward Process (拡散過程) -->

マルコフ性の利点： $T$ ステップの反復計算は不要。
$\alpha_t := 1 - \beta_t$, $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$ とおくと、
$\mathbf{x}_0$ から任意の $\mathbf{x}_t$ を一発でサンプリング可能（Reparameterization Trick）。

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
\\
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon} \quad (\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}))
$$

これは **訓練時に極めて重要** となる。

![w:700 center](../images/DDPM.png "The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: Ho et al. 2020 with a few additional annotations)")

---

<!-- _header: 2.2. Reverse Process (逆拡散過程) -->

### $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ : ノイズからデータを復元する

Forward Processの逆をたどる。
$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ からスタートし、ノイズを除去（Denoising）しながら $\mathbf{x}_0$ を復元する。

この逆過程 $p_\theta$ をニューラルネットワーク (NN) で近似する。
$$
p_\theta(\mathbf{x}{0:T}) := p(\mathbf{x}T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)
\\
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
学習対象: NN  $\boldsymbol{\mu}_\theta$ が真の逆過程の平均 $\tilde{\boldsymbol{\mu}}$ を予測するように学習する

![w:400 center](../images/diffusion-example.png "An example of training a diffusion model for modeling a 2D swiss roll data. (Image source: Sohl-Dickstein et al., 2015)")

---

<!-- _class: lead -->

# 3. 理論と定式化

---

<!-- _header: 3.1. 損失関数の導出 -->

**どうやって学習するか** 
目標： 尤度 $p_\theta(\mathbf{x}_t)$ を最大化したい。
→ 真の逆過程 $q$ とNNによる近似 $p_\theta$ のKL divergenceを最小化したい

変分下界 (ELBO) を用いて損失 $L$ を定義する
$$
L = \mathbb{E}_q[-\log p_\theta(\mathbf{x}_{0:T}) + \log q(\mathbf{x}_{1:T}|\mathbf{x}_0)]
$$

これを整理すると各ステップのKL divergenceの和になる

$$
L = \mathbb{E}_q[ \underbrace{D_{KL}(q(\mathbf{x}_T) || p(\mathbf{x}_T))}_{L_T} + \sum_{t>1} \underbrace{D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0} ]
$$

---

<!-- _header: 3.1. 損失関数の導出 -->

$L_{t-1}$ 項にある $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ のように $\mathbf{x}_0$ で条件付けると解析的に計算可能になる

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$
ここで
$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) := \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right)
$$

（ $\boldsymbol{\epsilon}_t$ は $\mathbf{x}_t$ の生成に使われたノイズ）

$L_{t-1}$ は、この **真の平均 $\tilde{\boldsymbol{\mu}}_t$** と **NNの予測 $\boldsymbol{\mu}_\theta$** の差を測る項になる。

// TODO: なぜ元のままだと計算が解析的にできないのか

---

<!-- _header: 3.2 損失関数の簡略化  -->

### $L_\mathrm{simple}$ : $\boldsymbol{\epsilon}$-prediction
// TODO: ε-predictionって何？手法の名前？何を指しているのか

$L_{t-1}$ は $p_\theta$ と $q$ の平均 $\boldsymbol{\mu}$ のL2距離（MSE）として計算できる。

$$
L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \lVert \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) \rVert^2 \right] + C
$$

さらに、 $\boldsymbol{\mu}$ を直接予測するのではなく、
$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ の関係を使って、ノイズ $\boldsymbol{\epsilon}$ を予測する 問題に置き換える。

NNで $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ が、真のノイズ $\boldsymbol{\epsilon}$ を予測するように学習する。

---
<!-- _header: 3.2 損失関数の簡略化  -->

$L_\mathrm{simple}$: 最終的な損失関数
論文では重み係数を無視した以下の単純な損失関数の方が安定しており性能も良かったと報告している

$$
L^\mathrm{simple}_t = \mathbb{E}_{t\sim[1, T], \epsilon_t} \left[\lVert \epsilon_t - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t, t) \|^2 \right] \tag{6}
$$

ここまでをまとめると
ニューラルネット $\boldsymbol{\epsilon}_\theta$ は、入力されたノイズ画像 $\mathbf{x}_t$ と時刻 $t$ から、そこに含まれるノイズ成分 $\boldsymbol{\epsilon}$ を予測するように学習すればよい。

![w:650 center](../images/DDPM-algo.png "The training and sampling algorithms in DDPM (Image source: Ho et al. 2020)")

---


<!-- _header: 3.3 Score Matching との関連性 (発展) -->

損失関数 $L_\mathrm{simple}$ は、Denoising Score Matching[^1]の目的関数と密接に関連している。

データ分布の勾配 $\nabla_{\mathbf{x}} \log q(\mathbf{x})$ を「Score」と呼ぶ。
Langevin Dynamics（スコアベースのサンプリング手法）において、Scoreの推定が重要。

DDPMのノイズ予測 $\boldsymbol{\epsilon}_\theta$ は、
このScore $s_\theta$ と以下の関係にあることが示されている。
（ノイズ予測は、実質的にデータの勾配（Score）を推定している）

$$
\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

> [1^] https://ieeexplore.ieee.org/abstract/document/6795935

---

<!-- _class: lead -->

# 4. 実験とアーキテクチャ

---

<!-- _header: 4.1. アーキテクチャ (U-Net) -->

### ノイズ予測器 $\boldsymbol{\epsilon}_\theta$ の実装

**ノイズ予測 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ のNNアーキテクチャ**
- 入力 ($\mathbf{x}_t$) と 出力 ($\boldsymbol{\epsilon}$) の解像度が同じである必要がある。
- 論文では U-Net（ResNetブロック + Attention）を採用。

**時刻 $t$ の入力方法 (Time Embedding)**

- $t$ は離散値だが、そのままNNに入力しない。
- TransformerのPositional Encodingと同様の手法で高次元ベクトルに変換し、
- U-Netの各ResNetブロックに加算する。

---

<!-- _header: 4.2. サンプリング（生成） -->

学習とは逆に、Reverse Process $p_\theta$ を $T$ ステップ実行する。

1. $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ （ガウスノイズ）からスタート
2. $t = T, \dots, 1$ について以下を反復：
    - $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $t > 1$, else $\mathbf{z} = \mathbf{0}$
    - NN $\boldsymbol{\epsilon}_\theta$ を使って $\boldsymbol{\mu}_\theta$ を計算
    - $\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}$
    （$\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$）
3. $\mathbf{x}_0$ が生成画像

---

<!-- _header: 4.3. 生成結果 -->

**生成結果 (CIFAR-10, CelebA)**
- 評価指標: FID (Fréchet Inception Distance), Inception Score
    - 当時のSOTA（特にGAN）に匹敵、あるいは凌駕するスコアを達成。
- Ablation Study:
  - $\boldsymbol{\mu}$ 予測より $\boldsymbol{\epsilon}$ 予測の方が性能が良いことを確認。
  - 損失の重み付けを無視した  $L_\mathrm{simple}$ の方が性能が良いことを確認。

---

<!-- _class: lead -->

# 5. 議論とまとめ

---

<!-- _header: 5.1. 議論とまとめ -->

**本研究の貢献**
- 高品質な画像生成: GANに匹敵する高忠実度な画像生成を達成。
- 安定した学習: 敵対的学習が不要で、安定した尤度ベースの学習が可能。
- 理論的背景: 変分推論とScore Matchingに裏打ちされた堅牢な理論。
- 単純な実装: 最終的な損失関数は「ノイズ予測のMSE」というシンプルな形。


---

<!-- _header: 5.1. 議論とまとめ -->

**課題**
サンプリング速度: $T$ ステップ (e.g., 1000〜4000) の反復計算が必要なため、推論（生成）が非常に遅い


---

<!-- _header: 5.1. 議論とまとめ -->

**影響**
- DDPMの成功と課題（速度）が、爆発的な後続研究を生み出した。
  - DDIM (2020): サンプリングを高速化 (e.g., 50ステップ)
  - Latent Diffusion (2021): 高解像度化（→ Stable Diffusionの基礎）

---

<!-- _header: 参考文書 -->

* Ho et al., "Denoising Diffusion Probabilistic Models", 2020. https://arxiv.org/abs/2006.11239
* https://arxiv.org/abs/1312.6114
* Song & Ermon, "Score-Based Generative Modeling", 2019. https://arxiv.org/abs/2011.13456
* What are Diffusion Models?. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
