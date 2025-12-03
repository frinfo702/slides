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



---

<!-- _class : lead -->







---

<!-- _header: 参考文献 -->




# Note (this is only for me to understand the paper)

latent空間でdiffusionするというのが肝
- 計算量が激減し、10~100倍早くなる
stable diffusionなどをPCで動かせる理由

latent構造には**意味的構造**がすでに圧縮されている
autoencoderが
- high-level な形状
- semantic consistency
- perceptual similarity
をlatentに押し込めるためdiffusionが学習するべき分布が扱いやすくなる

画像以外への応用が容易
- audio latent
- video latent
- 3D latent
- text latent (in progress)
- text latent (in progress)
高次元データの生成モデルはlatent表現を持つのが常識になりつつある

逆方向の理論 (score, 情報保存) が理解しやすくなる
pixel空間よりlatent空間の方が生成すべき確立分布が平滑なのでsocre (∇logp(x))の推定が安定

以下は論文の重要だと思った部分を抜き出したもの

## Abstract

既存のDMsはpixel空間で計算をしていたので大量のGPUを使用し、連続評価により推論は高価だった
LDMは latent空間で計算をすることで複雑性の減少と詳細維持・忠実性の間のほぼ最適と言えるバランスに到達した(という触れ込み)
- cross attention layer
- covolutional manner
などの導入により全般的な条件付き入力（テキストとかやbounding boxes）を可能にし、また高画質な画像の生成も同時に達成している

## Introduction
既存のdiffusion model
- 尤度ベースのクラスに属する
- GPUによる電力消費が激しい (e.g. 150-1000 V in 100 days)
これによって以下の状況をもたらした
1. GPUがたくさん必要
2. すでに学習されたモデルの評価にも、時間的・金銭的に負担がかかる。同じアーキテクチャのものでも最初から同じことをしないといけない (e.g. 25-1000 steps)
訓練とサンプリング（生成）の両方の複雑性を排除しつつ、性能は低下させないようにできないかというモチベーション

### Departure to Latent Space
