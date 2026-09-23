> ## Documentation Index
> Fetch the complete documentation index at: https://docs.fireworks.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Serverless Pricing

> Per-token serverless pricing for text, vision, and embedding models, including Priority and Fast serverless modes

## Overview

Serverless inference is priced per token. For how Standard, Priority, and Fast modes work and how to select one, see [Serverless Modes](/serverless/serverless-modes).

Every text or vision request is billed across three dimensions:

* **Input tokens** — what you send to the model.
* **Cached input tokens** — input tokens served from [prompt cache](/guides/prompt-caching), priced lower.
* **Output tokens** — what the model generates.

Embeddings are billed only on input tokens.

## How pricing works

* Prices below are **per 1 million tokens** in US dollars.
* **Batch inference** is billed at **50% of serverless pricing** on both input and output. See [Batch inference](/guides/batch-inference).

## Text and vision models

Per-model pricing for headline models. Fast variants appear as adjacent rows. In each **Standard** or **Priority** cell, prices are **input / cached input / output** (USD per 1M tokens), in that order.

| Model                                                                                                             | Standard                   | Priority                      |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------------------- |
| [Kimi K3](https://app.fireworks.ai/models/fireworks/kimi-k3)                                                      | \$3.00 / \$0.30 / \$15.00  | \$3.75 / \$0.375 / \$18.75    |
| [Kimi K3 Fast](https://app.fireworks.ai/models/fireworks/kimi-k3)                                                 | \$4.50 / \$0.45 / \$22.50  | —                             |
| [Kimi K3 (US)](https://app.fireworks.ai/models/fireworks/kimi-k3)                                                 | \$4.50 / \$0.45 / \$22.50  | \$5.625 / \$0.5625 / \$28.125 |
| [Kimi K2.7 Code](https://app.fireworks.ai/models/fireworks/kimi-k2p7-code)                                        | \$0.95 / \$0.19 / \$4.00   | \$1.425 / \$0.285 / \$6.00    |
| [Kimi K2.6](https://app.fireworks.ai/models/fireworks/kimi-k2p6)                                                  | \$0.95 / \$0.16 / \$4.00   | \$1.50 / \$0.22 / \$6.00      |
| [DeepSeek V4.1 Flash](https://app.fireworks.ai/models/fireworks/deepseek-v4p1-flash)                              | \$0.30 / \$0.006 / \$1.20  | \$0.375 / \$0.0075 / \$1.50   |
| [DeepSeek V4.1 Flash (US)](https://app.fireworks.ai/models/fireworks/deepseek-v4p1-flash)                         | \$0.45 / \$0.009 / \$1.80  | \$0.5625 / \$0.01125 / \$2.25 |
| [DeepSeek V4 Pro (0813)](https://app.fireworks.ai/models/fireworks/deepseek-v4-pro-0813)                          | \$1.32 / \$0.044 / \$3.96  | \$1.65 / \$0.055 / \$4.95     |
| [DeepSeek V4 Flash (0731)](https://app.fireworks.ai/models/fireworks/deepseek-v4-flash-0731)                      | \$0.22 / \$0.007 / \$0.66  | \$0.275 / \$0.00875 / \$0.825 |
| [DeepSeek V4 Flash Vision Exp](https://app.fireworks.ai/models/fireworks/deepseek-v4-flash-vision-exp)            | \$0.22 / \$0.007 / \$0.66  | \$0.275 / \$0.00875 / \$0.825 |
| [GLM 5.3 Flash](https://app.fireworks.ai/models/fireworks/glm-5p3-flash)                                          | \$0.15 / \$0.03 / \$0.50   | \$0.1875 / \$0.0375 / \$0.625 |
| [GLM 5.3 Flash (US)](https://app.fireworks.ai/models/fireworks/glm-5p3-flash)                                     | \$0.225 / \$0.045 / \$0.75 | —                             |
| [GLM 5.3](https://app.fireworks.ai/models/fireworks/glm-5p3)                                                      | \$1.40 / \$0.26 / \$4.40   | \$1.75 / \$0.325 / \$5.50     |
| [GLM 5.3 (US)](https://app.fireworks.ai/models/fireworks/glm-5p3)                                                 | \$2.10 / \$0.39 / \$6.60   | —                             |
| [GLM 5.3 Fast](https://app.fireworks.ai/models/fireworks/glm-5p3)                                                 | \$2.10 / \$0.39 / \$6.60   | —                             |
| [GLM 5.2](https://app.fireworks.ai/models/fireworks/glm-5p2)                                                      | \$1.40 / \$0.14 / \$4.40   | \$1.75 / \$0.18 / \$5.50      |
| [GLM 5.2 Fast](https://app.fireworks.ai/models/fireworks/glm-5p2)                                                 | \$2.10 / \$0.21 / \$6.60   | —                             |
| [GLM 5.2 Fast US](https://app.fireworks.ai/models/fireworks/glm-5p2)                                              | \$2.10 / \$0.21 / \$6.60   | —                             |
| [Qwen 3.8 Max](https://app.fireworks.ai/models/fireworks/qwen3p8-max)                                             | \$2.00 / \$0.25 / \$6.00   | \$3.00 / \$0.375 / \$9.00     |
| [MiniMax M3](https://app.fireworks.ai/models/fireworks/minimax-m3)                                                | \$0.30 / \$0.06 / \$1.20   | \$0.45 / \$0.09 / \$1.80      |
| [OpenAI GPT OSS 120B](https://app.fireworks.ai/models/fireworks/gpt-oss-120b)                                     | \$0.15 / \$0.015 / \$0.60  | \$0.18 / \$0.018 / \$0.72     |
| [Muse Glimmer 30B](https://app.fireworks.ai/models/fireworks/muse-glimmer-30b)                                    | \$0.35 / \$0.04 / \$1.50   | \$0.525 / \$0.06 / \$2.25     |
| [NVIDIA Nemotron 3.5 Lightning 30B A3B](https://app.fireworks.ai/models/fireworks/nemotron-lightning-3p5-30b-a3b) | \$0.05 / \$0.01 / \$0.20   | —                             |
| [NVIDIA Nemotron 3 Ultra (Preview)](https://app.fireworks.ai/models/fireworks/nemotron-3-ultra-nvfp4)             | \$0.60 / \$0.12 / \$2.40   | —                             |

**—** in the Priority column means Priority is not available for that model. This pricing table is the source of truth for Priority availability.

## Other base models — by size and architecture

For any text or vision model not listed individually, pricing is set by parameter count and architecture. These size-based prices apply uniformly to input and output (no separate cached-input rate):

| Model                                                  | \$ / 1M tokens |
| ------------------------------------------------------ | -------------- |
| Less than 4B parameters                                | \$0.10         |
| 4B – 16B parameters                                    | \$0.20         |
| More than 16B parameters                               | \$0.90         |
| MoE up to 56B parameters (e.g. Mixtral 8x7B)           | \$0.50         |
| MoE 56.1B – 176B parameters (e.g. DBRX, Mixtral 8x22B) | \$1.20         |

## Embeddings

Embeddings are billed per 1M input tokens.

| Base model parameter count | \$ / 1M input tokens |
| -------------------------- | -------------------- |
| up to 150M                 | \$0.008              |
| 150M – 350M                | \$0.016              |
| Qwen3 8B                   | \$0.10               |

## Notes

* Beginning September 1, 2026, launched [US-only Serverless](/serverless/us-only-serverless) models are priced at 1.5x the base model serverless prices. Kimi K3 (US) already includes this premium, while GLM 5.2 Fast US is an exception and matches global GLM 5.2 Fast pricing.
* For account-level controls (spend tiers, monthly spend limits, on-demand GPU quotas), see [Account quotas](/guides/quotas_usage/account-quotas).
