> ## Documentation Index
> Fetch the complete documentation index at: https://docs.together.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Available models

> Browse the catalog of available models for instant inference.

Serverless models are the fastest way to run inference on Together. You call any supported model through a shared per-token API, with no provisioning, no replicas to size, and no minimum cost. Pay only for the tokens you process.

## Models

If you're not sure which model to use, see [Recommended models](/docs/inference/recommended-models) for our picks by use case.

<Columns cols={4}>
  <Card title="Chat" icon="message-circle" horizontal href="#chat-models" />

  <Card title="Image" icon="photo" horizontal href="#image-models" />

  <Card title="Vision" icon="eye" horizontal href="#vision-models" />

  <Card title="Video" icon="video" horizontal href="#video-models" />

  <Card title="Audio" icon="volume" horizontal href="#audio-models" />

  <Card title="Embedding" icon="vector-bezier-2" horizontal href="#embedding-models" />

  <Card title="Rerank" icon="arrows-sort" horizontal href="#rerank-models" />

  <Card title="Moderation" icon="shield-check" horizontal href="#moderation-models" />
</Columns>

<Note>
  Serverless and dedicated model inference support different sets of models. See the [dedicated model inference catalog](/docs/dedicated-endpoints/models) for details.
</Note>

For rate limits and pricing, see the [Serverless overview](/docs/serverless/overview).

## Chat models

| Organization      | Model name                   | API model string                        | Context length | Input pricing (per 1M tokens) | Cached input pricing (per 1M tokens) | Output pricing (per 1M tokens) | Quantization | Function calling | Structured outputs |
| :---------------- | :--------------------------- | :-------------------------------------- | :------------- | :---------------------------- | :----------------------------------- | :----------------------------- | :----------- | :--------------- | :----------------- |
| Thinking Machines | Inkling                      | thinkingmachines/Inkling                | 524288         | \$1.00                        | \$0.17                               | \$4.05                         | NVFP4        | Yes              | Yes                |
| Minimax           | Minimax M3                   | MiniMaxAI/MiniMax-M3                    | 524288         | \$0.30                        | \$0.06                               | \$1.20                         | FP4          | Yes              | Yes                |
| Qwen              | Qwen3.8-2.4T-A95B            | Qwen/Qwen3.8-2.4T-A95B                  | -              | \$2.00                        | \$0.50                               | \$6.00                         | FP4          | -                | -                  |
| Qwen              | Qwen3.7 Max                  | Qwen/Qwen3.7-Max                        | -              | \$1.50                        | \$0.50                               | \$4.50                         | -            | -                | -                  |
| Qwen              | Qwen3.6 Plus                 | Qwen/Qwen3.6-Plus                       | 1000000        | \$0.50                        | -                                    | \$3.00                         | -            | -                | -                  |
| Qwen              | Qwen3.5 9B                   | Qwen/Qwen3.5-9B                         | 262144         | \$0.17                        | -                                    | \$0.25                         | FP8          | Yes              | Yes                |
| Moonshot          | Kimi K3                      | moonshotai/Kimi-K3                      | 1048576        | \$3.00                        | \$0.30                               | \$15.00                        | -            | Yes              | Yes                |
| Z.ai              | GLM-5.3                      | zai-org/GLM-5.3                         | 1048575        | \$1.40                        | \$0.26                               | \$4.40                         | FP4          | Yes              | Yes                |
| Z.ai              | GLM-5.3 Flash                | zai-org/GLM-5.3-Flash                   | 1048575        | \$0.15                        | \$0.03                               | \$0.50                         | FP8          | Yes              | Yes                |
| Z.ai              | GLM-5.2                      | zai-org/GLM-5.2                         | 1048575        | \$1.40                        | \$0.26                               | \$4.40                         | FP4          | Yes              | Yes                |
| OpenAI            | GPT-OSS 120B                 | openai/gpt-oss-120b                     | 131072         | \$0.15                        | -                                    | \$0.60                         | MXFP4        | Yes              | Yes                |
| DeepSeek          | DeepSeek-V4-Flash-0731       | deepseek-ai/DeepSeek-V4-Flash-0731      | 1048576        | \$0.14                        | \$0.03                               | \$0.28                         | FP4          | Yes              | Yes                |
| DeepSeek          | DeepSeek V4 Pro 0813         | deepseek-ai/DeepSeek-V4-Pro-0813        | 1048576        | \$1.32                        | \$0.13                               | \$3.96                         | NVFP4        | Yes              | Yes                |
| Meta              | Llama 3.3 70B Instruct Turbo | meta-llama/Llama-3.3-70B-Instruct-Turbo | 131072         | \$1.04                        | -                                    | \$1.04                         | FP8          | Yes              | Yes                |
| Qwen              | Qwen3.7 Plus                 | Qwen/Qwen3.7-Plus                       | 1000000        | \$0.32                        | -                                    | \$1.28                         | -            | -                | -                  |
| Prism ML          | Ternary Bonsai 27B           | Prism-ML/Ternary-Bonsai-27B             | 262144         | Free                          | -                                    | Free                           | -            | -                | -                  |
| Meta              | Muse Glimmer 30B             | meta-models/Muse-Glimmer-30B            | 131072         | \$0.35                        | \$0.04                               | \$1.50                         | FP8          | -                | -                  |
| Qwen              | Qwen3.8 Flash                | Qwen/Qwen3.8-Flash                      | 1000000        | \$0.09                        | -                                    | \$0.282                        | -            | -                | -                  |
| DeepSeek          | DeepSeek V4.1 Flash          | deepseek-ai/DeepSeek-V4.1-Flash         | 1000000        | \$0.30                        | \$0.006                              | \$1.20                         | FP8          | Yes              | Yes                |

**Chat model examples**

* [PDF to chat app](https://www.pdftochat.com/): Chat with your PDFs (blogs, textbooks, papers).
* [Open deep research notebook](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Together_Open_Deep_Research_CookBook.ipynb): Generate long form reports using a single prompt.
* [RAG with reasoning models notebook](https://github.com/togethercomputer/together-cookbook/blob/main/RAG_with_Reasoning_Models.ipynb): RAG with DeepSeek-R1.
* [Fine-tuning chat models notebook](https://github.com/togethercomputer/together-cookbook/blob/main/Finetuning/Finetuning_Guide.ipynb): Tune language models for conversation.
* [Building agents](https://github.com/togethercomputer/together-cookbook/tree/main/Agents): Agent workflows with language models.

## Image models

Use our [Images](/reference/post-images-generations) endpoint for image models.

| Organization      | Model name                                       | Model string for API                     | Price per MP | Default steps |
| :---------------- | :----------------------------------------------- | :--------------------------------------- | :----------- | :------------ |
| Google            | Imagen 4.0 Preview                               | google/imagen-4.0-preview                | \$0.04       | -             |
| Google            | Imagen 4.0 Fast                                  | google/imagen-4.0-fast                   | \$0.02       | -             |
| Google            | Imagen 4.0 Ultra                                 | google/imagen-4.0-ultra                  | \$0.06       | -             |
| Google            | Flash Image 2.5 (Nano Banana)                    | google/flash-image-2.5                   | \$0.039      | -             |
| Google            | Gemini 3 Pro Image (Nano Banana Pro)             | google/gemini-3-pro-image                | \$0.134      | -             |
| Black Forest Labs | Flux1.1 \[pro]                                   | black-forest-labs/FLUX.1.1-pro           | \$0.04       | -             |
| Black Forest Labs | Flux.1 Kontext \[pro]                            | black-forest-labs/FLUX.1-kontext-pro     | \$0.04       | 28            |
| Black Forest Labs | Flux.1 Kontext \[max]                            | black-forest-labs/FLUX.1-kontext-max     | \$0.08       | 28            |
| Black Forest Labs | FLUX.2 \[pro]                                    | black-forest-labs/FLUX.2-pro             | \$0.03       | -             |
| Black Forest Labs | FLUX.2 \[dev]                                    | black-forest-labs/FLUX.2-dev             | \$0.0154     | -             |
| Black Forest Labs | FLUX.2 \[flex]                                   | black-forest-labs/FLUX.2-flex            | \$0.03       | -             |
| ByteDance         | Seedream 3.0                                     | ByteDance-Seed/Seedream-3.0              | \$0.018      | -             |
| ByteDance         | Seedream 4.0                                     | ByteDance-Seed/Seedream-4.0              | \$0.03       | -             |
| ByteDance         | Seedream 5.0 Lite                                | ByteDance/Seedream-5.0-lite              | \$0.035      | -             |
| Qwen              | Qwen Image                                       | Qwen/Qwen-Image                          | \$0.0058     | -             |
| RunDiffusion      | Juggernaut Pro Flux                              | RunDiffusion/Juggernaut-pro-flux         | \$0.0049     | -             |
| RunDiffusion      | Juggernaut Lightning Flux                        | Rundiffusion/Juggernaut-Lightning-Flux   | \$0.0017     | -             |
| Ideogram          | Ideogram 3.0                                     | ideogram/ideogram-3.0                    | \$0.06       | -             |
| Stability AI      | SD XL                                            | stabilityai/stable-diffusion-xl-base-1.0 | \$0.0019     | -             |
| Black Forest Labs | FLUX.2 \[max]                                    | black-forest-labs/FLUX.2-max             | \$0.07       | 50            |
| Google            | Gemini 3.1 Flash Image (Nano Banana 2)           | google/flash-image-3.1                   | \$0.05       | -             |
| OpenAI            | GPT Image 1.5                                    | openai/gpt-image-1.5                     | \$0.034      | -             |
| Qwen              | Qwen Image 2.0                                   | Qwen/Qwen-Image-2.0                      | \$0.035      | -             |
| Qwen              | Qwen Image 2.0 Pro                               | Qwen/Qwen-Image-2.0-Pro                  | \$0.075      | -             |
| Wan-AI            | Wan 2.6 Image                                    | Wan-AI/Wan2.6-image                      | \$0.03       | -             |
| ideogram          | Ideogram 4.0                                     | ideogram/ideogram-4.0                    | \$0.06       | -             |
| OpenAI            | GPT Image 2                                      | openai/gpt-image-2                       | \$0.053      | -             |
| Google            | Gemini 3.1 Flash-Lite Image (Nano Banana 2 Lite) | google/flash-image-3.1-lite              | \$0.069      | -             |
| Pruna AI          | P-Image-Ideogram                                 | prunaai/p-image-ideogram                 | \$0.00225    | -             |

<Note>
  Calling image models requires a positive credit balance.
</Note>

### **Image model examples**

* [Blinkshot.io](https://www.blinkshot.io/): A realtime AI image playground built with Flux Schnell.
* [Logo creator](https://www.logo-creator.io/): A logo generator that creates professional logos in seconds using Flux Pro 1.1.
* [PicMenu](https://www.picmenu.co/): A menu visualizer that takes a restaurant menu and generates nice images for each dish.
* [Flux LoRA inference notebook](https://github.com/togethercomputer/together-cookbook/blob/main/Flux_LoRA_Inference.ipynb): Using LoRA fine-tuned image generations models.

**FLUX pricing**

For FLUX models (excluding pro models) pricing is based on the size of generated images in megapixels and the number of steps used (if the number of steps exceeds the default steps).

* **Default pricing:** The listed per megapixel prices are for the default number of steps.
* **Using more or fewer steps:** Costs are adjusted based on the number of steps used **only if you go above the default steps**. If you use more steps, the cost increases proportionally using the formula below. If you use fewer steps, the cost *does not* decrease and is based on the default rate.

Here's a formula to calculate cost:

Cost = MP × Price per MP × (Steps ÷ Default Steps)

Where:

* MP = (Width × Height ÷ 1,000,000).
* Price per MP = Cost for generating one megapixel at the default steps.
* Steps = The number of steps used for the image generation. This is only factored in if going above default steps.

### **Gemini 3 Pro Image** pricing

Gemini 3 Pro Image offers pricing based on the resolution of the image.

* 1080p and 2K: \$0.134/image.
* 4K resolution: \$0.24/image.

Supported dimensions: 1K: 1024×1024 (1:1), 1264×848 (3:2), 848×1264 (2:3), 1200×896 (4:3), 896×1200 (3:4), 928×1152 (4:5), 1152×928 (5:4), 768×1376 (9:16), 1376×768 (16:9), 1548×672 or 1584×672 (21:9).

2K: 2048×2048 (1:1), 2528×1696 (3:2), 1696×2528 (2:3), 2400×1792 (4:3), 1792×2400 (3:4), 1856×2304 (4:5), 2304×1856 (5:4), 1536×2752 (9:16), 2752×1536 (16:9), 3168×1344 (21:9).

4K: 4096×4096 (1:1), 5096×3392 or 5056×3392 (3:2), 3392×5096 or 3392×5056 (2:3), 4800×3584 (4:3), 3584×4800 (3:4), 3712×4608 (4:5), 4608×3712 (5:4), 3072×5504 (9:16), 5504×3072 (16:9), 6336×2688 (21:9).

## Vision models

If you're not sure which vision model to use, start with **Qwen3.5 9B** (`Qwen/Qwen3.5-9B`). For model-specific rate limits, see [Rate limits](/docs/serverless/rate-limits).

| Organization | Model name | API model string     | Context length | Input pricing (per 1M tokens) | Output pricing (per 1M tokens) |
| :----------- | :--------- | :------------------- | :------------- | :---------------------------- | :----------------------------- |
| Qwen         | Qwen3.5 9B | Qwen/Qwen3.5-9B      | 262144         | \$0.17                        | \$0.25                         |
| Minimax      | Minimax M3 | MiniMaxAI/MiniMax-M3 | 524288         | \$0.30                        | \$1.20                         |
| Moonshot     | Kimi K3    | moonshotai/Kimi-K3   | 1048576        | \$3.00                        | \$15.00                        |

### **Vision model examples**

* [LlamaOCR](https://llamaocr.com/): A tool that takes documents (like receipts) and outputs markdown.
* [Wireframe to code](https://www.napkins.dev/): A wireframe to app tool that takes in a UI mockup of a site and gives you React code.
* [Extracting structured data from images](https://github.com/togethercomputer/together-cookbook/blob/main/Structured_Text_Extraction_from_Images.ipynb): Extract information from images as JSON.

## Video models

| Organization      | Model name          | Model string for API        | Price per video | Resolution / duration |
| :---------------- | :------------------ | :-------------------------- | :-------------- | :-------------------- |
| MiniMax           | MiniMax 01 Director | minimax/video-01-director   | \$0.28          | 720p / 5s             |
| MiniMax           | MiniMax Hailuo 02   | minimax/hailuo-02           | \$0.49          | 768p / 10s            |
| Google            | Veo 2.0             | google/veo-2.0              | \$2.50          | 720p / 5s             |
| ByteDance         | Seedance 1.0 Pro    | ByteDance/Seedance-1.0-pro  | \$0.57          | 1080p / 5s            |
| PixVerse          | PixVerse v5         | pixverse/pixverse-v5        | \$0.30          | 1080p / 5s            |
| Kuaishou          | Kling 2.1 Standard  | kwaivgI/kling-2.1-standard  | \$0.18          | 720p / 5s             |
| Vidu              | Vidu Q1             | vidu/vidu-q1                | \$0.22          | 1080p / 5s            |
| OpenAI            | Sora 2              | openai/sora-2               | \$0.80          | 720p / 8s             |
| OpenAI            | Sora 2 Pro          | openai/sora-2-pro           | \$2.40          | 1080p / 8s            |
| PixVerse          | PixVerse v5.6       | pixverse/pixverse-v5.6      | \$0.1326        | -                     |
| Wan-AI            | Wan 2.7 T2V         | Wan-AI/wan2.7-t2v           | \$0.10          | -                     |
| Google            | Veo 3.1 Debug Test  | google/veo-3.1-test-debug   | \$0.08          | -                     |
| Vidu              | Vidu Q3             | vidu/vidu-q3                | \$0.0975        | -                     |
| Vidu              | Vidu Q3 Turbo       | vidu/vidu-q3-turbo          | \$0.195         | -                     |
| Wan-AI            | Wan 2.7 I2V         | Wan-AI/wan2.7-i2v           | \$0.10          | -                     |
| Wan-AI            | Wan 2.7 R2V         | Wan-AI/wan2.7-r2v           | \$0.10          | -                     |
| PixVerse          | PixVerse v6         | pixverse/pixverse-v6        | \$0.09          | -                     |
| Alibaba           | HappyHorse 1.0 T2V  | alibaba/happyhorse-1.0-t2v  | \$0.24          | -                     |
| ByteDance         | Seedance 2.0        | ByteDance/Seedance-2.0      | \$0.16          | -                     |
| Alibaba           | HappyHorse 1.0 I2V  | alibaba/happyhorse-1.0-i2v  | \$0.24          | -                     |
| Alibaba           | HappyHorse 1.0 R2V  | alibaba/happyhorse-1.0-r2v  | \$0.24          | -                     |
| Google            | Veo 3.1             | google/veo-3.1              | \$0.08          | -                     |
| Google            | Veo 3.1 Lite        | google/veo-3.1-lite         | \$0.05          | -                     |
| Alibaba           | HappyHorse 1.1 I2V  | alibaba/happyhorse-1.1-i2v  | \$0.14          | -                     |
| Alibaba           | HappyHorse 1.1 R2V  | alibaba/happyhorse-1.1-r2v  | \$0.14          | -                     |
| Alibaba           | HappyHorse 1.1 T2V  | alibaba/happyhorse-1.1-t2v  | \$0.14          | -                     |
| Black Forest Labs | FLUX 3              | black-forest-labs/FLUX-3    | \$0.17          | -                     |
| ByteDance         | Seedance 2.5        | ByteDance/Seedance-2.5      | \$0.115         | -                     |
| ByteDance         | Seedance 1.0 Lite   | ByteDance/Seedance-1.0-lite | \$0.143         | 720p / 5s             |
| MiniMax           | MiniMax H3          | MiniMaxAI/MiniMax-H3        | \$0.1391        | 2k / 1s               |

## Audio models

Use our [Audio](/reference/audio-speech) endpoint for text-to-speech models. For speech-to-text models see [Transcription](/reference/audio-transcriptions) and [Translations](/reference/audio-translations).

| Organization | Modality       | Model name                             | Model string for API                   | Pricing                |
| :----------- | :------------- | :------------------------------------- | :------------------------------------- | :--------------------- |
| Canopy Labs  | Text-to-Speech | Orpheus 3B                             | canopylabs/orpheus-3b-0.1-ft           | \$15.00 per 1M chars   |
| Kokoro       | Text-to-Speech | Kokoro                                 | hexgrad/Kokoro-82M                     | \$4.00 per 1M chars    |
| Cartesia     | Text-to-Speech | Cartesia Sonic 3                       | cartesia/sonic-3                       | \$65.00 per 1M chars   |
| Cartesia     | Text-to-Speech | Cartesia Sonic 2                       | cartesia/sonic-2                       | \$65.00 per 1M chars   |
| Cartesia     | Text-to-Speech | Cartesia Sonic                         | cartesia/sonic                         | \$65.00 per 1M chars   |
| OpenAI       | Speech-to-Text | Whisper Large v3                       | openai/whisper-large-v3                | \$0.0015 per audio min |
| NVIDIA       | Speech-to-Text | Parakeet TDT 0.6B v3                   | nvidia/parakeet-tdt-0.6b-v3            | \$0.0015 per audio min |
| NVIDIA       | Speech-to-Text | NVIDIA Nemotron 3 ASR Streaming 0.6B   | nvidia/nemotron-3-asr-streaming-0.6b   | \$0.0015 per audio min |
| NVIDIA       | Speech-to-Text | NVIDIA Nemotron 3.5 ASR Streaming 0.6B | nvidia/nemotron-3.5-asr-streaming-0.6b | \$0.0015 per audio min |

**Audio model examples**

* [PDF to podcast notebook](https://github.com/togethercomputer/together-cookbook/blob/main/PDF_to_Podcast.ipynb): Generate a NotebookLM style podcast given a PDF.
* [Audio podcast agent workflow](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Serial_Chain_Agent_Workflow.ipynb): Agent workflow to generate audio files given input content.

## Embedding models

There are currently no embedding models offered via serverless.

### **Embedding model examples**

* [Code generation agent](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Looping_Agent_Workflow.ipynb): An agent workflow to generate and iteratively improve code.
* [Multimodal search and image generation](https://github.com/togethercomputer/together-cookbook/blob/main/Multimodal_Search_and_Conditional_Image_Generation.ipynb): Search for images and generate more similar ones.
* [Visualizing embeddings](https://github.com/togethercomputer/together-cookbook/blob/main/Embedding_Visualization.ipynb): Visualizing and clustering vector embeddings.

## Rerank models

There are currently no rerank models offered via serverless. Rerank models like `mixedbread-ai/mxbai-rerank-large-v2` are only available with [dedicated model inference](/docs/dedicated-endpoints/models).

### **Rerank model examples**

* [Search and reranking](https://github.com/togethercomputer/together-cookbook/blob/main/Search_with_Reranking.ipynb): Simple semantic search pipeline improved using a reranker.
* [Implementing hybrid search notebook](https://github.com/togethercomputer/together-cookbook/blob/main/Open_Contextual_RAG.ipynb): Implementing semantic + lexical search along with reranking.

## Moderation models

There are currently no moderation models offered via serverless.
