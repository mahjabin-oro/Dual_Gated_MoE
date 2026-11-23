# MoE-TTA: Continual Test-Time Adaptation for Vision-Language Models through Mixture of Experts

Helping a large vision-language model (**CLIP**) adapt to **new, changing visual domains while it is being used**, with no labels and no full-model retraining, while keeping its general zero-shot ability.

> 🚧 The paper describes the full method. The code in this repository is the early experimental codebase it was developed from.

---

## The problem

Models in the real world keep meeting data that looks different from what they were trained on. Examples include a self-driving car in unexpected weather, a medical system seeing a new patient population, or photos turning into sketches, corrupted images or satellite views. Retraining every time is impractical, and at test time there are **no labels and no access to the original training data**.

**Continual Test-Time Domain Adaptation (CTTDA)** tackles this. The model adapts on the fly across a *sequence* of shifting domains, without forgetting what it already knows. It's hard because:

- Updating a large model for every new domain is **computationally expensive**
- Parameter updates can **overwrite earlier knowledge** and damage zero-shot ability
- Existing test-time methods (batch-norm updates, pseudo-labelling) can **build up errors** and struggle to tell apart features that are specific to one domain from features that hold across all domains

## Key idea

To our knowledge, MoE-TTA is the **first work to use a Mixture of Experts (MoE) for continual test-time adaptation on vision-language data.**

The text encoder of CLIP stays frozen and provides prompts such as *"A photo of a {class}"*. Only small **adapter blocks** inside the vision encoder learn.

### Inside each adapter block
1. **Feature disentanglement**: each input is split into
   - **domain-invariant features**: what stays the same across domains, kept stable to protect generalisation
   - **domain-specific features**: what is unique to the current domain
2. **Mixture of LoRA experts**: the domain-specific features go to a **router** that picks a few specialised **LoRA experts**. Only a small part of the network is active and updated, which keeps adaptation cheap.
3. **Gated feature fusion**: a small learned gate decides how much each feature type contributes to the final representation, balancing adaptability with generalisation.

### Two-stage training
- **Contrastive warm-up (before deployment)**: augmented views of images simulate several domains. A contrastive loss pulls the *domain-invariant* features of these views together and pushes the *domain-specific* features apart, so the two modules learn their separate roles.
- **Test-time adaptation (during deployment)**: with no labels, the model minimises **entropy loss** over its own predictions, averaged across augmented views of each test image. This guides it towards confident, consistent predictions in each new domain.

```
Test image → augmented views → CLIP vision encoder
     each transformer block: frozen attention + MLP
                             + adapter block:
                                 domain-invariant module ─┐
                                 domain-specific encoder → router → LoRA experts ─┤→ gated fusion
→ image feature ↔ cosine similarity ↔ frozen text features ("A photo of a {class}")
→ entropy loss (no labels) → update only the active adapter parameters
```

## Experiments

Evaluated in a continual test-time adaptation setting on two benchmark groups:

- **Set 1: ImageNet domain shifts**: ImageNet, ImageNet-C (15 corruption types), ImageNet-A (adversarial), ImageNet-V2, ImageNet-R, ImageNet-Sketch and ImageNet-D109
- **Set 2: fine-grained classification**: Aircraft, Caltech101, Stanford Cars, DTD (textures), EuroSAT (satellite), Flowers102, Food101, Oxford Pets, SUN397 (scenes) and UCF101 (human actions)

Compared against Zero-shot CLIP, TPT, CoOp, CoCoOp, ProDA, PromptAlign and VTE Ensemble.

## Results

| Benchmark | MoE-TTA average accuracy |
|---|---|
| ImageNet domain shifts (7 datasets) | **36.43%** |
| Fine-grained classification (10 datasets) | **32.93%** |

Highlights:
- **EuroSAT: 51.67%**, and **ImageNet-C: 73.4%**
- On fine-grained tasks, MoE-TTA **outperforms CoCoOp, ProDA, PromptAlign and VTE Ensemble on average**, and beats several baselines on Flowers102, SUN397 and UCF101
- Feature visualisations confirm that the model behaves as designed. Domain-invariant features mix across domains while staying separated by class, and domain-specific features separate clearly by domain.
- Expert-usage analysis shows that different experts are activated across different transformer layers

The model is not yet ahead of the strongest baselines on the ImageNet shifts. Future work focuses on better feature disentanglement and smarter expert selection.

## What's in this repository

The early experimental codebase: a frozen CLIP (ViT-B/16) with LoRA-style adapter experts, noisy top-k routing with load balancing, and experiment scripts and configs for multi-dataset and class-incremental settings.

## Tech stack

Python · PyTorch · OpenAI CLIP · LoRA · Mixture-of-Experts · Hydra

## Acknowledgements

The codebase started from **MoE-Adapters** (Yu et al., *"Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters"*, CVPR 2024), [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL) (Apache-2.0), which builds on OpenAI CLIP and ZSCL.
