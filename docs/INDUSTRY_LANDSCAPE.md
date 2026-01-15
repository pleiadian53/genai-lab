# Industry Landscape: Generative AI in Drug Discovery & Computational Biology

This document surveys companies and platforms pioneering generative AI, foundation models, and machine learning approaches for drug discovery, treatment response prediction, and biological research.

> **Last Updated:** December 2024  
> **Purpose:** Track industry developments, identify research directions, and gather ideas for this project.

---

## Table of Contents

1. [DNA Foundation Models & Sequence Generation](#dna-foundation-models--sequence-generation)
2. [Single-Cell Foundation Models](#single-cell-foundation-models)
3. [Gene Expression & Multi-Omics](#gene-expression--multi-omics-foundation-models)
4. [Splicing & RNA Processing](#splicing--rna-processing)
5. [Protein & Structure-Based Discovery](#protein--structure-based-discovery)
6. [Gene Editing & CRISPR](#gene-editing--crispr)
7. [Clinical & Treatment Response](#clinical--treatment-response)
8. [AI-Driven Target Discovery](#ai-driven-target-discovery)
9. [Key Observations & Research Directions](#key-observations--research-directions)

---

## DNA Foundation Models & Sequence Generation

Foundation models for DNA sequence understanding and generation — enabling synthetic genome design, variant effect prediction, and regulatory element discovery.

### Evo 2 (Arc Institute)

| | |
|---|---|
| **Website** | [arcinstitute.org/tools/evo](https://arcinstitute.org/tools/evo) |
| **Focus** | DNA foundation model for generalist prediction and design |
| **Key Technology** | **Evo 2** (40B parameters, 1M context length) |
| **Partners** | NVIDIA |

**What They Do:**

- Genomic foundation model for DNA, RNA, and protein tasks
- Single-nucleotide resolution with near-linear scaling
- Trained on 9+ trillion nucleotides from 128,000+ species (all domains of life)
- Generative design of synthetic DNA sequences

**Technical Details:**

- 40 billion parameters
- 1 megabase (1M tokens) context length
- Frontier deep learning architecture
- Evo Designer tool for sequence generation
- Open source on GitHub and Hugging Face

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- State-of-the-art in DNA generation
- Long-range context is critical for genomics
- Open source enables direct experimentation

**Resources:**

- [Evo 2 Preprint](https://arcinstitute.org/manuscripts/Evo2)
- [Evo 1 in Science (Nov 2024)](https://www.science.org/doi/10.1126/science.ado9336)
- [GitHub](https://github.com/arcinstitute/evo2)

---

### Nucleotide Transformer (InstaDeep)

| | |
|---|---|
| **Website** | [instadeep.com](https://www.instadeep.com/) |
| **Focus** | DNA foundation models for molecular phenotype prediction |
| **Key Technology** | **Nucleotide Transformer** (up to 2.5B parameters) |
| **Published** | Nature Methods (Nov 2024) |

**What They Do:**

- Foundation models pre-trained on DNA sequences
- Transfer learning for genomic tasks with limited labeled data
- Multi-species genome understanding

**Technical Details:**

- Models from 50M to 2.5B parameters
- Multispecies 2.5B outperforms single-species models
- 18 benchmark tasks for evaluation
- Interactive leaderboard for comparison

**Relevance to This Project:** ⭐⭐⭐⭐

- Established benchmark for DNA foundation models
- Transfer learning approach applicable to expression

**Resources:**

- [GitHub](https://github.com/instadeepai/nucleotide-transformer)
- [Nature Methods Paper](https://www.nature.com/articles/s41592-024-02523-z)

---

### HyenaDNA (Stanford/Hazy Research)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/HazyResearch/hyena-dna) |
| **Focus** | Long-range genomic sequence modeling |
| **Key Technology** | **HyenaDNA** (up to 1M context) |
| **Architecture** | Hyena (sub-quadratic attention alternative) |

**What They Do:**

- Genomic foundation model with ultra-long context
- Single nucleotide resolution
- Pre-trained on human reference genome

**Technical Details:**

- Up to 1 million token context (500x increase over dense attention)
- Hyena operator for efficient long-range modeling
- Fine-tunable for downstream tasks

**Relevance to This Project:** ⭐⭐⭐⭐

- Efficient architecture for long sequences
- Applicable to regulatory element prediction

---

### Caduceus (Cornell)

| | |
|---|---|
| **Website** | [caduceus-dna.github.io](https://caduceus-dna.github.io/) |
| **Focus** | Bi-directional DNA sequence modeling |
| **Key Technology** | **Caduceus** (BiMamba + RC equivariance) |
| **Architecture** | Mamba-based |

**What They Do:**

- First family of RC (reverse complement) equivariant DNA models
- Bi-directional modeling for upstream/downstream context
- Long-range sequence modeling

**Technical Details:**

- BiMamba: bi-directional Mamba block
- MambaDNA: RC equivariant extension
- Handles biological symmetry of DNA strands

**Relevance to This Project:** ⭐⭐⭐⭐

- Novel architecture addressing DNA-specific challenges
- Mamba efficiency for long sequences

---

### DNABERT / DNABERT-2

| | |
|---|---|
| **Website** | [GitHub](https://github.com/MAGICS-LAB/DNABERT_2) |
| **Focus** | Pre-trained bidirectional encoder for DNA |
| **Key Technology** | **DNABERT-2** (ICLR 2024) |
| **Benchmark** | Genome Understanding Evaluation (GUE) |

**What They Do:**

- BERT-style pre-training for DNA sequences
- Multi-species genome understanding
- DNABERT-S for DNA embeddings that cluster by genome

**Technical Details:**

- Efficient foundation model
- 28 datasets in GUE benchmark
- Transfer learning to downstream tasks

**Relevance to This Project:** ⭐⭐⭐

- Established baseline for DNA language models
- Comprehensive benchmark suite

---

## Single-Cell Foundation Models

Foundation models specifically designed for single-cell transcriptomics data.

### Geneformer

| | |
|---|---|
| **Website** | [Hugging Face](https://huggingface.co/ctheodoris/Geneformer) |
| **Focus** | Transfer learning for single-cell biology |
| **Key Technology** | **Geneformer-V2** (104M-316M parameters) |
| **Published** | Nature (2023) |

**What They Do:**

- Foundation model for context-specific gene network predictions
- Pre-trained on ~104M human single-cell transcriptomes
- Zero-shot and fine-tuning capabilities

**Technical Details:**

- Geneformer-V2-316M: latest model (Dec 2024)
- Input size: 4096 tokens
- Vocabulary: ~20K protein-coding genes
- Rank-value encoding of gene expression

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Primary inspiration for transformer-based expression modeling
- Representation learning that can be extended to generation

**Resources:**

- [Nature Paper](https://www.nature.com/articles/s41586-023-06139-9)
- [Hugging Face](https://huggingface.co/ctheodoris/Geneformer)

---

### scGPT

| | |
|---|---|
| **Website** | [GitHub](https://github.com/bowang-lab/scGPT) |
| **Focus** | Generative pre-trained transformer for single-cell |
| **Key Technology** | **scGPT** |
| **Published** | Nature Methods (2024) |

**What They Do:**

- Foundation model for single-cell multi-omics
- Generative capabilities for cell state prediction
- Pre-trained on 33+ million cells

**Technical Details:**

- Generative pre-trained transformer architecture
- Multi-task learning: cell type annotation, perturbation prediction, multi-omics integration
- Attention-based gene-gene interaction modeling

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Directly relevant — generative model for single-cell
- Perturbation prediction aligns with counterfactual goals

**Resources:**

- [Nature Methods Paper](https://www.nature.com/articles/s41592-024-02201-0)

---

## Gene Expression & Multi-Omics Foundation Models

Companies building foundation models specifically for gene expression, transcriptomics, and multi-omics data.

### Synthesize Bio

| | |
|---|---|
| **Website** | https://www.synthesize.bio/ |
| **Focus** | Generative foundation models for gene expression |
| **Key Technology** | **GEM-1** — Gene Expression Model |
| **Approach** | Generate biologically realistic gene expression data in silico |

**What They Do:**

- Build generative models that can simulate gene expression profiles under various conditions
- Enable in silico experimentation that bridges wet lab and computation
- Multi-omics and public + private data harmonization
- Platform for drug discovery, hypothesis testing, and clinical decision support

**Relevance to This Project:** ⭐⭐⭐⭐⭐ (Primary inspiration)
- Direct alignment with our cVAE and counterfactual simulation goals
- Their GEM-1 represents the state-of-the-art we're studying

**Key Blog Posts:**

- https://www.synthesize.bio/blog

---

### Deep Genomics

| | |
|---|---|
| **Website** | https://www.deepgenomics.com/ |
| **Focus** | RNA biology and therapeutics |
| **Key Technology** | **BigRNA** (~2 billion parameters) |
| **Founded** | 2014 (Toronto) |

**What They Do:**

- First transformer neural network engineered specifically for transcriptomics
- Predicts tissue-specific regulatory mechanisms of RNA expression
- Predicts binding sites of proteins and microRNAs
- Predicts effects of genetic variants and therapeutic candidates

**Technical Details:**

- ~2 billion adjustable parameters
- Trained on thousands of datasets (>1 trillion genomic signals)
- Designed to understand complex RNA interactions

**Relevance to This Project:** ⭐⭐⭐⭐⭐
- BigRNA is a foundation model for RNA/transcriptomics — directly relevant
- Their approach to predicting variant effects aligns with counterfactual reasoning

---

### Helical

| | |
|---|---|
| **Website** | https://www.helical.ai/ |
| **Focus** | Open-source DNA/RNA foundation models |
| **Key Technology** | **Helix-mRNA** (hybrid foundation model) |
| **Founded** | 2023 (Luxembourg) |
| **Funding** | €2.2M seed (June 2024) |

**What They Do:**

- First open-source platform dedicated to bio foundation models for DNA and RNA
- Democratize access to advanced AI tools for pharma/biotech
- Library of Bio AI Agents for tasks like biomarker discovery and target prediction

**Technical Details:**

- Helix-mRNA: hybrid foundation model for mRNA therapeutics
- Outperforms prior methods in modeling UTRs and long-sequence regions
- Uses only ~10% of parameters of comparable models
- Available on AWS Marketplace

**Relevance to This Project:** ⭐⭐⭐⭐
- Open-source focus aligns with our goals
- Could potentially integrate their models or learn from their architecture

---

### Noetik

| | |
|---|---|
| **Website** | https://www.noetik.ai/ |
| **Focus** | Cancer biology and treatment prediction |
| **Key Technology** | **OCTO** model |
| **Founded** | 2022 (San Francisco) |
| **Funding** | $40M Series A (2024) |

**What They Do:**

- AI model that acts like a virtual lab for cancer research
- Predicts how different cancer treatments might play out in real patients
- Tests "what if" scenarios for treatment optimization

**Technical Details:**

- OCTO trained on thousands of tumor samples
- Integrates gene expression, protein data, and cell images
- Predicts how tweaking a single gene could change protein levels across a tumor
- In vivo CRISPR Perturb-Map platform for validation

**Relevance to This Project:** ⭐⭐⭐⭐
- Their "what if" scenario testing is exactly counterfactual reasoning
- Multi-modal integration (expression + protein + images) is advanced

---

### BioMap

| | |
|---|---|
| **Website** | https://www.biomap.com/ |
| **Focus** | Multi-modal biological foundation models |
| **Key Technology** | **xTrimo** (~210 billion parameters) |
| **Partnerships** | Sanofi ($10M upfront, >$1B potential milestones) |

**What They Do:**

- World's largest life science foundation model
- Supports DNA, RNA, protein, cellular, and systems-level modalities
- Designed to understand and predict biological behavior across multiple modalities

**Technical Details:**

- ~210 billion parameters (as of 2025)
- Cross-Modal Transformer Representation of Interactome and Multi-Omics
- GPU-accelerated deployment using multi-expert architectures and FP8 precision

**Relevance to This Project:** ⭐⭐⭐⭐
- Multi-modal approach is the future direction
- Scale demonstrates what's possible with sufficient resources

---

## Splicing & RNA Processing

AI models for predicting and understanding alternative splicing, splice site recognition, and RNA processing — directly relevant to the meta-spliceai project.

### SpliceAI (Illumina)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/Illumina/SpliceAI) |
| **Focus** | Deep learning for splice site prediction |
| **Key Technology** | **SpliceAI** |
| **Published** | Cell (2019) |

**What They Do:**

- Predict splicing alterations from DNA sequence
- Identify cryptic splice sites created by variants
- Score variant pathogenicity based on splicing impact

**Technical Details:**

- Deep residual neural network
- 10,000 nucleotide context window
- Predicts donor/acceptor gain/loss
- Pre-computed scores available for all SNVs

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Foundation for meta-spliceai project
- Demonstrates deep learning for splicing prediction
- Well-established benchmark

**Resources:**

- [GitHub](https://github.com/Illumina/SpliceAI)
- [Cell Paper](https://www.cell.com/cell/fulltext/S0092-8674(18)31629-5)

---

### Splam (Johns Hopkins)

| | |
|---|---|
| **Website** | [GitHub](https://github.com/Kuanhao-Chao/splam) |
| **Focus** | Splice junction recognition |
| **Key Technology** | **Splam** |
| **Published** | 2024 |

**What They Do:**

- Improved splice junction recognition over SpliceAI
- Better accuracy with less genomic data
- Cross-species generalization

**Technical Details:**

- Deep learning model for splice sites
- Outperforms SpliceAI on benchmarks
- Generalizes across species

**Relevance to This Project:** ⭐⭐⭐⭐

- State-of-the-art in splice prediction
- Cross-species transfer learning

---

### Pangolin

| | |
|---|---|
| **Focus** | Tissue-specific splicing prediction |
| **Key Technology** | **Pangolin** |

**What They Do:**

- Predict tissue-specific alternative splicing
- Model splicing quantitative trait loci (sQTLs)

**Relevance to This Project:** ⭐⭐⭐⭐

- Tissue-specific conditioning aligns with our cVAE approach
- Splicing as a form of gene regulation

---

## Gene Editing & CRISPR

Generative AI for designing gene editors and optimizing CRISPR systems.

### Profluent Bio

| | |
|---|---|
| **Website** | [profluent.bio](https://www.profluent.bio/) |
| **Focus** | AI-designed gene editors |
| **Key Technology** | **OpenCRISPR-1** |
| **Published** | Nature (2025) |

**What They Do:**

- First AI-designed gene editor to edit human genome
- Generative AI creates novel CRISPR proteins
- Open-source release of OpenCRISPR-1

**Technical Details:**

- Large language models trained on CRISPR-Cas sequences
- Generated novel, functional genome editors
- Improved properties vs natural systems
- RNA-programmable with NGG PAM preference

**Relevance to This Project:** ⭐⭐⭐⭐⭐

- Demonstrates generative AI for protein design
- Open-source enables experimentation
- LLM approach to biological sequences

**Resources:**

- [GitHub](https://github.com/Profluent-AI/OpenCRISPR)
- [Nature Paper](https://www.nature.com/articles/s41586-024-08178-6)

---

## Protein & Structure-Based Discovery

Companies focused on protein structure prediction, design, and protein-based therapeutics.

### Isomorphic Labs

| | |
|---|---|
| **Website** | https://www.isomorphiclabs.com/ |
| **Focus** | AI-first drug discovery |
| **Key Technology** | **AlphaFold 3** |
| **Parent** | Alphabet (DeepMind spin-off, 2021) |

**What They Do:**

- Reimagining drug discovery from first principles with AI-first approach
- Expanded from small molecules to biologics
- Internal pipeline focused on oncology and immunology

**Technical Details:**

- AlphaFold 3: predicts structure of proteins, DNA, RNA, ligands, and their interactions
- Released in collaboration with Google DeepMind
- Nobel Prize-winning foundation (AlphaFold 2)

**Relevance to This Project:** ⭐⭐⭐
- Structure prediction complements expression modeling
- Their approach to "AI-first" drug discovery is instructive

---

### EvolutionaryScale

| | |
|---|---|
| **Website** | https://www.evolutionaryscale.ai/ |
| **Focus** | Protein design and engineering |
| **Key Technology** | **ESM3** (generative protein model) |
| **Founded** | 2024 (Meta FAIR spin-off) |
| **Funding** | $142M |

**What They Do:**

- Programmable biology for protein engineering
- Target cancer cells, find alternatives to plastics, environmental mitigations

**Technical Details:**

- ESM3: simultaneously reasons over sequence, structure, and function of proteins
- Third-generation ESM model
- Trained on NVIDIA H100 GPUs
- ESM Cambrian: parallel model family for protein understanding

**Relevance to This Project:** ⭐⭐⭐
- Generative approach to proteins parallels our approach to expression
- Their training methodology is instructive

---

### Generate:Biomedicines

| | |
|---|---|
| **Website** | https://generatebiomedicines.com/ |
| **Focus** | Protein therapeutics via generative AI |
| **Key Technology** | **Generative Biology™** platform |

**What They Do:**

- Pioneer of "Generative Biology" — generating custom protein therapeutics
- From peptides to antibodies, enzymes, gene therapies
- Generate, build, measure, learn loop

**Clinical Progress:**

- GB-0895: Phase 3 for severe asthma (anti-TSLP antibody)
- GB-0669: Phase 1 completed with positive results

**Relevance to This Project:** ⭐⭐⭐
- Their generate-build-measure-learn loop is a good framework
- Demonstrates clinical translation of generative approaches

---

### Chai Discovery

| | |
|---|---|
| **Website** | https://www.chaidiscovery.com/ |
| **Focus** | Molecular structure prediction and antibody design |
| **Key Technology** | **Chai-1** (structure), **Chai-2** (antibody design) |
| **Funding** | $100M total ($70M Series A, 2025) |
| **Investors** | Thrive Capital, OpenAI |

**What They Do:**

- Open-source multi-modal foundation model for molecular structure
- Unifies predictions across proteins, small molecules, DNA, RNA, covalent modifications

**Technical Details:**

- Chai-1: 77% success rate on PoseBusters (vs 76% AlphaFold3)
- Can operate without MSAs (reduces compute demands)
- Chai-2: ~16% hit rate for de novo antibody design across 52 novel antigens

**Relevance to This Project:** ⭐⭐⭐
- Open-source approach is valuable
- Zero-shot antibody design is impressive generative capability

---

### Recursion

| | |
|---|---|
| **Website** | https://www.recursion.com/ |
| **Focus** | Phenomics + drug discovery |
| **Key Technology** | **Phenom-Beta**, **BioHive-2** supercomputer |

**What They Do:**

- Merge AI with massive biological datasets
- Process cellular microscopy images into general-purpose embeddings
- In-silico fluorescent staining from brightfield images

**Technical Details:**

- Phenom-Beta: vision transformer (ViT) with masked autoencoders
- Trained on RxRx3 dataset (~2.2M images, ~17K knockouts, 1,674 chemicals)
- BioHive-2: 504 NVIDIA H100 GPUs
- Partnership with MIT for open-source protein co-folding model

**Relevance to This Project:** ⭐⭐⭐
- Phenomics (image-based) complements transcriptomics
- Their self-supervised approach is relevant

---

## Clinical & Treatment Response

Companies focused on clinical trials, treatment optimization, and patient response prediction.

### Insilico Medicine

| | |
|---|---|
| **Website** | https://insilico.com/ |
| **Focus** | End-to-end AI drug discovery |
| **Key Technology** | **Pharma.AI**, **Precious3GPT** |
| **Founded** | 2014 |
| **Funding** | $110M Series E (2025) |

**What They Do:**

- Fully integrated drug discovery suite
- PandaOmics: discover and prioritize novel targets
- Chemistry42: generate novel molecules
- InClinico: design and predict clinical trials

**Technical Details:**

- Precious3GPT: multi-omics, cross-species foundation transformer for aging research
- Ingests data from rats, monkeys, humans across transcriptomics, proteomics, methylation
- Enables virtual experiments to forecast compound effects on aging hallmarks
- Available on Hugging Face

**Clinical Progress:**

- Rentosertib (ISM001-055): AI-discovered drug, Phase 2a results published in Nature Medicine
- First AI-discovered drug to show clinical proof-of-concept

**Relevance to This Project:** ⭐⭐⭐⭐⭐
- Precious3GPT is directly relevant — multi-omics generative model
- Their end-to-end approach shows the full pipeline
- Clinical validation demonstrates real-world impact

---

### Tempus

| | |
|---|---|
| **Website** | https://www.tempus.com/ |
| **Focus** | Precision medicine with real-world data |
| **Key Technology** | **Tempus One** (AI platform) |

**What They Do:**

- AI-enabled precision medicine
- Predict response to therapies with greater accuracy
- Uncover novel biomarkers from real-world data

**Technical Details:**

- Integrates clinical data with AI-driven algorithms
- Neural-network-based high-throughput drug screening
- Generative AI capabilities for querying healthcare data

**Relevance to This Project:** ⭐⭐⭐
- Real-world data integration is important for validation
- Treatment response prediction aligns with our goals

---

### Owkin

| | |
|---|---|
| **Website** | https://www.owkin.com/ |
| **Focus** | Clinical trials and digital pathology |
| **Key Technology** | **Federated learning**, **SecureFedYJ** |

**What They Do:**

- AI models for drug discovery and clinical trial optimization
- Federated learning: train AI without centralizing data
- Digital pathology AI diagnostics

**Technical Details:**

- Owkin Studio: federated learning platform (40% of revenue)
- Owkin Connect: AI models for drug discovery (35% of revenue)
- SecureFedYJ: secure federated learning algorithm

**Partnerships:**

- Amgen: cardiovascular prediction
- AstraZeneca: AI tool for gBRCA mutation screening

**Relevance to This Project:** ⭐⭐⭐
- Federated learning is important for privacy-preserving multi-site studies
- Clinical trial optimization is downstream application

---

### Retro Biosciences

| | |
|---|---|
| **Website** | https://www.retro.bio/ |
| **Focus** | Cellular reprogramming and longevity |
| **Key Technology** | **GPT-4b micro** (with OpenAI) |
| **Funding** | $1B (led by Sam Altman) |

**What They Do:**

- Interventions to slow or reverse cellular aging
- Focus on neurodegeneration
- Combine wet-lab biology with computational methods

**Technical Details:**

- GPT-4b micro: biology-specialized foundation model
- Trained on protein sequences, biological literature, tokenized 3D structural data
- Redesigned Yamanaka transcription factors (RetroSOX, RetroKLF)
- 50-fold increases in pluripotency marker expression

**Relevance to This Project:** ⭐⭐⭐⭐
- Demonstrates LLM approach to biology
- Reprogramming is a form of counterfactual intervention

---

## AI-Driven Target Discovery

Companies using AI/ML for target identification and validation (not necessarily generative).

### Ochre Bio

| | |
|---|---|
| **Website** | https://www.ochre-bio.com/ |
| **Focus** | RNA therapeutics for liver disease |
| **Key Technology** | **OBELiX** platform |
| **Headquarters** | Oxford, UK |

**What They Do:**

- Developing RNA medicines for chronic liver diseases
- Built one of world's largest human liver functional genomics datasets (~120,000 samples)
- Combine machine learning with human validation models

**Technical Details:**

- Proprietary gene perturbation atlases + patient disease atlases
- Make causal predictions about drug targets
- Human validation: perfused livers, diseased tissue slices, primary cells
- In-house RNA chemistry

**Partnerships:**

- GSK: functional genomics and single-cell datasets
- Boehringer Ingelheim: chronic liver disease research

**Relevance to This Project:** ⭐⭐⭐
- Large-scale functional genomics data is valuable
- Causal predictions align with our counterfactual goals
- **Note:** Not building generative models, but using ML for target discovery

---

### Atomic AI

| | |
|---|---|
| **Website** | https://atomic.ai/ |
| **Focus** | RNA structure and drug discovery |
| **Key Technology** | **ATOM-1**, **PARSE** platform |
| **Funding** | ~$42M (seed + Series A) |

**What They Do:**

- AI-driven RNA drug discovery with atomic precision
- Predict RNA structural and functional properties
- Optimize RNA-targeted and RNA-based modalities

**Technical Details:**

- ATOM-1: large language model for RNA structure prediction
- PARSE: Platform for AI-driven RNA Structure Exploration
- Combined foundation-model + wet-lab loop

**Relevance to This Project:** ⭐⭐⭐
- RNA structure is complementary to expression
- Their foundation model + wet lab loop is a good paradigm

---

### Enveda Biosciences

| | |
|---|---|
| **Website** | https://enveda.com/ |
| **Focus** | Natural product drug discovery |
| **Key Technology** | **PRISM** foundation model |
| **Funding** | $300M+ (Series C + D, unicorn valuation) |
| **Investors** | Sanofi |

**What They Do:**

- Enhance molecular structure identification from natural products
- Self-supervised learning on mass spectrometry data

**Technical Details:**

- PRISM: Pretrained Representations Informed by Spectral Masking
- Trained on 1.2 billion small molecule mass spectra
- Masked peak modeling (similar to masked LM in NLP)

**Relevance to This Project:** ⭐⭐
- Different modality (mass spec vs expression)
- Self-supervised approach is transferable

---

## Key Observations & Research Directions

### Trends in the Industry

1. **Foundation Models Are Dominant**
   - Most companies are building large-scale foundation models
   - Parameters range from millions to 210 billion (BioMap xTrimo)
   - Self-supervised pretraining is standard

2. **Multi-Modal Integration**
   - Leading platforms integrate multiple data types
   - Expression + protein + structure + images
   - Cross-species and cross-tissue modeling

3. **Generative vs. Predictive**
   - Clear distinction between:
     - **Generative**: Synthesize Bio, Generate:Biomedicines, EvolutionaryScale
     - **Predictive**: Ochre Bio, Tempus, Owkin
   - Generative models enable counterfactual reasoning

4. **Clinical Translation**
   - Insilico Medicine leads with AI-discovered drugs in clinic
   - Validation in human systems is critical
   - Regulatory pathway is becoming clearer

5. **Open Source Movement**
   - Helical, Chai Discovery pushing open-source
   - Democratization of bio AI tools
   - Opportunity for academic contribution

### Research Directions for This Project

Based on industry analysis, priority areas:

1. **Conditional Generation with Biological Constraints**
   - Tissue/disease/batch conditioning (current focus)
   - Pathway-level constraints
   - Gene regulatory network priors

2. **Counterfactual Reasoning**
   - Treatment response prediction
   - Perturbation effect simulation
   - Causal inference integration

3. **Multi-Modal Extension**
   - Expression + protein (pseudobulk bridging)
   - Integration with structure predictions
   - Image-based phenomics

4. **Evaluation Frameworks**
   - DE agreement metrics
   - Pathway concordance
   - Batch leakage tests
   - Clinical validation proxies

5. **Scalability**
   - Efficient architectures (see Helical's 10% parameter efficiency)
   - Latent diffusion for high-dimensional data
   - Federated learning for multi-site data

---

## References

- [17 Companies Pioneering AI Foundation Models in Pharma and Biotech](https://www.biopharmatrend.com/business-intelligence/14-companies-pioneering-ai-foundation-models-in-pharma-and-biotech/)
- [NVIDIA BioNeMo Platform](https://blogs.nvidia.com/blog/drug-discovery-bionemo-generative-ai/)
- [12 AI Drug Discovery Companies You Should Know](https://www.labiotech.eu/best-biotech/ai-drug-discovery-companies/)
- Individual company websites and press releases
