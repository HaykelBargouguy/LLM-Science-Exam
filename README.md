# LLM Science Examination Task

## Project Overview
This project introduces the world of Large Language Models (LLMs), with the primary goal of developing a model capable of passing science examinations and responding accurately to complex science queries. This project enhances understanding of how to manipulate structured textual data, including prompts and contextual information.

## Data Description
- **Prompt**: User input which could either be a question or an instruction, identified by the column `is_question`.
- **Context**: Contains scraped information from Wikipedia to assist the model in formulating accurate responses.
- **A/B/C/D**: Four possible answers to the prompt.
- **Answer**: Specifies the correct answer from the options provided.

## Tasks
1. **Explore Strategies**: Managing LLM inputs, including prompt engineering and Retrieval-Augmented Generation (RAG).
2. **Train an LLM**: Assist students in answering challenging scientific questions.
3. **Experiment**: Fine-tuning techniques such as Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA), and Quantized Low-Rank Adaptation (QLoRA).
4. **Format Predictions**: As an array `[A, B, C, D]`, where A is the most likely correct answer and D is the least likely.
5. **Evaluate**: Using the Mean Average Precision (MAP@3) metric.

### Retrieval Augmented Generation (RAG)
RAG is an advanced architectural strategy that enhances the capabilities of large language models (LLM) by integrating external or custom data. The approach combines the power of neural networks with the efficiency of information retrieval techniques, enabling the model to access a vast repository of information that can significantly improve its responses.

### Key Libraries Used
1. **Sentence Transformers**:
   - **Purpose**: Converts text into high-dimensional vector embeddings. These embeddings are crucial for the subsequent similarity search as they capture the semantic information of the text.
   - **Usage in Code**: It is used to embed both the prompts (questions and potential answers) and the retrieved Wikipedia text into a 384-dimensional vector space for comparison. The specific model used is `sentence-transformers_all-MiniLM-L6-v2`, which is noted for its efficiency and performance in generating compact sentence embeddings.

2. **FAISS (Facebook AI Similarity Search)**:
   - **Purpose**: Provides efficient similarity search and clustering of dense vectors. It is optimized for speed and scalability, which is essential when working with large datasets like Wikipedia.
   - **Usage in Code**: Utilized to quickly retrieve the most similar sentences from the embedded Wikipedia sentences, based on the Euclidean distance (L2 norm).

3. **Scipy (specifically `cdist` from `scipy.spatial.distance`)**:
   - **Purpose**: Used for computing distances between sets of points. In this case, it helps in calculating the distances between embeddings.
   - **Usage in Code**: Calculates the cosine distances between the question embeddings and sentences from Wikipedia to find the closest matches.
![image](https://github.com/user-attachments/assets/a9285111-76d6-4dce-a1c5-d50f3459d01a)

### Workflow Description
1. **Data Initialization and Conditional Checks**:
   - The code starts by checking if the dataset size is a specific value (200). This appears to be a condition for either interactive mode or for a specific batch processing scenario.

2. **Embedding Generation**:
   - Text inputs are transformed into embeddings using the Sentence Transformer model. This includes both the user's prompts and potential answers.

3. **Similarity Search**:
   - These embeddings are then used to search for the most relevant sentences from a pre-indexed Wikipedia dataset using FAISS.

4. **Data Retrieval and Context Construction**:
   - Relevant sentences are retrieved, and the context is constructed based on the nearest sentences to the input query.

5. **Memory Management**:
   - Throughout the process, there are multiple calls to clear and manage memory, ensuring efficiency and preventing overload, especially important when working with large datasets on limited hardware resources.

## Data Preprocessing
- Convert between the answer options ('A', 'B', 'C', 'D') and numerical indices for easier manipulation during model training.
- Prepares two parts of input:
  - **first_sentence**: Array consisting of the context prefixed with a special token "[CLS]".
  - **second_sentences**: Array consisting of the concatenation of the prompt and each answer option, separated by special tokens.
- Utilizes a tokenizer to convert text inputs into a format suitable for model processing, ensuring inputs adhere to maximum length constraints.
- Labels each tokenized example with the corresponding index of the correct answer.

## Custom Data Collator
- **DataCollatorForMultipleChoice**: A custom data collator class that handles the aggregation of multiple examples into a single batch for processing by the model. This class is crucial for ensuring that data is appropriately padded and batched.
  - Dynamically adjusts padding based on the longest sequence in a batch to optimize computation.
  - Reshapes tensor dimensions to accommodate the structure required for multiple-choice evaluation.
  - Converts lists of features into tensors, which are the format required for training the model.

## Mean Average Precision at 3 (MAP@3)
Evaluate the effectiveness of a question-answering model by measuring its precision in predicting correct answers within the top 3 predictions for each question.
- **Precision at k**: Measures the precision at the top k predictions.
- **MAP@3**: A statistic used to evaluate the accuracy of a model in information retrieval tasks, focusing on the top 3 predictions made by the model.

  
![image](https://github.com/user-attachments/assets/6fe5189c-fb9e-48b0-ae00-c2c8ae1daea4)

## Parameter-Efficient Fine-Tuning (PEFT)

### Overview
PEFT is a library designed to efficiently adapt large pretrained models to various downstream applications without the need to fine-tune all of a model’s parameters. This approach is particularly beneficial because fully fine-tuning large models can be prohibitively costly in terms of both computation and storage. PEFT methods only fine-tune a small number of extra model parameters, significantly reducing computational and storage costs while maintaining performance comparable to fully fine-tuned models. This makes it more feasible to train and store large language models (LLMs) on consumer hardware.

### Integration
PEFT is integrated with several libraries, including Transformers, Diffusers, and Accelerate, providing a faster and easier way to load, train, and use large models for inference.

### Adapters
Adapters are a core component of PEFT. They are layers added after multi-head attention and feed-forward layers in the transformer architecture. During fine-tuning, only the parameters of these adapter layers are updated, while the rest of the model’s parameters remain frozen. This method addresses the inference latency problem, which is resolved by the LoRA approach.
![image](https://github.com/user-attachments/assets/b701b652-fdaf-4b10-8c94-09c764e01e9b)

### Low-Rank Adaptation (LoRA)
LoRA is a specific PEFT technique used to train LLMs on specific tasks or domains. It introduces trainable rank decomposition matrices into each layer of the transformer architecture, reducing the number of trainable parameters while keeping the pre-trained weights frozen. LoRA can minimize the number of trainable parameters by up to 10,000 times and reduce GPU memory requirements by 3 times, without compromising on the quality of the model's performance.
![image](https://github.com/user-attachments/assets/73dee637-107b-4fa0-a77f-37ba76f6dc15)

#### Advantages of LoRA
- **Efficiency**: Drastically reduces the number of trainable parameters.
- **Portability**: The original pre-trained weights are kept frozen, allowing for multiple lightweight and portable LoRA models for various downstream tasks.
- **Combination**: LoRA is orthogonal to many other parameter-efficient methods and can be combined with them.
- **Performance**: Comparable to fully fine-tuned models.
- **No Inference Latency**: Adapter weights can be merged with the base model, eliminating any additional inference latency.

#### Application of LoRA
LoRA can be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters. Typically, in transformer models, LoRA is applied to attention blocks to maximize parameter efficiency. The resulting number of trainable parameters in a LoRA model depends on the size of the low-rank update matrices, determined by the rank (r) and the shape of the original weight matrix.

### Merging LoRA Weights
Although LoRA is smaller and faster to train, there might be latency issues during inference due to separately loading the base model and the LoRA model. To eliminate this latency, the `merge_and_unload()` function merges the adapter weights with the base model, allowing the newly merged model to be used as a standalone model.

#### Mathematical Representation
LoRA adapts the weight matrix W of a layer by creating two smaller matrices, A and B, whose product approximates the modifications to W. This adaptation is expressed as Y = W + AB, where A and B are the low-rank matrices. If W is an mxn matrix, A might be mxr and B is rxn, where r is the rank and much smaller than m and n. During fine-tuning, only A and B are adjusted, enabling the model to learn task-specific features.

### Configuration Parameters for LoRA
- **r**: The rank of the update matrices. Lower rank results in smaller update matrices with fewer trainable parameters.
- **target_modules**: The modules (e.g., attention blocks) to apply the LoRA update matrices.
- **lora_alpha**: The LoRA scaling factor.
- **bias**: Specifies if the bias parameters should be trained. Can be 'none', 'all', or 'lora_only'.
- **use_rslora**: When set to True, uses Rank-Stabilized LoRA, which sets the adapter scaling factor to lora_alpha/math.sqrt(r), as this has been shown to work better. Otherwise, it uses the default value of lora_alpha/r.
- **modules_to_save**: List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include the model’s custom head that is randomly initialized for the fine-tuning task.
- **layers_to_transform**: List of layers to be transformed by LoRA. If not specified, all layers in target_modules are transformed.
- **layers_pattern**: Pattern to match layer names in target_modules, if layers_to_transform is specified. By default, PeftModel will look at common layer patterns (e.g., layers, h, blocks).
- **rank_pattern**: The mapping from layer names or regular expression (regexp) to ranks different from the default rank specified by r.
- **alpha_pattern**: The mapping from layer names or regexp to alphas different from the default alpha specified by lora_alpha.

### Quantized Low-Rank Adaptation (QLoRA)

QLoRA is an advanced version of LoRA (Low-Rank Adaptation) specifically designed to further optimize the fine-tuning of large language models (LLMs) by significantly reducing their memory demands. This technique enables these powerful models to be more accessible by allowing them to run on less powerful hardware, such as consumer GPUs.
![image](https://github.com/user-attachments/assets/b8c87947-7750-48a0-bcee-a75309e9606f)

#### Core Idea of QLoRA
1. **Quantization to 4-bit Precision**:
   - **Standard Precision**: Typically, the weight parameters in neural networks are stored in a 32-bit floating-point format. This high precision ensures accuracy but requires a lot of memory.
   - **QLoRA's Approach**: QLoRA compresses these parameters to a 4-bit format, drastically reducing the amount of memory needed to store and process these weights. This lower precision is sufficient for many tasks and allows the model to be fine-tuned and run on much less powerful hardware.

#### Key Innovations in QLoRA
1. **4-bit NormalFloat**:
   - **Optimal Quantization for Normal Distribution**: The 4-bit NormalFloat is a data type created for optimally quantizing weights that typically follow a normal distribution in neural networks. This quantization method is more effective and yields better results than using standard 4-bit integers or floats.

2. **Double Quantization**:
   - **Further Memory Reduction**: This technique involves quantizing the quantization constants themselves, which are used in the compression process. By doing this, QLoRA saves an additional average of 0.37 bits per parameter. For a very large model, such as one with 65 billion parameters, this can save approximately 3 GB of memory.

3. **Paged Optimizers**:
   - **Managing Memory Spikes**: In training neural networks, especially with large mini-batches or long sequences, memory usage can spike significantly due to gradient checkpointing (a technique used to save intermediate outputs to ease computation load). QLoRA uses a strategy with NVIDIA's unified memory to manage these spikes better, spreading out the memory load over time and preventing overwhelming the GPU.

#### Summary of QLoRA Benefits
- **Memory Efficiency**: Drastically reduces memory use without sacrificing performance.
- **Accessibility**: Enables training large models on hardware with limited memory capacity, such as consumer GPUs.
- **Performance**: Maintains high performance on par with or better than fully fine-tuned models.
- **No Additional Inference Latency**: Adapter weights can be merged with the base model, eliminating any additional inference latency.

For more details, you can refer to the [QLoRA paper](https://doi.org/10.48550/arXiv.2312.03732).


## Advanced Techniques
### Model Sharding
- Model Sharding splits the model into smaller, manageable pieces to run on devices with constrained GPU resources without compromising performance. It involves dynamic loading and unloading of model components, efficient buffer management, and parallel processing to reduce memory usage.

## Large Language Models (LLMs)

### DeBERTa: Decoding-enhanced BERT with Disentangled Attention
The DeBERTa model, proposed in the paper "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" by Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen, builds on Google's BERT and Facebook's RoBERTa models. It introduces two novel techniques to improve the efficiency and performance of pre-trained language models on NLP tasks.

1. **Disentangled Attention Mechanism**:
   - Each word is represented using two vectors that encode its content and position, respectively.
   - Attention weights among words are computed using disentangled matrices based on their contents and relative positions.
   - This improves the model's ability to understand the dependency between words based on their positions in the sentence.

2. **Enhanced Mask Decoder**:
   - Replaces the output softmax layer to predict masked tokens for model pretraining.
   - This technique significantly improves model pretraining efficiency and performance on downstream tasks.

**Performance**:
- DeBERTa outperforms RoBERTa-Large on various NLP tasks with only half the training data, achieving notable improvements on MNLI, SQuAD v2.0, and RACE benchmarks.

**Technical Implementation**:
- Integrates absolute word positions into the model's architecture by embedding them right before the softmax layer.
- Uses a disentangled attention mechanism to separate the understanding of word content from their relative positions.
- Enhances fine-tuning for downstream NLP tasks through virtual adversarial training, improving the model's generalization capabilities.

**References**:
- [DeBERTa GitHub](https://github.com/microsoft/DeBERTa)

### T5 (Text-to-Text Transfer Transformer)
T5 explores transfer learning techniques for NLP by converting every language problem into a text-to-text format. This unified framework facilitates systematic study and comparison of pretraining objectives, architectures, and transfer approaches across various NLP tasks.

**Key Features**:
- Converts all language tasks into a text-to-text format.
- Compares pretraining objectives, architectures, and transfer approaches on dozens of NLP tasks.
- Achieves state-of-the-art results on benchmarks like summarization, question answering, and text classification.
- Utilizes the "Colossal Clean Crawled Corpus" for training.

**References**:
- [T5 Paper](https://arxiv.org/pdf/1910.10683.pdf)

### Llama2
Llama 2 is a collection of pretrained and fine-tuned generative text models developed by Meta, ranging from 7 billion to 70 billion parameters. These models are optimized for dialogue use cases and achieve high performance on various benchmarks.

**Model Details**:
- **Variations**: Comes in parameter sizes of 7B, 13B, and 70B, with pretrained and fine-tuned versions.
- **Input/Output**: Models input text and generate text.
- **Architecture**: Uses an optimized transformer architecture with supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF).

**Performance**:
- Outperforms open-source chat models on most benchmarks.
- Comparable to popular closed-source models like ChatGPT and PaLM in human evaluations for helpfulness and safety.

**References**:
- [Llama 2 Download](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

### Phi3
The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter lightweight open model trained using the Phi-3 datasets, which include synthetic data and filtered publicly available website data. This model emphasizes high-quality and reasoning-dense properties and supports long context lengths of up to 128K tokens.

**Post-Training Process**:
- Underwent supervised fine-tuning and direct preference optimization to enhance its ability to follow instructions and adhere to safety measures.

**Performance**:
- Demonstrates robust and state-of-the-art performance on benchmarks testing common sense, language understanding, mathematics, coding, long-term context, and logical reasoning among models with fewer than 13 billion parameters.
  
  **References**:
- [Phi-3-Mini-128K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

# Results :
![image](https://github.com/user-attachments/assets/afe90da5-a3f7-4933-bf95-59aeda331e21)



## References
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://doi.org/10.48550/arXiv.2312.03732)
- [DeBERTa Paper](https://arxiv.org/abs/2006.03654)
- [Llama2 Model](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [Phi3 Model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
