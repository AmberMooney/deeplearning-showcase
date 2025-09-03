# deeplearning-showcase
Gallery of deep learning visuals with content accessible to those who are new to deep learning and those who wish to find ways to visualize concepts. This repository explores interpretation and visualization of deep learning models.

# Train Boundary Project
## Files
-train_boundary.py
-requirements_train_boundary.txt
-decision_boundary.gif
## Deep Visual: Evolving descision boundary (PyTorch)
- Trains a **2-layer MLP** on a synthetic dataset and records an **animated decision boundary** as the model learns. No external data needed.
- After running `python thrain_boundary.py`, you'll get `outputs/decision_boundary.gif` that looks like a colorful mal morphing as the network learns. View decision_boundary.gif as a preview
- You get to **watch the model carve up the plane** as it learns over epochs.
- Points are plotted over the plan so you can see where classifications fail and succeed over epochs.

# Attention Heatmap
## Files
-attention_heatmap.py
-attention_heatmap.png
## Deep Visual: A practical way to read an attention heatmap
-Transformer models used in NLP and LLM use attention heads to learn the next "token" or word/phrase in the generated sentence.
-The heatmap shows for each token(row), where it focuses(columns). Bright spots mark strong attention. Bands near the diagonal mean local context, and vertical bars mark hub tokens that many others attend to. Blocks suggest phrase-level grouping.
-Each head has its own map. Don't over interpret a single head. If you average heads, you get a smoother, less diagnostic map which is great for big picture visuals.
-Check the stats to confirm it is diffuse attention (entropy $\approx$ 2.69 - 2.75) means very spread out.
