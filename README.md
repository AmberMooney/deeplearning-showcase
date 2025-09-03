# deeplearning-showcase
Gallery of deep learning visuals with content accessible to those who are new to deep learning and those who wish to find ways to visualize concepts. This repository explores interpretation and visualization of deep learning models.

# 1. Train Boundary Project
## Files
- train_boundary.py
- requirements_train_boundary.txt
- decision_boundary.gif
## Deep Visual: Evolving descision boundary (PyTorch)
- Trains a **2-layer MLP** on a synthetic dataset and records an **animated decision boundary** as the model learns. No external data needed.
- After running `python thrain_boundary.py`, you'll get `outputs/decision_boundary.gif` that looks like a colorful mal morphing as the network learns. View decision_boundary.gif as a preview
- You get to **watch the model carve up the plane** as it learns over epochs.
- Points are plotted over the plan so you can see where classifications fail and succeed over epochs.

# 2. Attention Heatmap
## Files
- attention_heatmap.py
- attention_heatmap.png
## Deep Visual: A practical way to read an attention heatmap
- Transformer models used in NLP and LLM use attention heads to learn the next "token" or word/phrase in the generated sentence.
- The heatmap shows for each token(row), where it focuses(columns). Bright spots mark strong attention. Bands near the diagonal mean local context, and vertical bars mark hub tokens that many others attend to. Blocks suggest phrase-level grouping.
- Each head has its own map. Don't over interpret a single head. If you average heads, you get a smoother, less diagnostic map which is great for big picture visuals.
- Check the stats to confirm it is diffuse attention (entropy $\approx$ 2.69 - 2.75) means very spread out.

# 3. Loss Landscape
## Files
- loss_explorer.py
- loss_landscape.gif
## Deep Visual: Visualize the "ball rolling down the hill" as discussed in **Lambers, James V., Amber Sumner Mooney, Vivian A. Montiforte, and James Quinlan. Explorations in Numerical Analysis and Machine Learning with Julia. World Scientific, 2025. Chapter 16. https://doi.org/10.1142/14443**
- Explores the loss landscape as we freeze a 2D slice of the weight space around the trained model.
- Picks two random directions through that point and uses Gram-Schmidt to make them independent and at right angles (orthonormal set), so each axis shows a different, non-overlapping direction.
- Paints the loss over the plane as a countour map (darker = lower loss/valleys, lighter = higher loss/ridges).
- With the current training parameters we actually see the ball get stuck in the loss landscape! The goal is to minimize that loss in training and move out of the saddle point or local minimum. 
