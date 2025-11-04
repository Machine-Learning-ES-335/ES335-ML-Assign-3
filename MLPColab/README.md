# MLP Next-Word Predictor (War and Peace + Linux Kernel)


Files:
- data_prep.py : download & preprocess datasets, build vocab, save tokens
- train_mlp.py : train MLP next-word model (saves checkpoints)
- visualize_embeddings.py : t-SNE visualization of word embeddings
- streamlit_app.py : interactive text generation UI (Streamlit)
- dataset.py, utils.py : helper modules


1. Upload to Colab at `/content/MLPColab/MLPColab/`.
2. Install requirements: `!pip install -r requirements.txt`.
3. Prepare data: `!python data_prep.py --dataset warpeace`
4. Train (small test): `!python train_mlp.py --dataset warpeace --epochs 10 --batch_size 128`
5. Visualize: `!python visualize_embeddings.py --dataset warpeace --model_path models/warpeace/best.pt --out models/warpeace/warpeace_tsne.png`
6. Run Streamlit via ngrok 
This repo uses simple MLPs (embedding + MLP) to learn next-word prediction and extracts embeddings for interpretation.
