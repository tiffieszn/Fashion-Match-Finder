import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
from tqdm import tqdm
import io
import os


#setup
st.set_page_config(page_title="Fashion Match % Finder 👗", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            -45deg,
            #fbc2eb, 
            #a6c1ee, 
            #fbc2eb,
            #ff9a9e
        );
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .center-title {
        text-align: center;
        margin-bottom: 10px;
    }

    .center-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #333;
        margin-bottom: 30px;
    }      

    /*side card text*/
    .side-card {
        color: #333333 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .side-card h3 {
        margin-top: 0;
        color: #222222 !important;
    }

    /*card*/
    .side-card {
        background: rgba(255,255,255,0.85);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }

     /*selectbox + file uploader container */
    div[data-baseweb="select"],
    div[data-baseweb="input"],
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 12px !important;
    }

    /*actual input text */
    input, textarea {
        background-color: transparent !important;
        color: #333333 !important;
        font-weight: 500;
    }

    /*dropdown text */
    div[data-baseweb="select"] span {
        color: #333333 !important;
    }

    /*file uploader label */
    label {
        color: #333333 !important;
        font-weight: 600;
    }

    /*button styling*/
    button {
        background-color: #ffffffcc !important;
        color: #333333 !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600;
    }

    button:hover {
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <h1 class="center-title">👚 Fashion Match % Finder</h1>
    <div class="center-subtitle">
        Find how well fashion items match, or search for best combinations!
    </div>
    """,
    unsafe_allow_html=True
)
left, center, right = st.columns([1.2, 3.8, 1.2])
with left:
    st.markdown(
        """
        <div class="side-card">
        <h4>👗 About</h4>
        This app uses <b>CLIP embeddings</b> to measure visual similarity
        between fashion items.

        <br><br>
        <b>🌸Dataset:</b><br>
        FashionMNIST<br><br>

        <b>💐Method:</b><br>
        Cosine similarity
        </div>
        """,
        unsafe_allow_html=True
    )


with right:
    st.markdown(
        """
        <div class="side-card">
        <h4>✨ Tips</h4>
        • Use clear clothing images<br>
        • Plain backgrounds work best<br>
        • Similar texture = higher %

        <br><br>
        <b>🌹Use cases:</b><br>
        • Outfit matching<br>
        • AI demo<br>
        • Portfolio project
        </div>
        """,
        unsafe_allow_html=True
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_clip_model()


#load dataset and cache embeddings
@st.cache_data
def get_embeddings():
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    #check if saved embeddings already exist
    if os.path.exists("fashion_embeddings.npz"):
        data = np.load("fashion_embeddings.npz")
        return data["embeddings"], data["labels"], dataset

    #generate embeddings if not saved yet
    embeddings = []
    labels = []

    with st.spinner("Extracting image features... Please wait ⏳"):
        for img, label in tqdm(dataset, total=len(dataset)):
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            embeddings.append(image_features.cpu().numpy())
            labels.append(label)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    #save to disk
    np.savez("fashion_embeddings.npz", embeddings=embeddings, labels=labels)

    return embeddings, labels, dataset
embeddings, labels, dataset = get_embeddings()

label_map = {
    0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
    5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"
}

#helper functions
def match_percent(embed1, embed2):
    sim = cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))[0][0]
    return round((sim + 1) / 2 * 100, 2)

def best_matches_for(category_label, top_k=5):
    category_indices = np.where(labels == category_label)[0]
    others = np.where(labels != category_label)[0]

    mean_embed = embeddings[category_indices].mean(axis=0, keepdims=True)
    sims = cosine_similarity(mean_embed, embeddings[others])[0]
    top_idx = others[np.argsort(sims)[::-1][:top_k]]

    results = [
        (label_map[labels[i]], round((sims[np.where(others==i)][0] + 1)/2*100, 2), dataset[i][0])
        for i in top_idx
    ]
    return results

def embed_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return F.normalize(features, p=2, dim=-1).cpu().numpy()


#main Interface
with center:
    st.subheader("🔍 Option 1: Find best matches for a clothing type")
    selected_label = st.selectbox("Choose a clothing type:", list(label_map.values()))
    if st.button("Find Matches"):
        idx = list(label_map.values()).index(selected_label)
        results = best_matches_for(idx)
        st.markdown(
            f"""
            <div style="
                color:#000000;
                font-size:1.1rem;
                font-weight:700;
                margin:15px 0 10px 0;
            ">
                Best matches for {selected_label}:
            </div>
            """,
            unsafe_allow_html=True
        )
        for name, score, img in results:
            pil_img = to_pil_image(img)
            st.image(pil_img, width=120)
            st.markdown(
                f"""
                <div style="
                    text-align:left;
                    color:#000000;
                    font-weight:600;
                    margin-bottom:20px;
                ">
                    {name} — {score}%
                </div>
                """,
                unsafe_allow_html=True
            )



    st.divider()

    st.subheader("🖼️ Option 2: Upload your own fashion image")
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Item", width=200)
        user_embed = embed_uploaded_image(uploaded_file)

        #compare with all dataset embeddings
        sims = cosine_similarity(user_embed, embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]

        st.markdown(
            """
            <div style="
                color:#000000;
                font-size:1.1rem;
                font-weight:700;
                margin:20px 0 10px 0;
            ">
                Top 5 Matches from Dataset:
            </div>
            """,
            unsafe_allow_html=True
        )
        for i in top_idx:
            pil_img = to_pil_image(dataset[i][0])
            percent = (sims[i] + 1) / 2 * 100
            st.image(pil_img, width=120)
            st.markdown(
                f"""
                <div style="
                    text-align:left;
                    color:#000000;
                    font-weight:600;
                    margin-bottom:20px;
                ">
                    {label_map[labels[i]]} — {percent:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )


