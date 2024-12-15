import streamlit as st
from PyPDF2 import PdfReader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
import torch
from htmlTemplates import css, bot_template, user_template

# CSS and HTML templates for chat messages


def chat_message_template(template, msg):
    return template.replace("{{MSG}}", msg)


@st.cache_resource
def load_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return tokenizer, model, embedder


def extract_text_from_pdfs(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        for page in range(num_pages):
            page_obj = pdf_reader.pages[page]
            text += page_obj.extract_text()
    return text


def generate_response(text, question, tokenizer, model):
    input_text = f"Context: {text}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer.encode(
        input_text, return_tensors="pt", max_length=512, truncation=True
    )
    outputs = model.generate(
        inputs,
        max_length=400,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,
        top_p=0.9,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def process_embeddings(text, embedder):
    sentences = text.split("\n")
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings


def find_relevant_text(question, sentences, embeddings, embedder):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)
    relevant_text = "\n".join([sentences[idx] for idx in top_results.indices])
    return relevant_text


def main():
    st.title("Transformer based CUI using Langchain")
    st.write("Upload the domain specific documents here.")

    st.markdown(css, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDFs", type="pdf", accept_multiple_files=True
    )
    question = st.text_input("Enter your question here")

    if uploaded_files and question:
        tokenizer, model, embedder = load_models()
        with st.spinner("Extracting text from PDFs..."):
            text = extract_text_from_pdfs(uploaded_files)

        with st.spinner("Processing embeddings..."):
            sentences, embeddings = process_embeddings(text, embedder)

        with st.spinner("Finding relevant text..."):
            relevant_text = find_relevant_text(
                question, sentences, embeddings, embedder
            )

        with st.spinner("Generating response..."):
            answer = generate_response(relevant_text, question, tokenizer, model)

        st.markdown(
            chat_message_template(user_template, question), unsafe_allow_html=True
        )
        st.markdown(chat_message_template(bot_template, answer), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
