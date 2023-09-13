import io
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import timm

MODEL_PATH = 'TBModel.pt'
LABELS_PATH = 'model_classes.txt'

def load_image(key):
    uploaded_file = st.file_uploader(label='Carregar uma radiografia de tórax', key=key)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

# Função para carregar o modelo
def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

# Função para carregar os rótulos
def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

# Restante do código para predição

def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    all_prob, all_catid = torch.topk(probabilities, len(categories))
    _, max_catid = torch.max(probabilities, dim=0)
    max_class = categories[max_catid]
    st.write(f"Resultado Previsto: {max_class}")

# Função para exibir a página de diagnóstico
def show_diagnosis_page():
    st.title('Diagnóstico de Tuberculose a partir de imagens de Radiografia de Tórax')

    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)  # Carregar rótulos aqui
    image = load_image("main_image")  # Chave única para o widget nesta página
    result = st.button('Executar imagem')
    if result:
        st.write('Calculando resultados...')
        predict(model, categories, image)

# Função para exibir a página de dicas
def show_tuberculosis_page():
    st.title("Ilustrações e Dicas sobre Tuberculose")

    st.markdown("Video postado no Canal da OMS onde a Ministra da Saúde Silvia Lutukuta fala sobre a Malaria em Angola:")
    st.video("https://www.youtube.com/watch?v=4IcQTEXjf44&autoplay=1")


    st.markdown("Aqui estão algumas imagens de radiografias de tórax que podem indicar tuberculose, "
                "acompanhadas de dicas importantes:")

    tb_images = [
        ("image1.jpg", "Imagem 1: Essa imagem mostra um padrão de sombras nos pulmões que pode indicar a presença de tuberculose. Procure assistência médica se você tiver sintomas."),
        ("image2.jpg", "Imagem 2: Aqui, podemos ver áreas com maior densidade nos pulmões, o que pode ser um sinal de tuberculose ativa. Consulte um médico para avaliação."),
        ("image3.jpg", "Imagem 3: A radiografia revela lesões nos pulmões que podem ser causadas pela tuberculose. Não ignore sintomas como tosse prolongada e febre."),
    ]

    for i, (image_path, image_tip) in enumerate(tb_images):
        st.image(image_path, width=300, caption="Radiografia de Tórax")
        st.write(image_tip)
        st.write("---")

# Função principal para controlar a seleção de página
def main():
    st.sidebar.title("Navegação")
    page_selection = st.sidebar.radio("Selecione uma página", ["Diagnóstico de Tuberculose", "Dicas sobre Tuberculose"])

    if page_selection == "Diagnóstico de Tuberculose":
        show_diagnosis_page()
    elif page_selection == "Dicas sobre Tuberculose":
        show_tuberculosis_page()

if __name__ == '__main__':
    main()