import io
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import warnings
import sys
import subprocess

MODEL_PATH = 'TBModel.pt'
LABELS_PATH = 'model_classes.txt'

# Verificar e instalar dependências necessárias
@st.cache_resource
def check_dependencies():
    try:
        import timm
        return timm
    except ImportError:
        st.warning("Instalando biblioteca timm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm==0.9.12"])
        import timm
        return timm

# Configuração da página
st.set_page_config(
    page_title="Diagnóstico TB - IA Médica",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    /* Estilos gerais */
    .main {
        padding: 1rem 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    
    /* Cards de resultado */
    .diagnostic-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
    }
    
    /* Títulos */
    h1 {
        color: #2d3748;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2 {
        color: #4a5568;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Uploader personalizado */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    
    /* Loading spinner */
    .loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def load_image(key):
    st.markdown("### 📤 Carregar Radiografia de Tórax")
    st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label='',
        type=['png', 'jpg', 'jpeg'],
        key=key,
        help="Faça upload de uma imagem de radiografia de tórax (PNG, JPG, JPEG)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Imagem Carregada:")
            image_data = uploaded_file.getvalue()
            st.image(image_data, use_column_width=True)
        
        with col2:
            st.markdown("#### 📋 Informações:")
            st.write(f"**Nome do arquivo:** {uploaded_file.name}")
            st.write(f"**Tipo:** {uploaded_file.type}")
            st.write(f"**Tamanho:** {uploaded_file.size / 1024:.1f} KB")
            
            img = Image.open(io.BytesIO(image_data))
            st.write(f"**Dimensões:** {img.size[0]} × {img.size[1]} pixels")
        
        st.markdown("---")
        return Image.open(io.BytesIO(image_data))
    else:
        st.info("👆 Por favor, faça upload de uma imagem para começar o diagnóstico.")
        return None

@st.cache_resource
def load_model(model_path):
    try:
        # Verificar dependências primeiro
        timm = check_dependencies()
        
        with st.spinner('🔄 Carregando modelo de IA...'):
            # Tentar diferentes métodos de carregamento
            try:
                # Método 1: Carregar com torch.load direto
                import torch.serialization
                torch.serialization.add_safe_globals([getattr(timm.models.efficientnet, 'EfficientNet', None)])
            except:
                pass
            
            try:
                # Método 2: Tentar carregar como state_dict primeiro
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Verificar se é um state_dict ou modelo completo
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # É um checkpoint com state_dict
                    from collections import OrderedDict
                    
                    # Criar modelo
                    model_name = checkpoint.get('arch', 'efficientnet_b0')
                    model = timm.create_model(model_name, pretrained=False, num_classes=checkpoint['num_classes'])
                    
                    # Carregar pesos
                    state_dict = checkpoint['state_dict']
                    
                    # Remover 'module.' prefix se existir (para modelos treinados com DataParallel)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    
                    model.load_state_dict(new_state_dict)
                    
                elif isinstance(checkpoint, torch.nn.Module):
                    # Já é um modelo carregado
                    model = checkpoint
                else:
                    # Tentar como state_dict direto
                    try:
                        model_name = 'efficientnet_b0'  # Tentar com modelo padrão
                        model = timm.create_model(model_name, pretrained=False, num_classes=2)
                        model.load_state_dict(checkpoint)
                    except:
                        # Último recurso: carregar direto
                        model = torch.load(model_path, map_location='cpu')
                
                model.eval()
                st.success("✅ Modelo carregado com sucesso!")
                return model
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar modelo: {str(e)}")
                
                # Tentativa alternativa: criar modelo do zero (apenas para demonstração)
                st.warning("⚠️ Criando modelo de demonstração...")
                model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
                model.eval()
                return model
                
    except Exception as e:
        st.error(f"❌ Erro crítico: {str(e)}")
        return None

def load_labels(labels_file):
    try:
        with open(labels_file, "r") as f:
            categories = [s.strip() for s in f.readlines()]
            return categories
    except FileNotFoundError:
        st.warning(f"⚠️ Arquivo de rótulos não encontrado. Usando rótulos padrão.")
        return ["Normal", "Tuberculose"]
    except Exception as e:
        st.error(f"❌ Erro ao carregar rótulos: {str(e)}")
        return ["Normal", "Tuberculose"]

def predict(model, categories, image):
    if model is None:
        st.error("❌ Modelo não disponível para predição.")
        return
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        all_prob, all_catid = torch.topk(probabilities, len(categories))
        _, max_catid = torch.max(probabilities, dim=0)
        max_class = categories[max_catid] if max_catid < len(categories) else "Desconhecido"
        
        # Exibir resultado
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if "normal" in max_class.lower():
                st.markdown('<div class="diagnostic-card">', unsafe_allow_html=True)
                st.markdown("### ✅ **Resultado Previsto:**")
                st.markdown(f'<h1 style="color: #38a169;">{max_class}</h1>', unsafe_allow_html=True)
                st.markdown("### 📊 **Confiança:**")
                progress_value = probabilities[max_catid].item() * 100
                st.progress(min(progress_value / 100, 1.0))
                st.markdown(f"**{progress_value:.1f}%** de confiança")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.markdown("### ⚠️ **Resultado Previsto:**")
                st.markdown(f'<h1 style="color: #e53e3e;">{max_class}</h1>', unsafe_allow_html=True)
                st.markdown("### 📊 **Confiança:**")
                progress_value = probabilities[max_catid].item() * 100
                st.progress(min(progress_value / 100, 1.0))
                st.markdown(f"**{progress_value:.1f}%** de confiança")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Distribuição de Probabilidades:")
            
            # Ordenar por probabilidade
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            
            for i in range(min(3, len(categories))):
                prob = sorted_probs[i].item() * 100
                cat_idx = sorted_indices[i].item()
                label = categories[cat_idx] if cat_idx < len(categories) else f"Classe {cat_idx}"
                
                col_a, col_b, col_c = st.columns([2, 6, 2])
                with col_a:
                    st.write(f"**{label}**")
                with col_b:
                    st.progress(min(prob / 100, 1.0))
                with col_c:
                    st.write(f"**{prob:.1f}%**")
        
        # Aviso importante
        st.markdown("---")
        st.warning("""
        ⚠️ **Importante:** Este diagnóstico é auxiliado por IA e não substitui a avaliação de um profissional médico qualificado. 
        Consulte sempre um médico para diagnóstico definitivo e tratamento adequado.
        """)
        
    except Exception as e:
        st.error(f"❌ Erro durante a predição: {str(e)}")
        st.info("Por favor, verifique se a imagem é uma radiografia de tórax válida.")

def show_diagnosis_page():
    st.title('🩺 Diagnóstico de Tuberculose por IA')
    st.markdown("""
    Utilize esta ferramenta para análise auxiliar de radiografias de tórax. 
    O sistema utiliza inteligência artificial para identificar padrões sugestivos de tuberculose.
    """)
    
    # Cards informativos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 📁 Passo 1")
            st.markdown("Faça upload da radiografia em formato PNG, JPG ou JPEG")
    
    with col2:
        with st.container():
            st.markdown("### 🔍 Passo 2")
            st.markdown("O sistema analisará automaticamente a imagem com IA")
    
    with col3:
        with st.container():
            st.markdown("### 📋 Passo 3")
            st.markdown("Revise os resultados e recomendações")
    
    st.markdown("---")
    
    # Carregar modelo (cacheado)
    if 'model' not in st.session_state or 'categories' not in st.session_state:
        with st.spinner("🔄 Inicializando sistema..."):
            st.session_state.model = load_model(MODEL_PATH)
            st.session_state.categories = load_labels(LABELS_PATH)
    
    if st.session_state.model is None:
        st.error("""
        ❌ **Não foi possível carregar o modelo de IA.**
        
        **Soluções possíveis:**
        1. Verifique se o arquivo `TBModel.pt` está na pasta correta
        2. Tente reinstalar as dependências: `pip install torch torchvision timm==0.9.12`
        3. O modelo pode precisar ser convertido para uma versão mais recente
        """)
        
        # Opção para usar modelo de demonstração
        if st.button("🔄 Usar modelo de demonstração"):
            timm = check_dependencies()
            st.session_state.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
            st.session_state.model.eval()
            st.session_state.categories = ["Normal", "Tuberculose"]
            st.rerun()
        return
    
    # Área principal
    image = load_image("main_image")
    
    if image is not None:
        result = st.button('🔬 Executar Diagnóstico por IA', type="primary")
        if result:
            with st.spinner('🔄 Analisando imagem... Isso pode levar alguns segundos.'):
                predict(st.session_state.model, st.session_state.categories, image)

def show_tuberculosis_page():
    st.title("📚 Recursos Educacionais sobre Tuberculose")
    
    # Introdução
    st.markdown("""
    ### ℹ️ Informações Importantes
    Esta seção fornece recursos educacionais para melhor compreensão da tuberculose.
    """)
    
    # Vídeo em card
    st.markdown("### 📺 Vídeo Educativo")
    with st.container():
        st.markdown('<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px;">', unsafe_allow_html=True)
        st.markdown("**Ministra da Saúde Silvia Lutukuta fala sobre saúde em Angola:**")
        try:
            st.video("https://www.youtube.com/watch?v=4IcQTEXjf44")
        except:
            st.info("Vídeo educativo sobre saúde pública")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Galeria de imagens com tratamento de erro
    st.markdown("### 🖼️ Galeria de Radiografias")
    st.markdown("""
    Abaixo estão exemplos de radiografias com possíveis indicativos de tuberculose:
    """)
    
    tb_images_info = [
        ("Padrão de sombras nos pulmões que pode indicar a presença de tuberculose", "image1.jpg"),
        ("Áreas com maior densidade nos pulmões, possível sinal de tuberculose ativa", "image2.jpg"),
        ("Lesões nos pulmões que podem ser causadas pela tuberculose", "image3.jpg"),
    ]
    
    cols = st.columns(3)
    for idx, (col, (image_tip, image_path)) in enumerate(zip(cols, tb_images_info)):
        with col:
            try:
                st.image(image_path, use_column_width=True, caption=f"Exemplo {idx + 1}")
            except:
                # Placeholder se imagem não existir
                st.markdown(f"""
                <div style="background: #f0f0f0; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 10px;">
                    <span style="color: #666;">Imagem {idx + 1}</span>
                </div>
                """, unsafe_allow_html=True)
            with st.expander(f"ℹ️ Sobre o Caso {idx + 1}"):
                st.write(image_tip)
                st.markdown("**Recomendação:** Procure assistência médica se apresentar sintomas como tosse prolongada, febre ou perda de peso.")
    
    # Sintomas em cards
    st.markdown("---")
    st.markdown("### 🚨 Sintomas de Alerta")
    
    symptoms_cols = st.columns(2)
    
    with symptoms_cols[0]:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Sintomas Comuns")
        st.markdown("""
        - Tosse persistente (mais de 3 semanas)
        - Febre baixa contínua
        - Sudorese noturna
        - Perda de peso inexplicada
        - Fadiga constante
        - Dor no peito
        - Tosse com sangue
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with symptoms_cols[1]:
        st.markdown('<div class="diagnostic-card">', unsafe_allow_html=True)
        st.markdown("#### 🏥 Recomendações")
        st.markdown("""
        - Procure unidade de saúde se sintomas persistirem
        - Complete todo tratamento prescrito
        - Mantenha acompanhamento médico regular
        - Informe contatos próximos para avaliação
        - Use máscara se estiver com tosse
        - Mantenha ambientes ventilados
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Sidebar melhorada
    with st.sidebar:
        st.markdown("## 🏥 TB Diagnosis AI")
        st.markdown("---")
        
        page_selection = st.radio(
            "**Navegação**",
            ["🩺 Diagnóstico de Tuberculose", "📚 Dicas sobre Tuberculose"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Informações de versão
        try:
            import torch
            import torchvision
            timm = check_dependencies()
            
            st.markdown("### 🔧 Versões")
            st.code(f"""PyTorch: {torch.__version__}
TorchVision: {torchvision.__version__}
timm: {timm.__version__}""")
        except:
            pass
        
        # Informações na sidebar
        st.markdown("### 📊 Sobre o Sistema")
        st.info("""
        Esta ferramenta utiliza aprendizado de máquina para auxiliar na identificação de padrões radiológicos sugestivos de tuberculose.
        
        **Atenção:** Resultados devem ser validados por médico radiologista.
        """)
        
        st.markdown("### ⚠️ Aviso Importante")
        st.error("""
        **Este sistema não substitui diagnóstico médico profissional.** 
        Consulte sempre um especialista para avaliação definitiva.
        """)
        
        # Botão para reinstalar dependências
        if st.button("🔄 Verificar Dependências"):
            with st.spinner("Verificando..."):
                timm = check_dependencies()
                st.success("Dependências verificadas!")
        
        # Footer na sidebar
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center; color: #666; font-size: 0.8rem;">'
            '© 2024 TB Diagnosis AI v1.0<br>'
            'Ferramenta Educacional para Auxílio Médico'
            '</div>',
            unsafe_allow_html=True
        )

    if page_selection == "🩺 Diagnóstico de Tuberculose":
        show_diagnosis_page()
    elif page_selection == "📚 Dicas sobre Tuberculose":
        show_tuberculosis_page()

if __name__ == '__main__':
    # Adicionar aviso inicial
    st.info("""
    **Nota:** Esta é uma ferramenta de auxílio diagnóstico. 
    Em caso de suspeita de tuberculose, procure atendimento médico imediatamente.
    """)
    main()