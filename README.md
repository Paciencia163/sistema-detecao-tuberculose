# 🩺 Sistema de Diagnóstico de Tuberculose por IA

Sistema de auxílio diagnóstico baseado em inteligência artificial para análise de radiografias de tórax, desenvolvido com PyTorch e Streamlit.

## 🚀 Funcionalidades

### 🎯 Diagnóstico por IA
- **Análise automática** de radiografias de tórax
- **Classificação binária**: Normal vs Tuberculose
- **Probabilidades detalhadas** com confiança
- **Interface intuitiva** para upload de imagens

### 📚 Recursos Educacionais
- Galeria de exemplos radiológicos
- Informações sobre sintomas
- Recomendações médicas
- Vídeos educativos

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **PyTorch** - Framework de deep learning
- **Streamlit** - Interface web
- **timm** - Modelos pré-treinados
- **TorchVision** - Processamento de imagens
- **Pillow** - Manipulação de imagens

## 📁 Estrutura do Projeto

TB_Diagnosis_AI/
│
├── app.py                 # Aplicação principal Streamlit
├── TBModel.pt             # Modelo treinado de IA
├── model_classes.txt      # Rótulos das classes
├── requirements.txt       # Dependências do projeto
├── README.md              # Este arquivo
│
├── images/                # Imagens de exemplo (opcional)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── utils/                 # Utilitários (opcional)
    └── model_utils.py

## ⚙️ Instalação

### 1. Clone o repositório
git clone https://github.com/Paciencia163/sistema-detecao-tuberculose.git
cd sistema-detecao-tuberculose

### 2. Crie um ambiente virtual (recomendado)
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

### 3. Instale as dependências
pip install -r requirements.txt

Ou instale manualmente:
pip install torch torchvision timm==0.9.12 streamlit Pillow

## 🚀 Execução

streamlit run app.py

Acesse no navegador:
http://localhost:8501

## 📄 Arquivo requirements.txt

streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
timm==0.9.12
Pillow==10.0.0
numpy==1.24.3
pandas==2.0.3

## 🏥 Avisos Importantes

Este sistema é apenas uma ferramenta de AUXÍLIO ao diagnóstico e não substitui avaliação médica profissional.

---

**AVISO:** Consulte sempre um profissional de saúde para diagnóstico definitivo.
