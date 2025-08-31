import streamlit as st
import joblib
import numpy as np

# --- 1. CARREGAR O SEU MODELO ---
# Esta função carrega seu modelo salvo e o guarda em cache para não recarregar a cada interação.
@st.cache_resource
def carregar_modelo():
    
    try:
        modelo = joblib.load('Bagging_DT_dNone.pkl')
        return modelo
    except FileNotFoundError:
        st.error("Arquivo do modelo ('modelo_sementes.pkl') não encontrado. Verifique se o arquivo está na mesma pasta que o app.py.")
        return None

# Carrega o modelo ao iniciar a página
modelo = carregar_modelo()


# --- 2. INTERFACE VISUAL DA PÁGINA ---
st.title("Classificador de Sementes")

st.header("Insira as características da semente:")


col1 = st.columns(0)

with col1:
    Lux_str = st.text_input("Lux", value="15.0")
    Temperatura_str = st.text_input("Temperatura", value="14.0")
    vibracao_str = st.text_input("Vibração", value="0.0")



# --- 3. LÓGICA DE PREVISÃO ---
# O código abaixo só roda se o modelo foi carregado com sucesso
if modelo:
    # Botão para executar a previsão, agora na página principal
    if st.button("Fazer Previsão"):
        
        # Junta todos os inputs de texto em uma lista
        inputs_str = [Lux_str, Temperatura_str, vibracao_str]
        
        try:
            # Tenta converter todos os textos para números (float)
            # A linha .replace(',', '.') ajuda a aceitar tanto vírgula quanto ponto como decimal
            features = [float(s.replace(',', '.')) for s in inputs_str]
            
            # Converte a lista para o formato que o modelo espera (array 2D)
            dados_para_prever = np.array(features).reshape(1, -1)

            # Faz a previsão usando o modelo carregado
            resultado_numerico = modelo.predict(dados_para_prever)[0]

            # Mapeia o resultado numérico para o nome da semente
            mapa_classes = {1: 'equipamento 1', 2: 'equipamento 2', 3: 'equipamento 3'} 
           
            
            nome_do_equipamento = mapa_classes.get(resultado_numerico, "Resultado desconhecido")

            # Exibe o resultado final de forma clara
            st.subheader("O equipamento é:")
            st.success(f"**{nome_do_equipamento}**")

        except ValueError:
            # Se a conversão para número falhar, mostra uma mensagem de erro
            st.error("Erro: Por favor, insira apenas números válidos em todos os campos. Use ponto (.) como separador decimal.")