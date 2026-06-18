import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA & ESTILO
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Detector de Spam IA",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Detector de Spam com TensorFlow")
st.write("Cole o conteúdo do e-mail abaixo para verificar se a mensagem é segura ou Spam.")

# -------------------------------------------------------------------------
# 2. DADOS DE EXEMPLO & TREINAMENTO DA IA
# -------------------------------------------------------------------------
# Exemplos base para inicializar a inteligência do modelo
MENSAGENS_EXEMPLO = [
    # Não Spam (Ham)
    ("Olá, tudo bem? Segue em anexo a ata da nossa reunião de ontem.", 0),
    ("Seu relatório semanal de engenharia está pronto. Por favor, revise.", 0),
    ("Oi mãe, almoçamos juntos amanhã? Me avisa que horas você pode.", 0),
    ("Confirmação de recebimento do seu pedido #12345. Obrigado por comprar conosco.", 0),
    ("Lembrete: Sua consulta com o médico está marcada para quinta-feira às 14h.", 0),
    # Spam
    ("GANHE DINHEIRO FÁCIL! CLIQUE AQUI E REIVINDIQUE SEU PRÊMIO DE 10 MIL REAIS AGORA!!", 1),
    ("🔥 TRABALHE EM CASA GANHANDO R$ 500 POR DIA!! VAGAS LIMITADAS CADASTRAR JÁ 🔥", 1),
    ("Seu cartão de crédito foi bloqueado por segurança. Acesse o link urgente para desbloquear.", 1),
    ("Compre remédios sem receita médica com 80% de desconto. Entrega imediata.", 1),
    ("Você foi o sorteado da nossa loteria internacional! Envie seus dados bancários.", 1),
]

textos_treino = [item[0] for item in MENSAGENS_EXEMPLO]
labels_treino = np.array([item[1] for item in MENSAGENS_EXEMPLO], dtype=np.float32)

# -------------------------------------------------------------------------
# 3. CONSTRUÇÃO E TREINAMENTO DO MODELO TENSORFLOW (CACHED)
# -------------------------------------------------------------------------
@st.cache_resource
def inicializar_modelo():
    """Cria, treina e retorna o pipeline de NLP com TensorFlow"""
    # Camada de Vetorização de Texto
    max_tokens = 1000
    sequence_length = 50
   
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    vectorizer.adapt(textos_treino)
   
    # Arquitetura da Rede Neural
    model = Sequential([
        vectorizer,
        Embedding(input_dim=max_tokens, output_dim=16, input_length=sequence_length),
        GlobalAveragePooling1D(),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid') # Saída binária: 0 (Ham) ou 1 (Spam)
    ])
   
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   
    # Treinamento rápido (overfitting intencional nos dados de exemplo para demonstração)
    model.fit(np.array(textos_treino), labels_treino, epochs=50, verbose=0)
   
    return model

# Inicializa a IA
modelo_spam = inicializar_modelo()

# -------------------------------------------------------------------------
# 4. INTERFACE DO USUÁRIO (STREAMLIT)
# -------------------------------------------------------------------------

# Sidebar com exemplos para o usuário testar
with st.sidebar:
    st.header("💡 Exemplos para Testar")
    st.subheader("Exemplos de Não Spam:")
    st.caption("• *Olá, tudo bem? Segue em anexo a ata da nossa reunião de ontem.*")
    st.caption("• *Seu relatório semanal de engenharia está pronto.*")
   
    st.subheader("Exemplos de Spam:")
    st.caption("• *GANHE DINHEIRO FÁCIL! CLIQUE AQUI E REIVINDIQUE SEU PRÊMIO AGORA!!*")
    st.caption("• *Seu cartão de crédito foi bloqueado. Acesse o link urgente.*")

# Área de Input Principal
email_input = st.text_area(
    "Cole a mensagem do e-mail aqui:",
    height=200,
    placeholder="Digite ou cole o texto do e-mail suspeito..."
)

# Botão de Classificação
if st.button("Analisar Mensagem", use_container_width=True):
    if email_input.strip() == "":
        st.warning("⚠️ Por favor, insira alguma mensagem antes de analisar.")
    else:
        with st.spinner("A inteligência artificial está analisando o texto..."):
            # Predição do modelo TensorFlow
            # O modelo espera um array/lista de strings
            predicao = modelo_spam.predict(np.array([email_input]))[0][0]
           
            # Formatação do resultado (limiar de 0.5)
            st.write("---")
            st.subheader("Resultado da Análise:")
           
            if predicao >= 0.5:
                # Se for maior ou igual a 50%, é classificado como Spam
                confianca = predicao * 100
                st.error(f"🚨 **Esta mensagem é um SPAM!** (Confiança da IA: {confianca:.2f}%)")
                st.markdown(
                    "> **Cuidado:** Esta mensagem possui padrões altamente suspeitos semelhantes a golpes, "
                    "links maliciosos ou promoções falsas. Não clique em nenhum link."
                )
            else:
                # Se for menor que 50%, é classificado como seguro (Ham)
                confianca = (1 - predicao) * 100
                st.success(f"✅ **Mensagem Segura!** (Confiança da IA: {confianca:.2f}%)")
                st.markdown(
                    "> **Observação:** O modelo identificou que esta mensagem possui características de comunicação "
                    "legítima e segura."
                )
