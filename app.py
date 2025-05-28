import streamlit as st
import pandas as pd
import openai
import plotly.express as px

st.set_page_config(page_title="Painel Clínico Avançado", layout="wide")
st.title("🏥 Painel Inteligente para Clínicas de Saúde")

uploaded_file = st.file_uploader("📁 Envie um CSV com atendimentos detalhados", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['data'])

    st.subheader("🔍 Dados carregados")
    st.write(df.head())

    # Filtros
    especialidades = df["especialidade"].unique()
    especialidade_selecionada = st.selectbox("Filtrar por especialidade", ["Todas"] + list(especialidades))

    if especialidade_selecionada != "Todas":
        df = df[df["especialidade"] == especialidade_selecionada]

    # KPIs
    total_atendimentos = df["atendimentos"].sum()
    receita_total = df["receita_total"].sum()
    media_duracao = df["duracao_media_minutos"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total de atendimentos", f"{total_atendimentos}")
    col2.metric("💰 Receita total", f"R$ {receita_total:,.2f}")
    col3.metric("⏱️ Duração média", f"{media_duracao:.1f} min")

    # Gráfico de linha: Atendimentos por dia
    st.subheader("📈 Evolução dos atendimentos")
    df_diario = df.groupby("data").agg({"atendimentos": "sum"}).reset_index()
    fig1 = px.line(df_diario, x="data", y="atendimentos", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico de barras: Receita por especialidade
    st.subheader("💸 Receita por especialidade")
    receita_esp = df.groupby("especialidade").agg({"receita_total": "sum"}).reset_index()
    fig2 = px.bar(receita_esp, x="especialidade", y="receita_total", color="especialidade", title="Receita Total por Especialidade")
    st.plotly_chart(fig2, use_container_width=True)

    # Geração de resumo com IA
    st.subheader("🧠 Geração de resumo com IA")

    if st.button("Gerar resumo com IA"):
        # Dados de entrada para IA
        resumo = (
            f"- Total de atendimentos: {total_atendimentos}\n"
            f"- Receita total: R$ {receita_total:,.2f}\n"
            f"- Duração média: {media_duracao:.1f} minutos\n"
            f"- Período: de {df['data'].min().date()} a {df['data'].max().date()}\n"
        )

        openai.api_key = st.secrets["OPENAI_API_KEY"]
        prompt = (
            "Você é um assistente de BI para clínicas de saúde. Gere um resumo claro, direto e estratégico com base nesses dados:\n"
            + resumo
        )

        resposta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um assistente de dados para gestores de clínicas."},
                {"role": "user", "content": prompt}
            ]
        )

        st.success("Resumo gerado pela IA:")
        st.write(resposta['choices'][0]['message']['content'])

else:
    st.info("🔄 Envie um CSV com as colunas: data, atendimentos, especialidade, tipo_atendimento, duracao_media_minutos, receita_total.")
