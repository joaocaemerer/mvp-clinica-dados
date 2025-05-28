import streamlit as st
import pandas as pd
import openai
from prophet import Prophet
import plotly.graph_objs as go

# Título
st.set_page_config(page_title="Painel Clínico com IA", layout="wide")
st.title("📊 Painel Inteligente para Clínicas de Saúde")

# Upload de CSV
uploaded_file = st.file_uploader("📁 Envie um arquivo CSV com os dados de atendimentos", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['data'])
    st.subheader("🔍 Dados carregados")
    st.write(df.head())

    # Gráfico de atendimentos por dia
    st.subheader("📈 Volume diário de atendimentos")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['data'], y=df['atendimentos'], mode='lines+markers'))
    fig.update_layout(xaxis_title='Data', yaxis_title='Atendimentos')
    st.plotly_chart(fig, use_container_width=True)

    # IA - Resumo automático
    st.subheader("🧠 Geração de resumo com IA")
    if st.button("Gerar resumo"):
        resumo_texto = f"Resumo dos dados do período:\n"
        resumo_texto += f"- Total de atendimentos: {df['atendimentos'].sum()}\n"
        resumo_texto += f"- Média diária: {df['atendimentos'].mean():.1f}\n"
        dia_mais = df.loc[df['atendimentos'].idxmax()]['data'].strftime('%d/%m/%Y')
        resumo_texto += f"- Dia com maior volume: {dia_mais} ({df['atendimentos'].max()} atendimentos)\n"

        prompt = f"""
Gere um resumo amigável e direto para gestores de clínicas baseado nos dados abaixo. Destaque o total de atendimentos, o dia com maior volume e a média diária:

{resumo_texto}
"""

        openai.api_key = st.secrets["OPENAI_API_KEY"]
        resposta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um assistente de BI para clínicas."},
                {"role": "user", "content": prompt}
            ]
        )
        st.success("Resumo gerado:")
        st.write(resposta['choices'][0]['message']['content'])

    # Previsão com Prophet
    st.subheader("📅 Previsão de atendimentos")
    df_prophet = df.rename(columns={"data": "ds", "atendimentos": "y"})
    modelo = Prophet()
    modelo.fit(df_prophet)

    futuro = modelo.make_future_dataframe(periods=30)
    previsao = modelo.predict(futuro)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=previsao['ds'], y=previsao['yhat'], mode='lines', name='Previsão'))
    fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Histórico'))
    fig2.update_layout(title="Previsão de atendimentos (próximos 30 dias)", xaxis_title="Data", yaxis_title="Atendimentos")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("🔄 Envie um CSV com as colunas: `data` e `atendimentos`.")
