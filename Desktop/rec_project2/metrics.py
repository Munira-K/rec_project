import streamlit as st
import pandas as pd
import plotly.express as px

st.title('📊 Сравнение моделей рекомендательных систем')
df_collab = pd.read_csv('Desktop/rec_project2/metrics_table.csv')
df_content = pd.read_csv('Desktop/rec_project2/content_metrics_table.csv')
df_hybrid = pd.read_csv('Desktop/rec_project2/hybrid_metrics.csv')


st.header('1. Коллаборативная фильтрация')
st.subheader('Метрики регрессии')
st.dataframe(df_collab.style.highlight_min(axis=0, subset=['RMSE', 'MAE']), 
             use_container_width=True)

st.write("""На основе представленных метрик видно, что модель SVD показала наилучшие результаты среди всех протестированных подходов:
- Лучший RMSE (0.706) — значительно точнее предсказывает оценки по сравнению с другими методами
- Конкурентный MAE (0.524) — уступает только UserBasedCF, но с небольшим отрывом
- Стабильность — демонстрирует сбалансированную точность по обеим метрикам""")

st.header('2. Контентная фильтрация')
st.subheader('Метрики ранжирования')
st.dataframe(df_content.style.highlight_max(axis=0, subset=['Precision@10', 'Recall@10']),  use_container_width=True)

st.write("""Несмотря на нулевые метрики в тестах, Doc2Vec был выбран потому что:
- Справляется с новыми курсами (cold-start) — там, где коллаборативная фильтрация бессильна
- Анализирует контент — учитывает описание и темы, а не только оценки
- Дополняет SVD — даёт рекомендации, когда недостаточно данных о пользователе""")

st.header('3. Гибридная модель (SVD + Doc2Vec)')
df_hybrid = pd.read_csv('Desktop/rec_project2/compare_table.csv')

st.subheader('Сравнение')
st.dataframe(df_hybrid.style.highlight_max(axis=0, subset=['Precision@10', 'Recall@10']),use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig_precision = px.bar(df_hybrid, x='Model', y='Precision@10',
                          title='Precision@10 сравнение',
                          color='Model', text_auto='.3f')
    st.plotly_chart(fig_precision, use_container_width=True)

with col2:
    fig_recall = px.bar(df_hybrid, x='Model', y='Recall@10',
                       title='Recall@10 сравнение',
                       color='Model', text_auto='.3f')
    st.plotly_chart(fig_recall, use_container_width=True)


col3, col4 = st.columns(2)
with col3:
    fig_precision = px.bar(df_hybrid, x='Model', y='RMSE',
                          title='RMSE сравнение',
                          color='Model', text_auto='.3f')
    st.plotly_chart(fig_precision, use_container_width=True)

with col4:
    fig_recall = px.bar(df_hybrid, x='Model', y='MAE',
                       title='MAE сравнение',
                       color='Model', text_auto='.3f')
    st.plotly_chart(fig_recall, use_container_width=True)


with st.expander("ℹ️ О метриках"):
    st.markdown("""
    - **Precision@10**: Доля релевантных курсов в топ-10 рекомендаций
    - **Recall@10**: Доля рекомендованных релевантных курсов от всех релевантных
    - **RMSE (Root Mean Squared Error)**: Среднеквадратичная ошибка — показывает среднее квадратичное отклонение предсказанных рейтингов от фактических. Более чувствительна к большим ошибкам.
    - **MAE (Mean Absolute Error)**: Средняя абсолютная ошибка — среднее значение модулей отклонений между предсказанными и фактическими рейтингами. Менее чувствительна к выбросам по сравнению с RMSE.
    """)
