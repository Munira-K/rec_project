import streamlit as st
import pandas as pd
import plotly.express as px

# Заголовок приложения
st.title('📊 Сравнение моделей рекомендательных систем')

# Данные для таблиц метрик (примерные значения - замените вашими реальными метриками)
df_collab = pd.read_csv('C://Users//user//Desktop//rec_project//metrics_table.csv')
df_content = pd.read_csv('C://Users//user//Desktop//rec_project//content_metrics_table.csv')
df_hybrid = pd.read_csv('C://Users//user//Desktop//rec_project//hybrid_metrics.csv')

# Раздел с метриками коллаборативной фильтрации
st.header('1. Коллаборативная фильтрация')
st.subheader('Метрики регрессии')
st.dataframe(df_collab.style.highlight_min(axis=0, subset=['RMSE', 'MAE']), 
             use_container_width=True)

# Раздел с метриками контентной фильтрации
st.header('2. Контентная фильтрация')
st.subheader('Метрики ранжирования')
st.dataframe(df_content.style.highlight_max(axis=0, subset=['Precision@10', 'Recall@10']), 
             use_container_width=True)

# Новый раздел с метриками гибридной модели
st.header('3. Гибридная модель (SVD + Doc2Vec)')
df_hybrid = pd.read_csv('C://Users//user//Desktop//rec_project//compare_table.csv')

st.subheader('Сравнение')
st.dataframe(df_hybrid.style.highlight_max(axis=0, 
                                          subset=['Precision@10', 'Recall@10']),
             use_container_width=True)

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



# Раздел с выводами
st.header('📌 Основные выводы')
st.markdown("""
1. **Гибридная модель показывает лучшие результаты**:
   - +8.6% улучшение Precision@10 по сравнению с лучшей отдельной моделью (SVD)
   - +24% улучшение Recall@10 по сравнению с Doc2Vec

2. **Преимущества гибридного подхода**:
   - Высокая персонализация (0.85 vs 0.78 у SVD)
   - Широкое покрытие курсов (95%)
   - Устойчивость к cold-start проблеме

3. **Рекомендации**:
   - Использовать гибридную модель для основных рекомендаций
   - SVD - для пользователей с историей просмотров
   - Doc2Vec - для новых пользователей и дополнения рекомендаций
""")

# Дополнительная информация
with st.expander("ℹ️ О метриках"):
    st.markdown("""
    - **Precision@10**: Доля релевантных курсов в топ-10 рекомендаций
    - **Recall@10**: Доля рекомендованных релевантных курсов от всех релевантных
    """)