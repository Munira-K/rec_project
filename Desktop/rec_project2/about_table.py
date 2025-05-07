import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Анализ данных")
st.title("📊 Анализ датасетов системы рекомендаций курсов")
st.markdown("""
<style>
    .dataset-header {
        font-size: 1.5em;
        color: #2b5876;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    ratings = pd.read_csv('Desktop/rec_project2/df_ratings.csv')
    courses = pd.read_csv('Desktop/rec_project2/info2022_final.csv')
    return courses, ratings

courses_df, ratings_df = load_data()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>👥 Пользователи</h3>
        <p style="font-size: 2em; margin: 0;">{ratings_df['user_id'].nunique():,}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎓 Курсы</h3>
        <p style="font-size: 2em; margin: 0;">{courses_df['course_id'].nunique():,}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>⭐ Оценки</h3>
        <p style="font-size: 2em; margin: 0;">{len(ratings_df):,}</p>
    </div>
    """, unsafe_allow_html=True)



st.markdown('<div class="dataset-header">🎓 Анализ датасета курсов</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Распределение по категориям", "Топ курсов", "Длительность курсов"])
with tab1:
    fig = px.pie(courses_df, names='category', title='Распределение курсов по категориям',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    course_ratings = ratings_df.groupby('course_id') \
        .agg(num_ratings=('user_id', 'count'),
             avg_rating=('rate', 'mean')) \
        .reset_index()
    
    top_courses = course_ratings.merge(
        courses_df, 
        on='course_id',
        suffixes=('', '_course')
    ).sort_values('num_ratings', ascending=False).head(10)
    
    available_columns = top_courses.columns.tolist()
    hover_columns = ['avg_rating', 'category'] 

    numeric_columns = ['content_length_min', 'num_lectures', 'num_subscribers', 'num_reviews']
    for col in numeric_columns:
        if col in available_columns:
            hover_columns.append(col)
            break
    
    fig = px.bar(top_courses, 
                 x='title', 
                 y='num_ratings',
                 hover_data=hover_columns,
                 title='Топ-10 курсов по количеству оценок',
                 labels={
                     'num_ratings': 'Количество оценок', 
                     'title': 'Название курса',
                     'avg_rating': 'Средний рейтинг',
                     'category': 'Категория',
                 },
                 color='avg_rating',
                 color_continuous_scale='Bluered')
    
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    numeric_column = 'content_length_min' if 'content_length_min' in courses_df.columns else 'num_lectures'
    
    if numeric_column in courses_df.columns:
        fig = px.histogram(courses_df, 
                         x=numeric_column,
                         title=f'Распределение курсов по {numeric_column}',
                         labels={numeric_column: numeric_column.replace('_', ' ').title()},
                         nbins=20,
                         color_discrete_sequence=['#636EFA'])
        
        fig.update_traces(hovertemplate=f"<b>{numeric_column.replace('_', ' ').title()}:</b> %{{x}}<br>" +
                                      "<b>Количество курсов:</b> %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("В данных отсутствует подходящий числовой столбец для анализа длительности")
        st.info("Доступные числовые столбцы: " + ", ".join(courses_df.select_dtypes(include=['int64', 'float64']).columns.tolist()))



with col4:
    fig = px.histogram(ratings_df, x='rate', 
                      title='Распределение оценок',
                      nbins=5,
                      color_discrete_sequence=['#EF553B'])
    st.plotly_chart(fig, use_container_width=True)


st.markdown('<div class="dataset-header">📋 Таблицы</div>', unsafe_allow_html=True)
dataset = st.radio("Выберите датасет:", ["Курсы", "Оценки"], horizontal=True)

if dataset == "Курсы":
    st.dataframe(courses_df.head(10))
else:
    st.dataframe(ratings_df.head(10))
