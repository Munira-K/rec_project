import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from surprise import KNNBaseline, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
nltk.data.path.append("./nltk_data")

# Инициализация
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Настройки страницы
st.set_page_config(
    page_title="Рекомендательная система курсов",
    page_icon="🎓",
    layout="wide"
)

# Загрузка данных
@st.cache_data
def load_data():
    try:
        df_ratings = pd.read_csv('Desktop/rec_project2/df_ratings.csv')
        df_courses = pd.read_csv('Desktop/rec_project2/info2022_final.csv')
        
        # Проверяем и переименовываем необходимые колонки
        if 'Course ID' in df_courses.columns:
            df_courses = df_courses.rename(columns={'Course ID': 'course_id'})
        if 'Course title' in df_courses.columns:
            df_courses = df_courses.rename(columns={'Course title': 'title'})
        
        return df_ratings, df_courses
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        st.stop()

df_ratings, df_courses = load_data()

string_cols = df_courses.select_dtypes(include='object')
string_cols = string_cols.fillna('').apply(lambda col: col.str.lower())
df_courses['description'] = string_cols.apply(lambda row: ' '.join(filter(None, row)), axis=1)

# Функция предобработки текста
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words and word not in set(string.punctuation)
    ]
    return tokens

# Создание или загрузка Doc2Vec модели
@st.cache_resource
def get_doc2vec_model(df_courses):
    model_path = Path("doc2vec_model.model")
    
    if model_path.exists():
        try:
            return Doc2Vec.load(str(model_path))
        except Exception as e:
            st.warning(f"Ошибка загрузки модели: {str(e)}. Создаем новую модель.")
    
    # Создаем новую модель
    with st.spinner("Создание Doc2Vec модели... Это может занять несколько минут."):
        # Подготовка данных
        documents = [
            TaggedDocument(
                words=preprocess_text(row['description']),
                tags=[str(idx)]
            )
            for idx, row in df_courses.iterrows()
        ]
        
        # Обучение модели
        model = Doc2Vec(
            vector_size=50,
            min_count=2,
            epochs=20,
            workers=4
        )
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        
        # Сохраняем модель
        model.save(str(model_path))
        st.success("Doc2Vec модель успешно создана и сохранена.")
        return model

# Загрузка KNN модели
@st.cache_resource
def get_knn_model(df_ratings):
    try:
        # Проверяем и переименовываем колонки
        if 'userId' in df_ratings.columns:
            df_ratings = df_ratings.rename(columns={'userId': 'user_id'})
        if 'course_id' not in df_ratings.columns and 'Course ID' in df_ratings.columns:
            df_ratings = df_ratings.rename(columns={'Course ID': 'course_id'})
        if 'rate' not in df_ratings.columns and 'rating' in df_ratings.columns:
            df_ratings = df_ratings.rename(columns={'rating': 'rate'})
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_ratings[['user_id', 'course_id', 'rate']], reader)
        trainset = data.build_full_trainset()
        
        model = KNNBaseline(
            k=10,
            sim_options={'name': 'pearson_baseline', 'user_based': True},
            bsl_options={'method': 'als', 'n_epochs': 5, 'reg_u': 15, 'reg_i': 5}
        )
        model.fit(trainset)
        return model
    except Exception as e:
        st.error(f"Ошибка инициализации KNN модели: {str(e)}")
        st.stop()

# Загрузка моделей
doc2vec_model = get_doc2vec_model(df_courses)
knn_model = get_knn_model(df_ratings)

# Класс рекомендателя
class HybridRecommender:
    def __init__(self, cf_model, doc2vec_model, df_courses, df_ratings, n_candidates=100):
        self.cf_model = cf_model
        self.doc2vec_model = doc2vec_model
        self.df_courses = df_courses
        self.df_ratings = df_ratings
        self.n_candidates = n_candidates
        self.course_id_to_idx = {cid: idx for idx, cid in enumerate(df_courses['course_id'])}

    def _get_cf_candidates(self, user_id):
        valid_items = self.df_courses['course_id'].unique()
        scores = [self.cf_model.predict(user_id, item).est for item in valid_items]
        top_indices = np.argsort(scores)[-self.n_candidates:][::-1]
        return valid_items[top_indices], np.array(scores)[top_indices]

    def _get_content_scores(self, user_id, candidates):
        user_history = self.df_ratings[self.df_ratings['user_id'] == user_id]['course_id']
        if len(user_history) == 0:
            return np.zeros(len(candidates))
        
        user_vector = np.mean([
            self.doc2vec_model.dv[str(self.course_id_to_idx[cid])]
            for cid in user_history
            if cid in self.course_id_to_idx
        ], axis=0)
        
        candidate_vectors = [
            self.doc2vec_model.dv[str(self.course_id_to_idx[cid])]
            for cid in candidates
            if cid in self.course_id_to_idx
        ]
        
        if len(candidate_vectors) == 0:
            return np.zeros(len(candidates))
            
        return cosine_similarity([user_vector], candidate_vectors)[0]

    def recommend(self, user_id, top_k=10):
        candidates, cf_scores = self._get_cf_candidates(user_id)
        content_scores = self._get_content_scores(user_id, candidates)
        combined_scores = 0.7 * cf_scores + 0.3 * content_scores
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        result = pd.DataFrame({
            'course_id': candidates[top_indices],
            'score': combined_scores[top_indices]
        }).merge(self.df_courses, on='course_id')
        
        return result.sort_values('score', ascending=False)

# Инициализация рекомендателя
recommender = HybridRecommender(
    cf_model=knn_model,
    doc2vec_model=doc2vec_model,
    df_courses=df_courses,
    df_ratings=df_ratings
)

# Интерфейс Streamlit
st.title("🎓 Гибридная рекомендательная система курсов")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    user_list = df_ratings['user_id'].unique().tolist()
    selected_user = st.selectbox("Выберите пользователя", user_list)
    num_recommendations = st.slider("Количество рекомендаций", 3, 20, 10)

# Основное содержимое
st.header(f"Рекомендации для пользователя {selected_user}")

try:
    recommendations = recommender.recommend(selected_user, top_k=num_recommendations)
    
    if recommendations.empty:
        st.warning("Не найдено рекомендаций для этого пользователя.")
    else:
        # Отображение карточек
        cols = st.columns(3)
        for idx, row in recommendations.iterrows():
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    background: #f8f9fa;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4>{row['title'] if 'title' in row else 'Название недоступно'}</h4>
                    <p>📊 Оценка: <b>{row['score']:.2f}</b></p>
                    <p>🏷️ Категория: {row.get('category', 'N/A')}</p>
                    <p>⭐ Рейтинг: {row.get('avg_rating', '?')}/5</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Таблица с деталями
        with st.expander("Подробная информация"):
            display_columns = ['title', 'score']
            if 'category' in recommendations.columns:
                display_columns.append('category')
            if 'avg_rating' in recommendations.columns:
                display_columns.append('avg_rating')
            if 'language' in recommendations.columns:
                display_columns.append('language')
            if 'instructor_name' in recommendations.columns:
                display_columns.append('instructor_name')
            if 'course_url' in recommendations.columns:
                display_columns.append('course_url')



            st.dataframe(
                recommendations[display_columns],
                column_config={
                    "title": "Название курса",
                    "score": st.column_config.NumberColumn("Оценка", format="%.2f"),
                    "category": "Категория",
                    "avg_rating": st.column_config.NumberColumn("Рейтинг", format="%.1f"),
                    "language": 'Язык',
                    "instructor_name" : "Инструктор",
                    "course_url" : "Ccылка на курс",
                },
                hide_index=True,
                use_container_width=True
            )
        
except Exception as e:
    st.error(f"Ошибка при генерации рекомендаций: {str(e)}")
