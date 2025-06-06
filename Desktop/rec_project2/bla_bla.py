import streamlit as st

st.set_page_config(layout="wide", page_title="О проекте")
st.title("🎯Рекомендательная система курсов")

st.header("Цели проекта")
st.markdown("""
🔹 **Основная задача**: Помочь пользователям находить подходящие курсы без долгого поиска исходя из его истории
🔹 **Что решаем**: Проблему переизбытка выбора и низкого качества рекомендаций  
🔹 **Для кого**: Для учащихся и платформ онлайн-образования  
""")
st.divider()




st.header("🔧 Проделанные шаги")
steps = [
    {"icon": "📊", "title": "Анализ данных", 
     "text": "Изучено распределение оценок, популярность курсов и активность пользователей"},
    
    {"icon": "🧹", "title": "Подготовка данных", 
     "text": "Очистка от выбросов, обработка текстовых описаний курсов (токенизация, лемматизация)"},
    
    {"icon": "🤖", "title": "Реализация моделей", 
     "text": "Разработка отдельно коллаборативную (SVD), контентную (Doc2Vec) модели и baseline"},
    
    {"icon": "🔄", "title": "Интеграция подходов", 
     "text": "Создание комбинированной системы: сначала SVD, потом дополнение рекомендациями Doc2Vec"},
    
    {"icon": "�", "title": "Оценка качества", 
     "text": "Сравнение метрик (Precision@10, Recall@10, MAE, RMSE) с базовыми подходами"},
    
    {"icon": "📱", "title": "Визуализация", 
     "text": "Деплой в streamlit для тестирования рекомендаций"}
]

for step in steps:
    with st.container():
        cols = st.columns([1, 10])
        with cols[0]:
            st.subheader(step["icon"])
        with cols[1]:
            st.markdown(f"**{step['title']}**  \n{step['text']}")
        st.write("")

st.divider()

st.header("📊Baseline")
st.markdown("""
Для сравнения использовали простые методы:
- **Топ популярных курсов**: Рекомендации одинаковы для всех  
- **Средние оценки курсов**: Без учета персональных предпочтений  

*Результат: Комбинированная модель на 7-15% точнее baseline.*
""")
st.divider()



st.header("😶‍🌫️Комбинировать подходы??")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🤝 Сначала коллаборативная фильтрация")
    st.markdown("""
    - Анализирует поведение похожих пользователей («Кто смотрел это, также смотрел то»)
    - Плюсы:
      - Учитывает реальные предпочтения
      - Хорошо работает при наличии данных
    """)

with col2:
    st.subheader("📝 Затем контентная фильтрация")
    st.markdown("""
    - Анализирует содержание курсов
    - Плюсы:
      - Работает для новых курсов
      - Не требует истории оценок
    """)

st.markdown("""
**→ Комбинация дает лучший результат**: личные предпочтения + широкий охват контента.
""")

st.divider()

st.header("Как можно улучшить систему?")
st.markdown("""
🔹 **Добавить больше данных**: Поведение пользователей (просмотры, время изучения)

🔹 **Усовершенствовать модель**:  
   - Тестировать другие алгоритмы (LightFM, NLF)  
   - Добавить временные фильтры (новые vs старые курсы) 
   - Использовать комментарии пользователей 

🔹 **Улучшить интерфейс**:  
   - Возможность уточнения запроса («Хочу курс на 2 недели»)  
   - Объяснение рекомендаций («Потому что вы смотрели Python») 
""")

st.divider()

st.header("Польза проекта")
st.markdown("""
✅ **Для учащихся**:  
   - Экономия времени на поиск курсов  
   - Персонализированные рекомендации  

✅ **Для платформ**:  
   - Увеличение вовлеченности пользователей  
   - Снижение оттока учащихся  

✅ **Для преподавателей**:  
   - Лучшее понимание аудитории  
   - Возможность адаптировать контент  
""")
