import streamlit as st
import pandas as pd

pages = [
    st.Page('bla_bla.py', title = 'О проекте 🫡'), 
    st.Page('about_table.py', title = 'Обзор 🔍'),
    st.Page('metrics.py', title = 'Метрики ⚖️'),
    st.Page('streamlit_proj.py', title = 'Модель💃')

]
pg_h = st.navigation(pages)
pg_h.run()