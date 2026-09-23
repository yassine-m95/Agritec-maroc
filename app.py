import streamlit as st
import pickle
import numpy as np
import base64
import plotly.graph_objects as go
import pandas as pd
import folium
from streamlit_folium import st_folium

# =============================================
# CHARGEMENT
# =============================================
@st.cache_resource
def load_models():
    model  = pickle.load(open('model.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    le     = pickle.load(open('label_encoder.pkl','rb'))
    return model, scaler, le

model, scaler, le = load_models()

# =============================================
# PALETTE OPTION A — VERT FORÊT
# =============================================
BG     = "#1B4332"
CARD   = "#2D6A4F"
CARD2  = "#1F5140"
ACCENT = "#40916C"
MINT   = "#74C69D"
LIGHT  = "#D8F3DC"
WHITE  = "#F0FFF4"
YELLOW = "#F9C74F"
ORANGE = "#F8961E"
RED    = "#F94144"
TXT    = "#F0FFF4"
GREY   = "#95D5B2"
BORDER = "rgba(116,198,157,0.25)"
SHADOW = "rgba(0,0,0,0.2)"

# =============================================
# TRADUCTIONS
# =============================================
T = {
'fr':{
    'title':'AgriTech Maroc',
    'subtitle':'Plateforme de Recommandation Agricole',
    'powered':'Propulsé par Random Forest ML',
    'acc':'Précision','crops':'Cultures',
    'obs':'Observations','trees':'Arbres RF',
    'regions':'Régions','reg_t':'📍 Régions marocaines',
    'tab1':'🔍 Recommandation',
    'tab2':'🗺️ Carte & Comparaison',
    'tab3':'📊 Analyse EDA',
    'tab4':'ℹ️ À propos',
    'sol_comp':'Composition du sol',
    'clim':'Paramètres climatiques',
    'N':'Azote (N) — kg/ha',
    'P':'Phosphore (P) — kg/ha',
    'K':'Potassium (K) — kg/ha',
    'temp':'Température (°C)',
    'hum':'Humidité (%)',
    'ph':'pH du sol',
    'rain':'Pluviométrie (mm)',
    'btn':'🌱 Analyser mon sol',
    'result':'Résultat de l\'analyse',
    'recommended':'CULTURE RECOMMANDÉE',
    'conf':'Confiance',
    'radar':'Profil du sol',
    'top5':'Top 5 cultures',
    'hist':'Historique des analyses',
    'evol':'Évolution des analyses',
    'map_title':'Carte des cultures au Maroc',
    'comp_title':'Comparaison de sols',
    'comp_sub':'Paramètres de 3 sols à comparer',
    'sol1':'Sol 1','sol2':'Sol 2','sol3':'Sol 3',
    'comp_btn':'⚖️ Comparer les 3 sols',
    'comp_res':'Résultats',
    'eda':'Analyse Exploratoire des Données',
    'dist':'Distribution des cultures',
    'corr':'Matrice de corrélation',
    'box':'Boxplot','stats':'Statistiques',
    'box_sel':'Choisissez une variable',
    'obj_t':'Objectif du projet',
    'obj_txt':"Plateforme PFE en Data Science pour assister les agriculteurs marocains dans leurs choix culturaux via 7 paramètres pédoclimatiques.",
    'ml_t':'Modèle Machine Learning',
    'ml_txt':'Random Forest — 100 arbres — 2200 observations — 22 cultures — ~99%.',
    'tech_t':'Technologies','data_t':'Dataset',
    'perf':'Performance','last':'Dernière recommandation',
    'sol':'Sol','var':'7 Variables',
    'warn':'⚠️ Fichier CSV introuvable.',
    'cats':['Azote','Phosphore','Potassium','Température','Humidité','pH','Pluie'],
    'del_all':'🗑️ Supprimer tout',
    'del_one':'🗑️',
    'detail_title':'Détails',
    'show':'▼ Ouvrir',
    'hide':'▲ Fermer',
    'regions_list':[
        ("🌴","Souss-Massa"),("💧","Gharb-Loukkos"),
        ("🍇","Meknès-Fès"),("🏜️","Tafilalet"),
        ("☀️","Doukkala"),("🏔️","Moyen Atlas"),
    ],
},
'en':{
    'title':'AgriTech Morocco',
    'subtitle':'Agricultural Recommendation Platform',
    'powered':'Powered by Random Forest ML',
    'acc':'Accuracy','crops':'Crops',
    'obs':'Observations','trees':'RF Trees',
    'regions':'Regions','reg_t':'📍 Moroccan Regions',
    'tab1':'🔍 Recommendation',
    'tab2':'🗺️ Map & Comparison',
    'tab3':'📊 EDA Analysis',
    'tab4':'ℹ️ About',
    'sol_comp':'Soil Composition',
    'clim':'Climatic Parameters',
    'N':'Nitrogen (N) — kg/ha',
    'P':'Phosphorus (P) — kg/ha',
    'K':'Potassium (K) — kg/ha',
    'temp':'Temperature (°C)',
    'hum':'Humidity (%)',
    'ph':'Soil pH',
    'rain':'Rainfall (mm)',
    'btn':'🌱 Analyze My Soil',
    'result':'Analysis Result',
    'recommended':'RECOMMENDED CROP',
    'conf':'Confidence',
    'radar':'Soil Profile',
    'top5':'Top 5 Crops',
    'hist':'Analysis History',
    'evol':'Analysis Evolution',
    'map_title':'Crop Map of Morocco',
    'comp_title':'Soil Comparison',
    'comp_sub':'Parameters for 3 soils',
    'sol1':'Soil 1','sol2':'Soil 2','sol3':'Soil 3',
    'comp_btn':'⚖️ Compare 3 Soils',
    'comp_res':'Results',
    'eda':'Exploratory Data Analysis',
    'dist':'Crop Distribution',
    'corr':'Correlation Matrix',
    'box':'Boxplot','stats':'Statistics',
    'box_sel':'Choose a variable',
    'obj_t':'Project Objective',
    'obj_txt':'Data Science FYP to assist Moroccan farmers in crop selection via 7 pedoclimatic parameters.',
    'ml_t':'Machine Learning Model',
    'ml_txt':'Random Forest — 100 trees — 2200 observations — 22 crops — ~99%.',
    'tech_t':'Technologies','data_t':'Dataset',
    'perf':'Performance','last':'Last Recommendation',
    'sol':'Soil','var':'7 Variables',
    'warn':'⚠️ CSV file not found.',
    'cats':['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall'],
    'del_all':'🗑️ Delete All',
    'del_one':'🗑️',
    'detail_title':'Details',
    'show':'▼ Open',
    'hide':'▲ Close',
    'regions_list':[
        ("🌴","Souss-Massa"),("💧","Gharb-Loukkos"),
        ("🍇","Meknès-Fès"),("🏜️","Tafilalet"),
        ("☀️","Doukkala"),("🏔️","Middle Atlas"),
    ],
},
'ar':{
    'title':'التكنولوجيا الزراعية في المغرب',
    'subtitle':'منصة التوصية الزراعية الذكية',
    'powered':'مدعوم بتقنية Random Forest',
    'acc':'الدقة','crops':'المحاصيل',
    'obs':'ملاحظات','trees':'أشجار RF',
    'regions':'المناطق','reg_t':'📍 المناطق المغربية',
    'tab1':'🔍 التوصية',
    'tab2':'🗺️ الخريطة والمقارنة',
    'tab3':'📊 تحليل البيانات',
    'tab4':'ℹ️ حول المشروع',
    'sol_comp':'تركيبة التربة',
    'clim':'المعاملات المناخية',
    'N':'النيتروجين (N) — كغ/هكتار',
    'P':'الفوسفور (P) — كغ/هكتار',
    'K':'البوتاسيوم (K) — كغ/هكتار',
    'temp':'درجة الحرارة (°م)',
    'hum':'الرطوبة (%)',
    'ph':'حموضة التربة',
    'rain':'التساقطات (ملم)',
    'btn':'🌱 تحليل تربتي',
    'result':'نتيجة التحليل',
    'recommended':'المحصول الموصى به',
    'conf':'نسبة الثقة',
    'radar':'ملف التربة',
    'top5':'أفضل 5 محاصيل',
    'hist':'سجل التحليلات',
    'evol':'تطور التحليلات',
    'map_title':'خريطة المحاصيل في المغرب',
    'comp_title':'مقارنة التربات',
    'comp_sub':'معاملات 3 تربات للمقارنة',
    'sol1':'تربة 1','sol2':'تربة 2','sol3':'تربة 3',
    'comp_btn':'⚖️ مقارنة 3 تربات',
    'comp_res':'النتائج',
    'eda':'التحليل الاستكشافي للبيانات',
    'dist':'توزيع المحاصيل',
    'corr':'مصفوفة الارتباط',
    'box':'المخطط الصندوقي',
    'stats':'الإحصاءات',
    'box_sel':'اختر متغيراً',
    'obj_t':'هدف المشروع',
    'obj_txt':'منصة مشروع نهاية الدراسة لمساعدة المزارعين المغاربة في اختيار المحاصيل بناءً على 7 معاملات علمية.',
    'ml_t':'نموذج التعلم الآلي',
    'ml_txt':'الغابة العشوائية — 100 شجرة — 2200 ملاحظة — 22 محصول — دقة 99%.',
    'tech_t':'التقنيات','data_t':'البيانات',
    'perf':'الأداء','last':'آخر توصية',
    'sol':'تربة','var':'7 متغيرات',
    'warn':'⚠️ ملف CSV غير موجود.',
    'cats':['النيتروجين','الفوسفور','البوتاسيوم','الحرارة','الرطوبة','pH','التساقطات'],
    'del_all':'🗑️ حذف الكل',
    'del_one':'🗑️',
    'detail_title':'التفاصيل',
    'show':'▼ فتح',
    'hide':'▲ إغلاق',
    'regions_list':[
        ("🌴","سوس-ماسة"),("💧","الغرب-لوكوس"),
        ("🍇","مكناس-فاس"),("🏜️","تافيلالت"),
        ("☀️","دكالة"),("🏔️","أطلس المتوسط"),
    ],
},
}

C = {
'apple'      :{'fr':'🍎 Pomme','en':'🍎 Apple','ar':'🍎 تفاح',
               'rfr':'Moyen Atlas','ren':'Middle Atlas','rar':'أطلس المتوسط',
               'lat':31.17,'lon':-7.33,'color':'#E74C3C'},
'banana'     :{'fr':'🍌 Banane','en':'🍌 Banana','ar':'🍌 موز',
               'rfr':'Souss','ren':'Souss','rar':'سوس',
               'lat':30.42,'lon':-9.59,'color':'#F1C40F'},
'blackgram'  :{'fr':'🫘 Haricot noir','en':'🫘 Black Gram','ar':'🫘 فاصوليا سوداء',
               'rfr':'Gharb','ren':'Gharb','rar':'الغرب',
               'lat':34.55,'lon':-5.99,'color':'#74C69D'},
'chickpea'   :{'fr':'🟡 Pois chiche','en':'🟡 Chickpea','ar':'🟡 حمص',
               'rfr':'Meknes-Fes','ren':'Meknes-Fes','rar':'مكناس-فاس',
               'lat':33.89,'lon':-5.54,'color':'#F8961E'},
'coconut'    :{'fr':'🥥 Cocotier','en':'🥥 Coconut','ar':'🥥 جوز الهند',
               'rfr':'Cote Atlantique','ren':'Atlantic Coast','rar':'الساحل الأطلسي',
               'lat':33.57,'lon':-7.58,'color':'#8B5E3C'},
'coffee'     :{'fr':'☕ Cafe','en':'☕ Coffee','ar':'☕ قهوة',
               'rfr':'Zones humides','ren':'Humid Zones','rar':'المناطق الرطبة',
               'lat':34.02,'lon':-5.00,'color':'#6F4E37'},
'cotton'     :{'fr':'🌿 Coton','en':'🌿 Cotton','ar':'🌿 قطن',
               'rfr':'Gharb-Loukkos','ren':'Gharb-Loukkos','rar':'الغرب-لوكوس',
               'lat':34.30,'lon':-6.20,'color':'#95D5B2'},
'grapes'     :{'fr':'🍇 Raisin','en':'🍇 Grapes','ar':'🍇 عنب',
               'rfr':'Meknes','ren':'Meknes','rar':'مكناس',
               'lat':33.70,'lon':-5.80,'color':'#C77DFF'},
'jute'       :{'fr':'🌾 Jute','en':'🌾 Jute','ar':'🌾 جوت',
               'rfr':'Gharb','ren':'Gharb','rar':'الغرب',
               'lat':34.55,'lon':-6.30,'color':'#D4AC0D'},
'kidneybeans':{'fr':'🫘 Haricot rouge','en':'🫘 Kidney Beans','ar':'🫘 فاصوليا حمراء',
               'rfr':'Souss-Massa','ren':'Souss-Massa','rar':'سوس-ماسة',
               'lat':30.00,'lon':-9.80,'color':'#F94144'},
'lentil'     :{'fr':'🟤 Lentille','en':'🟤 Lentil','ar':'🟤 عدس',
               'rfr':'Meknes-Fes','ren':'Meknes-Fes','rar':'مكناس-فاس',
               'lat':33.60,'lon':-5.20,'color':'#A04000'},
'maize'      :{'fr':'🌽 Mais','en':'🌽 Maize','ar':'🌽 ذرة',
               'rfr':'Doukkala','ren':'Doukkala','rar':'دكالة',
               'lat':32.29,'lon':-8.50,'color':'#F9C74F'},
'mango'      :{'fr':'🥭 Mangue','en':'🥭 Mango','ar':'🥭 مانجو',
               'rfr':'Souss','ren':'Souss','rar':'سوس',
               'lat':29.80,'lon':-9.20,'color':'#FF8C00'},
'mothbeans'  :{'fr':'🫘 Haricot Moth','en':'🫘 Moth Beans','ar':'🫘 فاصوليا موث',
               'rfr':'Oriental','ren':'Oriental','rar':'الشرقية',
               'lat':34.68,'lon':-1.90,'color':'#7D6608'},
'mungbean'   :{'fr':'🌱 Haricot mungo','en':'🌱 Mung Bean','ar':'🌱 فاصوليا خضراء',
               'rfr':'Gharb','ren':'Gharb','rar':'الغرب',
               'lat':34.20,'lon':-6.50,'color':'#52B788'},
'muskmelon'  :{'fr':'🍈 Melon','en':'🍈 Muskmelon','ar':'🍈 شمام',
               'rfr':'Souss-Massa','ren':'Souss-Massa','rar':'سوس-ماسة',
               'lat':30.60,'lon':-9.40,'color':'#48CAE4'},
'orange'     :{'fr':'🍊 Orange','en':'🍊 Orange','ar':'🍊 برتقال',
               'rfr':'Souss-Massa','ren':'Souss-Massa','rar':'سوس-ماسة',
               'lat':30.20,'lon':-9.60,'color':'#F8961E'},
'papaya'     :{'fr':'🍑 Papaye','en':'🍑 Papaya','ar':'🍑 بابايا',
               'rfr':'Souss','ren':'Souss','rar':'سوس',
               'lat':29.60,'lon':-9.80,'color':'#F0B27A'},
'pigeonpeas' :{'fr':'🫛 Pois Angole','en':'🫛 Pigeon Peas','ar':'🫛 بازلاء الحمام',
               'rfr':'Sud Maroc','ren':'South Morocco','rar':'جنوب المغرب',
               'lat':28.41,'lon':-11.08,'color':'#7DCEA0'},
'pomegranate':{'fr':'🍎 Grenade','en':'🍎 Pomegranate','ar':'🍎 رمان',
               'rfr':'Tafilalet','ren':'Tafilalet','rar':'تافيلالت',
               'lat':31.93,'lon':-4.42,'color':'#F94144'},
'rice'       :{'fr':'🌾 Riz','en':'🌾 Rice','ar':'🌾 أرز',
               'rfr':'Gharb-Loukkos','ren':'Gharb-Loukkos','rar':'الغرب-لوكوس',
               'lat':34.80,'lon':-5.80,'color':'#74C69D'},
'watermelon' :{'fr':'🍉 Pasteque','en':'🍉 Watermelon','ar':'🍉 بطيخ',
               'rfr':'Doukkala','ren':'Doukkala','rar':'دكالة',
               'lat':32.00,'lon':-8.80,'color':'#40916C'},
}

def gn(c,l): return C.get(c,{}).get(l,c)
def gr(c,l):
    k={'fr':'rfr','en':'ren','ar':'rar'}.get(l,'rfr')
    return C.get(c,{}).get(k,'Maroc')

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="AgriTech Maroc",page_icon="🌱",
    layout="wide",initial_sidebar_state="expanded"
)

# =============================================
# SESSION STATE
# =============================================
for k,v in [('lang','fr'),('hist',[]),('sel',None),
            ('open_sol',[True,False,False])]:
    if k not in st.session_state:
        st.session_state[k]=v

# =============================================
# LANGUE
# =============================================
lc=st.columns([5,1,1,1])
with lc[1]:
    if st.button("🇫🇷 FR"): st.session_state.lang='fr'
with lc[2]:
    if st.button("🇬🇧 EN"): st.session_state.lang='en'
with lc[3]:
    if st.button("🇲🇦 AR"): st.session_state.lang='ar'

lang=st.session_state.lang
t=T[lang]
rtl=lang=='ar'
ds="direction:rtl;text-align:right;" if rtl else ""
ff="Tajawal" if rtl else "Nunito"

# =============================================
# CSS OPTION A — VERT FORÊT
# =============================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

*{{font-family:{ff},sans-serif!important;}}

/* FOND GLOBAL — PAYSAGE AGRICOLE */
[data-testid="stAppViewContainer"]{{
    background:
      linear-gradient(180deg, rgba(11,61,46,0.45) 0%, rgba(27,67,50,0.62) 55%, rgba(27,67,50,0.75) 100%),
      url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%201440%20900%22%20preserveAspectRatio%3D%22xMidYMid%20slice%22%3E%3Cdefs%3E%3ClinearGradient%20id%3D%22sky%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%230B3D2E%22/%3E%3Cstop%20offset%3D%220.45%22%20stop-color%3D%22%231B5E43%22/%3E%3Cstop%20offset%3D%220.75%22%20stop-color%3D%22%232D6A4F%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2340916C%22/%3E%3C/linearGradient%3E%3CradialGradient%20id%3D%22sung%22%20cx%3D%220.5%22%20cy%3D%220.5%22%20r%3D%220.5%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%23FFF4D6%22/%3E%3Cstop%20offset%3D%220.35%22%20stop-color%3D%22%23FFE89C%22/%3E%3Cstop%20offset%3D%220.7%22%20stop-color%3D%22%23F9C74F%22%20stop-opacity%3D%220.7%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23F9C74F%22%20stop-opacity%3D%220%22/%3E%3C/radialGradient%3E%3ClinearGradient%20id%3D%22hill1%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2374C69D%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2352B788%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id%3D%22hill2%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2352B788%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2340916C%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id%3D%22hill3%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2340916C%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%232D6A4F%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id%3D%22field%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2395D5B2%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2352B788%22/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect%20width%3D%221440%22%20height%3D%22900%22%20fill%3D%22url%28%23sky%29%22/%3E%3C%21--%20soleil%20--%3E%3Ccircle%20cx%3D%221180%22%20cy%3D%22180%22%20r%3D%22220%22%20fill%3D%22url%28%23sung%29%22/%3E%3Ccircle%20cx%3D%221180%22%20cy%3D%22180%22%20r%3D%2262%22%20fill%3D%22%23FFF4D6%22/%3E%3Cg%20stroke%3D%22%23FFE89C%22%20stroke-width%3D%225%22%20stroke-linecap%3D%22round%22%20opacity%3D%220.55%22%3E%3Cline%20x1%3D%221180%22%20y1%3D%2270%22%20x2%3D%221180%22%20y2%3D%2220%22/%3E%3Cline%20x1%3D%221180%22%20y1%3D%22290%22%20x2%3D%221180%22%20y2%3D%22340%22/%3E%3Cline%20x1%3D%221070%22%20y1%3D%22180%22%20x2%3D%221020%22%20y2%3D%22180%22/%3E%3Cline%20x1%3D%221290%22%20y1%3D%22180%22%20x2%3D%221340%22%20y2%3D%22180%22/%3E%3Cline%20x1%3D%221103%22%20y1%3D%22103%22%20x2%3D%221068%22%20y2%3D%2268%22/%3E%3Cline%20x1%3D%221257%22%20y1%3D%22103%22%20x2%3D%221292%22%20y2%3D%2268%22/%3E%3Cline%20x1%3D%221103%22%20y1%3D%22257%22%20x2%3D%221068%22%20y2%3D%22292%22/%3E%3Cline%20x1%3D%221257%22%20y1%3D%22257%22%20x2%3D%221292%22%20y2%3D%22292%22/%3E%3C/g%3E%3C%21--%20oiseaux%20--%3E%3Cg%20stroke%3D%22%231B4332%22%20stroke-width%3D%223%22%20fill%3D%22none%22%20opacity%3D%220.5%22%20stroke-linecap%3D%22round%22%3E%3Cpath%20d%3D%22M300%2C150%20q15%2C-15%2030%2C0%20q15%2C-15%2030%2C0%22/%3E%3Cpath%20d%3D%22M420%2C200%20q12%2C-12%2024%2C0%20q12%2C-12%2024%2C0%22/%3E%3Cpath%20d%3D%22M250%2C230%20q10%2C-10%2020%2C0%20q10%2C-10%2020%2C0%22/%3E%3C/g%3E%3C%21--%20collines%20lointaines%20--%3E%3Cpath%20d%3D%22M0%2C430%20Q360%2C330%20720%2C400%20T1440%2C380%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23hill1%29%22%20opacity%3D%220.7%22/%3E%3Cpath%20d%3D%22M0%2C500%20Q300%2C420%20640%2C480%20T1440%2C470%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23hill2%29%22%20opacity%3D%220.85%22/%3E%3Cpath%20d%3D%22M0%2C580%20Q380%2C520%20760%2C575%20T1440%2C560%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23hill3%29%22/%3E%3C%21--%20champs%20color%C3%A9s%20en%20rangs%20%28perspective%29%20--%3E%3Cg%20opacity%3D%220.9%22%3E%3Cpath%20d%3D%22M0%2C640%20L1440%2C640%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23field%29%22/%3E%3Cg%20stroke%3D%22%2340916C%22%20stroke-width%3D%223%22%20opacity%3D%220.6%22%3E%3Cline%20x1%3D%22120%22%20y1%3D%22900%22%20x2%3D%22500%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%22320%22%20y1%3D%22900%22%20x2%3D%22600%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%22520%22%20y1%3D%22900%22%20x2%3D%22700%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%22760%22%20y1%3D%22900%22%20x2%3D%22800%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%221000%22%20y1%3D%22900%22%20x2%3D%22900%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%221240%22%20y1%3D%22900%22%20x2%3D%221000%22%20y2%3D%22650%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23F9C74F%22%20opacity%3D%220.55%22%3E%3Cellipse%20cx%3D%22200%22%20cy%3D%22720%22%20rx%3D%2240%22%20ry%3D%2210%22/%3E%3Cellipse%20cx%3D%22560%22%20cy%3D%22700%22%20rx%3D%2234%22%20ry%3D%228%22/%3E%3Cellipse%20cx%3D%22980%22%20cy%3D%22710%22%20rx%3D%2238%22%20ry%3D%229%22/%3E%3Cellipse%20cx%3D%221280%22%20cy%3D%22730%22%20rx%3D%2242%22%20ry%3D%2210%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23E76F51%22%20opacity%3D%220.4%22%3E%3Cellipse%20cx%3D%22380%22%20cy%3D%22760%22%20rx%3D%2230%22%20ry%3D%228%22/%3E%3Cellipse%20cx%3D%22780%22%20cy%3D%22755%22%20rx%3D%2232%22%20ry%3D%228%22/%3E%3Cellipse%20cx%3D%221120%22%20cy%3D%22770%22%20rx%3D%2230%22%20ry%3D%228%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20ARBRES%20FRUITIERS%20d%C3%A9taill%C3%A9s%20--%3E%3C%21--%20oranger%20--%3E%3Cg%20transform%3D%22translate%28180%2C470%29%22%3E%3Crect%20x%3D%22-9%22%20y%3D%2240%22%20width%3D%2218%22%20height%3D%2290%22%20rx%3D%225%22%20fill%3D%22%236B4226%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%2262%22%20fill%3D%22%2352B788%22/%3E%3Ccircle%20cx%3D%22-40%22%20cy%3D%2220%22%20r%3D%2244%22%20fill%3D%22%2340916C%22/%3E%3Ccircle%20cx%3D%2240%22%20cy%3D%2220%22%20r%3D%2244%22%20fill%3D%22%2374C69D%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%22-30%22%20r%3D%2240%22%20fill%3D%22%2374C69D%22/%3E%3Cg%20fill%3D%22%23F8961E%22%3E%3Ccircle%20cx%3D%22-20%22%20cy%3D%22-5%22%20r%3D%228%22/%3E%3Ccircle%20cx%3D%2222%22%20cy%3D%220%22%20r%3D%228%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2225%22%20r%3D%228%22/%3E%3Ccircle%20cx%3D%22-35%22%20cy%3D%2230%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%2238%22%20cy%3D%2232%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%2210%22%20cy%3D%22-20%22%20r%3D%227%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20pommier%20--%3E%3Cg%20transform%3D%22translate%28520%2C510%29%22%3E%3Crect%20x%3D%22-8%22%20y%3D%2235%22%20width%3D%2216%22%20height%3D%2280%22%20rx%3D%225%22%20fill%3D%22%236B4226%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%2255%22%20fill%3D%22%2374C69D%22/%3E%3Ccircle%20cx%3D%22-34%22%20cy%3D%2218%22%20r%3D%2238%22%20fill%3D%22%2352B788%22/%3E%3Ccircle%20cx%3D%2234%22%20cy%3D%2218%22%20r%3D%2238%22%20fill%3D%22%2395D5B2%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%22-26%22%20r%3D%2234%22%20fill%3D%22%2352B788%22/%3E%3Cg%20fill%3D%22%23F94144%22%3E%3Ccircle%20cx%3D%22-18%22%20cy%3D%220%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%2220%22%20cy%3D%225%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2222%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%22-30%22%20cy%3D%2225%22%20r%3D%226%22/%3E%3Ccircle%20cx%3D%2232%22%20cy%3D%2228%22%20r%3D%226%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20olivier%20--%3E%3Cg%20transform%3D%22translate%281280%2C540%29%22%3E%3Crect%20x%3D%22-8%22%20y%3D%2230%22%20width%3D%2216%22%20height%3D%2278%22%20rx%3D%225%22%20fill%3D%22%235C4033%22/%3E%3Cellipse%20cx%3D%220%22%20cy%3D%22-5%22%20rx%3D%2250%22%20ry%3D%2260%22%20fill%3D%22%2352B788%22/%3E%3Cellipse%20cx%3D%220%22%20cy%3D%22-5%22%20rx%3D%2230%22%20ry%3D%2242%22%20fill%3D%22%2374C69D%22/%3E%3Cg%20fill%3D%22%232D6A4F%22%3E%3Ccircle%20cx%3D%22-15%22%20cy%3D%220%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%2216%22%20cy%3D%228%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2220%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%22-10%22%20cy%3D%22-25%22%20r%3D%225%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20palmier%20--%3E%3Cg%20transform%3D%22translate%28880%2C520%29%22%3E%3Cpath%20d%3D%22M-7%2C40%20Q-2%2C-10%208%2C-50%20L18%2C-48%20Q6%2C-8%207%2C40%20Z%22%20fill%3D%22%236B4226%22/%3E%3Cg%20fill%3D%22%2352B788%22%3E%3Cpath%20d%3D%22M8%2C-50%20Q-40%2C-70%20-75%2C-55%20Q-38%2C-48%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q50%2C-72%2086%2C-58%20Q44%2C-48%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q-30%2C-95%20-55%2C-110%20Q-12%2C-78%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q40%2C-96%2066%2C-112%20Q22%2C-78%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q4%2C-100%208%2C-128%20Q14%2C-98%208%2C-50%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23F8961E%22%3E%3Ccircle%20cx%3D%226%22%20cy%3D%22-44%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%22-4%22%20cy%3D%22-40%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%2214%22%20cy%3D%22-40%22%20r%3D%225%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20petit%20arbuste%20--%3E%3Cg%20transform%3D%22translate%28700%2C580%29%22%3E%3Crect%20x%3D%22-5%22%20y%3D%2220%22%20width%3D%2210%22%20height%3D%2240%22%20rx%3D%224%22%20fill%3D%22%236B4226%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%2234%22%20fill%3D%22%2374C69D%22/%3E%3Ccircle%20cx%3D%22-20%22%20cy%3D%2210%22%20r%3D%2222%22%20fill%3D%22%2352B788%22/%3E%3Ccircle%20cx%3D%2220%22%20cy%3D%2210%22%20r%3D%2222%22%20fill%3D%22%2395D5B2%22/%3E%3Cg%20fill%3D%22%23E76F51%22%3E%3Ccircle%20cx%3D%22-10%22%20cy%3D%222%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%2212%22%20cy%3D%226%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2215%22%20r%3D%225%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")!important;
    background-size:cover!important;
    background-position:center top!important;
    background-attachment:fixed!important;
}}
.main,.block-container{{background:transparent!important;}}
[data-testid="stHeader"]{{background:transparent!important;}}
[data-testid="stSidebar"]{{
    background:
      linear-gradient(180deg, rgba(31,81,64,0.92) 0%, rgba(27,67,50,0.95) 100%),
      url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%201440%20900%22%20preserveAspectRatio%3D%22xMidYMid%20slice%22%3E%3Cdefs%3E%3ClinearGradient%20id%3D%22sky%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%230B3D2E%22/%3E%3Cstop%20offset%3D%220.45%22%20stop-color%3D%22%231B5E43%22/%3E%3Cstop%20offset%3D%220.75%22%20stop-color%3D%22%232D6A4F%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2340916C%22/%3E%3C/linearGradient%3E%3CradialGradient%20id%3D%22sung%22%20cx%3D%220.5%22%20cy%3D%220.5%22%20r%3D%220.5%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%23FFF4D6%22/%3E%3Cstop%20offset%3D%220.35%22%20stop-color%3D%22%23FFE89C%22/%3E%3Cstop%20offset%3D%220.7%22%20stop-color%3D%22%23F9C74F%22%20stop-opacity%3D%220.7%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23F9C74F%22%20stop-opacity%3D%220%22/%3E%3C/radialGradient%3E%3ClinearGradient%20id%3D%22hill1%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2374C69D%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2352B788%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id%3D%22hill2%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2352B788%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2340916C%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id%3D%22hill3%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2340916C%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%232D6A4F%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id%3D%22field%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22%2395D5B2%22/%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%2352B788%22/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect%20width%3D%221440%22%20height%3D%22900%22%20fill%3D%22url%28%23sky%29%22/%3E%3C%21--%20soleil%20--%3E%3Ccircle%20cx%3D%221180%22%20cy%3D%22180%22%20r%3D%22220%22%20fill%3D%22url%28%23sung%29%22/%3E%3Ccircle%20cx%3D%221180%22%20cy%3D%22180%22%20r%3D%2262%22%20fill%3D%22%23FFF4D6%22/%3E%3Cg%20stroke%3D%22%23FFE89C%22%20stroke-width%3D%225%22%20stroke-linecap%3D%22round%22%20opacity%3D%220.55%22%3E%3Cline%20x1%3D%221180%22%20y1%3D%2270%22%20x2%3D%221180%22%20y2%3D%2220%22/%3E%3Cline%20x1%3D%221180%22%20y1%3D%22290%22%20x2%3D%221180%22%20y2%3D%22340%22/%3E%3Cline%20x1%3D%221070%22%20y1%3D%22180%22%20x2%3D%221020%22%20y2%3D%22180%22/%3E%3Cline%20x1%3D%221290%22%20y1%3D%22180%22%20x2%3D%221340%22%20y2%3D%22180%22/%3E%3Cline%20x1%3D%221103%22%20y1%3D%22103%22%20x2%3D%221068%22%20y2%3D%2268%22/%3E%3Cline%20x1%3D%221257%22%20y1%3D%22103%22%20x2%3D%221292%22%20y2%3D%2268%22/%3E%3Cline%20x1%3D%221103%22%20y1%3D%22257%22%20x2%3D%221068%22%20y2%3D%22292%22/%3E%3Cline%20x1%3D%221257%22%20y1%3D%22257%22%20x2%3D%221292%22%20y2%3D%22292%22/%3E%3C/g%3E%3C%21--%20oiseaux%20--%3E%3Cg%20stroke%3D%22%231B4332%22%20stroke-width%3D%223%22%20fill%3D%22none%22%20opacity%3D%220.5%22%20stroke-linecap%3D%22round%22%3E%3Cpath%20d%3D%22M300%2C150%20q15%2C-15%2030%2C0%20q15%2C-15%2030%2C0%22/%3E%3Cpath%20d%3D%22M420%2C200%20q12%2C-12%2024%2C0%20q12%2C-12%2024%2C0%22/%3E%3Cpath%20d%3D%22M250%2C230%20q10%2C-10%2020%2C0%20q10%2C-10%2020%2C0%22/%3E%3C/g%3E%3C%21--%20collines%20lointaines%20--%3E%3Cpath%20d%3D%22M0%2C430%20Q360%2C330%20720%2C400%20T1440%2C380%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23hill1%29%22%20opacity%3D%220.7%22/%3E%3Cpath%20d%3D%22M0%2C500%20Q300%2C420%20640%2C480%20T1440%2C470%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23hill2%29%22%20opacity%3D%220.85%22/%3E%3Cpath%20d%3D%22M0%2C580%20Q380%2C520%20760%2C575%20T1440%2C560%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23hill3%29%22/%3E%3C%21--%20champs%20color%C3%A9s%20en%20rangs%20%28perspective%29%20--%3E%3Cg%20opacity%3D%220.9%22%3E%3Cpath%20d%3D%22M0%2C640%20L1440%2C640%20L1440%2C900%20L0%2C900%20Z%22%20fill%3D%22url%28%23field%29%22/%3E%3Cg%20stroke%3D%22%2340916C%22%20stroke-width%3D%223%22%20opacity%3D%220.6%22%3E%3Cline%20x1%3D%22120%22%20y1%3D%22900%22%20x2%3D%22500%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%22320%22%20y1%3D%22900%22%20x2%3D%22600%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%22520%22%20y1%3D%22900%22%20x2%3D%22700%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%22760%22%20y1%3D%22900%22%20x2%3D%22800%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%221000%22%20y1%3D%22900%22%20x2%3D%22900%22%20y2%3D%22650%22/%3E%3Cline%20x1%3D%221240%22%20y1%3D%22900%22%20x2%3D%221000%22%20y2%3D%22650%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23F9C74F%22%20opacity%3D%220.55%22%3E%3Cellipse%20cx%3D%22200%22%20cy%3D%22720%22%20rx%3D%2240%22%20ry%3D%2210%22/%3E%3Cellipse%20cx%3D%22560%22%20cy%3D%22700%22%20rx%3D%2234%22%20ry%3D%228%22/%3E%3Cellipse%20cx%3D%22980%22%20cy%3D%22710%22%20rx%3D%2238%22%20ry%3D%229%22/%3E%3Cellipse%20cx%3D%221280%22%20cy%3D%22730%22%20rx%3D%2242%22%20ry%3D%2210%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23E76F51%22%20opacity%3D%220.4%22%3E%3Cellipse%20cx%3D%22380%22%20cy%3D%22760%22%20rx%3D%2230%22%20ry%3D%228%22/%3E%3Cellipse%20cx%3D%22780%22%20cy%3D%22755%22%20rx%3D%2232%22%20ry%3D%228%22/%3E%3Cellipse%20cx%3D%221120%22%20cy%3D%22770%22%20rx%3D%2230%22%20ry%3D%228%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20ARBRES%20FRUITIERS%20d%C3%A9taill%C3%A9s%20--%3E%3C%21--%20oranger%20--%3E%3Cg%20transform%3D%22translate%28180%2C470%29%22%3E%3Crect%20x%3D%22-9%22%20y%3D%2240%22%20width%3D%2218%22%20height%3D%2290%22%20rx%3D%225%22%20fill%3D%22%236B4226%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%2262%22%20fill%3D%22%2352B788%22/%3E%3Ccircle%20cx%3D%22-40%22%20cy%3D%2220%22%20r%3D%2244%22%20fill%3D%22%2340916C%22/%3E%3Ccircle%20cx%3D%2240%22%20cy%3D%2220%22%20r%3D%2244%22%20fill%3D%22%2374C69D%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%22-30%22%20r%3D%2240%22%20fill%3D%22%2374C69D%22/%3E%3Cg%20fill%3D%22%23F8961E%22%3E%3Ccircle%20cx%3D%22-20%22%20cy%3D%22-5%22%20r%3D%228%22/%3E%3Ccircle%20cx%3D%2222%22%20cy%3D%220%22%20r%3D%228%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2225%22%20r%3D%228%22/%3E%3Ccircle%20cx%3D%22-35%22%20cy%3D%2230%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%2238%22%20cy%3D%2232%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%2210%22%20cy%3D%22-20%22%20r%3D%227%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20pommier%20--%3E%3Cg%20transform%3D%22translate%28520%2C510%29%22%3E%3Crect%20x%3D%22-8%22%20y%3D%2235%22%20width%3D%2216%22%20height%3D%2280%22%20rx%3D%225%22%20fill%3D%22%236B4226%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%2255%22%20fill%3D%22%2374C69D%22/%3E%3Ccircle%20cx%3D%22-34%22%20cy%3D%2218%22%20r%3D%2238%22%20fill%3D%22%2352B788%22/%3E%3Ccircle%20cx%3D%2234%22%20cy%3D%2218%22%20r%3D%2238%22%20fill%3D%22%2395D5B2%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%22-26%22%20r%3D%2234%22%20fill%3D%22%2352B788%22/%3E%3Cg%20fill%3D%22%23F94144%22%3E%3Ccircle%20cx%3D%22-18%22%20cy%3D%220%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%2220%22%20cy%3D%225%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2222%22%20r%3D%227%22/%3E%3Ccircle%20cx%3D%22-30%22%20cy%3D%2225%22%20r%3D%226%22/%3E%3Ccircle%20cx%3D%2232%22%20cy%3D%2228%22%20r%3D%226%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20olivier%20--%3E%3Cg%20transform%3D%22translate%281280%2C540%29%22%3E%3Crect%20x%3D%22-8%22%20y%3D%2230%22%20width%3D%2216%22%20height%3D%2278%22%20rx%3D%225%22%20fill%3D%22%235C4033%22/%3E%3Cellipse%20cx%3D%220%22%20cy%3D%22-5%22%20rx%3D%2250%22%20ry%3D%2260%22%20fill%3D%22%2352B788%22/%3E%3Cellipse%20cx%3D%220%22%20cy%3D%22-5%22%20rx%3D%2230%22%20ry%3D%2242%22%20fill%3D%22%2374C69D%22/%3E%3Cg%20fill%3D%22%232D6A4F%22%3E%3Ccircle%20cx%3D%22-15%22%20cy%3D%220%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%2216%22%20cy%3D%228%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2220%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%22-10%22%20cy%3D%22-25%22%20r%3D%225%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20palmier%20--%3E%3Cg%20transform%3D%22translate%28880%2C520%29%22%3E%3Cpath%20d%3D%22M-7%2C40%20Q-2%2C-10%208%2C-50%20L18%2C-48%20Q6%2C-8%207%2C40%20Z%22%20fill%3D%22%236B4226%22/%3E%3Cg%20fill%3D%22%2352B788%22%3E%3Cpath%20d%3D%22M8%2C-50%20Q-40%2C-70%20-75%2C-55%20Q-38%2C-48%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q50%2C-72%2086%2C-58%20Q44%2C-48%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q-30%2C-95%20-55%2C-110%20Q-12%2C-78%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q40%2C-96%2066%2C-112%20Q22%2C-78%208%2C-50%22/%3E%3Cpath%20d%3D%22M8%2C-50%20Q4%2C-100%208%2C-128%20Q14%2C-98%208%2C-50%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23F8961E%22%3E%3Ccircle%20cx%3D%226%22%20cy%3D%22-44%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%22-4%22%20cy%3D%22-40%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%2214%22%20cy%3D%22-40%22%20r%3D%225%22/%3E%3C/g%3E%3C/g%3E%3C%21--%20petit%20arbuste%20--%3E%3Cg%20transform%3D%22translate%28700%2C580%29%22%3E%3Crect%20x%3D%22-5%22%20y%3D%2220%22%20width%3D%2210%22%20height%3D%2240%22%20rx%3D%224%22%20fill%3D%22%236B4226%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%2234%22%20fill%3D%22%2374C69D%22/%3E%3Ccircle%20cx%3D%22-20%22%20cy%3D%2210%22%20r%3D%2222%22%20fill%3D%22%2352B788%22/%3E%3Ccircle%20cx%3D%2220%22%20cy%3D%2210%22%20r%3D%2222%22%20fill%3D%22%2395D5B2%22/%3E%3Cg%20fill%3D%22%23E76F51%22%3E%3Ccircle%20cx%3D%22-10%22%20cy%3D%222%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%2212%22%20cy%3D%226%22%20r%3D%225%22/%3E%3Ccircle%20cx%3D%220%22%20cy%3D%2215%22%20r%3D%225%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")!important;
    background-size:cover!important;
    background-position:left top!important;
    border-right:1px solid {BORDER}!important;
}}

/* HEADER */
.hdr{{
    position:relative;overflow:hidden;
    background:linear-gradient(135deg, rgba(8,28,21,0.50) 0%, rgba(45,106,79,0.38) 100%);
    backdrop-filter:blur(3px);
    border-radius:24px;padding:22px 28px 26px 28px;
    text-align:center;margin-bottom:24px;
    border:1px solid rgba(255,255,255,0.20);
    box-shadow:0 12px 40px rgba(0,0,0,0.42);{ds}
}}
.hdr-content{{position:relative;z-index:2;}}
/* Barre des logos — fond translucide doré/vert, sans blanc */
.logobar{{
    display:flex;justify-content:center;align-items:center;gap:36px;flex-wrap:wrap;
    background:linear-gradient(90deg, rgba(255,255,255,0.14), rgba(255,255,255,0.22), rgba(255,255,255,0.14));
    border:1px solid rgba(255,255,255,0.28);
    border-radius:20px;padding:18px 30px;margin-bottom:18px;
    box-shadow:0 4px 18px rgba(0,0,0,0.25);
}}
.logobar img{{height:92px;width:auto;
    background:#FFFFFF;
    border:3px solid rgba(255,255,255,0.9);
    border-radius:16px;padding:10px 16px;
    box-shadow:0 6px 20px rgba(0,0,0,0.35);}}
/* Barre du titre */
.titlebar{{
    background:linear-gradient(90deg, rgba(27,67,50,0.55), rgba(45,106,79,0.45), rgba(27,67,50,0.55));
    border:1px solid rgba(255,255,255,0.18);
    border-radius:16px;padding:12px 24px;margin-bottom:14px;
    box-shadow:0 4px 16px rgba(0,0,0,0.28);
}}
.hdr h1{{color:#FFFFFF;font-size:2.9rem;font-weight:900;
    margin:0 0 4px 0;letter-spacing:0.5px;
    text-shadow:0 2px 0 #1B4332,0 4px 14px rgba(0,0,0,0.5);}}
.hdr h1 em{{color:{MINT};font-style:normal;font-weight:900;
    text-shadow:0 2px 0 #1B4332,0 4px 14px rgba(0,0,0,0.5);}}
.hdr p{{color:#EAFBF1;font-size:1rem;margin:0;font-weight:600;
    text-shadow:0 1px 6px rgba(0,0,0,0.5);}}
.bdgwrap{{display:flex;justify-content:center;flex-wrap:wrap;gap:6px;}}
.bdg{{display:inline-block;background:rgba(116,198,157,0.30);
    border:1px solid rgba(116,198,157,0.55);color:#EAFBF1;
    padding:6px 15px;border-radius:20px;font-size:0.79rem;
    margin:3px;font-weight:800;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);}}
.bdg2{{display:inline-block;background:rgba(249,199,79,0.28);
    border:1px solid rgba(249,199,79,0.55);color:#FFE89C;
    padding:6px 15px;border-radius:20px;font-size:0.79rem;
    margin:3px;font-weight:800;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);}}

/* SECTION TITLE */
.st{{color:{MINT};font-size:0.97rem;font-weight:800;
    margin:0 0 14px 0;padding:11px 16px;
    background:{CARD};border-radius:12px;
    {'border-right:4px solid '+MINT+';' if rtl else 'border-left:4px solid '+MINT+';'}
    box-shadow:0 2px 8px {SHADOW};{ds}}}

/* CARTES */
.crd{{background:{CARD};border-radius:16px;
    padding:20px;margin-bottom:14px;
    border:1px solid {BORDER};
    box-shadow:0 4px 16px {SHADOW};
    color:{TXT};{ds}}}
.crd-t{{color:{MINT};font-size:0.9rem;font-weight:800;
    margin-bottom:12px;padding-bottom:8px;
    border-bottom:1px solid {BORDER};}}

/* RÉSULTAT */
.res{{background:linear-gradient(135deg,#081C15,{CARD},{ACCENT});
    border-radius:18px;padding:30px;text-align:center;
    margin-bottom:18px;border:1px solid {BORDER};
    box-shadow:0 8px 28px rgba(0,0,0,0.3);{ds}}}
.res .lbl{{color:{GREY};font-size:0.72rem;font-weight:800;
    letter-spacing:3px;text-transform:uppercase;}}
.res h2{{color:{WHITE};font-size:2.6rem;font-weight:900;margin:10px 0;}}
.res .sub{{color:rgba(240,255,244,0.6);font-size:0.82rem;}}

/* MÉTRIQUES */
.mrow{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0;
    {'flex-direction:row-reverse;' if rtl else ''}}}
.mit{{flex:1;background:{CARD2};border-radius:12px;
    padding:12px;text-align:center;
    border:1px solid {BORDER};min-width:70px;}}
.mit .v{{font-size:1.15rem;font-weight:900;color:{MINT};}}
.mit .l{{font-size:0.65rem;color:{GREY};margin-top:3px;
    font-weight:700;text-transform:uppercase;}}

/* TOP 5 */
.tc{{background:{CARD2};border-radius:12px;padding:12px;
    margin:6px 0;
    {'border-right:4px solid '+MINT+';' if rtl else 'border-left:4px solid '+MINT+';'}
    display:flex;justify-content:space-between;
    align-items:center;color:{TXT};
    box-shadow:0 2px 8px {SHADOW};
    {'flex-direction:row-reverse;' if rtl else ''}}}

/* HISTORIQUE */
.hi-btn{{background:{CARD};border-radius:10px;
    padding:11px 15px;margin:5px 0;
    color:{TXT};font-size:0.84rem;font-weight:600;
    border:1px solid {BORDER};cursor:pointer;
    box-shadow:0 2px 6px {SHADOW};
    display:flex;justify-content:space-between;
    align-items:center;{ds}}}
.hi-sel{{background:{ACCENT};border-radius:10px;
    padding:11px 15px;margin:5px 0;
    color:{WHITE};font-size:0.84rem;font-weight:700;
    border:1px solid {MINT};
    box-shadow:0 4px 12px rgba(0,0,0,0.25);{ds}}}
.det{{background:{CARD};border-radius:14px;padding:18px;
    border:1px solid {BORDER};
    box-shadow:0 4px 16px {SHADOW};margin:10px 0;{ds}}}
.det-row{{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px;
    {'flex-direction:row-reverse;' if rtl else ''}}}
.det-item{{background:{CARD2};border-radius:10px;
    padding:10px 12px;text-align:center;flex:1;min-width:58px;
    border:1px solid {BORDER};}}
.det-item .dv{{font-size:1.1rem;font-weight:800;color:{MINT};}}
.det-item .dl{{font-size:0.64rem;color:{GREY};font-weight:700;
    text-transform:uppercase;margin-top:2px;}}

/* SOL PANELS — REMPLACEMENT EXPANDER */
.sol-header{{background:{CARD};border-radius:12px;
    padding:13px 18px;margin-bottom:2px;
    border:1px solid {BORDER};
    color:{MINT};font-weight:800;font-size:0.9rem;
    cursor:pointer;
    box-shadow:0 2px 8px {SHADOW};
    display:flex;justify-content:space-between;
    align-items:center;{ds}}}
.sol-header:hover{{background:{ACCENT};color:{WHITE};}}
.sol-open{{background:{CARD2};border-radius:0 0 12px 12px;
    padding:16px;margin-bottom:10px;
    border:1px solid {BORDER};border-top:none;}}

/* SIDEBAR */
.sb{{background:{CARD};border-radius:12px;padding:13px;
    text-align:center;margin:6px 0;
    border:1px solid {BORDER};}}
.sb .n{{font-size:1.65rem;font-weight:900;color:{MINT};}}
.sb .d{{font-size:0.71rem;color:{GREY};margin-top:2px;
    font-weight:700;text-transform:uppercase;}}

/* ABOUT */
.ab{{background:{CARD};border-radius:14px;padding:22px;
    border:1px solid {BORDER};color:{TXT};
    margin-bottom:14px;box-shadow:0 4px 16px {SHADOW};{ds}}}
.ab h3{{color:{MINT}!important;font-weight:800!important;
    font-size:1rem;margin-bottom:10px;}}
.ab p{{color:{GREY};line-height:1.75;font-size:0.9rem;}}

/* COMP RESULT */
.cc{{background:{CARD2};border-radius:12px;
    padding:13px;margin-bottom:9px;
    {'border-right:4px solid '+MINT+';' if rtl else 'border-left:4px solid '+MINT+';'}
    color:{TXT};box-shadow:0 2px 8px {SHADOW};{ds}}}

/* BOUTON */
.stButton>button{{
    background:{ACCENT}!important;color:{WHITE}!important;
    border:1px solid {MINT}!important;
    border-radius:12px!important;
    padding:12px 28px!important;font-size:0.92rem!important;
    font-weight:800!important;width:100%!important;
    box-shadow:0 4px 16px rgba(0,0,0,0.25)!important;
    transition:all 0.2s!important;}}
.stButton>button:hover{{
    background:{MINT}!important;color:#081C15!important;
    transform:translateY(-2px)!important;
    box-shadow:0 8px 24px rgba(0,0,0,0.3)!important;}}

/* TABS */
.stTabs [data-baseweb="tab-list"]{{
    background:{CARD2}!important;border-radius:14px!important;
    padding:4px!important;border:1px solid {BORDER}!important;
    gap:3px!important;}}
.stTabs [data-baseweb="tab"]{{
    color:{GREY}!important;border-radius:10px!important;
    font-weight:700!important;font-size:0.84rem!important;
    padding:9px 14px!important;}}
.stTabs [aria-selected="true"]{{
    color:#081C15!important;background:{MINT}!important;
    font-weight:800!important;}}

/* SLIDERS */
.stSlider label{{color:{LIGHT}!important;font-weight:700!important;font-size:0.84rem!important;}}
/* SELECTBOX */
.stSelectbox label{{color:{LIGHT}!important;font-weight:700!important;}}
/* INPUT */
.stNumberInput label{{color:{LIGHT}!important;}}

/* LANGUE */
div[data-testid="column"] .stButton>button{{
    background:{CARD}!important;color:{MINT}!important;
    border:1px solid {BORDER}!important;
    border-radius:10px!important;
    padding:8px 10px!important;font-size:0.78rem!important;
    font-weight:800!important;}}
div[data-testid="column"] .stButton>button:hover{{
    background:{MINT}!important;color:#081C15!important;}}

/* TITRES */
h3{{color:{MINT}!important;font-weight:800!important;}}
h4{{color:{GREY}!important;font-weight:700!important;}}
hr{{border-color:{BORDER}!important;margin:16px 0!important;}}
p{{color:{TXT}!important;}}

/* FOOTER */
.ftr{{text-align:center;padding:18px;color:{GREY};
    font-size:0.73rem;font-weight:600;
    border-top:1px solid {BORDER};margin-top:20px;{ds}}}

/* EFFET VERRE — laisser transparaître le paysage */
.crd,.sb,.ab,.tc,.mit,.cc,.det,.det-item,.sol-header,.sol-open,.hi-btn,.st{{
    background-color:rgba(45,106,79,0.82)!important;
    backdrop-filter:blur(3px)!important;
    border:1px solid rgba(255,255,255,0.10)!important;
}}
[data-testid="stSidebar"] .stButton>button{{
    backdrop-filter:blur(2px);
}}
/* Onglets translucides */
.stTabs [data-baseweb="tab-list"]{{
    background-color:rgba(31,81,64,0.78)!important;
    backdrop-filter:blur(3px)!important;
    border-radius:14px!important;
}}
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================
_logo_data = base64.b64encode(open('logo_ensa.png','rb').read()).decode()
if lang=='fr': ttl='AgriTech&nbsp;<em>Maroc</em>'
elif lang=='en': ttl='AgriTech&nbsp;<em>Morocco</em>'
else: ttl=f'<em>{t["title"]}</em>'

# --- HEADER (un seul bloc structuré) ---
_header_html = (
    '<div class="hdr">'
      '<div class="hdr-content">'
        '<div class="logobar">'
          f'<img src="data:image/png;base64,{_logo_data}"/>'        '<div class="titlebar">'
          f'<h1>🌱 {ttl}</h1>'
          f'<p>{t["subtitle"]} — {t["powered"]}</p>'
        '</div>'
        '<div class="bdgwrap">'
          f'<span class="bdg">🎯 {t["acc"]} ~99%</span>'
          f'<span class="bdg">🌿 22 {t["crops"]}</span>'
          f'<span class="bdg2">📍 6 {t["regions"]}</span>'
          f'<span class="bdg">{t["var"]}</span>'
          f'<span class="bdg2">🤖 Random Forest</span>'
        '</div>'
      '</div>'
    '</div>'
)
st.markdown(_header_html, unsafe_allow_html=True)

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0;{ds}">
        <div style="font-size:2.8rem">🌿</div>
        <div style="color:{MINT};font-size:1.1rem;
        font-weight:900;margin:6px 0;">{t['title']}</div>
        <div style="color:{GREY};font-size:0.76rem;
        font-weight:600;">{t['subtitle']}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER}!important;margin:6px 0!important;'>",unsafe_allow_html=True)
    for num,desc in [('~99%',t['acc']),('22',t['crops']),
                     ('2200',t['obs']),('100',t['trees'])]:
        st.markdown(f"""<div class="sb">
            <div class="n">{num}</div>
            <div class="d">{desc}</div>
        </div>""",unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER}!important;margin:6px 0!important;'>",unsafe_allow_html=True)
    st.markdown(f"<div style='color:{MINT};font-weight:800;margin-bottom:8px;font-size:0.86rem;{ds}'>{t['reg_t']}</div>",unsafe_allow_html=True)
    for e,r in t['regions_list']:
        st.markdown(f"<div style='color:{LIGHT};padding:4px 0;font-size:0.8rem;font-weight:600;border-bottom:1px solid {BORDER};{ds}'>{e} {r}</div>",unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER}!important;margin:6px 0!important;'>",unsafe_allow_html=True)
    st.markdown(f"<div style='color:{GREY};font-size:0.69rem;text-align:center;font-weight:700;'>PFE 2025/2026</div>",unsafe_allow_html=True)

# =============================================
# ONGLETS
# =============================================
tab1,tab2,tab3,tab4=st.tabs([t['tab1'],t['tab2'],t['tab3'],t['tab4']])

# =============================================
# ONGLET 1
# =============================================
with tab1:
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f'<div class="crd"><div class="crd-t">🧪 {t["sol_comp"]}</div></div>',unsafe_allow_html=True)
        N =st.slider(t['N'],  0,  140, 50)
        P =st.slider(t['P'],  5,  145, 50)
        K =st.slider(t['K'],  5,  205, 50)
    with c2:
        st.markdown(f'<div class="crd"><div class="crd-t">🌦️ {t["clim"]}</div></div>',unsafe_allow_html=True)
        TM=st.slider(t['temp'],  8.0,44.0,25.0,0.1)
        HU=st.slider(t['hum'], 14.0,100.0,70.0,0.1)
        PH=st.slider(t['ph'],   3.5,  9.9, 6.5,0.1)
        RA=st.slider(t['rain'],20.0,300.0,100.0,0.1)

    st.markdown(f"""
    <div class="mrow">
        <div class="mit"><div class="v">{N}</div><div class="l">N</div></div>
        <div class="mit"><div class="v">{P}</div><div class="l">P</div></div>
        <div class="mit"><div class="v">{K}</div><div class="l">K</div></div>
        <div class="mit"><div class="v">{TM}°</div><div class="l">T°C</div></div>
        <div class="mit"><div class="v">{HU}%</div><div class="l">H%</div></div>
        <div class="mit"><div class="v">{PH}</div><div class="l">pH</div></div>
        <div class="mit"><div class="v">{RA}</div><div class="l">mm</div></div>
    </div>""",unsafe_allow_html=True)

    st.markdown("---")
    _,cb,_=st.columns([1,2,1])
    with cb:
        go_btn=st.button(t['btn'])

    if go_btn:
        feat  =np.array([[N,P,K,TM,HU,PH,RA]])
        feat_s=scaler.transform(feat)
        pred  =model.predict(feat_s)
        cen   =le.inverse_transform(pred)[0]
        prob  =model.predict_proba(feat_s)[0]
        conf  =round(max(prob)*100,1)
        nom   =gn(cen,lang)
        reg   =gr(cen,lang)
        lat   =C.get(cen,{}).get('lat',31.79)
        lonc  =C.get(cen,{}).get('lon',-7.09)

        st.markdown("---")
        st.markdown(f'<div class="st">✅ {t["result"]}</div>',unsafe_allow_html=True)
        st.markdown(f"""
        <div class="res">
            <p class="lbl">{t['recommended']}</p>
            <h2>{nom}</h2>
            <div style="margin:10px 0;">
                <span class="bdg">📍 {reg}</span>
                <span class="bdg">🎯 {t['conf']} : {conf}%</span>
            </div>
            <p class="sub">N={N} · P={P} · K={K} · T={TM}°C · H={HU}% · pH={PH} · {RA}mm</p>
        </div>""",unsafe_allow_html=True)

        r1,r2=st.columns(2)
        with r1:
            st.markdown(f'<div class="st">🕸️ {t["radar"]}</div>',unsafe_allow_html=True)
            vn=[N/140*100,P/145*100,K/205*100,
                (TM-8)/36*100,(HU-14)/86*100,
                (PH-3.5)/6.4*100,(RA-20)/280*100]
            cats=t['cats']
            fig_r=go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vn+[vn[0]],theta=cats+[cats[0]],
                fill='toself',
                fillcolor='rgba(116,198,157,0.15)',
                line=dict(color=MINT,width=2.5),
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True,range=[0,100],
                        gridcolor=BORDER,
                        tickfont=dict(color=GREY,size=8)),
                    angularaxis=dict(
                        tickfont=dict(color=LIGHT,size=9),
                        gridcolor=BORDER),
                    bgcolor=CARD2),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,height=340,
                margin=dict(t=25,b=25)
            )
            st.plotly_chart(fig_r,use_container_width=True)

        with r2:
            st.markdown(f'<div class="st">🏆 {t["top5"]}</div>',unsafe_allow_html=True)
            top5i=np.argsort(prob)[::-1][:5]
            top5c=le.inverse_transform(top5i)
            top5p=prob[top5i]*100
            meds=['🥇','🥈','🥉','4️⃣','5️⃣']
            bcols=[YELLOW,GREY,'#CD7F32',MINT,ACCENT]

            for i,(cu,pr) in enumerate(zip(top5c,top5p)):
                nd=gn(cu,lang); rd=gr(cu,lang)
                st.markdown(f"""
                <div class="tc" style="
                {'border-right' if rtl else 'border-left'}:4px solid {bcols[i]};">
                    <span style="font-size:1.25rem;min-width:30px;">{meds[i]}</span>
                    <span style="flex:1;padding:0 10px;{ds}">
                        <strong style="color:{LIGHT};font-size:0.88rem;">{nd}</strong><br>
                        <span style="font-size:0.68rem;color:{GREY};font-weight:600;">📍 {rd}</span>
                    </span>
                    <span style="font-weight:900;color:{MINT};font-size:0.95rem;">{pr:.1f}%</span>
                </div>""",unsafe_allow_html=True)

        st.session_state.hist.append({
            'id':len(st.session_state.hist),
            'culture':nom,'region':reg,'conf':conf,'cen':cen,
            'N':N,'P':P,'K':K,'T':TM,'H':HU,'PH':PH,'R':RA,
            'lat':lat,'lon':lonc
        })
        st.session_state.sel=len(st.session_state.hist)-1

    # HISTORIQUE
    if st.session_state.hist:
        st.markdown("---")
        st.markdown(f'<div class="st">📋 {t["hist"]}</div>',unsafe_allow_html=True)

        cd,_=st.columns([1,4])
        with cd:
            if st.button(t['del_all']):
                st.session_state.hist=[]
                st.session_state.sel=None
                st.rerun()

        for i,h in enumerate(reversed(st.session_state.hist)):
            ri=len(st.session_state.hist)-1-i
            is_sel=(st.session_state.sel==ri)
            ch,cd2=st.columns([7,1])
            with ch:
                lbl=f"{'⭐' if is_sel else '📊'} {t['sol']} {ri+1} — {h['culture']} | 📍 {h['region']} | 🎯 {h['conf']}%"
                if st.button(lbl,key=f"hb{ri}"):
                    st.session_state.sel=None if is_sel else ri
                    st.rerun()
            with cd2:
                if st.button(t['del_one'],key=f"hd{ri}"):
                    st.session_state.hist.pop(ri)
                    if st.session_state.sel==ri:
                        st.session_state.sel=None
                    st.rerun()

        if st.session_state.sel is not None and \
           st.session_state.sel<len(st.session_state.hist):
            h=st.session_state.hist[st.session_state.sel]
            st.markdown(f"""
            <div class="det">
                <div style="color:{MINT};font-weight:800;font-size:0.95rem;margin-bottom:4px;{ds}">
                🔍 {t['detail_title']} — {h['culture']}</div>
                <div style="color:{GREY};font-size:0.82rem;{ds}">
                📍 {h['region']} | 🎯 {h['conf']}%</div>
                <div class="det-row">
                    <div class="det-item"><div class="dv">{h['N']}</div><div class="dl">N</div></div>
                    <div class="det-item"><div class="dv">{h['P']}</div><div class="dl">P</div></div>
                    <div class="det-item"><div class="dv">{h['K']}</div><div class="dl">K</div></div>
                    <div class="det-item"><div class="dv">{h['T']}°</div><div class="dl">T°C</div></div>
                    <div class="det-item"><div class="dv">{h['H']}%</div><div class="dl">H%</div></div>
                    <div class="det-item"><div class="dv">{h['PH']}</div><div class="dl">pH</div></div>
                    <div class="det-item"><div class="dv">{h['R']}</div><div class="dl">mm</div></div>
                </div>
            </div>""",unsafe_allow_html=True)

        if len(st.session_state.hist)>=2:
            st.markdown(f'<div class="st">📈 {t["evol"]}</div>',unsafe_allow_html=True)
            dh=pd.DataFrame(st.session_state.hist)
            dh['i']=range(1,len(dh)+1)
            fig_e=go.Figure()
            fig_e.add_trace(go.Scatter(
                x=dh['i'],y=dh['conf'],
                mode='lines+markers+text',
                line=dict(color=MINT,width=2.5),
                marker=dict(size=11,color=ACCENT,
                    line=dict(color=MINT,width=2)),
                text=dh['culture'],
                textposition='top center',
                textfont=dict(color=LIGHT,size=8),
            ))
            fig_e.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=CARD2,font=dict(color=TXT),
                xaxis=dict(title='#',gridcolor=BORDER,tickfont=dict(color=GREY)),
                yaxis=dict(title='%',gridcolor=BORDER,tickfont=dict(color=GREY),range=[0,105]),
                height=260,margin=dict(t=15,b=15),showlegend=False
            )
            st.plotly_chart(fig_e,use_container_width=True)

# =============================================
# ONGLET 2 — CARTE & COMPARAISON
# =============================================
with tab2:
    m1,m2=st.columns(2)
    with m1:
        st.markdown(f'<div class="st">🗺️ {t["map_title"]}</div>',unsafe_allow_html=True)
        mp=folium.Map(location=[29.5,-8.0],zoom_start=5,
                      tiles='OpenStreetMap',width='100%')
        for cult,info in C.items():
            nc=gn(cult,lang); rc=gr(cult,lang)
            color=info.get('color',MINT)
            folium.CircleMarker(
                location=[info['lat'],info['lon']],
                radius=8,color='white',weight=2,
                fill=True,fill_color=color,fill_opacity=0.9,
                tooltip=f"{nc} — {rc}",
                popup=folium.Popup(f"<b>{nc}</b><br>📍 {rc}",max_width=200)
            ).add_to(mp)
            folium.Marker(
                location=[info['lat'],info['lon']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:700;color:#1B4332;white-space:nowrap;margin-top:-18px;margin-left:10px;">{nc}</div>',
                    icon_size=(120,20),icon_anchor=(0,10)
                )
            ).add_to(mp)
        if st.session_state.hist:
            dl=st.session_state.hist[-1]
            folium.Marker(
                location=[dl['lat'],dl['lon']],
                icon=folium.DivIcon(
                    html='<div style="font-size:24px;">⭐</div>',
                    icon_size=(30,30),icon_anchor=(15,15)
                ),
                tooltip=f"⭐ {dl['culture']}",
                popup=folium.Popup(f"<b>⭐ {dl['culture']}</b><br>📍 {dl['region']}<br>🎯 {dl['conf']}%",max_width=200)
            ).add_to(mp)
        st_folium(mp,width=None,height=450,returned_objects=[])

        if st.session_state.hist:
            dl=st.session_state.hist[-1]
            st.markdown(f"""
            <div class="crd" style="text-align:center;margin-top:10px;">
                <div class="crd-t" style="justify-content:center;">⭐ {t['last']}</div>
                <div style="font-size:1.5rem;color:{MINT};font-weight:900;">{dl['culture']}</div>
                <div style="color:{GREY};margin-top:5px;font-weight:600;">
                📍 {dl['region']} | <span style="color:{YELLOW};">{dl['conf']}%</span></div>
            </div>""",unsafe_allow_html=True)

    with m2:
        st.markdown(f'<div class="st">⚖️ {t["comp_title"]}</div>',unsafe_allow_html=True)
        st.markdown(f'<p style="color:{GREY};font-size:0.86rem;font-weight:600;margin-bottom:12px;{ds}">{t["comp_sub"]}</p>',unsafe_allow_html=True)

        # PANNEAUX SOLS SANS EXPANDER
        sol_names=[t['sol1'],t['sol2'],t['sol3']]
        sols_data=[]

        for i in range(3):
            # Bouton toggle
            col_hdr,_=st.columns([1,1])
            open_key=f"open_sol_{i}"
            if open_key not in st.session_state:
                st.session_state[open_key]=(i==0)

            arrow="▲" if st.session_state[open_key] else "▼"
            if st.button(f"🌱 {sol_names[i]}  {arrow}",key=f"sol_hdr_{i}"):
                st.session_state[open_key]=not st.session_state[open_key]
                st.rerun()

            if st.session_state[open_key]:
                st.markdown(f'<div class="sol-open">',unsafe_allow_html=True)
                cc1,cc2=st.columns(2)
                with cc1:
                    n_=st.slider("N (kg/ha)",0,140,[50,80,30][i],key=f"N{i}")
                    p_=st.slider("P (kg/ha)",5,145,[50,40,90][i],key=f"P{i}")
                    k_=st.slider("K (kg/ha)",5,205,[50,60,40][i],key=f"K{i}")
                with cc2:
                    t_=st.slider("T (°C)",8.0,44.0,[25.0,28.0,20.0][i],0.1,key=f"T{i}")
                    h_=st.slider("H (%)",14.0,100.0,[70.0,80.0,60.0][i],0.1,key=f"H{i}")
                    ph_=st.slider("pH",3.5,9.9,[6.5,7.0,6.0][i],0.1,key=f"PH{i}")
                    r_=st.slider("mm",20.0,300.0,[100.0,150.0,80.0][i],0.1,key=f"R{i}")
                st.markdown('</div>',unsafe_allow_html=True)
                sols_data.append([n_,p_,k_,t_,h_,ph_,r_])
            else:
                sols_data.append([50,50,50,25.0,70.0,6.5,100.0])

        _,cbb,_=st.columns([1,2,1])
        with cbb:
            cbtn=st.button(t['comp_btn'])

        if cbtn:
            st.markdown(f'<div class="st">📊 {t["comp_res"]}</div>',unsafe_allow_html=True)
            rc=[]
            for i,sol in enumerate(sols_data):
                fs=scaler.transform(np.array([sol]))
                cu=le.inverse_transform(model.predict(fs))[0]
                pr=model.predict_proba(fs)[0]
                cf=round(max(pr)*100,1)
                nm=gn(cu,lang); rg=gr(cu,lang)
                rc.append({'s':sol_names[i],'c':nm,'r':rg,'f':cf})
                st.markdown(f"""
                <div class="cc">
                    <strong style="color:{MINT};">{sol_names[i]}</strong> →
                    <strong style="color:{LIGHT};">{nm}</strong> |
                    📍 {rg} |
                    🎯 <strong style="color:{YELLOW};">{cf}%</strong>
                </div>""",unsafe_allow_html=True)

            dc=pd.DataFrame(rc)
            fig_c=go.Figure(go.Bar(
                x=dc['s'],y=dc['f'],
                text=dc['c']+'<br>'+dc['f'].astype(str)+'%',
                textposition='outside',
                textfont=dict(color=LIGHT,size=10),
                marker=dict(color=[MINT,YELLOW,ORANGE],line=dict(width=0))
            ))
            fig_c.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor=CARD2,
                font=dict(color=TXT),
                xaxis=dict(tickfont=dict(color=GREY)),
                yaxis=dict(title='%',gridcolor=BORDER,tickfont=dict(color=GREY),range=[0,110]),
                height=260,margin=dict(t=15,b=5),showlegend=False
            )
            st.plotly_chart(fig_c,use_container_width=True)

# =============================================
# ONGLET 3 — EDA
# =============================================
with tab3:
    st.markdown(f'<div class="st">📊 {t["eda"]}</div>',unsafe_allow_html=True)

    @st.cache_data
    def load_csv():
        try:    return pd.read_csv('Crop_recommendation.csv')
        except:
            try: return pd.read_excel('Crop_recommendation_clean.xlsx')
            except: return None

    df=load_csv()
    cn=['N','P','K','temperature','humidity','ph','rainfall']

    if df is not None:
        e1,e2=st.columns(2)
        with e1:
            st.markdown(f'<div class="st">🌾 {t["dist"]}</div>',unsafe_allow_html=True)
            cnt=df['label'].value_counts()
            fig_d=go.Figure(go.Bar(
                x=cnt.index,y=cnt.values,
                marker=dict(color=cnt.values,
                    colorscale=[[0,CARD2],[0.5,ACCENT],[1,MINT]],
                    line=dict(width=0)),
                text=cnt.values,textposition='outside',
                textfont=dict(color=LIGHT,size=9)
            ))
            fig_d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor=CARD2,
                font=dict(color=TXT),
                xaxis=dict(tickangle=45,tickfont=dict(color=GREY,size=8)),
                yaxis=dict(gridcolor=BORDER,tickfont=dict(color=GREY)),
                height=360,margin=dict(t=10,b=90),showlegend=False
            )
            st.plotly_chart(fig_d,use_container_width=True)

        with e2:
            st.markdown(f'<div class="st">🔥 {t["corr"]}</div>',unsafe_allow_html=True)
            cr=df[cn].corr().round(2)
            fig_h=go.Figure(go.Heatmap(
                z=cr.values,x=cn,y=cn,
                colorscale=[[0,ORANGE],[0.5,CARD],[1,MINT]],
                zmid=0,text=cr.values,
                texttemplate='%{text}',
                textfont=dict(size=10,color=WHITE),
            ))
            fig_h.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor=CARD2,
                font=dict(color=TXT),
                xaxis=dict(tickfont=dict(color=GREY)),
                yaxis=dict(tickfont=dict(color=GREY)),
                height=360,margin=dict(t=10,b=15)
            )
            st.plotly_chart(fig_h,use_container_width=True)

        e3,e4=st.columns(2)
        with e3:
            st.markdown(f'<div class="st">📦 {t["box"]}</div>',unsafe_allow_html=True)
            vc=st.selectbox(t['box_sel'],cn)
            fig_bx=go.Figure()
            for cu in df['label'].unique():
                fig_bx.add_trace(go.Box(
                    y=df[df['label']==cu][vc],name=cu,
                    marker_color=YELLOW,line_color=MINT,
                    fillcolor='rgba(116,198,157,0.15)'
                ))
            fig_bx.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor=CARD2,
                font=dict(color=TXT),
                xaxis=dict(tickangle=45,tickfont=dict(color=GREY,size=7)),
                yaxis=dict(gridcolor=BORDER,tickfont=dict(color=GREY)),
                height=380,margin=dict(t=10,b=90),showlegend=False
            )
            st.plotly_chart(fig_bx,use_container_width=True)

        with e4:
            st.markdown(f'<div class="st">📈 {t["stats"]}</div>',unsafe_allow_html=True)
            st_df=df[cn].describe().round(2)
            fig_st=go.Figure(data=[go.Table(
                header=dict(
                    values=['Stat']+cn,fill_color=ACCENT,
                    align='center',font=dict(color=WHITE,size=10),
                    line_color=BORDER,
                ),
                cells=dict(
                    values=[st_df.index]+[st_df[c] for c in cn],
                    fill_color=[[CARD,CARD2]*5],
                    align='center',font=dict(color=LIGHT,size=9.5),
                    height=28,line_color=BORDER,
                )
            )])
            fig_st.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=380,margin=dict(t=5,b=5)
            )
            st.plotly_chart(fig_st,use_container_width=True)
    else:
        st.warning(t['warn'])

# =============================================
# ONGLET 4 — À PROPOS
# =============================================
with tab4:
    a1,a2=st.columns(2)
    with a1:
        st.markdown(f"""
        <div class="ab">
            <h3>🎯 {t['obj_t']}</h3>
            <p>{t['obj_txt']}</p>
        </div>
        <div class="ab">
            <h3>🤖 {t['ml_t']}</h3>
            <p>{t['ml_txt']}</p>
        </div>""",unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="ab">
            <h3>🛠️ {t['tech_t']}</h3>
            <p>
            • <strong style="color:{MINT};">Python</strong> — Langage principal<br>
            • <strong style="color:{MINT};">Scikit-Learn</strong> — Machine Learning<br>
            • <strong style="color:{MINT};">Streamlit</strong> — Application Web<br>
            • <strong style="color:{MINT};">Plotly</strong> — Visualisations interactives<br>
            • <strong style="color:{MINT};">Folium</strong> — Carte interactive<br>
            • <strong style="color:{MINT};">Pandas / NumPy</strong> — Traitement<br>
            • <strong style="color:{MINT};">Power BI</strong> — Tableau de bord<br>
            • <strong style="color:{MINT};">Jupyter Notebook</strong> — Analyse
            </p>
        </div>
        <div class="ab">
            <h3>📊 {t['data_t']}</h3>
            <p>
            <strong style="color:{MINT};">Crop Recommendation Dataset</strong><br>
            • 2 200 observations — 22 cultures<br>
            • 7 variables : N, P, K, T°, H%, pH, mm<br>
            • 100 observations par culture
            </p>
        </div>""",unsafe_allow_html=True)

    st.markdown(f'<div class="st">📈 {t["perf"]}</div>',unsafe_allow_html=True)
    p1,p2,p3,p4=st.columns(4)
    for col,num,desc in zip([p1,p2,p3,p4],
        ['~99%','22','2200','100'],
        [t['acc'],t['crops'],t['obs'],t['trees']]):
        with col:
            st.markdown(f"""
            <div class="sb" style="padding:20px;">
                <div class="n">{num}</div>
                <div class="d">{desc}</div>
            </div>""",unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown(f"""
<div class="ftr">
    🌱 {t['title']} — {t['subtitle']}<br>
    PFE 2025/2026 | Random Forest ~99% | Stratégie Génération Green 2020-2030<br>
    <span style="color:{MINT};font-weight:800;font-size:0.82rem;">✨ Développée par Yassine MALIKI ✨</span>
</div>""",unsafe_allow_html=True)
