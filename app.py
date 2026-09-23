import streamlit as st
import pickle
import numpy as np
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
_LOGO_FEG = "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"
_LOGO_UNIV = "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mWVZ1dXVhYWF/fv3ZxEopayoqKioqKivr2cU9t6L/FBPPPHEpk2bHnrooX79+v3qV79avnw560T2Ke69997bbrutrKzsxz/+8bBhw2prazdu3PjnP/9506ZNH//4x4uLi1m9PvLII9dccw0A/Pa3vz355JOLi4s5uMfr+thjjz3++OPR7z7//PO33377gAEDJk+ePGPGjIqKiurq6tLS0tLSUtd1o1zSe2WoqR07dqxbt47jjLxmHGwBgGQyqbUeNmzYypUro/Tudddd97vf/a68vPxnP/vZueee67rulVdeWVNTc+WVV37ta18TQnR3d/NGZy+OzczIlWWRwFKdowIAsGDBgj/+8Y+7du36wx/+UF1dHUkOx3E45sGmIjs1vu93dHQUFxcXFRWxxs3n867rsk/oum4ymbzwwgtnz55dUFAwdOjQ+vr6u+++e+HChY2Nja+99hoAlJaWKqUuvvjiH//4x2VlZWVlZZG4jsfjffr02bt3r+d58Xj8oMlqa2ubP3/+NddcU1BQ8MMf/vDPf/5znz59Vq1aNXfu3Gw2y7v/3HPPbWpqev3110eOHHnaaaft37+fLRXbtrPZbCwWe/rppx977LHvf//7d91112233XbccccNGjSooKCA5y2bzR533HHs0o8ZM0YI8cgjjzz66KOrV69evXr1HXfcEY/Htdann3763XffzRKFKQfv7Nz+G9dtt912+umnc9w0m82yPE+n052dnel0moh+9atfzZkzp6WlhYhWrFhx2mmn3X///ccee2xZWdny5ct5B9fV1c2ZM+foo4+eMGHCSy+9xBHiurq6qVOnXnbZZVFqgMN5jOhcLsfpMSJat27dnDlzTjnllPr6erauPc9bv379Y489xsqFHX0iam5u/ulPf3r++eezpfLUU09dffXVRLRgwYKvfvWrbw39dnZ2Pvzww+PHj2fdd9RRR51wwgklJSUDBgyYP38+f+acc84577zzopTel7/85WnTpmUymUg9RTquqalJKTV+/PiSkpJRo0ZdccUVffv2veSSS3qHro0xd999d1lZGfMHiouLb7nllra2Nn7wjo6OGTNmDB8+fOfOndddd93w4cPr6uqIaPTo0eeee2544OIfbW1tffHFF4855hjHcSoqKkaMGDFx4sSTTjppxowZ1157LYvP6KeDIHhPsiFqzZo1Y8aMsW3b8zxW2LxRMpmM67q5XG7ZsmXV1dUlJSUAsGHDBsbE9OnTly5dumjRogkTJhhj+vTp881vfvNb3/pWSUkJa3e2IUpKSjKZTBTbiKIuzEvwfZ/XftSoUc8+++zWrVv79OnDD2aMWbx48U9+8hPHccaNG8d7a8WKFY899tiDDz44derU7u7ul19++Ze//OXGjRvPOeec22+//fHHH581a9bMmTMBIJ1Or1+/fs2aNU8++eT27dunTJlyww03zJ49u6KiorS09Otf//qrr77qum5NTc3GjRt37Nhx9tlnR0GhkSNHLl26NPJOe+uL4uLi//7v/168ePHxxx9/ySWXfOxjH9u1a9djjz02c+bMU045pba2dvPmzQsXLnz11Ve7u7v79+/f3t6eSCQAYOfOnWVlZb7v//KXv+zo6JBSjhkzRil19NFHFxUVAcAZZ5zxu9/97qWXXurXr5/v+y0tLYsWLfrb3/62bNmySZMmXXHFFXPnzh09enRRURG7JLFYjI2tKPjxH1rlb15z5sx54YUXwjBk8ZBOp9kJ5tc/+MEPxo0bd8899/Buq6mpmTZtWvTdSy65JAzDTCZjjGlpaTnxxBMHDhy4bdu2KLD/2c9+9vzzz2cJwcvP2y6bzT766KM1NTXs2XN4oLfk0Fr/7//+LyKOHj16zpw5Y8eOHTlyJAD07dv3vPPOGzp06LRp04YNG6aUqqio6N+/v23b1dXViURiypQp48aNGzRoEMeRLrjggmeeeaa+vj7KXoZh+Nprr7FtNHXq1FgsNmTIkE2bNrFWIqL77rtv5MiRDQ0NvSVE9N0gCBoaGtra2nhCnnvuuVQq5TjO5MmTR4wYwabMqFGjbr311sWLF//oRz8qLCwEgOrq6unTpw8fPty27d/97nd79+798Y9//LGPfeyhhx7iydmyZUt5eXllZeWECROGDh3Kwv/EE0/81a9+tXfv3u7u7t5J82iK+HUkJ94TCQFjx4596aWXfN/PZrP5fJ5hQUS7d+++8sorE4nEN7/5TQ6i8Zw2NDQsXLhwyZIlX/jCFz73uc91dXVxmG/v3r0TJkw47bTTWPDy5H7729/+6le/ynPteR6POAiCpqamT3ziE+vXr/c8jx8pl8vxF/lpiWjPnj3z5s1LJBL9+/cvKys75ZRTfv7zn2/YsGHfvn3HH388ABx77LHr1q277LLLRowYcd99923YsOH444+vqqqaMGHCWWed9Zvf/GbHjh2REuEYNj+F1vr555+fO3dunz59Pv3pT7/22ms8m/zsL7744sknn9zZ2Rlpit75Sb5PLpfLZDK5XM7zvMcff3z27NkDBgwYNGjQJZdccvfdd+/evZv3Q3d391NPPfW1r31t2rRp1dXVw4YN++lPf5rP53kLMZ0gm81y3Pq+++476qij+vfvf/LJJ994442LFi1iHPBsdHR0NDY28j+ZpxNl2yMixXtCM4DRo0dff/31/A/+7a6urscff3zSpEkAcPnll69fv54HwasVffPss8/+/e9/n06nfd/fvXv3ySefXFBQsHDhQr4Pz8hTTz1177338mZiPceoeu21184999z9+/ez5OAHZmXR3d2dz+fb2tqMMddff/3HP/7xzs7O7u7uaIt7ntfY2FhfX89rxvPLf8rn811dXblcrvdeeaty5YnjD/OfPM+LlPfmzZtvu+22jo6OtwLioChkbz5RJpPhZT6IU8JfyeVyvPz8sDyfvVkRfGWz2Uwm09sS4vXO5/MPPvjghAkT5s+fz0G2Q0eegEsvvTQej//+979fuXLlq6+++pe//OUTn/gEMyK//vWvr1u3jre753nMjCKi2traz3zmM6WlpStXriSibdu2nX/++cXFxVdcccVB3KH6+vo9e/bwo0YEEyJ64IEHLr/8cpZJPDvZbJYnjsUyr/EFF1xw5pln9p4aDkv3luEH8U0O2tD/bNP0fj8acLStOSL0f95w0Rf/GaOitxH6DmyJ6ANtbW2/+MUvOGG0fv16nq5DBYh169aNHz+e9RxHGgBg6tSpv/nNbxjyLANyuRwP4pVXXjnhhBMA4Ne//jVTKSdPnlxYWPjb3/6Wt+xB3kTEFeMn5C3yk5/85Jvf/KYxpru7O6LbsLRgT1JrvWDBgsGDB990003Rluptgfee/d6T+2+t4kEfZixyqPvf3YLvZoEj0fLPiFX/7Cae5+3bt++b3/zm//t//6+rq+uQktCAiNavX//973//rLPOOuOMM772ta898MAD9fX1LAm7urrS6TRriu7u7kcffbSyshIAfvSjHxHRq6++OnPmzM997nNLly7lBeO8Pn+YN3Rvsl20p3/4wx+eddZZLHXYhmDU83p0dnbeddddffr0OeGEE1jAHERGMv94vXuS0j9by97YZTvjbSl07yGN798aM09aR0cHm7q9Nc57D4jOzk5++NbWVtYIfLG1mE6nWXrX1NRcddVVsVjMdd2bb76ZTaqtW7e+/PLL/AHO37e3t/NY+QXbjL0nnZ/tjjvuGD58+GuvvXaQtM9kMs8999yVV15ZUFAwYMCA5557ju24SGz0ZsG8V5PCxk10z4OCAe/59X8AMZsRjINDTbODIAgi5yIy8iNeCbMmH3744eHDhwNAnz597rrrLraN+YuRpGXFz2me3kZD77mO7Lu1a9cWFxefd955b7zxRmdnZ319/auvvnrPPfecc845ZWVliUTiE5/4xPPPP8+zFhF3D90isYJj7/cgm/SQwuJdfpiHxJwJHuehgwXwLDMOIiYBv5nNZu+8887x48fbtj1w4MCLL7549erVkQ/Z0dHBcj6SAdGjctyTbxKpZMYNT0Q2m/3qV7/K5JqysrI+ffqw7ZJKpcaOHfvDH/6Qv5tOpyOLL7rDQUZ49H5v6dpbubyD9RAZv/l8nrdg5FhG0uggsRSFUt56/7cuUu/wZW8JFL3PU/3PwNH759jSYm/oIGv6vb0w8hI5ucUjs217+fLlDz/88F/+8pdZs2Ydd9xxJ554IgsJXmleyyjTGBXHRXwkzplxgorrwHgX8oc9z+vu7n7ggQdee+217du3JxKJWCw2derUSZMmTZkyZciQIZxljjiGUZUYpy04/pPL5SJGp+d5HMwPgoDfyeVynAazbZsTMb2pHjzUXC7HhXhMD7Msi1O+zGCTUjIjN0rO8fwwY4VLtfiLnAnjr/CT8hj4k1GSjBcyyi3zBEZZnojLE72OMMc1cJyX5qR59JjvPc+Icee6bjqdVkrxBCHiwoULW1tbZ8+e3bdvX04v5XI5njV+ZsY1M0H4UflFRHDl6eMbdnd3J5NJTlfy+9Gi1tXVFRQUcOQ/kUjwQkaTCwD5fJ4jtb7vJxIJ3/d5dVkHMbxisZjv+0zMCYKA8z1RTpyxzgDlkTPLy3Ec27ZzuRyDhieaw+qImE6nuchOStnR0ZFIJCzL4j3KtBf+JOMpwpxt2/zTB2Ug2SyLx+MMF8ZcNLGcJYjH45y95NnjNG9UOMncVV6CQ0e0xEgi2bbN8+J5HrODelPH+E+RcHMch6nrvD+i0t6IYsn/jdaV7YBYLMZZCU51WpbVm4fIY2AQ8M7jaY3mNwgCriPlwTBJqTcHSQiRzWbj8Tgjj2eNZ5Mz5r1ZVVFNc5RNjcp2Gei8EiyceQ14tbjKOWJLsGEUcbF42SJaCU8vw5FHy7/I+f1/VlXMH2NZy1KKS4H5w6ybDqGEaG9vv+WWW4477jgOMLD4Ze/L9/1f/OIX7e3tl1566eTJk9PpdDKZXLRo0RNPPPG9732voqIiYvz5vv/QQw8lEolzzz23oaHhrrvuOv7446dOnSqlTKfTDz74YDqdvvzyy7dt2zZ//vzPfvazAwcO5E2WTqefeOKJwYMHz507lzOut9566zHHHDN27Ng777xz1qxZ8+bNW7t27dNPP33RRReVlpay/IwgeM011+zZs+fCCy88/fTTeZa5enPlypVr1qz5zGc+k0gkojVmFsVLL70UpToty5o3b14mk7ngggt++ctfnnLKKXV1de3t7S0tLWedddbIkSM5fXXuuecOHDiQ7yOE2L9//29+85t9+/Z997vfnTBhQnd39913393d3X3VVVdFtaORjmBGuO/7Dz744COPPHLyySdfcskllmXdcMMN69atY9JNv379PvWpTz355JMc0k2n0/Pmzbvqqqs4bNPZ2XnjjTc2NzezJXHsscdedNFFBQUFh6rEpa2tbeTIkbfddlvkCLC/kM1mTznllD59+gwaNGj48OGrV69mt+KPf/wjAPz973+PcgQs9y655JIrrriCbzhkyJDHH3+8ubn56aef/uQnPwkAX//6140xa9asKSoqYoczyor913/918033xxZ+yNGjLjrrrvq6upmz57929/+loj+9Kc/jRkzhnNUnIfj3Mell15aXFw8cODAsrKyxx9/PDJ4iejxxx//1Kc+xQEVztREYewf/vCHAFBWVlZcXDxp0qQf/OAHc+bMWbx48ezZs5977rnTTjvtW9/61ujRox9++GGuZ5w2bdqSJUvYyk6n0+l0+swzzywqKurTp8+UKVN27dq1cOHCadOm9e/f/4knnuDwfGSQRrp1/vz5ADB+/HhE/OEPf6i1PuWUU2zbHjVqVGFh4fjx42+99daysjLmKrMLFiUTVq1aZVnWoEGDKisrCwsLL7root7prvf2UlETCM7LcSU/22L333//hg0bnn322erq6jPPPPPhhx/+0Y9+pLXmyoWdO3cec8wxEe2FNQ5LXaVUPB7P5XK33377tdde269fv+uvv/7ss89mcVJdXc2KlkVoUVERJwzZ5GZTrqysbPfu3Vu3bi0vL2fWE5OF2CgTQjiOs3Dhwvnz599zzz0nnHDCl7/85T/96U8f+9jHWAzYts0V/pFAjhpa5HI5ADjnnHOuvPJK1j6JRGLRokVXXXVVa2vrH//4x7Vr137lK19ZvHhxRCRj64FvEovFbrvttjVr1ixYsKC8vPyMM8647777KisrL7vsstbW1o0bN37yk5+MUuds4nBC+IYbbrj88stvueWW//mf/3n00UdPO+20ioqKL3zhCxdccAH/yubNmydMmPC1r30tFoulUqnCwsJI+iqlBgwYcOONNxYVFVmWVVxcHGmo955kyyjm+jXu5RAFbhctWvTlL395ypQpffv2/f73v7927dq2trbm5uba2tohQ4asXbs2qj5g/RrV5HCY0hgzevRo13VPOumkq6++etSoUUqpjo4OREwkEvx55p5HdCxWVWwQzJ8/v7W1lVX16tWrJ0yYEGVcWbn+/e9/P/HEE0844YR4PP6jH/2IKY3s73Bcjw31yLxl+Sel9DyPE+hHH330jBkzxowZc9ZZZ73++utbt26dP3/+CSecMG/ePN7okWkSsQ2CIHjjjTc+//nPT58+ffDgwZdffvnf/va3/v3733PPPddee+3UqVN7V/MxCoUQGzdu7Orq+v73vx+G4VVXXVVeXv7aa695npdMJlesWJFMJmfNmsXG1s6dOzdt2jR9+vSjjjqKnTXWUwUFBfX19atXr544ceL48eMPXZMF9eCDD77xxhtNTU0PPPBATU3Nd77zndLSUp6ClpaWY445ZunSpTU1NZMnT2apW1tba1nWRRdd9Oqrr3JRRmSjZbNZlhBFRUVMb/zUpz519dVX/+pXvxowYMAll1zypz/9adOmTbW1td/61reOPvroL33pS9ygIyKkc/sRAHj66aeXL19eXV0dhmFLS8uePXs8z2P7ICKaNjU1DRs2rK2t7Z577rngggssy9qwYcPUqVMjDn4+n2cJwYwSdkYiLZnJZNjlsSzri1/8YlFR0Xe+851LLrnkm9/8Jm8JttoiR5opfZyxHDZsGIN4xIgRlZWV48aNO+uss+bNmzd79mw2JCMvg23JPXv2DB061Lbthx9++Nhjjx02bFhraysiPvjgg+Xl5fPnz3/ggQfKyspqamruv//+tra28vLyc845hyeTDfz9+/ffcccdbW1tuVzuO9/5DvflORS+hli3bt2f/vSn1tbW1atX33777Q0NDXfcccd3v/vdbdu2IeKAAQMWLVr00EMPlZWVMYdq5cqVe/fu/cxnPlNbW7tjxw4AuP3227/xjW+0tbUVFxdzfiudTmezWV7jM844Qwjx17/+taOj469//evjjz/e3t6+aNGixYsXs5/S1dUVhiHXO0T+6p133jlmzJiJEyeGYbhnz57m5uZXX321ra2NC8y/9KUvbd26tbS0lOnLXMDPJTRr1qz54he/uGbNmrKyMjbvGUOe5/3P//zPpZdeWl9fz75MxMAOgqCgoODkk09OJpPz5s0rLy/vHVRFxEwmw0O66KKL9u3bx0wwXvLS0lKOlV188cXf+973lFJtbW3sInLUhN0TIiopKens7HzxxRdbWloGDRrEHmxdXd22bdtefvnl9vb2Pn36tLa2rl+/fu/evRs3bnQcZ+/evV/84hd//etfp1KpMAy3bNmyb9++5cuXRyU9h0RCfPe73z3ttNOuvPLKY4899vLLL9+zZ8+tt97K4ystLV23bt1XvvKVL37xi3V1dclkMgiClpaWoqKiYcOGxePxPXv2dHd333vvvbW1tSNHjgzDkE1fy7KYhOd53pVXXomIf/rTnwYNGnTPPfesWrXq2muvveGGG8aNG1dRUcH9qTg8wD4hhx1LSkquuOKK2267raura8uWLY7jMKOfiG699db9+/d3dXVNmzZt27Ztc+bMueeee1pbW5nVfsMNNyxatCiXy33+85+Px+NRtGD+/PmPP/54S0vLCSecwCKEXRUe7ZIlSy6++OI9e/acddZZt95660UXXcQVCazOiouLV65c+cADD+zZs2fs2LHGmB07dnz84x8Pw3DDhg1MdTnttNN27ty5Zs2aoUOHZjKZVCoFAIlEggNZ8Xh806ZN/fr1u/rqq4cPH3711VdPnDixtLT0S1/6UiKROOaYY4466qj777+fuV6FhYXf+MY3giC48847X3jhhVdeeaW8vHzy5MkDBw7M5XLXXnttKpXqXZrwHkuI8vLyOXPmGGMmTJgwaNCgVatWnXTSSY8//rht2yUlJS+++KLjOOXl5XfeeWdVVVUul2ttbT322GMBYOLEiVu3bn399dfPPffc+++/v7W1tbm5mQ1SZgMnEomf/OQnjz/++Ny5cysqKowxo0aNmjhxomVZo0ePHjFiBEvdVColhOjs7OSVY2LERRddNGrUqPb2dqXU5s2bv/a1r40dO/ahhx7as2fP8ccf/9RTTyUSiXg8vmvXrr179w4bNmzhwoVhGMbj8QEDBjzxxBOVlZWNjY3xeJyrCgBg06ZNZ5xxxqOPPrp06dJsNsvsbVbSvu//5Cc/qampGTBgQCKRuOGGG7Zv3661ZoktpYzH488999xll132q1/9av369YWFhU899RTrx5deesl13ZdeeunCCy/83Oc+t3DhQu4WtXnz5j/84Q87d+7kaMScOXPS6fTixYuHDx++a9eu9vb2cePGGWPGjh176qmnTpo0iaOQlmWdddZZxxxzTJ8+fdrb22tra1944YVPfvKT27dvl1J+7GMfO/fcc4cMGfLPWmC9NxJCCMFVIvwblmWtXr26qampqqrqS1/60vTp0y+55JIRI0b8+te/vuOOO9hjvOKKKwBg3LhxS5cunTx58qJFi15++eWJEycmEon58+d3dHSk0+nm5uYwDKdMmTJy5MgHHnhg69at3//+9z/1qU9xdptb0nBcj7lSPPu+7xcUFLBu5sdubGx86qmnbrnlltra2gcffPD8889/7bXX1q9fn8vlrr/++kcfffTCCy8844wz7rjjDi7j5zKy3bt3T5o0ia0Tfs6ioqIXXnhh9erVM2fObG1tXbBgAbNHBw0aNHDgwMbGxh/96EfPP//8cccdd/fdd//973+vrKz8+c9//vLLL69Zs4ZbIfz1r3/t7OwcOnTohRde+Nhjj1166aVDhgx54IEH7rvvvtbW1v3797e3t3d3d7MBu2bNmj//+c/l5eUDBw5kdu706dMvvfTSK6+8kjP7p5566sMPP7xixYqTTz75jjvuGDduHNsQDz74oOM4jz322KmnnppIJH7+85+vWLGCpdfjjz/uuu599913+umn/9d//Rd7hYckDtHR0XHMMcdwh7l9+/adcsop/fr1W7BgARHdfvvtXM3y7W9/O5vNLl++fNasWZs3byaiN954Y86cOcuWLTvzzDPLy8uXL1/Opn5JSUlRUVH//v2ZTrdt27af/vSnQ4cO5dYRS5YsOeGEE9atWxcl8To7Oz//+c//8Y9/ZBZFEATHH3/8Qw89lM/nL7/88ksuuWTu3Lk7d+7cvn37Jz7xiZdffvnqq68uLi6+8cYbiejRRx8dO3bsgAEDzjnn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p9P33XffH//4x9mzZxcWFkbdMjjVvH379h//+MfPPvvs2WefrZR68cUXf/3rX48dO7aiouKnP/3pvn37Vq9ePWXKlEOU31JTp07ltiZVVVX81siRI8vKyoYOHcrzePfdd6fTaW6MwiGByspKDgxUV1cnk8kwDCdOnFhUVPTcc8/de++9LBU5BCmlvPjii+PxOBFt3br1iiuuOPPMM1muzp07VwhRXV196623/vCHP7z55pvnzZt3zDHHPPXUU88///yzzz5bUlKyYsWKTCZTXV09cODAtra2KVOmCCG++93vcn+Fe++9d8KECT/+8Y+51JOzAFu3bv3BD37Q3d29e/fug+Qtvxg5cmRjY+Ps2bMHDhzI/YcGDBjQ2trar18/lmHFxcXbt29/4YUX4vH4SSedZIxxXXfdunXPPPPMU0891a9fv2XLlrFk+u1vf/uVr3zlwgsv5CpyZppls9mnn346k8nU1NRws07uYfLtb387Ho9/73vfa21tLSsrKy8v3759++DBgw9q99HU1PT888/fe++93NeAvRiuleLOblwCH7Gu33NAiMgoi2ggTH6MItmDBw9etGjRN77xjcbGRh4ii2X+JDdt4djOJz/5ySAI5s+fX1BQ0N7e/tJLL5155pnxeLy1tTWVSrEIHTRoEIellVIcjeGWBJx7jKTlww8/nM/neRN7nscZSESMx+Njx46Nx+OsMkaPHm2MKSkpYfZANpsdPny4MaaiomLmzJmRRdz7gROJhOM4TOfk1BqHp1hhM5t38eLFxcXFr7/+OqdzmVZYX19/77335nK5SZMmVVVVMZnj/vvv37Jly7Rp08aNG8e0IO4GVFtbu2HDhsiK9H1/6tSpN954449+9KPS0lIiSiaTgwcPZt+YiRcMiFQq9dnPfvbss89mb5ap4YWFhYMGDXJdN5PJrF27lunBhygUIfL5PLOb0ul0RC6K/tvW1nbaaaf95je/eeSRR6699lrOcbPXwB9mFc6bfsaMGdOnT3/uuee6u7vb29vXr1//qU99ihNd8Xi8qanplltuufrqq1esWBGZb62trVprrvyfPn16GIannXbaxIkTr7jiiltvvZVVO69cMpns6uriZoC8OTgJwjuMdbDrups3b77rrruuvfbaRYsWRW3nel+cy/7b3/7285///Be/+MVPfvKT3//+95ws5ZYS27Zt279//5///GdjzCOPPMLvT5ky5WMf+9jVV19966238lwppa655poFCxYwUZTpnOzxXnfdddOnT7/jjju6u7t5j6XT6csuu+yCCy647rrrbrrpJvZmOaHVO2JGRBw5rauri5ajtLQ0FovxWR4lJSVXXXVVVVXV/fff/1bH8L1RGewsWJaVSqUirz3yILjV13nnndfV1XXllVd2d3f//Oc/79OnD3Oy2Xhmy8uyLNu2P/3pT19yySVbt25ds2YNqz2ugWltba2srDzxxBNffPHFvXv3Tpkyhbf7+vXrv/GNb7z88ssPPfQQsxYKCgpuvvnmgQMHXnvttVrra665JpFItLW1cd6LhRaHs5hdzRY7I1IIUVxcPHny5HXr1kVsq7c6I7FYbODAgZzjRUTOe0XLs3DhQuaBGmP+/ve/f/Ob32Teww9+8IN4PP7973+/tbX1xhtvFEKcddZZd91115e//OUvfelLt99++5gxYxobG5csWTJnzpzq6uo777xz//79lZWVtm03NTVNmTLlt7/9red5V1111YgRIwYPHsx+I7tC0dFUbEwwYpjaM3bs2O9973u2bffp0+d73/teOp2eMmVKR0fHIQpPqai2JCo4zOfzLEv5E+3t7blc7stf/nI2m73lllu+9KUvDRo0iPP93HyIO+AzveDMM8/84x//eP311+dyueuuu4756b7vFxUVtbS0jB8/nol0EcvthBNOuPTSSxcuXPjzn//89ttvtyyru7t7+PDhN910UyaTue+++84555xx48bx2vOOZyciErPsGzOImWs0ZsyY2bNnH9QT6M0HViqbzY4bN+7444+Pym9Y5XH/smeeeWbo0KFLliwZMWLEggULuIFQJpMZPnz4zTffzCbkxRdfPGzYsHw+f9FFF0kpL7300meeeWbkyJEvvfRSJpNZsmRJQUFBUVHRn/70p1tuucX3/cLCQs6M3HTTTWEYXnHFFTNnzoxY/72jO1HIletsea+OGTOGazey2WxpaWk+n+/Xr9+hYkwxe3jt2rVNTU2xWExKuXfvXtu2eTtymofrKb7xjW9cd911K1asWLZsWYRunsdUKsUWQGFh4Wc+85knn3xy5syZM2fOjHizLDY5393S0sKHs3GhwZgxY372s58tWLDgD3/4g5Ry4cKF7e3tsVjs7LPPLioq4tuy18qWF0MhlUqVlpZGPWxZoRYV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"
_LOGO_ID = "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"

if lang=='fr': ttl='AgriTech&nbsp;<em>Maroc</em>'
elif lang=='en': ttl='AgriTech&nbsp;<em>Morocco</em>'
else: ttl=f'<em>{t["title"]}</em>'

# --- HEADER (un seul bloc structuré) ---
_header_html = (
    '<div class="hdr">'
      '<div class="hdr-content">'
        '<div class="logobar">'
          f'<img src="data:image/png;base64,{_LOGO_UNIV}"/>'
          f'<img src="data:image/png;base64,{_LOGO_FEG}"/>'
          f'<img src="data:image/png;base64,{_LOGO_ID}"/>'
        '</div>'
        '<div class="titlebar">'
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
    <span style="color:{MINT};font-weight:800;font-size:0.82rem;">✨ Développée par Kaoutar MALIKI ✨</span>
</div>""",unsafe_allow_html=True)