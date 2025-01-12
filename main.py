# main.py

import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import yaml
import json

# Настройка страницы (заголовок, макет).
st.set_page_config(page_title="Экономическая модель склада (Проект-конвейер №2)", layout="wide")

from data_model import WarehouseParams, validate_inputs
from calculations import (
    calculate_additional_metrics,
    calculate_roi,
    calculate_irr,
    calculate_total_bep,
    monte_carlo_simulation,
    calculate_financials,
    min_loan_amount_for_bep,
    calculate_monthly_bep,
    calculate_areas,
    calculate_npv
)
from utils import (
    normalize_shares,
    load_params_from_file,
    save_params_to_file,
    load_css,
)
from streamlit_ui import (
    MetricDisplay,
    ChartDisplay,
    TableDisplay,
    display_tab1_header,
    display_tab1_metrics,
    display_tab1_bep,
    display_tab1_chart,
    display_tab1_analysis,
    display_tab1,
    display_tab2_header,
    display_tab2_basic_forecast,
    display_tab2_ml_forecast,
    display_tab2_monte_carlo,
    display_tab3_header,
    display_tab3_bep_info,
    display_tab3_monthly_bep,
    display_tab3_sensitivity,
    display_tab4_header,
    display_tab4_area_metrics,
    display_tab4_storage_table,
    display_tab4_profit_table,
    display_tab4_results,
    compare_params,
    display_tab5_header,
    display_tab5_dashboard
)
from ml_models import prepare_ml_data, load_ml_model, train_ml_model
from app_state import AppState


def save_trained_model_to_file(model, filename="trained_model.pkl"):
    """
    Сохраняет обученную ML-модель в буфер, чтобы пользователь мог скачать её в виде .pkl.
    """
    import io
    import joblib
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    st.download_button(
        label="Скачать ML-модель",
        data=buffer,
        file_name=filename,
        mime="application/octet-stream"
    )


@st.cache_data
def load_help_texts():
    """
    Загружает подсказки из help_text.yaml (если нужно выводить в интерфейсе).
    """
    try:
        with open("help_text.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.warning(f"Не удалось загрузить help_text.yaml: {e}")
        return {}

help_texts = load_help_texts()
load_css("style.css")

st.markdown("# Экономическая модель склада (Проект-конвейер №2)")
st.markdown(
    "Приложение для оценки доходности, расходов и ключевых показателей склада. "
    "Параметры задаются в боковой панели. Результаты отражаются на вкладках."
)

app_state = AppState()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

default_params = config["default_params"]

selected_forecast_method = app_state.get("forecast_method") or "Базовый"
poly_degree = app_state.get("poly_degree") or 2
n_estimators = app_state.get("n_estimators") or 100
df_for_ml = app_state.get("df_for_ml")
ml_model = app_state.get("ml_model")
features = app_state.get("features") or ["Месяц", "Lag_1", "Lag_2", "Rolling_Mean_3", "Rolling_Mean_5"]
auto_feature_selection = app_state.get("auto_feature_selection") or False

def reset_params():
    """
    Сброс параметров к настройкам по умолчанию, очистка загруженных файлов и текущего метода прогноза.
    """
    app_state.load_default_state()
    st.session_state["uploaded_file"] = None
    st.session_state["df_for_ml"] = None
    st.session_state["ml_model"] = None
    st.session_state["forecast_method"] = "Базовый"
    st.rerun()

query_params = st.query_params
if query_params and "params" in query_params:
    try:
        loaded_params_str = query_params["params"]
        loaded_params = json.loads(loaded_params_str)
        for key, value in loaded_params.items():
            if key in default_params:
                app_state.set(key, value)
        if "shares" in loaded_params:
            app_state.shares.update(loaded_params["shares"])
        st.success("Параметры успешно загружены из URL.")
    except Exception as e:
        st.error(f"Ошибка загрузки из URL: {e}")

with st.sidebar:
    st.markdown("## Ввод параметров")
    if st.button("🔄 Сбросить параметры"):
        reset_params()
    
    with st.sidebar.expander("### Настройки отображения", expanded=False):
        # Добавлен селектбокс для выбора формата чисел
        format_options = ["Без разделителей", "С разделителями"]
        current_format = app_state.get("selected_format") or "С разделителями"
        format_index = format_options.index(current_format) if current_format in format_options else 1
        selected_format = st.selectbox(
            "Формат чисел",
            format_options,
            index=format_index,
            help = "Выберите формат отображения чисел: с разделителем тысяч или без.",
        )
        app_state.set("selected_format", selected_format)
         
        decimal_places_options = ["0", "1", "2", "3", "4"]
        current_decimal = app_state.get("selected_decimal") or "2"
        decimal_index = decimal_places_options.index(current_decimal) if current_decimal in decimal_places_options else 2
        selected_decimal = st.selectbox(
            "Знаков после запятой",
            decimal_places_options,
            index=decimal_index,
            help = "Выберите количество знаков после запятой.",
        )
        app_state.set("selected_decimal", selected_decimal)
        
        theme_options = ["Стандартная", "Темная"]
        current_theme = app_state.get("selected_theme") or "Стандартная"
        theme_index = theme_options.index(current_theme) if current_theme in theme_options else 0
        selected_theme = st.selectbox(
            "Тема интерфейса",
            theme_options,
            index=theme_index,
            help=help_texts.get("theme", "")
        )
        app_state.set("selected_theme", selected_theme)

        main_color = st.color_picker(
            "Основной цвет интерфейса",
            value=app_state.get("main_color") or "#007bff",
            help=help_texts.get("main_color", "")
        )
        app_state.set("main_color", main_color)

        if selected_theme == "Темная":
            load_css("dark_style.css")
        else:
            load_css("style.css")
    
    
    with st.sidebar.expander("### Основные параметры", expanded=False):
        total_area = st.number_input(
            "Общая площадь (м²)",
            value=app_state.get("total_area"),
            step=10,
            min_value=1,
            format="%i",
            help=help_texts.get("total_area", "Общая арендуемая площадь склада в квадратных метрах."),
        )
        app_state.set("total_area", total_area)
        if total_area <= 0:
            st.error("Общая площадь должна быть больше нуля.")

        rental_cost_per_m2 = st.number_input(
            "Стоимость аренды (руб./м²/мес.)",
            value=app_state.get("rental_cost_per_m2"),
            step=50,
            min_value=1,
            format="%i",
            help=help_texts.get("rental_cost_per_m2", "Ежемесячная аренда за один квадратный метр."),
        )
        app_state.set("rental_cost_per_m2", rental_cost_per_m2)
        if rental_cost_per_m2 <= 0:
            st.error("Арендная ставка должна быть > 0.")

        useful_area_ratio_slider = st.slider(
            "Доля полезной площади (%)",
            40,
            80,
            int(app_state.get("useful_area_ratio") * 100),
            5,
             help=help_texts.get("useful_area_ratio", "Процент полезной площади от общей площади склада.")
        )
        useful_area_ratio = useful_area_ratio_slider / 100.0
        app_state.set("useful_area_ratio", useful_area_ratio)

    with st.sidebar.expander("### Распределение площади", expanded=False):
        mode = st.radio(
            "Режим распределения",
            ["Ручной", "Автоматический"],
            index=0,
            help=help_texts.get("mode", "Выберите режим распределения площадей: ручной или автоматический"),
        )
        app_state.set("mode", mode)

        if mode == "Ручной":
            st.markdown("#### Ручной ввод (м²)")
            temp_usable = total_area * useful_area_ratio

            col1, col2 = st.columns(2)
            storage_area_manual = col1.number_input(
                "Простое, м²",
                value=app_state.get("storage_area_manual"),
                step=10.0,
                min_value=0.0,
                format="%.2f",
                help=help_texts.get("storage_area_manual", "Площадь под простое хранение."),
            )
            app_state.set("storage_area_manual", storage_area_manual)

            loan_area_manual = col2.number_input(
                "Займы, м²",
                value=app_state.get("loan_area_manual"),
                step=10.0,
                min_value=0.0,
                format="%.2f",
                help=help_texts.get("loan_area_manual", "Площадь под займы."),
            )
            app_state.set("loan_area_manual", loan_area_manual)

            col3, col4 = st.columns(2)
            vip_area_manual = col3.number_input(
                "VIP, м²",
                value=app_state.get("vip_area_manual"),
                step=10.0,
                min_value=0.0,
                format="%.2f",
                 help=help_texts.get("vip_area_manual", "Площадь под VIP-хранение.")
            )
            app_state.set("vip_area_manual", vip_area_manual)

            short_term_area_manual = col4.number_input(
                "Краткосрочное, м²",
                value=app_state.get("short_term_area_manual"),
                step=10.0,
                min_value=0.0,
                format="%.2f",
                help=help_texts.get("short_term_area_manual", "Площадь под краткосрочное хранение."),
            )
            app_state.set("short_term_area_manual", short_term_area_manual)

            total_manual_set = (
                storage_area_manual
                + loan_area_manual
                + vip_area_manual
                + short_term_area_manual
            )
            leftover = temp_usable - total_manual_set
            st.write(f"Не распределено: {leftover:.2f} м² из {temp_usable:.2f} м².")
        else:
            st.write("Автоматическое распределение по указанным долям.")

    with st.sidebar.expander("### Тарифы и плотности", expanded=False):
        storage_fee = st.number_input(
            "Тариф простого (руб./м²/мес.)",
            value=app_state.get("storage_fee"),
            step=100,
            min_value=0,
            format="%i",
            help=help_texts.get("storage_fee", "Ежемесячный тариф за простой склад (руб./м²)."),
        )
        app_state.set("storage_fee", storage_fee)

        col1, col2 = st.columns(2)
        shelves_per_m2 = col1.number_input(
            "Полок на 1 м²",
            value=app_state.get("shelves_per_m2"),
            step=1,
            min_value=1,
            max_value=100,
            format="%i",
            help=help_texts.get("shelves_per_m2", "Количество полок на 1 м²."),
        )
        app_state.set("shelves_per_m2", shelves_per_m2)

        short_term_daily_rate = col2.number_input(
            "Тариф краткоср. (руб./день/м²)",
            value=app_state.get("short_term_daily_rate"),
            step=10.0,
            min_value=0.0,
            format="%.2f",
            help=help_texts.get("short_term_daily_rate", "Тариф за 1 м² краткосрочного хранения в день."),
        )
        app_state.set("short_term_daily_rate", short_term_daily_rate)

        vip_extra_fee = st.number_input(
            "Наценка VIP (руб./м²/мес.)",
            value=app_state.get("vip_extra_fee"),
            step=50.0,
            min_value=0.0,
            format="%.2f",
            help=help_texts.get("vip_extra_fee", "Дополнительная наценка для VIP-хранения (руб./м²)."),
        )
        app_state.set("vip_extra_fee", vip_extra_fee)

    with st.sidebar.expander("### Оценка и займы", expanded=False):
        item_evaluation_slider = st.slider(
            "Оценка вещи (%)",
            0,
            100,
            int(app_state.get("item_evaluation") * 100),
            5,
            help=help_texts.get("item_evaluation", "Процент оценки стоимости вещи, которую можно взять под залог."),
        )
        item_evaluation = item_evaluation_slider / 100.0
        app_state.set("item_evaluation", item_evaluation)

        item_realization_markup = st.number_input(
            "Наценка реализации (%)",
            value=app_state.get("item_realization_markup"),
            step=5.0,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
            help=help_texts.get("item_realization_markup", "Наценка, применяемая при продаже невостребованных вещей."),
        )
        app_state.set("item_realization_markup", item_realization_markup)

        average_item_value = st.number_input(
            "Средняя оценка вещи (руб.)",
            value=app_state.get("average_item_value"),
            step=500,
            min_value=0,
            format="%i",
            help=help_texts.get("average_item_value", "Средняя стоимость одной вещи (руб.)."),
        )
        app_state.set("average_item_value", average_item_value)

        loan_interest_rate = st.number_input(
            "Ставка займов (%/день)",
            value=app_state.get("loan_interest_rate"),
            step=0.01,
            min_value=0.0,
            format="%.3f",
           help=help_texts.get("loan_interest_rate", "Дневная процентная ставка для займов."),
        )
        app_state.set("loan_interest_rate", loan_interest_rate)

        loan_term_days = st.number_input(
            "Средний срок займа (дни)",
            value=app_state.get("loan_term_days"),
            step=1,
            min_value=1,
            format="%i",
             help=help_texts.get("loan_term_days", "Средний срок займа в днях."),
        )
        app_state.set("loan_term_days", loan_term_days)

    with st.sidebar.expander("### Реализация (%)", expanded=False):
        realization_share_storage_slider = st.slider(
            "Простое",
            0,
            100,
            int(app_state.get("realization_share_storage") * 100),
            5,
            help=help_texts.get("realization_share_storage", "Процент вещей из простого хранения, идущих на реализацию."),
        )
        realization_share_storage = realization_share_storage_slider / 100.0
        app_state.set("realization_share_storage", realization_share_storage)

        realization_share_loan_slider = st.slider(
            "Займы",
            0,
            100,
            int(app_state.get("realization_share_loan") * 100),
            5,
            help=help_texts.get("realization_share_loan", "Процент вещей из займов, идущих на реализацию."),
        )
        realization_share_loan = realization_share_loan_slider / 100.0
        app_state.set("realization_share_loan", realization_share_loan)

        realization_share_vip_slider = st.slider(
            "VIP",
            0,
            100,
            int(app_state.get("realization_share_vip") * 100),
            5,
             help=help_texts.get("realization_share_vip", "Процент вещей из VIP-хранения, идущих на реализацию."),
        )
        realization_share_vip = realization_share_vip_slider / 100.0
        app_state.set("realization_share_vip", realization_share_vip)

        realization_share_short_term_slider = st.slider(
            "Краткосрочное",
            0,
            100,
            int(app_state.get("realization_share_short_term") * 100),
            5,
            help=help_texts.get("realization_share_short_term", "Процент вещей из краткосрочного хранения на реализацию."),
        )
        realization_share_short_term = realization_share_short_term_slider / 100.0
        app_state.set("realization_share_short_term", realization_share_short_term)

    with st.sidebar.expander("### Процент заполняемости", expanded=False):
        storage_fill_rate_slider = st.slider(
            "Простое",
            0,
            100,
            int(app_state.get("storage_fill_rate") * 100),
            5,
           help=help_texts.get("storage_fill_rate", "Процент заполнения площади простого хранения."),
        )
        storage_fill_rate = storage_fill_rate_slider / 100.0
        app_state.set("storage_fill_rate", storage_fill_rate)

        loan_fill_rate_slider = st.slider(
            "Займы",
            0,
            100,
            int(app_state.get("loan_fill_rate") * 100),
            5,
            help=help_texts.get("loan_fill_rate", "Процент заполнения площади займов."),
        )
        loan_fill_rate = loan_fill_rate_slider / 100.0
        app_state.set("loan_fill_rate", loan_fill_rate)

        vip_fill_rate_slider = st.slider(
            "VIP",
            0,
            100,
            int(app_state.get("vip_fill_rate") * 100),
            5,
            help=help_texts.get("vip_fill_rate", "Процент заполнения VIP-секции."),
        )
        vip_fill_rate = vip_fill_rate_slider / 100.0
        app_state.set("vip_fill_rate", vip_fill_rate)

        short_term_fill_rate_slider = st.slider(
            "Краткосрочное",
            0,
            100,
            int(app_state.get("short_term_fill_rate") * 100),
            5,
             help=help_texts.get("short_term_fill_rate", "Процент заполнения краткосрочного хранения."),
        )
        short_term_fill_rate = short_term_fill_rate_slider / 100.0
        app_state.set("short_term_fill_rate", short_term_fill_rate)

    with st.sidebar.expander("### Плотность (вещей/м²)", expanded=False):
        storage_items_density = st.number_input(
            "Простое",
            value=app_state.get("storage_items_density"),
            step=1,
            min_value=1,
            format="%i",
             help=help_texts.get("storage_items_density", "Плотность хранения (вещей на м²) для простого хранения."),
        )
        app_state.set("storage_items_density", storage_items_density)

        loan_items_density = st.number_input(
            "Займы",
            value=app_state.get("loan_items_density"),
            step=1,
            min_value=1,
            format="%i",
             help=help_texts.get("loan_items_density", "Плотность хранения для займов (вещи/м²)."),
        )
        app_state.set("loan_items_density", loan_items_density)

        vip_items_density = st.number_input(
            "VIP",
            value=app_state.get("vip_items_density"),
            step=1,
            min_value=1,
            format="%i",
             help=help_texts.get("vip_items_density", "Плотность хранения для VIP (вещи/м²)."),
        )
        app_state.set("vip_items_density", vip_items_density)

        short_term_items_density = st.number_input(
            "Краткосрочное",
            value=app_state.get("short_term_items_density"),
            step=1,
            min_value=1,
            format="%i",
            help=help_texts.get("short_term_items_density", "Плотность хранения для краткосрочного (вещи/м²)."),
        )
        app_state.set("short_term_items_density", short_term_items_density)

    with st.sidebar.expander("### Отток клиентов/вещей (%)", expanded=False):
        storage_monthly_churn_num = st.number_input(
            "Простое (%)",
            value=app_state.get("storage_monthly_churn") * 100,
            step=0.1,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
            help=help_texts.get("storage_monthly_churn", "Ежемесячный отток клиентов из простого хранения."),
        )
        storage_monthly_churn = storage_monthly_churn_num / 100.0
        app_state.set("storage_monthly_churn", storage_monthly_churn)

        loan_monthly_churn_num = st.number_input(
            "Займы (%)",
            value=app_state.get("loan_monthly_churn") * 100,
            step=0.1,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
            help=help_texts.get("loan_monthly_churn", "Ежемесячный отток по займам."),
        )
        loan_monthly_churn = loan_monthly_churn_num / 100.0
        app_state.set("loan_monthly_churn", loan_monthly_churn)

        vip_monthly_churn_num = st.number_input(
            "VIP (%)",
            value=app_state.get("vip_monthly_churn") * 100,
            step=0.1,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
             help=help_texts.get("vip_monthly_churn", "Ежемесячный отток клиентов из VIP-секции."),
        )
        vip_monthly_churn = vip_monthly_churn_num / 100.0
        app_state.set("vip_monthly_churn", vip_monthly_churn)

        short_term_monthly_churn_num = st.number_input(
            "Краткосрочное (%)",
            value=app_state.get("short_term_monthly_churn") * 100,
            step=0.1,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
             help=help_texts.get("short_term_monthly_churn", "Ежемесячный отток клиентов по краткосрочному хранению."),
        )
        short_term_monthly_churn = short_term_monthly_churn_num / 100.0
        app_state.set("short_term_monthly_churn", short_term_monthly_churn)

    with st.sidebar.expander("### Финансы (ежемесячные)", expanded=False):
        salary_expense = st.number_input(
            "Зарплата (руб./мес.)",
            value=app_state.get("salary_expense"),
            step=10000,
            min_value=0,
            format="%i",
            help=help_texts.get("salary_expense", "Общие затраты на зарплату (руб./мес.)."),
        )
        app_state.set("salary_expense", salary_expense)

        miscellaneous_expenses = st.number_input(
            "Прочие (руб./мес.)",
            value=app_state.get("miscellaneous_expenses"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("miscellaneous_expenses", "Прочие ежемесячные расходы."),
        )
        app_state.set("miscellaneous_expenses", miscellaneous_expenses)

        depreciation_expense = st.number_input(
            "Амортизация (руб./мес.)",
            value=app_state.get("depreciation_expense"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("depreciation_expense", "Ежемесячная амортизация."),
        )
        app_state.set("depreciation_expense", depreciation_expense)

        marketing_expenses = st.number_input(
            "Маркетинг (руб./мес.)",
            value=app_state.get("marketing_expenses"),
            step=5000,
            min_value=0,
            format="%i",
             help=help_texts.get("marketing_expenses", "Затраты на маркетинг."),
        )
        app_state.set("marketing_expenses", marketing_expenses)

        insurance_expenses = st.number_input(
            "Страхование (руб./мес.)",
            value=app_state.get("insurance_expenses"),
            step=1000,
            min_value=0,
            format="%i",
            help=help_texts.get("insurance_expenses", "Ежемесячная страховка."),
        )
        app_state.set("insurance_expenses", insurance_expenses)

        taxes = st.number_input(
            "Налоги (руб./мес.)",
            value=app_state.get("taxes"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("taxes", "Налоговые отчисления (руб./мес.)."),
        )
        app_state.set("taxes", taxes)

        utilities_expenses = st.number_input(
            "Коммуналка (руб./мес.)",
            value=app_state.get("utilities_expenses"),
            step=5000,
            min_value=0,
            format="%i",
             help=help_texts.get("utilities_expenses", "Коммунальные услуги (руб./мес.)."),
        )
        app_state.set("utilities_expenses", utilities_expenses)

        maintenance_expenses = st.number_input(
            "Обслуживание (руб./мес.)",
            value=app_state.get("maintenance_expenses"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("maintenance_expenses", "Обслуживание склада (руб./мес)."),
        )
        app_state.set("maintenance_expenses", maintenance_expenses)

    with st.sidebar.expander("### Финансы (единовременные)", expanded=False):
        one_time_setup_cost = st.number_input(
            "Настройка (руб.)",
            value=app_state.get("one_time_setup_cost"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("one_time_setup_cost", "Единовременные затраты на настройку склада."),
        )
        app_state.set("one_time_setup_cost", one_time_setup_cost)

        one_time_equipment_cost = st.number_input(
            "Оборудование (руб.)",
            value=app_state.get("one_time_equipment_cost"),
            step=5000,
            min_value=0,
            format="%i",
             help=help_texts.get("one_time_equipment_cost", "Единовременные затраты на оборудование."),
        )
        app_state.set("one_time_equipment_cost", one_time_equipment_cost)

        one_time_other_costs = st.number_input(
            "Другие (руб.)",
            value=app_state.get("one_time_other_costs"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("one_time_other_costs", "Прочие единовременные расходы."),
        )
        app_state.set("one_time_other_costs", one_time_other_costs)

        one_time_legal_cost = st.number_input(
            "Юридические (руб.)",
            value=app_state.get("one_time_legal_cost"),
            step=5000,
            min_value=0,
            format="%i",
             help=help_texts.get("one_time_legal_cost", "Единовременные юридические расходы."),
        )
        app_state.set("one_time_legal_cost", one_time_legal_cost)

        one_time_logistics_cost = st.number_input(
            "Логистика (руб.)",
            value=app_state.get("one_time_logistics_cost"),
            step=5000,
            min_value=0,
            format="%i",
            help=help_texts.get("one_time_logistics_cost", "Единовременные логистические расходы."),
        )
        app_state.set("one_time_logistics_cost", one_time_logistics_cost)

    with st.sidebar.expander("### Переменные расходы", expanded=False):
        packaging_cost_per_m2 = st.number_input(
            "Упаковка (руб./м²)",
            value=app_state.get("packaging_cost_per_m2"),
            step=5.0,
            min_value=0.0,
            format="%.2f",
            help=help_texts.get("packaging_cost_per_m2", "Стоимость упаковки на 1 м² площади."),
        )
        app_state.set("packaging_cost_per_m2", packaging_cost_per_m2)

        electricity_cost_per_m2 = st.number_input(
            "Электричество (руб./м²)",
            value=app_state.get("electricity_cost_per_m2"),
            step=10.0,
            min_value=0.0,
            format="%.1f",
            help=help_texts.get("electricity_cost_per_m2", "Стоимость электроэнергии на 1 м²."),
        )
        app_state.set("electricity_cost_per_m2", electricity_cost_per_m2)

    with st.sidebar.expander("### Инфляция и рост", expanded=False):
        monthly_inflation_rate_val = st.number_input(
            "Инфляция (%/мес.)",
            value=app_state.get("monthly_inflation_rate") * 100,
            step=0.1,
            min_value=0.0,
            format="%.1f",
             help=help_texts.get("monthly_inflation_rate", "Ежемесячная инфляция (%)."),
        )
        monthly_inflation_rate = monthly_inflation_rate_val / 100.0
        app_state.set("monthly_inflation_rate", monthly_inflation_rate)

        monthly_rent_growth_val = st.number_input(
            "Рост аренды (%/мес.)",
            value=app_state.get("monthly_rent_growth") * 100,
            step=0.5,
            min_value=0.0,
            format="%.1f",
            help=help_texts.get("monthly_rent_growth", "Рост аренды в месяц (%)."),
        )
        monthly_rent_growth = monthly_rent_growth_val / 100.0
        app_state.set("monthly_rent_growth", monthly_rent_growth)

        monthly_salary_growth_val = st.number_input(
            "Рост зарплаты (%/мес.)",
            value=app_state.get("monthly_salary_growth") * 100,
            step=0.1,
            min_value=0.0,
            format="%.1f",
           help=help_texts.get("monthly_salary_growth", "Ежемесячный рост зарплаты (%)."),
        )
        monthly_salary_growth = monthly_salary_growth_val / 100.0
        app_state.set("monthly_salary_growth", monthly_salary_growth)

        monthly_other_expenses_growth_val = st.number_input(
            "Рост прочих расходов (%/мес.)",
            value=app_state.get("monthly_other_expenses_growth") * 100,
            step=0.1,
            min_value=0.0,
            format="%.1f",
            help=help_texts.get("monthly_other_expenses_growth", "Ежемесячный рост прочих расходов (%)."),
        )
        monthly_other_expenses_growth = monthly_other_expenses_growth_val / 100.0
        app_state.set("monthly_other_expenses_growth", monthly_other_expenses_growth)

    with st.sidebar.expander("### Расширенные параметры и прогнозирование", expanded=False):
        disable_extended = st.checkbox(
            "Отключить расширенные параметры",
            value=app_state.get("disable_extended"),
            help=help_texts.get("disable_extended", "Если включено, расширенные параметры не учитываются.")
        )
        app_state.set("disable_extended", disable_extended)

        amortize_one_time_expenses = st.checkbox(
            "Амортизировать единовременные расходы",
            value=app_state.get("amortize_one_time_expenses"),
            help=help_texts.get("amortize_one_time_expenses", "Если включено, единовременные расходы распределяются по всему горизонту прогноза.")
        )
        app_state.set("amortize_one_time_expenses", amortize_one_time_expenses)

        if not disable_extended:
            time_horizon_val = st.slider(
                "Горизонт прогноза (мес.)",
                1,
                24,
                value=app_state.get("time_horizon"),
                help=help_texts.get("time_horizon", "Сколько месяцев прогнозируем."),
            )
            app_state.set("time_horizon", time_horizon_val)

            default_probability_val = st.number_input(
                "Вероятность невозврата (%)",
                value=app_state.get("default_probability") * 100,
                step=1.0,
                min_value=0.0,
                max_value=100.0,
                format="%.1f",
                help=help_texts.get("default_probability", "Вероятность невозврата (для займов)."),
            )
            default_probability = default_probability_val / 100.0
            app_state.set("default_probability", default_probability)

            liquidity_factor_val = st.number_input(
                "Коэффициент ликвидности",
                value=app_state.get("liquidity_factor"),
                step=0.1,
                min_value=0.1,
                format="%.1f",
               help=help_texts.get("liquidity_factor", "Коэффициент ликвидности."),
            )
            app_state.set("liquidity_factor", liquidity_factor_val)

            safety_factor_val = st.number_input(
                "Коэффициент запаса",
                value=app_state.get("safety_factor"),
                step=0.1,
                min_value=0.1,
                format="%.1f",
                help=help_texts.get("safety_factor", "Коэффициент запаса для устойчивости."),
            )
            app_state.set("safety_factor", safety_factor_val)

            loan_grace_period_val = st.number_input(
                "Льготный период (мес.)",
                value=app_state.get("loan_grace_period"),
                step=1,
                min_value=0,
                format="%i",
                help=help_texts.get("loan_grace_period", "Льготный период по займам (мес)."),
            )
            app_state.set("loan_grace_period", loan_grace_period_val)

            monthly_income_growth_val = st.number_input(
                "Рост доходов (%/мес.)",
                value=app_state.get("monthly_income_growth") * 100,
                step=0.5,
                format="%.1f",
               help=help_texts.get("monthly_income_growth", "Предполагаемый рост доходов в месяц."),
            )
            monthly_income_growth = monthly_income_growth_val / 100.0
            app_state.set("monthly_income_growth", monthly_income_growth)

            monthly_expenses_growth_val = st.number_input(
                "Рост расходов (%/мес.)",
                value=app_state.get("monthly_expenses_growth") * 100,
                step=0.5,
                format="%.1f",
               help=help_texts.get("monthly_expenses_growth", "Предполагаемый рост расходов в месяц."),
            )
            monthly_expenses_growth = monthly_expenses_growth_val / 100.0
            app_state.set("monthly_expenses_growth", monthly_expenses_growth)
        else:
            app_state.set("time_horizon", 1)
            app_state.set("default_probability", 0.0)
            app_state.set("liquidity_factor", 1.0)
            app_state.set("safety_factor", 1.2)
            app_state.set("loan_grace_period", 0)
            app_state.set("monthly_income_growth", 0.0)
            app_state.set("monthly_expenses_growth", 0.0)

        fm_options = [
            "Базовый",
            "ML (линейная регрессия)",
            "ML (полиномиальная регрессия)",
            "Симуляция Монте-Карло",
            "ML (случайный лес)",
            "ML (SVR)",
        ]
        current_fm = app_state.get("forecast_method") or "Базовый"
        fm_index = fm_options.index(current_fm) if current_fm in fm_options else 0
        forecast_method_sel = st.selectbox(
            "Метод прогнозирования",
            fm_options,
            index=fm_index,
            help=help_texts.get("forecast_method", "Метод, используемый для построения прогноза."),
        )
        app_state.set("forecast_method", forecast_method_sel)

        if forecast_method_sel == "Симуляция Монте-Карло":
            monte_carlo_simulations_val = st.number_input(
                "Симуляций Монте-Карло",
                value=app_state.get("monte_carlo_simulations"),
                step=10,
                min_value=10,
                format="%i",
                help=help_texts.get("monte_carlo_simulations", "Число симуляций в Монте-Карло."),
            )
            app_state.set("monte_carlo_simulations", monte_carlo_simulations_val)

            monte_carlo_deviation_val = st.number_input(
                "Отклонения (0.1 = ±10%)",
                value=app_state.get("monte_carlo_deviation"),
                step=0.01,
                min_value=0.01,
                format="%.2f",
                help=help_texts.get("monte_carlo_deviation", "Отклонения для Монте-Карло (0.1 = ±10%)."),
            )
            app_state.set("monte_carlo_deviation", monte_carlo_deviation_val)

            monte_carlo_seed_val = st.number_input(
                "Значение Seed",
                value=app_state.get("monte_carlo_seed"),
                step=1,
                format="%i",
                 help=help_texts.get("monte_carlo_seed", "Зерно случайности (Монте-Карло)."),
            )
            app_state.set("monte_carlo_seed", monte_carlo_seed_val)

            mc_dist_options = ["Равномерное", "Нормальное", "Треугольное"]
            current_dist = app_state.get("monte_carlo_distribution") or "Равномерное"
            mc_dist_index = mc_dist_options.index(current_dist) if current_dist in mc_dist_options else 0
            monte_carlo_distribution_sel = st.selectbox(
                "Тип распределения",
                mc_dist_options,
                index=mc_dist_index,
                help=help_texts.get("monte_carlo_distribution", "Тип распределения для симуляции Монте-Карло."),
            )
            app_state.set("monte_carlo_distribution", monte_carlo_distribution_sel)

            if monte_carlo_distribution_sel == "Нормальное":
                mc_normal_mean_val = st.number_input(
                    "Среднее (норм. распр.)",
                    value=app_state.get("monte_carlo_normal_mean") or 0.0,
                    step=0.1,
                    format="%.1f",
                     help=help_texts.get("monte_carlo_normal_mean", "Среднее для нормального распределения."),
                )
                app_state.set("monte_carlo_normal_mean", mc_normal_mean_val)

                mc_normal_std_val = st.number_input(
                    "Ст. отклонение (норм. распр.)",
                    value=app_state.get("monte_carlo_normal_std") or 0.1,
                    step=0.01,
                    min_value=0.01,
                    format="%.2f",
                    help=help_texts.get("monte_carlo_normal_std", "Ст. отклонение для нормального распределения."),
                )
                app_state.set("monte_carlo_normal_std", mc_normal_std_val)

            if monte_carlo_distribution_sel == "Треугольное":
                mc_triang_left_val = st.number_input(
                    "Мин. значение (треуг. распр.)",
                    value=app_state.get("monte_carlo_triang_left") or 0.0,
                    step=0.1,
                    format="%.1f",
                     help=help_texts.get("monte_carlo_triang_left", "Мин. значение для треугольного распределения."),
                )
                app_state.set("monte_carlo_triang_left", mc_triang_left_val)

                mc_triang_mode_val = st.number_input(
                    "Мода (треуг. распр.)",
                    value=app_state.get("monte_carlo_triang_mode") or 1.0,
                    step=0.1,
                    format="%.1f",
                    help=help_texts.get("monte_carlo_triang_mode", "Мода для треугольного распределения."),
                )
                app_state.set("monte_carlo_triang_mode", mc_triang_mode_val)

                mc_triang_right_val = st.number_input(
                    "Макс. значение (треуг. распр.)",
                    value=app_state.get("monte_carlo_triang_right") or 2.0,
                    step=0.1,
                    format="%.1f",
                    help=help_texts.get("monte_carlo_triang_right", "Макс. значение для треугольного распределения."),
                )
                app_state.set("monte_carlo_triang_right", mc_triang_right_val)

        enable_ml_settings_val = st.checkbox(
            "Включить расширенный ML-прогноз",
            value=app_state.get("enable_ml_settings"),
            help=help_texts.get("enable_ml_settings", "Включает дополнительные настройки для ML-прогноза.")
        )
        app_state.set("enable_ml_settings", enable_ml_settings_val)

        if forecast_method_sel == "ML (полиномиальная регрессия)" and enable_ml_settings_val:
            poly_degree_val = st.number_input(
                "Степень полинома",
                min_value=1,
                max_value=5,
                value=app_state.get("poly_degree") or 2,
                step=1,
                format="%i",
                help=help_texts.get("poly_degree", "Степень полинома для полиномиальной регрессии."),
            )
            app_state.set("poly_degree", poly_degree_val)
        else:
            app_state.set("poly_degree", 2)

        if forecast_method_sel == "ML (случайный лес)" and enable_ml_settings_val:
            n_estimators_val = st.number_input(
                "Количество деревьев в RF",
                min_value=10,
                max_value=500,
                value=app_state.get("n_estimators") or 100,
                step=10,
                format="%i",
                help=help_texts.get("n_estimators", "Количество деревьев в RF."),
            )
            app_state.set("n_estimators", n_estimators_val)

            features_options = ["Месяц", "Lag_1", "Lag_2", "Rolling_Mean_3", "Rolling_Mean_5"]
            selected_features = st.multiselect(
                "Признаки для обучения",
                options=features_options,
                default=features_options,
                help=help_texts.get("features", "Признаки для обучения ML модели."),
            )
            app_state.set("features", selected_features)
        elif forecast_method_sel == "ML (SVR)" and enable_ml_settings_val:
            features_options = ["Месяц", "Lag_1", "Lag_2", "Rolling_Mean_3", "Rolling_Mean_5"]
            selected_features = st.multiselect(
                "Признаки для обучения",
                options=features_options,
                default=features_options,
                help=help_texts.get("features", "Признаки для обучения ML модели."),
            )
            app_state.set("features", selected_features)
        else:
            app_state.set("n_estimators", 100)
            app_state.set("features", ["Месяц", "Lag_1", "Lag_2", "Rolling_Mean_3", "Rolling_Mean_5"])

        if forecast_method_sel in ["ML (случайный лес)", "ML (SVR)"] and enable_ml_settings_val:
            param_search_options = ["Нет", "GridSearchCV", "RandomizedSearchCV"]
            param_search_method = st.selectbox(
                "Поиск параметров ML",
                param_search_options,
                index=0,
                 help=help_texts.get("param_search_method", "Метод поиска параметров ML."),
            )
            app_state.set("param_search_method", param_search_method)

            auto_feature_selection_val = st.checkbox(
                "Автоматический выбор признаков",
                value=app_state.get("auto_feature_selection"),
                 help=help_texts.get("auto_feature_selection", "Включить автоматический выбор признаков.")
            )
            app_state.set("auto_feature_selection", auto_feature_selection_val)
        else:
            app_state.set("param_search_method", "Нет")
            app_state.set("auto_feature_selection", False)

    if enable_ml_settings_val and forecast_method_sel.startswith("ML"):
        uploaded_file = st.file_uploader(
            "Загрузить данные (CSV/Excel) для ML",
            type=["csv", "xlsx"]
        )
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1]
            try:
                 with st.spinner("Чтение данных..."):
                    if file_extension == ".csv":
                        df_for_ml = pd.read_csv(uploaded_file)
                    elif file_extension == ".xlsx":
                        df_for_ml = pd.read_excel(uploaded_file)
                    else:
                        raise ValueError("Формат файла не поддерживается.")
                    st.success("Файл с данными для ML успешно загружен.")
                    app_state.set("df_for_ml", df_for_ml)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
                app_state.set("df_for_ml", None)
        else:
            app_state.set("df_for_ml", None)

        uploaded_model = st.file_uploader(
            "Загрузить готовую ML-модель (.pkl)",
            type=["pkl"]
        )
        app_state.set("uploaded_model", uploaded_model)
    else:
        app_state.set("df_for_ml", None)
        app_state.set("uploaded_model", None)

    st.sidebar.markdown("---")

    if st.sidebar.button("Сохранить параметры"):
        if "saved_params" not in st.session_state:
            st.session_state.saved_params = {}
        param_name = f"Сохраненные параметры {len(st.session_state.saved_params) + 1}"
        
        params_to_save = {k: app_state.get(k) for k in default_params.keys()}
        params_to_save["shares"] = dict(app_state.shares)
        
        st.session_state.saved_params[param_name] = params_to_save
        st.success(f"Параметры сохранены: {param_name}")
    
    
    if "saved_params" in st.session_state and st.session_state.saved_params:
      selected_param = st.selectbox(
            "Сравнить с ранее сохранёнными:",
            options=list(st.session_state.saved_params.keys()),
            index=0,
              )
      if st.button("Переименовать", key="rename_saved_params_button"):
          new_name = st.text_input("Новое имя:", key="rename_param_input")
          if new_name:
             st.session_state.saved_params[new_name] = st.session_state.saved_params.pop(selected_param)
             st.success(f"Параметры переименованы на {new_name}")
             st.rerun()

    else:
        selected_param = None

    uploaded_file_sess = st.sidebar.file_uploader(
        "Загрузить сохранённые параметры (JSON или YAML)",
        type=["json", "yaml", "yml"]
    )
    if uploaded_file_sess:
        try:
             with st.spinner("Чтение параметров..."):
                if uploaded_file_sess.name.endswith(".json"):
                    loaded_params = json.load(uploaded_file_sess)
                elif uploaded_file_sess.name.endswith((".yaml", ".yml")):
                    loaded_params = yaml.safe_load(uploaded_file_sess)
                else:
                    raise ValueError("Поддерживается только JSON или YAML.")
                for key, value in loaded_params.items():
                    if key in default_params:
                        app_state.set(key, value)
                if "shares" in loaded_params:
                    app_state.shares.update(loaded_params["shares"])
                st.success("Параметры успешно загружены.")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

    if st.sidebar.button("Сохранить в файл"):
        try:
            filename = st.text_input("Имя файла", "warehouse_params")
            file_format = st.selectbox("Формат", ["json", "yaml"])
            if filename:
                params_to_save = {k: app_state.get(k) for k in default_params.keys()}
                params_to_save["shares"] = dict(app_state.shares)
                file_data = save_params_to_file(params_to_save, filename, file_format)
                st.download_button(
                    label="Скачать параметры",
                    data=file_data,
                    file_name=f"{filename}.{file_format}",
                    mime=f"application/{file_format}",
                )
        except Exception as e:
            st.error(f"Ошибка сохранения: {e}")

    if st.sidebar.button("Сформировать быстрый отчёт"):
        one_time_sum = (
            app_state.get("one_time_setup_cost")
            + app_state.get("one_time_equipment_cost")
            + app_state.get("one_time_other_costs")
            + app_state.get("one_time_legal_cost")
            + app_state.get("one_time_logistics_cost")
        )
        monthly_sum = (
            app_state.get("salary_expense")
            + app_state.get("miscellaneous_expenses")
            + app_state.get("depreciation_expense")
            + app_state.get("marketing_expenses")
            + app_state.get("insurance_expenses")
            + app_state.get("taxes")
            + app_state.get("utilities_expenses")
            + app_state.get("maintenance_expenses")
        )
        st.sidebar.write(f"Единовременные расходы: {one_time_sum:,.2f} руб.")
        st.sidebar.write(f"Ежемесячные расходы: {monthly_sum:,.2f} руб.")

    if st.sidebar.button("Сформировать углублённый отчёт"):
        one_time_sum2 = (
            app_state.get("one_time_setup_cost")
            + app_state.get("one_time_equipment_cost")
            + app_state.get("one_time_other_costs")
            + app_state.get("one_time_legal_cost")
            + app_state.get("one_time_logistics_cost")
        )
        monthly_sum2 = (
            app_state.get("salary_expense")
            + app_state.get("miscellaneous_expenses")
            + app_state.get("depreciation_expense")
            + app_state.get("marketing_expenses")
            + app_state.get("insurance_expenses")
            + app_state.get("taxes")
            + app_state.get("utilities_expenses")
            + app_state.get("maintenance_expenses")
        )
        difference = one_time_sum2 - monthly_sum2
        st.sidebar.write("## Углублённый отчёт")
        st.sidebar.write(f"Единовременные vs. ежемесячные, разница: {difference:,.2f} руб.")
    
    if enable_ml_settings_val and forecast_method_sel.startswith("ML"):
       if st.session_state.get("uploaded_model") is not None:
             st.sidebar.write(f"Загруженная модель: {st.session_state.get('uploaded_model').name}")
       if st.session_state.get("df_for_ml") is not None:
             st.sidebar.write(f"Загруженный датасет: {st.session_state.get('df_for_ml').name if hasattr(st.session_state.get('df_for_ml'),'name') else 'неизвестно'}")
forecast_method = app_state.get("forecast_method") or "Базовый"
params = WarehouseParams(
    total_area=app_state.get("total_area"),
    rental_cost_per_m2=app_state.get("rental_cost_per_m2"),
    useful_area_ratio=app_state.get("useful_area_ratio"),
    mode=app_state.get("mode") or "Ручной",
    storage_share=app_state.shares["storage_share"],
    loan_share=app_state.shares["loan_share"],
    vip_share=app_state.shares["vip_share"],
    short_term_share=app_state.shares["short_term_share"],
    storage_area_manual=app_state.get("storage_area_manual"),
    loan_area_manual=app_state.get("loan_area_manual"),
    vip_area_manual=app_state.get("vip_area_manual"),
    short_term_area_manual=app_state.get("short_term_area_manual"),
    storage_fee=app_state.get("storage_fee"),
    shelves_per_m2=app_state.get("shelves_per_m2"),
    short_term_daily_rate=app_state.get("short_term_daily_rate"),
    vip_extra_fee=app_state.get("vip_extra_fee"),
    item_evaluation=app_state.get("item_evaluation"),
    item_realization_markup=app_state.get("item_realization_markup"),
    average_item_value=app_state.get("average_item_value"),
    loan_interest_rate=app_state.get("loan_interest_rate"),
    loan_term_days=app_state.get("loan_term_days"),
    realization_share_storage=app_state.get("realization_share_storage"),
    realization_share_loan=app_state.get("realization_share_loan"),
    realization_share_vip=app_state.get("realization_share_vip"),
    realization_share_short_term=app_state.get("realization_share_short_term"),
    storage_items_density=app_state.get("storage_items_density"),
    loan_items_density=app_state.get("loan_items_density"),
    vip_items_density=app_state.get("vip_items_density"),
    short_term_items_density=app_state.get("short_term_items_density"),
    storage_fill_rate=app_state.get("storage_fill_rate"),
    loan_fill_rate=app_state.get("loan_fill_rate"),
    vip_fill_rate=app_state.get("vip_fill_rate"),
    short_term_fill_rate=app_state.get("short_term_fill_rate"),
    storage_monthly_churn=app_state.get("storage_monthly_churn"),
    loan_monthly_churn=app_state.get("loan_monthly_churn"),
    vip_monthly_churn=app_state.get("vip_monthly_churn"),
    short_term_monthly_churn=app_state.get("short_term_monthly_churn"),
    salary_expense=app_state.get("salary_expense"),
    miscellaneous_expenses=app_state.get("miscellaneous_expenses"),
    depreciation_expense=app_state.get("depreciation_expense"),
    marketing_expenses=app_state.get("marketing_expenses"),
    insurance_expenses=app_state.get("insurance_expenses"),
    taxes=app_state.get("taxes"),
    utilities_expenses=app_state.get("utilities_expenses"),
    maintenance_expenses=app_state.get("maintenance_expenses"),
    one_time_setup_cost=app_state.get("one_time_setup_cost"),
    one_time_equipment_cost=app_state.get("one_time_equipment_cost"),
    one_time_other_costs=app_state.get("one_time_other_costs"),
    one_time_legal_cost=app_state.get("one_time_legal_cost"),
    one_time_logistics_cost=app_state.get("one_time_logistics_cost"),
    time_horizon=app_state.get("time_horizon"),
    monthly_rent_growth=app_state.get("monthly_rent_growth"),
    default_probability=app_state.get("default_probability"),
    liquidity_factor=app_state.get("liquidity_factor"),
    safety_factor=app_state.get("safety_factor"),
    loan_grace_period=app_state.get("loan_grace_period"),
    monthly_income_growth=app_state.get("monthly_income_growth"),
    monthly_expenses_growth=app_state.get("monthly_expenses_growth"),
    forecast_method=forecast_method,
    monte_carlo_simulations=app_state.get("monte_carlo_simulations"),
    monte_carlo_deviation=app_state.get("monte_carlo_deviation"),
    monte_carlo_seed=app_state.get("monte_carlo_seed"),
    enable_ml_settings=app_state.get("enable_ml_settings"),
    electricity_cost_per_m2=app_state.get("electricity_cost_per_m2"),
    monthly_inflation_rate=app_state.get("monthly_inflation_rate"),
    monthly_salary_growth=app_state.get("monthly_salary_growth"),
    monthly_other_expenses_growth=app_state.get("monthly_other_expenses_growth"),
    packaging_cost_per_m2=app_state.get("packaging_cost_per_m2"),
    poly_degree=poly_degree,
    n_estimators=n_estimators,
    features=features,
    monte_carlo_distribution=app_state.get("monte_carlo_distribution"),
    monte_carlo_normal_mean=app_state.get("monte_carlo_normal_mean"),
    monte_carlo_normal_std=app_state.get("monte_carlo_normal_std"),
    monte_carlo_triang_left=app_state.get("monte_carlo_triang_left"),
    monte_carlo_triang_mode=app_state.get("monte_carlo_triang_mode"),
    monte_carlo_triang_right=app_state.get("monte_carlo_triang_right"),
    auto_feature_selection=auto_feature_selection,
    param_search_method=app_state.get("param_search_method")
)
amortize_one_time_expenses = app_state.get("amortize_one_time_expenses")
is_valid, error_message = validate_inputs(params)
if not is_valid:
    st.error(f"Ошибка ввода: {error_message}")
else:
    areas = calculate_areas(params)
    for k, v in areas.items():
        setattr(params, k, v)

    st.markdown("---")
    st.markdown("### Выберите вкладку:")
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Общие результаты"

    tabs_list = [
        "Общие результаты",
        "Прогнозирование",
        "Точка безубыточности",
        "Детализация",
         "Сценарный анализ",
         "Диагностика",
         "Документация",
         "Дашборд"
    ]
    chosen = st.radio(
        "Вкладки:",
        tabs_list,
        index=tabs_list.index(st.session_state.active_tab),
        key="active_tab",
        horizontal=True
    )

    if chosen == "Общие результаты":
        base_financials = calculate_financials(params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        irr_val = calculate_irr(
            [
                -params.one_time_setup_cost - params.one_time_equipment_cost
                - params.one_time_other_costs - params.one_time_legal_cost
                - params.one_time_logistics_cost
            ]
            + [base_financials["profit"]] * params.time_horizon
        )
        pm, pr = calculate_additional_metrics(
            base_financials["total_income"],
            base_financials["total_expenses"],
            base_financials["profit"]
        )
        roi_val = calculate_roi(base_financials["total_income"], base_financials["total_expenses"])
        npv_val = calculate_npv(
            [
                -params.one_time_setup_cost - params.one_time_equipment_cost
                - params.one_time_other_costs - params.one_time_legal_cost
                - params.one_time_logistics_cost
            ]
            + [base_financials["profit"]] * params.time_horizon,
            0.05
        )
        tab_container = st.container()
        display_tab1(
            tab_container,
            base_financials,
            pm,
            pr,
            roi_val,
            irr_val,
            params,
            selected_param=None,
            main_color=app_state.get("main_color") or "#007bff",
            npv=npv_val
        )

    elif chosen == "Прогнозирование":
        tab_container = st.container()
        base_financials = calculate_financials(params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        display_tab2_header(tab_container)

        if params.forecast_method == "Базовый":
            display_tab2_basic_forecast(tab_container, base_financials, params)
        elif params.forecast_method.startswith("ML"):
            st.subheader("Управление ML-моделью")
            if st.button("Обучить модель"):
                if df_for_ml is not None:
                    try:
                        with st.spinner("Обучение модели..."):
                            new_model = train_ml_model(
                                df_for_ml,
                                target_column="Доходы",
                                model_type=params.forecast_method,
                                poly_degree=params.poly_degree,
                                n_estimators=params.n_estimators,
                                features=params.features,
                                param_search_method=params.param_search_method,
                                auto_feature_selection=params.auto_feature_selection
                            )
                            st.session_state["ml_model"] = new_model
                            st.success("Модель обучена!")
                    except Exception as e:
                        st.error(f"Ошибка обучения: {e}")
                else:
                    st.warning("Нет данных для обучения.")

            if st.session_state.get("ml_model") is not None:
                save_trained_model_to_file(
                    st.session_state["ml_model"],
                    filename="trained_model.pkl"
                )

            display_tab2_ml_forecast(
                tab_container,
                params.enable_ml_settings,
                selected_forecast_method,
                st.session_state.get("ml_model"),
                df_for_ml,
                params,
                base_financials
            )
        elif params.forecast_method == "Симуляция Монте-Карло":
            display_tab2_monte_carlo(tab_container, base_financials, params)

    elif chosen == "Точка безубыточности":
        tab_container = st.container()
        base_financials = calculate_financials(params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        display_tab3_header(tab_container)
        display_tab3_bep_info(tab_container, base_financials, params)
        display_tab3_monthly_bep(tab_container, base_financials, params)
        display_tab3_sensitivity(tab_container, params, disable_extended=False, help_texts=help_texts)

    elif chosen == "Детализация":
        tab_container = st.container()
        base_financials = calculate_financials(params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        irr_val = calculate_irr(
            [
                -params.one_time_setup_cost - params.one_time_equipment_cost
                - params.one_time_other_costs - params.one_time_legal_cost
                - params.one_time_logistics_cost
            ]
            + [base_financials["profit"]] * params.time_horizon
        )
        display_tab4_header(tab_container)
        display_tab4_area_metrics(tab_container, params)
        display_tab4_storage_table(tab_container, params, base_financials)
        display_tab4_profit_table(tab_container, params, base_financials)
        display_tab4_results(tab_container, base_financials, params, irr_val)

    elif chosen == "Сценарный анализ":
        st.header("Сценарный анализ (Гибкие настройки)")
        import copy

        st.write("Укажите изменения для Оптимистичного и Пессимистичного сценария:")

        with st.expander("Оптимистичный сценарий", expanded=True):
            opt_storage_fee_pct = st.slider(
                "Изменение тарифа хранения (%)",
                -50, 50, 20, step=5
            )
            opt_loan_rate_pct = st.slider(
                "Изменение ставки займов (%)",
                -50, 50, -20, step=5
            )
            opt_default_prob_pct = st.slider(
                "Изменение default_probability (%)",
                -50, 100, -20, step=10
            )

        with st.expander("Пессимистичный сценарий", expanded=True):
            pes_storage_fee_pct = st.slider(
                "Изменение тарифа хранения (%)",
                -50, 50, -20, step=5,
                key="pes_storage_fee_pct"
            )
            pes_loan_rate_pct = st.slider(
                "Изменение ставки займов (%)",
                -50, 50, 20, step=5,
                key="pes_loan_rate_pct"
            )
            pes_default_prob_pct = st.slider(
                "Изменение default_probability (%)",
                -50, 100, 30, step=10,
                key="pes_default_prob_pct"
            )

        st.write("Реалистичный сценарий остаётся без изменений.")

        def apply_changes(base_params, fee_pct, loan_pct, def_pct):
            p = copy.deepcopy(base_params)
            p.storage_fee *= (1 + fee_pct / 100.0)
            p.loan_interest_rate *= (1 + loan_pct / 100.0)
            factor = (1 + def_pct / 100.0)
            factor = max(factor, 0)
            p.default_probability *= factor
            return p

        opt_params = apply_changes(params, opt_storage_fee_pct, opt_loan_rate_pct, opt_default_prob_pct)
        pes_params = apply_changes(params, pes_storage_fee_pct, pes_loan_rate_pct, pes_default_prob_pct)
        real_params = copy.deepcopy(params)

        fin_opt = calculate_financials(opt_params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        fin_pes = calculate_financials(pes_params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        fin_real = calculate_financials(real_params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)

        scenarios_data = [
            {
                "Сценарий": "Оптимистичный",
                "Доход": fin_opt["total_income"],
                "Расход": fin_opt["total_expenses"],
                "Прибыль": fin_opt["profit"]
            },
            {
                "Сценарий": "Реалистичный",
                "Доход": fin_real["total_income"],
                "Расход": fin_real["total_expenses"],
                "Прибыль": fin_real["profit"]
            },
            {
                "Сценарий": "Пессимистичный",
                "Доход": fin_pes["total_income"],
                "Расход": fin_pes["total_expenses"],
                "Прибыль": fin_pes["profit"]
            }
        ]
        df_scenarios = pd.DataFrame(scenarios_data)
        st.dataframe(
            df_scenarios.style.format(
                {
                    "Доход": "{:,.2f}",
                    "Расход": "{:,.2f}",
                    "Прибыль": "{:,.2f}"
                }
            )
        )

        st.write("Сравнительный график для трёх сценариев:")
        from streamlit_ui import ChartDisplay
        df_plot = df_scenarios.melt(
            id_vars="Сценарий",
            value_vars=["Доход", "Расход", "Прибыль"],
            var_name="Показатель",
            value_name="Значение"
        )
        chart_scen = ChartDisplay("Сравнение: Доход, Расход, Прибыль", x_title="Сценарий", y_title="Рубли")
        chart_scen.display_bar(df_plot, x="Сценарий", y="Значение", color="Показатель")

        st.info("Пользователь сам выбирает процентные изменения для каждого сценария.")

    elif chosen == "Диагностика":
        st.header("Диагностика и отладка")

        base_fin = calculate_financials(params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)

        st.subheader("Площади")
        st.write(f"Полезная площадь: {params.usable_area:.2f} м²")
        st.write(f"Простое: {params.storage_area:.2f} м²")
        st.write(f"Займы: {params.loan_area:.2f} м²")
        st.write(f"VIP: {params.vip_area:.2f} м²")
        st.write(f"Краткосрочное: {params.short_term_area:.2f} м²")

        st.subheader("Доход по статьям")
        st.write(f"Простое: {base_fin['storage_income']:.2f} руб.")
        st.write(f"VIP: {base_fin['vip_income']:.2f} руб.")
        st.write(f"Краткосрочное: {base_fin['short_term_income']:.2f} руб.")
        st.write(f"Займы: {base_fin['loan_income']:.2f} руб.")
        st.write(f"Реализация: {base_fin['realization_income']:.2f} руб.")

        st.subheader("Расходы по статьям")
        monthly_rent = params.total_area * params.rental_cost_per_m2
        st.write(f"Аренда: {monthly_rent:.2f} руб.")
        packaging_cost = params.total_area * params.packaging_cost_per_m2
        st.write(f"Упаковка: {packaging_cost:.2f} руб.")
        electricity_cost = params.total_area * params.electricity_cost_per_m2
        st.write(f"Электричество: {electricity_cost:.2f} руб.")
        st.write(f"Зарплата: {params.salary_expense:.2f} руб.")
        st.write(f"Прочие: {params.miscellaneous_expenses:.2f} руб.")
        st.write(f"Амортизация: {params.depreciation_expense:.2f} руб.")
        st.write(f"Маркетинг: {params.marketing_expenses:.2f} руб.")
        st.write(f"Страхование: {params.insurance_expenses:.2f} руб.")
        st.write(f"Налоги: {params.taxes:.2f} руб.")
        st.write(f"Коммуналка: {params.utilities_expenses:.2f} руб.")
        st.write(f"Обслуживание: {params.maintenance_expenses:.2f} руб.")
        st.write(f"Итого расходов: {base_fin['total_expenses']:.2f} руб.")

        st.subheader("Промежуточная прибыль")
        income_minus_expenses = base_fin["total_income"] - base_fin["total_expenses"]
        st.write(f"Прибыль до учёта единовременных: {income_minus_expenses:.2f} руб.")
        st.write(f"Окончательная прибыль (с учётом единовременных расходов): {base_fin['profit']:.2f} руб.")
        st.info(
            "Эти данные помогают разобраться, на каком этапе могут возникать несостыковки "
            "и проверить корректность введённых параметров."
        )
    
    elif chosen == "Документация":
        st.header("Документация")
        
        with open("help.md", "r", encoding="utf-8") as f:
            help_text = f.read()

        # Поиск по документации
        search_term = st.text_input("Поиск в документации", "")
        if search_term:
                search_results = [line for line in help_text.splitlines() if search_term.lower() in line.lower()]
                if search_results:
                    for result in search_results:
                        st.markdown(result, unsafe_allow_html=True)
                else:
                    st.info("По вашему запросу ничего не найдено.")
        else:
             st.markdown(help_text, unsafe_allow_html=True)
        
        with st.expander("Часто задаваемые вопросы (FAQ)"):
             st.markdown("""
                **Q: Что такое точка безубыточности (BEP)?**
                A: Точка безубыточности — это уровень дохода, при котором общая прибыль становится равной общим расходам.

                **Q: Как интерпретировать отрицательное значение NPV?**
                A: Отрицательное значение NPV говорит о том, что текущая стоимость будущих денежных потоков не покрывает текущие инвестиции и проект может быть невыгоден.

                **Q: Как использовать режим автоматического распределения площади?**
                A: Укажите доли для каждого типа хранения (простое, займы, VIP, краткосрочное) , а остальное приложение рассчитает само.

                **Q: Какой тип распределения выбрать в методе Монте-Карло?**
                A: Выбор распределения зависит от ваших предположений о природе данных. Если у вас нет предпочтений, то используйте равномерное распределение. Нормальное распределение подходит для тех случаев, где ожидается, что большинство отклонений будут близки к среднему значению. Треугольное подойдет, когда есть минимальное, модальное (наиболее вероятное), и максимальное значения.

                **Q: Что такое RMSE, R² и MAE в ML-прогнозировании?**
                  A: RMSE - корень среднеквадратичной ошибки, оценивает среднеквадратическое отклонение прогноза от фактических значений. R² - коэффициент детерминации, показывает, как хорошо модель описывает данные. MAE - средняя абсолютная ошибка, оценивает среднее отклонение прогноза от фактических значений по модулю.

                **Q: Что делать, если ML-модель выдает плохие прогнозы?**
                A: Проверьте качество и достаточность данных, которые вы загрузили для обучения ML-модели. Попробуйте использовать другие признаки, или другой метод прогнозирования.

                **Q: Можно ли использовать приложение для разных видов складов?**
                A: Да, но нужно вводить параметры, которые соответствуют вашему типу склада, а не брать их "из воздуха".

                 **Q: Как влияют коэффициенты ликвидности и запаса на точку безубыточности?**
                A: Коэффициент ликвидности корректирует BEP на основе ликвидности активов, делая расчёт более консервативным, если активы могут быть быстро превращены в денежные средства. Коэффициент запаса — увеличивает расходы для обеспечения безопасности, создавая резерв.

                **Q: Почему важно амортизировать единовременные расходы?**
                A: Амортизация позволяет более равномерно распределить единовременные расходы по всему периоду прогноза, что дает более точную картину прибыльности в каждый месяц, и сглаживает колебания прибыльности.

                **Q: Как работают различные методы прогнозирования?**
                  A: Базовый метод предполагает линейный рост доходов и расходов. Модели ML пытаются уловить зависимость в данных. Монте-Карло моделирует различные сценарии случайных отклонений.

                **Q: Как использовать функцию сравнения параметров?**
                  A: Вы можете сохранить текущие настройки (кнопка "Сохранить параметры") и затем сравнить их с другими, ранее сохранёнными, что бы понять, какие именно параметры вы изменили, и как это влияет на результат. Это позволяет проследить динамику изменения параметров.

                **Q: Что такое "параметрический поиск" в ML моделях?**
                A: Это метод автоматического подбора оптимальных значений параметров (гиперпараметров) для ML-модели с помощью GridSearchCV или RandomizedSearchCV. Это повышает качество прогноза. GridSearchCV — это полный перебор, RandomizedSearchCV — случайный перебор, который позволяет найти хорошие параметры за меньшее время.

                 **Q: Что такое "автоматический выбор признаков" в ML моделях?**
                A: Это процесс, когда модель автоматически выбирает наиболее важные признаки для обучения, игнорируя менее значимые. Используется для того, чтобы упростить модель и повысить точность прогноза. Признаки (фичи) генерируются автоматически из данных, которые вы загрузили.

                 **Q: Как работает коэффициент дисконтирования в NPV?**
                 A: Коэффициент дисконтирования в NPV отражает стоимость денег во времени. Будущие денежные потоки дисконтируются (т.е. их текущая стоимость уменьшается) на величину, соответствующую ставке дисконтирования.

                 **Q: Как можно интерпретировать значения в таблице чувствительности?**
                  A: Таблица чувствительности показывает, как изменение одного или нескольких параметров влияют на общую прибыль. По таблице видно, какие параметры оказывают наибольшее влияние на финансовый результат.

                 **Q: Что такое "горизонт прогнозирования"?**
                 A: Горизонт прогнозирования – это период времени в месяцах, на который вы делаете прогноз. Чем больше горизонт, тем больше неопределенности в прогнозе.

                 **Q: Как часто нужно обновлять данные в приложении?**
                A: Рекомендуется обновлять данные в приложении не реже раза в месяц, или если произошли существенные изменения в параметрах работы склада, тарифах или экономике.

                 **Q: Зачем нужно "зерно случайности" в Монте-Карло?**
                 A: Значение "seed" позволяет зафиксировать состояние генератора случайных чисел для того, чтобы воспроизводить одни и те же результаты при моделировании. Это необходимо для отладки и сравнения результатов.
                
                **Q: Почему в некоторых графиках видны доверительные интервалы?**
                 A: Доверительные интервалы (диапазоны) на графиках ML-прогноза показывают неопределенность предсказаний. Это помогает понять, где прогноз точный, а где может быть большая погрешность.

             """)
    elif chosen == "Дашборд":
        tab_container = st.container()
        base_financials = calculate_financials(params, disable_extended=False, amortize_one_time_expenses=amortize_one_time_expenses)
        display_tab5_header(tab_container)
        display_tab5_dashboard(tab_container, base_financials, params,  help_texts=help_texts)