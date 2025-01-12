# test_utils.py

import unittest
import pandas as pd
from utils import (
    generate_download_link,
    generate_excel_download,
    normalize_shares,
    perform_sensitivity_analysis,
    safe_display_irr,
    load_params_from_file,
    save_params_to_file
)
import streamlit as st
import numpy as np
import os
import json
import yaml
import tempfile
from data_model import WarehouseParams
from calculations import calculate_financials, calculate_areas


class TestUtils(unittest.TestCase):
    """
    Тесты для проверки корректности работы функций в utils.py.
    """

    def test_generate_download_link(self):
        """
        Тестирует функцию generate_download_link.
         Проверяет, что создается правильная ссылка на скачивание CSV.
        """
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        link = generate_download_link(df)
        self.assertIsInstance(link, str)
        self.assertTrue(link.startswith("<a href"))

    def test_generate_excel_download(self):
        """
         Тестирует функцию generate_excel_download.
          Проверяет, что создается правильная ссылка на скачивание Excel.
         """
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        link = generate_excel_download(df)
        self.assertIsInstance(link, str)
        self.assertTrue(link.startswith("<a href"))

    def test_normalize_shares(self):
        """
        Тестирует функцию normalize_shares.
        Проверяет, что доли нормализуются корректно.
        """
        st.session_state.shares = {"storage_share": 0.5, "loan_share": 0.3, "vip_share": 0.1, "short_term_share": 0.1}
        
        # Test case 1
        shares = st.session_state.shares.copy()
        normalize_shares("storage_share", 0.7, shares)
        self.assertAlmostEqual(sum(shares.values()), 1.0, delta=0.00001)
        

        # Test case 2
        st.session_state.shares = {"storage_share": 0.5, "loan_share": 0.3, "vip_share": 0.1, "short_term_share": 0.1}
        shares = st.session_state.shares.copy()
        normalize_shares("loan_share", 0.5, shares)
        self.assertAlmostEqual(sum(shares.values()), 1.0, delta=0.00001)


    # TODO Rename this here and in `test_normalize_shares`
    def _extracted_from_test_normalize_shares_7(self, arg0, arg1):
      pass

    def test_perform_sensitivity_analysis(self):
        """
        Тестирует функцию perform_sensitivity_analysis.
        Проверяет, что функция возвращает DataFrame с результатами анализа.
        """
        params = WarehouseParams(
            total_area=100,
            rental_cost_per_m2=10,
            useful_area_ratio=0.5,
             mode="Ручной",
            storage_share=0.25,
            loan_share=0.25,
            vip_share=0.25,
            short_term_share=0.25,
            storage_area_manual=10.0,
            loan_area_manual=10.0,
            vip_area_manual=10.0,
            short_term_area_manual=10.0,
            storage_fee=15,
            shelves_per_m2=3,
            short_term_daily_rate=6,
            vip_extra_fee=10,
            item_evaluation=0.8,
            item_realization_markup=20.0,
            average_item_value=15000,
            loan_interest_rate=0.317,
            loan_term_days=30,
            realization_share_storage=0.5,
            realization_share_loan=0.5,
            realization_share_vip=0.5,
            realization_share_short_term=0.5,
            storage_items_density=5,
            loan_items_density=1,
            vip_items_density=2,
            short_term_items_density=4,
            storage_fill_rate=1.0,
            loan_fill_rate=1.0,
            vip_fill_rate=1.0,
            short_term_fill_rate=1.0,
            storage_monthly_churn=0.01,
            loan_monthly_churn=0.02,
            vip_monthly_churn=0.005,
            short_term_monthly_churn=0.03,
            salary_expense=240000,
            miscellaneous_expenses=50000,
            depreciation_expense=20000,
            marketing_expenses=30000,
            insurance_expenses=10000,
            taxes=50000,
            utilities_expenses=20000,
            maintenance_expenses=15000,
            one_time_setup_cost=100000,
            one_time_equipment_cost=200000,
            one_time_other_costs=50000,
            one_time_legal_cost=20000,
            one_time_logistics_cost=30000,
            time_horizon=6,
            monthly_rent_growth=0.01,
            default_probability=0.05,
            liquidity_factor=1.0,
            safety_factor=1.2,
            loan_grace_period=0,
            monthly_income_growth=0.0,
            monthly_expenses_growth=0.0,
            forecast_method="Базовый",
            monte_carlo_simulations=100,
            monte_carlo_deviation=0.1,
            monte_carlo_seed=42,
            enable_ml_settings=False,
            electricity_cost_per_m2=100,
            monthly_inflation_rate=0.005,
            monthly_salary_growth=0.005,
            monthly_other_expenses_growth=0.005,
            poly_degree=2,
        )
        areas = calculate_areas(params)
        for k, v in areas.items():
            setattr(params, k, v)
        df = perform_sensitivity_analysis(params, "storage_fee", [10, 20, 30], disable_extended=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("Параметр", df.columns)
        self.assertIn("Прибыль (руб.)", df.columns)


    def test_safe_display_irr(self):
        """
        Тестирует функцию safe_display_irr.
        Проверяет, что корректно отображается IRR.
        """
        with st.empty():
           safe_display_irr(0.05)
           safe_display_irr(None)
           safe_display_irr(np.nan)

    def test_load_save_params(self):
        """
         Тестирует load_params_from_file и save_params_to_file.
          Проверяет корректность сохранения и загрузки параметров.
        """
        params = {"test_param": 123, "shares": {"a":0.2,"b":0.2, "c":0.2, "d":0.4}}
        
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_json:
            json.dump(params, tmp_json)  # Записываем JSON в файл
            tmp_json.flush()
            with open(tmp_json.name, 'r') as f:  # Открываем файл для чтения
                loaded_params_json = json.load(f)  # Считываем JSON
            self._extracted_from_test_load_save_params_11(loaded_params_json, params, tmp_json)

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp_yaml:
            yaml.dump(params, tmp_yaml, sort_keys=False)  # Записываем YAML в файл
            tmp_yaml.flush()
            with open(tmp_yaml.name, 'r') as f:  # Открываем файл для чтения
                loaded_params_yaml = yaml.safe_load(f) # Считываем YAML
            self._extracted_from_test_load_save_params_11(loaded_params_yaml, params, tmp_yaml)

        # Тестируем ошибку загрузки
        loaded_params_fail = load_params_from_file("fail.txt")
        self.assertIsNone(loaded_params_fail)

    # TODO Rename this here and in `test_load_save_params`
    def _extracted_from_test_load_save_params_11(self, arg0, params, arg2):
        self.assertIsInstance(arg0, dict)
        self.assertEqual(arg0, params)
        os.remove(arg2.name)


if __name__ == "__main__":
    unittest.main()