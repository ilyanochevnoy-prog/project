import io
import traceback
import pandas as pd
import streamlit as st

from project_backend_v4 import read_excel_any, run_project_analysis, RiskParams


st.set_page_config(
    page_title="Проектный офис — CPM + риск к цели",
    layout="wide",
)

st.title("CPM + риск к выбранной цели (вехе)")
st.caption("Загрузите выгрузку задач (Excel), укажите цель расчёта и ИД целевой вехи. Вывод — рекомендации.")

with st.sidebar:
    st.header("Параметры")
    mode = st.selectbox(
        "Режим длительностей",
        options=["reconstruct_plan", "pure_cpm"],
        index=0,
        help=(
            "reconstruct_plan: длительности считаются из плановых дат (календарные дни). "
            "pure_cpm: длительности берутся из поля 'Длительность'."
        ),
    )

    st.subheader("Пороговые критерии для рекомендаций")
    rp = RiskParams()
    rp.high_impact_prob = st.slider("P(impact) ≥ ...", 0.05, 0.95, float(rp.high_impact_prob), 0.05)
    rp.high_delay_prob = st.slider("p_delay ≥ ...", 0.05, 0.95, float(rp.high_delay_prob), 0.05)
    rp.high_eimpact_days = st.number_input("E(impactDays) ≥ ... (дней)", min_value=0.0, value=float(rp.high_eimpact_days), step=0.5)
    rp.top_n = int(st.number_input("Top-N по E(impactDays)", min_value=3, max_value=50, value=int(rp.top_n), step=1))

st.divider()

col1, col2 = st.columns([2, 1], vertical_alignment="top")

with col1:
    goal_text = st.text_area(
        "Цель проекта / цель расчёта (обязательное поле)",
        placeholder="Напр.: Ввести объект в эксплуатацию / Достичь готовности оборудования / Завершить этап ...",
        height=90,
    )

with col2:
    goal_id = st.text_input(
        "ИД целевой вехи (обязательное поле)",
        placeholder="Напр.: 75",
        help="Введите ИД задачи-вехи из файла. Автоматический выбор цели отключён: без ИД расчёт не выполняется.",
    )

uploaded = st.file_uploader(
    "Загрузите файл (Excel: .xlsx/.xls). MPP пока не поддерживается напрямую — требуется выгрузка в Excel.",
    type=["xlsx", "xls", "xlsm", "mpp"],
)

run_btn = st.button("Рассчитать", type="primary", use_container_width=True)

def _as_int(x: str):
    x = (x or "").strip()
    if not x:
        return None
    try:
        return int(float(x))
    except Exception:
        return None

if run_btn:
    # Валидация
    gid = _as_int(goal_id)
    if not goal_text or str(goal_text).strip() == "" or gid is None:
        st.error("Расчёт не выполнен: заполните цель (текст) и ИД целевой вехи.")
        st.stop()

    if uploaded is None:
        st.error("Расчёт не выполнен: загрузите файл Excel с выгрузкой задач.")
        st.stop()

    if uploaded.name.lower().endswith(".mpp"):
        st.error(
            "MPP-файл загружен, но прямое чтение MPP в этом прототипе не поддержано.\n\n"
            "Решение: в MS Project выполните экспорт в Excel (таблица задач с предшественниками/датами) "
            "и загрузите .xlsx."
        )
        st.stop()

    try:
        # читаем Excel из памяти
        data = io.BytesIO(uploaded.getvalue())
        df = read_excel_any(data, sheet=0)

        result = run_project_analysis(
            df_raw=df,
            goal_id=gid,
            goal_name_text=str(goal_text).strip(),
            mode=mode,
            risk_params=rp,
            build_diagnostics_tables=False,  # UI выводит только рекомендации
        )

        st.success("Расчёт выполнен.")

        # вывод рекомендаций как таблицы (одна строка = один пункт)
        rec_df = pd.DataFrame({"Рекомендации": result.recommendations})
        st.subheader("Рекомендации")
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        # предупреждения — отдельно
        with st.expander("Технические предупреждения и допущения модели", expanded=False):
            st.write("**Warnings:**")
            for w in result.warnings:
                st.write(f"- {w}")

        # download
        txt = "\n".join(result.recommendations)
        st.download_button(
            label="Скачать рекомендации (TXT)",
            data=txt.encode("utf-8"),
            file_name="recommendations.txt",
            mime="text/plain",
            use_container_width=True,
        )

    except Exception as e:
        st.error("Ошибка при обработке файла или расчёте.")
        st.code(traceback.format_exc())
