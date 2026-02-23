"""
Проектный бэкэнд для календарно‑сетевого анализа (CPM) + расчёта риска по задачам.

Данный модуль рассчитан на работу с выгрузками MS Project/Project Server (и совместимых),
но допускает вариации имён колонок (например: Ид/ИД, Название/Название_задачи, Начало/Дата_начала).

Ключевые требования (зафиксированы по ТЗ пользователя):
- Целевая веха задаётся пользователем по ИД (duration = 0). Автовыбор цели НЕ выполняется.
- Если не задана целевая веха ИЛИ текст цели — расчёт не выполняется (ошибка валидации).
- Единица времени: календарные дни (включая дробные через время суток).
- Constraints учитываются в CPM (минимум: "Как можно раньше", "Начало не ранее", "Окончание не ранее").
- Для задач с пустыми предшественниками:
    * отсутствие предшественников НЕ считается отклонением для вех (D=0) и суммарных задач (в них логика обычно задаётся дочерними задачами),
    * для листовых НЕ‑вех задач без предшественников связи автоматически НЕ добавляются (политика обработки),
      такие задачи считаются стартовыми внутри своего раздела/подраздела; риск оценивается по их разделу и последователям.
- Связи на суммарные задачи раскрываются автоматически на уровень листовых задач.
- Риск‑модель: вероятность задержки p_delay + средняя задержка mu_delay (в днях),
  далее риск влияния на цель (R2): P(Δ > slack_to_goal) и E[(Δ - slack_to_goal)^+]
  при смеси: Δ=0 с вероятностью (1-p), иначе Δ ~ Exp(mean=mu).
- Учитывается ситуация "последователь выполнен, предшественник нет" (out‑of‑sequence):
    * формируется рекомендация,
    * риск по "сомнительно завершённой" задаче не обнуляется (ре‑ворк/перепроверка),
      и передаётся дальше по сети.

Выход:
- ProjectRunResult с перечнем рекомендаций + тех. предупреждениями.
  (Таблица остаётся как опциональный диагностический артефакт, UI может её не показывать.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Iterable, Any
import math
import re
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd
import numpy as np

# --------------------------
# Парсинг русских дат/длительностей
# --------------------------

_RU_MONTHS = {
    'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4, 'Май': 5, 'Июнь': 6,
    'Июль': 7, 'Август': 8, 'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12
}
_RU_MONTHS_GEN = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
    'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
}
_RU_WEEKDAYS_SHORT = {'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'}


def parse_ru_datetime(val: Any) -> Optional[datetime]:
    """Парсинг дат из форматов:
    - '01 Сентябрь 2025 8:00'
    - 'Пн 01.09.25'
    - '01.09.2025'
    - pandas Timestamp
    Возвращает datetime или None.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if s in ('', 'НД', 'nan', 'NaT'):
        return None

    m = re.match(r'(\d{1,2})\s+([А-Яа-яA-Za-z]+)\s+(\d{4})\s+(\d{1,2}):(\d{2})', s)
    if m:
        day = int(m.group(1))
        mon_name = m.group(2)
        year = int(m.group(3))
        hour = int(m.group(4))
        minute = int(m.group(5))
        mon = _RU_MONTHS.get(mon_name, _RU_MONTHS_GEN.get(mon_name.lower()))
        if mon is None:
            raise ValueError(f"Unknown month '{mon_name}' in '{s}'")
        return datetime(year, mon, day, hour, minute)

    parts = s.split()
    if len(parts) == 2 and parts[0] in _RU_WEEKDAYS_SHORT:
        s2 = parts[1]
    else:
        s2 = s

    m = re.match(r'(\d{2})\.(\d{2})\.(\d{2,4})(?:\s+(\d{1,2}):(\d{2}))?$', s2)
    if m:
        day = int(m.group(1))
        mon = int(m.group(2))
        year = int(m.group(3))
        if year < 100:
            year += 2000
        hour = int(m.group(4) or 0)
        minute = int(m.group(5) or 0)
        return datetime(year, mon, day, hour, minute)

    # pandas Timestamp fallback
    try:
        ts = pd.to_datetime(val)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception as e:
        raise ValueError(f"Cannot parse datetime '{s}'") from e


def parse_ru_duration_days(val: Any) -> float:
    """Парсинг длительностей вида:
    - '210,13 дней'
    - '47,13д'
    - '0д'
    - '7 дней'
    Возвращает float (дни).
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    s = str(val).strip()
    if s in ('', 'НД', 'nan'):
        return 0.0
    m = re.match(r'(-?\d+(?:[.,]\d+)?)\s*д', s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', '.'))
    m = re.match(r'(-?\d+(?:[.,]\d+)?)', s)
    if m:
        return float(m.group(1).replace(',', '.'))
    return 0.0


def parse_percent_complete_series(series: pd.Series) -> pd.Series:
    """Нормализует % завершения в диапазон 0..100.

    В выгрузках Project часто встречается 0..1 (доля), а не 0..100.
    Также возможны строки '49%' / '49,0%'.

    Правило:
    - парсим в float,
    - если максимальное значение <= 1.0 (с небольшим запасом), считаем что это доля и умножаем на 100.
    """
    if series is None:
        return series
    s = series.copy()

    def _to_float(x: Any) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float('nan')
        xs = str(x).strip()
        if xs in ('', 'НД', 'nan'):
            return float('nan')
        xs = xs.replace('%', '').replace(',', '.')
        try:
            return float(xs)
        except Exception:
            return float('nan')

    vals = s.map(_to_float)
    mx = np.nanmax(vals.values) if np.isfinite(vals.values).any() else float('nan')
    if np.isfinite(mx) and mx <= 1.0001:
        vals = vals * 100.0
    # clamp
    vals = vals.clip(lower=0.0, upper=100.0)
    return vals


def parse_predecessors(val: Any) -> List[Tuple[int, str, float, str]]:
    """Парсер поля 'Предшественники'.
    Возвращает список: (pred_id, rel_type, lag_days, raw_token),
    где rel_type ∈ {'FS','SS'}.
    Поддержано: суффикс 'НН' => SS. Иначе по умолчанию FS.
    Лаги вида '+2д'/'-1д' сейчас игнорируются (0), но заготовка оставлена.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if s in ('', 'НД', 'nan'):
        return []
    tokens = re.split(r'[;,]\s*', s)
    res: List[Tuple[int, str, float, str]] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        rel = 'FS'
        lag = 0.0
        # detect 'НН' suffix
        m = re.match(r'^(\d+)\s*(НН)?\s*$', tok, flags=re.IGNORECASE)
        if m:
            pid = int(m.group(1))
            if m.group(2):
                rel = 'SS'
            res.append((pid, rel, lag, tok))
            continue
        # more complex patterns can be added
        m = re.match(r'^(\d+)', tok)
        if m:
            pid = int(m.group(1))
            res.append((pid, rel, lag, tok))
    return res


# --------------------------
# Нормализация колонок
# --------------------------

_CANON_MAP = {
    "ИД": ["ИД", "Ид", "ID", "Id", "Task ID", "TaskID"],
    "Название_задачи": ["Название_задачи", "Название", "Имя", "Task Name", "Name"],
    "Уровень_структуры": ["Уровень_структуры", "Уровень", "Outline Level", "Уровень структуры"],
    "Суммарная_задача": ["Суммарная_задача", "Суммарная задача", "Сводная задача", "Summary", "Суммарная"],
    "Веха": ["Веха", "Milestone"],
    "Длительность": ["Длительность", "Duration"],
    "Предшественники": ["Предшественники", "Предшественник", "Predecessors", "Predecessor"],
    "Дата_начала": ["Дата_начала", "Начало", "Start", "Start Date"],
    "Дата_окончания": ["Дата_окончания", "Окончание", "Finish", "Finish Date"],
    "Тип_ограничения": ["Тип_ограничения", "Тип ограничения", "Constraint Type"],
    "Дата_ограничения": ["Дата_ограничения", "Дата ограничения", "Constraint Date"],
    "Базовое_начало": ["Базовое_начало", "Базовое начало", "Baseline Start"],
    "Базовое_окончание": ["Базовое_окончание", "Базовое окончание", "Baseline Finish"],
    "СДР": ["СДР", "WBS", "СДР код"],
    "Текст1": ["Текст1", "Ответственный", "Owner", "Исполнитель"],
    "Процент_завершения": ["Процент_завершения", "Процент завершения", "% завершения", "Percent Complete"],
    # optional
    "Последователи": ["Последователи", "Successors"],
}


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Переименовывает колонки в канонические имена, если находит эквиваленты."""
    col_map: Dict[str, str] = {}
    cols = list(df.columns)

    # quick access map by normalized string
    norm = {str(c).strip(): c for c in cols}

    for canon, variants in _CANON_MAP.items():
        for v in variants:
            if v in norm:
                col_map[norm[v]] = canon
                break

    out = df.rename(columns=col_map).copy()

    # нормализуем % завершения на уровне df (если есть)
    if "Процент_завершения" in out.columns:
        out["Процент_завершения"] = parse_percent_complete_series(out["Процент_завершения"])

    return out, col_map


# --------------------------
# Модели данных
# --------------------------

@dataclass
class Task:
    id: int
    name: str
    duration_days: float

    is_milestone: bool
    is_summary: bool
    level: int

    wbs: Optional[str] = None
    constraint_type: str = "Как можно раньше"
    constraint_date: Optional[datetime] = None

    planned_start: Optional[datetime] = None
    planned_finish: Optional[datetime] = None
    baseline_start: Optional[datetime] = None
    baseline_finish: Optional[datetime] = None

    owner: Optional[str] = None
    percent_complete: Optional[float] = None  # 0..100

    row_index: int = 0
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class Edge:
    pred: int
    succ: int
    rel: str  # 'FS' or 'SS'
    lag: float = 0.0
    inferred: bool = False
    source: str = "explicit"  # 'explicit'|'expanded'|'inferred'


# --------------------------
# Вспомогательные функции и иерархия
# --------------------------

def _build_outline_parent(df: pd.DataFrame,
                          level_col: str = 'Уровень_структуры',
                          id_col: str = 'ИД') -> Tuple[Dict[int, Optional[int]], List[int]]:
    stack: List[Tuple[int, int]] = []  # (level, task_id)
    parent: Dict[int, Optional[int]] = {}
    order: List[int] = []
    for idx, row in df.iterrows():
        tid = int(row[id_col])
        lvl = int(row[level_col])
        order.append(tid)

        while stack and stack[-1][0] >= lvl:
            stack.pop()

        parent[tid] = stack[-1][1] if stack else None
        stack.append((lvl, tid))
    return parent, order


def _nearest_summary_parent(tasks: Dict[int, Task], tid: int) -> Optional[int]:
    """Ближайший суммарный предок (по цепочке parent_id)."""
    cur = tasks[tid].parent_id
    while cur is not None:
        if tasks[cur].is_summary:
            return cur
        cur = tasks[cur].parent_id
    return None


def _ancestors_of_goal(goal: int, edges: Iterable[Edge]) -> Set[int]:
    rev = defaultdict(list)
    for e in edges:
        rev[e.succ].append(e.pred)
    anc: Set[int] = set()
    stack = [goal]
    while stack:
        n = stack.pop()
        if n in anc:
            continue
        anc.add(n)
        for p in rev.get(n, []):
            if p not in anc:
                stack.append(p)
    return anc


# --------------------------
# Построение задач из DataFrame
# --------------------------

def build_tasks_from_df(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, Task], List[int], Set[int], Set[int], List[str]]:
    """
    Создает задачи из DataFrame.
    Возвращает:
      df (нормализованный),
      tasks,
      order_ids,
      leaf_ids,
      blank_pred_leaf_ids,
      warnings
    """
    df, rename_map = normalize_columns(df_raw)
    warnings: List[str] = []

    required = ['ИД', 'Название_задачи', 'Уровень_структуры', 'Суммарная_задача', 'Веха',
                'Длительность', 'Предшественники', 'Дата_начала', 'Дата_окончания',
                'Тип_ограничения', 'Дата_ограничения']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    tasks: Dict[int, Task] = {}

    for idx, row in df.iterrows():
        tid = int(row['ИД'])
        name = str(row['Название_задачи'])

        dur = parse_ru_duration_days(row['Длительность'])

        is_summary = str(row['Суммарная_задача']).strip() == 'Да'
        is_milestone = (str(row['Веха']).strip() == 'Да') or (dur == 0.0)

        level = int(row['Уровень_структуры'])

        ctype = str(row['Тип_ограничения']) if not pd.isna(row['Тип_ограничения']) else "Как можно раньше"
        ctype = ctype.strip() if ctype else "Как можно раньше"
        cdate = parse_ru_datetime(row['Дата_ограничения']) if ctype != "Как можно раньше" else None

        ps = parse_ru_datetime(row['Дата_начала'])
        pf = parse_ru_datetime(row['Дата_окончания'])
        bs = parse_ru_datetime(row['Базовое_начало']) if 'Базовое_начало' in df.columns else None
        bf = parse_ru_datetime(row['Базовое_окончание']) if 'Базовое_окончание' in df.columns else None

        owner = None
        if 'Текст1' in df.columns and not pd.isna(row.get('Текст1')):
            owner = str(row.get('Текст1'))

        wbs = None
        if 'СДР' in df.columns and not pd.isna(row.get('СДР')):
            wbs = str(row.get('СДР'))

        pc = None
        if 'Процент_завершения' in df.columns and not pd.isna(row.get('Процент_завершения')):
            try:
                pc = float(row.get('Процент_завершения'))
            except Exception:
                pc = None

        tasks[tid] = Task(
            id=tid,
            name=name,
            duration_days=float(dur),
            is_milestone=is_milestone,
            is_summary=is_summary,
            level=level,
            wbs=wbs,
            constraint_type=ctype,
            constraint_date=cdate,
            planned_start=ps,
            planned_finish=pf,
            baseline_start=bs,
            baseline_finish=bf,
            owner=owner,
            percent_complete=pc,
            row_index=int(idx),
        )

    parent_map, order = _build_outline_parent(df)
    for tid, pid in parent_map.items():
        tasks[tid].parent_id = pid
        if pid is not None and pid in tasks:
            tasks[pid].children.append(tid)

    leaf_ids: Set[int] = {tid for tid, t in tasks.items() if not t.is_summary}

    blank_pred_ids = df.loc[
        df['Предшественники'].isna()
        | (df['Предшественники'].astype(str).str.strip().isin(['', 'НД', 'nan'])),
        'ИД'
    ].astype(int).tolist()
    blank_leaf_ids: Set[int] = {tid for tid in blank_pred_ids if tid in leaf_ids and (not tasks[tid].is_milestone)}

    # предупреждение о переименованиях
    if rename_map:
        warnings.append("Нормализация колонок выполнена (переименованы: " + ", ".join([f"'{k}'→'{v}'" for k, v in rename_map.items()]) + ").")

    return df, tasks, order, leaf_ids, blank_leaf_ids, warnings


def build_raw_edges_from_df(df: pd.DataFrame) -> Tuple[List[Edge], List[str]]:
    edges: List[Edge] = []
    warnings: List[str] = []
    missing_pred_refs = 0

    for idx, row in df.iterrows():
        succ = int(row['ИД'])
        preds = parse_predecessors(row['Предшественники'])
        for pred, rel, lag, rawtok in preds:
            edges.append(Edge(pred=pred, succ=succ, rel=rel, lag=lag, inferred=False, source='explicit'))
    # проверим ссылки на отсутствующие ИД (отложенно — в expand)
    # здесь просто считаем, чтобы выдать предупреждение позже
    return edges, warnings


# --------------------------
# Раскрытие связей на суммарные задачи
# --------------------------

def _leaf_descendants(tasks: Dict[int, Task], root_id: int, memo: Dict[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if root_id in memo:
        return memo[root_id]
    t = tasks[root_id]
    if not t.is_summary:
        memo[root_id] = (root_id,)
        return memo[root_id]
    res: List[int] = []
    for c in t.children:
        res.extend(_leaf_descendants(tasks, c, memo))
    memo[root_id] = tuple(res)
    return memo[root_id]


def _entry_exit_leafs_by_plan_dates(tasks: Dict[int, Task]):
    """
    Возвращает функции entry_leafs(summary_id) и exit_leafs(summary_id).
    Entry/Exit определяются по плановым датам внутри summary:
    - entry: листовые с минимальным planned_start
    - exit: листовые с максимальным planned_finish
    """
    memo_desc: Dict[int, Tuple[int, ...]] = {}

    def leafs(sid: int) -> Tuple[int, ...]:
        return _leaf_descendants(tasks, sid, memo_desc)

    def entry_leafs(sid: int) -> List[int]:
        ls = [lid for lid in leafs(sid) if lid in tasks and not tasks[lid].is_summary]
        if not ls:
            return []
        def key(lid: int):
            ps = tasks[lid].planned_start or datetime.min
            return (ps, tasks[lid].row_index)
        mn = min(ls, key=key)
        mn_ps = tasks[mn].planned_start or datetime.min
        return [lid for lid in ls if (tasks[lid].planned_start or datetime.min) == mn_ps]

    def exit_leafs(sid: int) -> List[int]:
        ls = [lid for lid in leafs(sid) if lid in tasks and not tasks[lid].is_summary]
        if not ls:
            return []
        def key(lid: int):
            pf = tasks[lid].planned_finish or datetime.min
            return (pf, tasks[lid].row_index)
        mx = max(ls, key=key)
        mx_pf = tasks[mx].planned_finish or datetime.min
        return [lid for lid in ls if (tasks[lid].planned_finish or datetime.min) == mx_pf]

    return entry_leafs, exit_leafs


def expand_edges_to_leaf_level(raw_edges: List[Edge], tasks: Dict[int, Task]) -> Tuple[List[Edge], Dict[str, int], List[str]]:
    """Преобразует связи, указанные на суммарные задачи, в связи между листовыми задачами.
    Возвращает: leaf_edges, stats, warnings
    """
    entry_leafs, exit_leafs = _entry_exit_leafs_by_plan_dates(tasks)
    stats = {"explicit_edges": 0, "expanded_edges": 0, "skipped_missing_ref": 0, "milestone_edges": 0}
    warnings: List[str] = []

    expanded: List[Edge] = []
    for e in raw_edges:
        if e.pred not in tasks or e.succ not in tasks:
            stats["skipped_missing_ref"] += 1
            continue

        stats["explicit_edges"] += 1

        if not tasks[e.pred].is_summary:
            pred_set = [e.pred]
        else:
            # FS от суммарной — берём выходные листы; SS — входные листы
            pred_set = exit_leafs(e.pred) if e.rel == 'FS' else entry_leafs(e.pred)

        if not tasks[e.succ].is_summary:
            succ_set = [e.succ]
        else:
            succ_set = entry_leafs(e.succ)

        for p in pred_set:
            for s in succ_set:
                if p == s:
                    continue
                expanded.append(Edge(pred=p, succ=s, rel=e.rel, lag=e.lag,
                                     inferred=False,
                                     source=('expanded' if (tasks[e.pred].is_summary or tasks[e.succ].is_summary) else 'explicit')))
                if tasks[p].is_milestone or tasks[s].is_milestone:
                    stats["milestone_edges"] += 1
                if tasks[e.pred].is_summary or tasks[e.succ].is_summary:
                    stats["expanded_edges"] += 1

    # dedup
    uniq: Dict[Tuple[int, int, str, float], Edge] = {}
    for ed in expanded:
        k = (ed.pred, ed.succ, ed.rel, ed.lag)
        if k not in uniq:
            uniq[k] = ed

    if stats["skipped_missing_ref"] > 0:
        warnings.append(f"Обнаружены ссылки на отсутствующие ИД в предшественниках: пропущено связей {stats['skipped_missing_ref']}. Проверьте исходную выгрузку/фильтры.")

    return list(uniq.values()), stats, warnings


# --------------------------
# Восстановление связей для задач с пустыми предшественниками
# --------------------------

def infer_links_for_blank_predecessors(tasks: Dict[int, Task],
                                       leaf_edges: List[Edge],
                                       blank_leaf_ids: Set[int],
                                       overlap_tolerance_days: float = 0.0) -> Tuple[List[Edge], List[Edge], List[str]]:
    """
    Восстановление недостающих связей (FS) для листовых задач без предшественников.

    Правило (по ТЗ):
    - "предыдущая задача по ИД" в пределах раздела/иерархии.
      Раздел = ближайший суммарный предок.
    - связь добавляется ТОЛЬКО если по плановым датам текущая задача не начинается раньше окончания предыдущей
      (иначе считаем, что задачи/разделы параллельны и связь навязывать нельзя).

    Возвращает:
      all_edges, inferred_edges, warnings
    """
    warnings: List[str] = []
    leaf_ids = {tid for tid, t in tasks.items() if not t.is_summary}

    incoming = {tid: 0 for tid in leaf_ids}
    for e in leaf_edges:
        if e.succ in incoming:
            incoming[e.succ] += 1

    # группируем по "разделу"
    by_section: Dict[Optional[int], List[int]] = defaultdict(list)
    for tid in leaf_ids:
        sec = _nearest_summary_parent(tasks, tid)
        by_section[sec].append(tid)
    for sec, lst in by_section.items():
        by_section[sec] = sorted(lst)

    inferred: List[Edge] = []

    for tid in sorted(blank_leaf_ids):
        if incoming.get(tid, 0) > 0:
            continue  # уже есть вход (после раскрытия), не трогаем

        sec = _nearest_summary_parent(tasks, tid)
        candidates = [x for x in by_section.get(sec, []) if x < tid]
        if not candidates:
            # fallback: предыдущая листовая по проекту
            candidates = [x for x in sorted(leaf_ids) if x < tid]
        if not candidates:
            warnings.append(f"Задача {tid} ('{tasks[tid].name}') без предшественников и без предыдущих задач — оставлена стартовой.")
            continue

        pred = max(candidates)

        allow = True
        ps = tasks[tid].planned_start
        pf = tasks[pred].planned_finish
        if ps is not None and pf is not None:
            delta = (ps - pf).total_seconds() / 86400.0
            if delta < -overlap_tolerance_days:
                allow = False

        if allow:
            inferred.append(Edge(pred=pred, succ=tid, rel='FS', lag=0.0, inferred=True, source='inferred'))
            incoming[tid] = 1
        else:
            warnings.append(
                f"Задача {tid} ('{tasks[tid].name}') без предшественников: "
                f"по ИД предыдущая листовая {pred} ('{tasks[pred].name}'), "
                f"но плановые даты перекрываются (Start<{pred} Finish). "
                f"Связь НЕ добавлена — требуется явная логика (FS/SS/без связи)."
            )

    # merge + dedup
    existing = {(e.pred, e.succ, e.rel, e.lag) for e in leaf_edges}
    new_unique: List[Edge] = []
    for e in inferred:
        k = (e.pred, e.succ, e.rel, e.lag)
        if k not in existing:
            new_unique.append(e)
            existing.add(k)

    all_edges = leaf_edges + new_unique
    return all_edges, new_unique, warnings


# --------------------------
# CPM (к цели) с constraints
# --------------------------

def _topological_sort(nodes: Set[int], edges: Iterable[Edge]) -> List[int]:
    indeg = {n: 0 for n in nodes}
    adj = defaultdict(list)
    for e in edges:
        if e.pred in nodes and e.succ in nodes:
            adj[e.pred].append(e.succ)
            indeg[e.succ] += 1

    q = deque([n for n, d in indeg.items() if d == 0])
    order: List[int] = []
    while q:
        n = q.popleft()
        order.append(n)
        for s in adj.get(n, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)

    if len(order) != len(nodes):
        return []
    return order


def _cpm_with_constraints(tasks: Dict[int, Task],
                          edges: List[Edge],
                          nodes: Set[int],
                          goal: int,
                          project_start: datetime) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], float]:
    """Возвращает ES, EF, LS, LF, TF (в днях от project_start), project_finish_offset."""
    inc: Dict[int, List[Edge]] = defaultdict(list)
    out: Dict[int, List[Edge]] = defaultdict(list)
    for e in edges:
        if e.pred in nodes and e.succ in nodes:
            inc[e.succ].append(e)
            out[e.pred].append(e)

    topo = _topological_sort(nodes, edges)
    if not topo:
        raise ValueError("Cycle detected in task network (cycle after expansion/inference).")

    ES = {n: 0.0 for n in nodes}
    EF = {n: 0.0 for n in nodes}

    # forward pass
    for n in topo:
        t = tasks[n]
        lb = 0.0
        if t.constraint_type != "Как можно раньше" and t.constraint_date is not None:
            cd = (t.constraint_date - project_start).total_seconds() / 86400.0
            if t.constraint_type == "Начало не ранее":
                lb = max(lb, cd)
            elif t.constraint_type == "Окончание не ранее":
                lb = max(lb, cd - t.duration_days)

        # якорим стартовые узлы по плановому началу, если у узла нет входящих связей в рассматриваемой сети
        # (требование: учитывать План_начало, особенно при неполной логике зависимостей)
        if not inc.get(n, []) and t.planned_start is not None:
            ps = (t.planned_start - project_start).total_seconds() / 86400.0
            lb = max(lb, ps)

        cand = lb
        for e in inc.get(n, []):
            if e.rel == 'FS':
                c = EF[e.pred] + e.lag
            elif e.rel == 'SS':
                c = ES[e.pred] + e.lag
            else:
                c = EF[e.pred] + e.lag
            cand = max(cand, c)

        ES[n] = cand
        EF[n] = ES[n] + t.duration_days

    project_finish = EF[goal]

    # backward pass (только по узлам, ведущим к цели — nodes уже ограничен ancestors)
    LS = {n: float('inf') for n in nodes}
    LF = {n: float('inf') for n in nodes}
    LF[goal] = project_finish
    LS[goal] = LF[goal] - tasks[goal].duration_days

    for n in reversed(topo):
        if n == goal:
            continue

        for e in out.get(n, []):
            s = e.succ
            if e.rel == 'FS':
                LF[n] = min(LF[n], LS[s] - e.lag)
            elif e.rel == 'SS':
                LS[n] = min(LS[n], LS[s] - e.lag)

        if LS[n] == float('inf') and LF[n] < float('inf'):
            LS[n] = LF[n] - tasks[n].duration_days
        if LF[n] == float('inf') and LS[n] < float('inf'):
            LF[n] = LS[n] + tasks[n].duration_days
        if LS[n] == float('inf') and LF[n] == float('inf'):
            # страховка: если узел по какой-то причине не получил ограничений (не должен случаться в nodes)
            LF[n] = project_finish
            LS[n] = LF[n] - tasks[n].duration_days

        # консистентность
        LF[n] = LS[n] + tasks[n].duration_days

    TF = {n: LS[n] - ES[n] for n in nodes}
    return ES, EF, LS, LF, TF, project_finish


# --------------------------
# Риск‑модель
# --------------------------

@dataclass
class RiskParams:
    # базовая логит‑модель p_delay
    beta0: float = -0.7
    beta_ln_dur: float = 0.55
    beta_constraint: float = 0.55
    beta_external: float = 0.45
    beta_deg: float = 0.25
    beta_baseline_slip: float = 0.03  # на 1 день slip

    # out-of-sequence / rework
    beta_pred_has_completed_succ: float = 0.65   # предшественник не завершён, но есть завершённый последователь
    p_rework_min: float = 0.25                   # минимум p_delay для "сомнительно завершённых" задач
    mu_rework_factor: float = 0.25               # доля от длительности (если задача "сомнительно завершена")

    # распространение по сети
    w_pre: float = 0.25
    w_succ: float = 0.10
    k_pre: float = 0.20
    k_succ: float = 0.10
    iter_max: int = 12

    # mu_delay
    mu_factor: float = 0.18        # средняя задержка ~ 18% длительности
    mu_min_days: float = 0.5
    mu_max_mul: float = 1.2

    # детекция "внешних"
    external_keywords: Tuple[str, ...] = ("внеш", "подряд", "контрагент", "постав", "vendor", "outsourc")

    # thresholds for recommendations
    high_impact_prob: float = 0.35
    high_delay_prob: float = 0.60
    high_eimpact_days: float = 2.0
    top_n: int = 10

    # numerical eps
    eps: float = 1e-6


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _is_external(owner: Optional[str], params: RiskParams) -> bool:
    if not owner:
        return False
    s = owner.lower()
    return any(k in s for k in params.external_keywords)


def detect_out_of_sequence(tasks: Dict[int, Task], edges: List[Edge]) -> List[Tuple[int, int]]:
    """Возвращает список пар (pred, succ), где succ завершён, а pred не завершён."""
    anomalies: List[Tuple[int, int]] = []
    for e in edges:
        pred = tasks.get(e.pred)
        succ = tasks.get(e.succ)
        if pred is None or succ is None:
            continue
        pred_done = (pred.percent_complete is not None and pred.percent_complete >= 99.9)
        succ_done = (succ.percent_complete is not None and succ.percent_complete >= 99.9)
        if succ_done and (not pred_done):
            anomalies.append((e.pred, e.succ))
    return anomalies


def compute_risk_table(tasks: Dict[int, Task],
                       edges: List[Edge],
                       nodes: Set[int],
                       TF: Dict[int, float],
                       params: RiskParams) -> pd.DataFrame:
    """Возвращает таблицу риск‑метрик по узлам nodes."""
    preds: Dict[int, List[int]] = defaultdict(list)
    succs: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        if e.pred in nodes and e.succ in nodes:
            preds[e.succ].append(e.pred)
            succs[e.pred].append(e.succ)

    # counts for degree
    pred_cnt = {tid: len(preds.get(tid, [])) for tid in nodes}
    succ_cnt = {tid: len(succs.get(tid, [])) for tid in nodes}

    completed = {tid for tid in nodes if (tasks[tid].percent_complete is not None and tasks[tid].percent_complete >= 99.9)}

    anomalies = detect_out_of_sequence(tasks, edges)
    anomalies_set = set(anomalies)
    # задачи, которые "завершены", но имеют незавершённые предшественники => не обнуляем риск (rework/validation)
    tainted_completed: Set[int] = set()
    for pred, succ in anomalies:
        if succ in completed:
            tainted_completed.add(succ)

    # у предшественников с завершёнными последователями повышаем риск (организационная/логическая проблема)
    pred_has_completed_succ: Set[int] = set()
    for pred, succ in anomalies:
        pred_has_completed_succ.add(pred)

    # базовые p и mu
    p_base: Dict[int, float] = {}
    mu_base: Dict[int, float] = {}

    for tid in nodes:
        t = tasks[tid]
        D = max(0.0, t.duration_days)

        constraint_flag = 1.0 if (t.constraint_type != "Как можно раньше") else 0.0
        external_flag = 1.0 if _is_external(t.owner, params) else 0.0
        deg = math.log(1.0 + pred_cnt[tid] + succ_cnt[tid])

        slip = 0.0
        if t.baseline_finish and t.planned_finish:
            slip = max(0.0, (t.planned_finish - t.baseline_finish).total_seconds() / 86400.0)

        score = (
            params.beta0
            + params.beta_ln_dur * math.log(1.0 + D)
            + params.beta_constraint * constraint_flag
            + params.beta_external * external_flag
            + params.beta_deg * deg
            + params.beta_baseline_slip * slip
        )

        if tid in pred_has_completed_succ:
            score += params.beta_pred_has_completed_succ

        p = _sigmoid(score)

        # completed tasks: p=0, но если task tainted (completed while predecessors not) — не обнуляем
        if tid in completed and tid not in tainted_completed:
            p = 0.0
        if tid in tainted_completed:
            p = max(p, params.p_rework_min)

        p_base[tid] = float(min(max(p, 0.0), 1.0))

        mu = max(params.mu_min_days, params.mu_factor * D)
        if D == 0.0:
            mu = params.mu_min_days

        if tid in tainted_completed:
            mu = max(mu, params.mu_rework_factor * max(1.0, D))

        mu_base[tid] = float(mu)

    # распространение p_delay (логит‑сглаживание по сети)
    p_eff = dict(p_base)
    for _ in range(params.iter_max):
        new_p = {}
        for tid in nodes:
            if tid in completed and tid not in tainted_completed:
                new_p[tid] = 0.0
                continue
            l0 = _logit(p_base[tid])
            pre = preds.get(tid, [])
            suc = succs.get(tid, [])
            pre_term = 0.0
            suc_term = 0.0
            if pre:
                pre_term = sum(_logit(p_eff[p]) for p in pre) / len(pre)
            if suc:
                suc_term = sum(_logit(p_eff[s]) for s in suc) / len(suc)
            l = l0 + params.w_pre * pre_term + params.w_succ * suc_term
            new_p[tid] = _sigmoid(l)
        p_eff = new_p

    # распространение mu (масса задержки)
    mu_eff = dict(mu_base)
    for _ in range(params.iter_max):
        new_mu = {}
        for tid in nodes:
            if tid in completed and tid not in tainted_completed:
                new_mu[tid] = params.mu_min_days
                continue

            D = max(0.0, tasks[tid].duration_days)
            pre = preds.get(tid, [])
            suc = succs.get(tid, [])

            add_pre = 0.0
            add_suc = 0.0
            if pre:
                add_pre = sum(p_eff[p] * mu_eff[p] for p in pre) / len(pre)
            if suc:
                add_suc = sum(p_eff[s] * mu_eff[s] for s in suc) / len(suc)

            mu = mu_base[tid] + params.k_pre * add_pre + params.k_succ * add_suc
            cap = params.mu_min_days + params.mu_max_mul * max(1.0, D)
            mu = min(max(params.mu_min_days, mu), cap)
            new_mu[tid] = float(mu)
        mu_eff = new_mu

    # влияние на цель (R2): P(Δ > slack) и E[(Δ - slack)+]
    p_impact: Dict[int, float] = {}
    e_impact: Dict[int, float] = {}
    for tid in nodes:
        slack = max(0.0, TF.get(tid, 0.0))
        mu = max(1e-6, mu_eff[tid])
        p = p_eff[tid]
        tail = math.exp(-slack / mu)  # P(Δ>slack | delay happened)
        p_imp = p * tail
        e_imp = p * mu * tail
        p_impact[tid] = float(min(max(p_imp, 0.0), 1.0))
        e_impact[tid] = float(max(0.0, e_imp))

    rows = []
    for tid in sorted(nodes):
        rows.append({
            "ИД": tid,
            "p_delay_base": p_base[tid],
            "mu_delay_base_days": mu_base[tid],
            "p_delay_eff": p_eff[tid],
            "mu_delay_eff_days": mu_eff[tid],
            "p_impact_goal": p_impact[tid],
            "p_noimpact_goal": 1.0 - p_impact[tid],
            "e_impact_goal_days": e_impact[tid],
            "is_completed": (tid in completed),
            "is_tainted_completed": (tid in tainted_completed),
            "has_out_of_sequence_issue": (tid in pred_has_completed_succ),
        })
    return pd.DataFrame(rows)


# --------------------------
# Рекомендации
# --------------------------

def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _sorted_by_es(nodes: Iterable[int], ES: Dict[int, float], tasks: Dict[int, Task]) -> List[int]:
    return sorted(list(nodes), key=lambda tid: (ES.get(tid, 0.0), tasks[tid].row_index, tid))


def generate_recommendations(tasks: Dict[int, Task],
                             nodes: Set[int],
                             leaf_ids: Set[int],
                             ES: Dict[int, float],
                             EF: Dict[int, float],
                             TF: Dict[int, float],
                             goal_id: int,
                             risk_df: pd.DataFrame,
                             expansion_stats: Dict[str, int],
                             blank_leaf_ids: Set[int],
                             other_warnings: List[str],
                             params: RiskParams) -> List[str]:
    rec: List[str] = []

    # 0) контекст
    excluded = [tid for tid in leaf_ids if tid not in nodes]
    rec.append(f"Расчёт выполнен к цели (goal_id={goal_id}): '{tasks[goal_id].name}'. В анализ включено {len(nodes)} листовых задач, исключено {len(excluded)} задач вне влияния на цель.")

    if excluded:
        # показываем только первые N, иначе будет шум
        ex_show = excluded[:min(10, len(excluded))]
        ex_str = "; ".join([f"{tid} '{tasks[tid].name}'" for tid in ex_show])
        tail = "" if len(excluded) <= 10 else f" (и ещё {len(excluded)-10})"
        rec.append(f"Исключённые задачи (пример): {ex_str}{tail}.")

    # 1) критический путь к цели
    cp = [tid for tid in nodes if TF.get(tid, 0.0) <= params.eps]
    cp_sorted = _sorted_by_es(cp, ES, tasks)
    rec.append(f"Критический путь к цели (TF≈0): {len(cp_sorted)} задач.")
    # list them
    for tid in cp_sorted:
        t = tasks[tid]
        dur = t.duration_days
        done = (t.percent_complete is not None and t.percent_complete >= 99.9)
        rec.append(f"— CP: ID {tid}: {t.name} | D={dur:.2f}д | ES={ES.get(tid,0):.2f}д | TF={TF.get(tid,0):.2f}д | {'DONE' if done else 'OPEN'}")

    # 2) высокий риск влияния на цель (R2)
    rdf = risk_df.set_index("ИД")
    # top by expected impact days
    rdf_nodes = risk_df.copy()
    rdf_nodes = rdf_nodes.sort_values(["e_impact_goal_days", "p_impact_goal"], ascending=[False, False])
    top = rdf_nodes.head(params.top_n)

    rec.append(f"Топ-{min(params.top_n, len(rdf_nodes))} задач по ожидаемому влиянию на срок цели E(impactDays):")
    for _, row in top.iterrows():
        tid = int(row["ИД"])
        rec.append(
            f"— RISK: ID {tid}: {tasks[tid].name} | TF={TF.get(tid,0):.2f}д | "
            f"p_delay={_fmt_pct(row['p_delay_eff'])} | "
            f"P(impact)={_fmt_pct(row['p_impact_goal'])} | "
            f"E(impact)={row['e_impact_goal_days']:.2f}д"
            + (" | ⚠ out-of-sequence" if bool(row.get("has_out_of_sequence_issue")) else "")
            + (" | ⚠ tainted-complete" if bool(row.get("is_tainted_completed")) else "")
        )

    # 3) зона внимания (пороговая)
    zone = rdf_nodes[
        (rdf_nodes["p_impact_goal"] >= params.high_impact_prob)
        | (rdf_nodes["p_delay_eff"] >= params.high_delay_prob)
        | (rdf_nodes["e_impact_goal_days"] >= params.high_eimpact_days)
        | (rdf_nodes["has_out_of_sequence_issue"] == True)
        | (rdf_nodes["is_tainted_completed"] == True)
    ].copy()

    if not zone.empty:
        rec.append(f"Зона повышенного внимания: {len(zone)} задач (критерии: P(impact)≥{int(params.high_impact_prob*100)}%, p_delay≥{int(params.high_delay_prob*100)}%, E(impact)≥{params.high_eimpact_days}д, либо out-of-sequence).")
        # show limited
        zone = zone.sort_values(["p_impact_goal", "e_impact_goal_days"], ascending=False).head(min(15, len(zone)))
        for _, row in zone.iterrows():
            tid = int(row["ИД"])
            hints = []
            if TF.get(tid, 0.0) <= params.eps:
                hints.append("ускорение/ресурс на критике")
            if row["p_delay_eff"] >= params.high_delay_prob:
                hints.append("вероятная задержка")
            if row["p_impact_goal"] >= params.high_impact_prob:
                hints.append("влияет на срок цели")
            if bool(row.get("has_out_of_sequence_issue")):
                hints.append("проверить факты/логические связи")
            if bool(row.get("is_tainted_completed")):
                hints.append("возможен rework/переприёмка")
            hint_txt = ", ".join(hints) if hints else "контроль"
            rec.append(f"— ACTION: ID {tid}: {tasks[tid].name} → {hint_txt} (TF={TF.get(tid,0):.2f}д, P(impact)={_fmt_pct(row['p_impact_goal'])}, E(impact)={row['e_impact_goal_days']:.2f}д).")

    # 4) качество входных данных / сетевой структуры
    if blank_leaf_ids:
        rec.append(f"Задачи без предшественников (листовые): {len(blank_leaf_ids)}. Для CPM это допустимо только как стартовые события разделов — иначе критичность может быть искажена.")
        # show first few
        show = sorted(list(blank_leaf_ids))[:10]
        rec.append("Пример: " + "; ".join([f"{tid} '{tasks[tid].name}'" for tid in show]) + ("…" if len(blank_leaf_ids) > 10 else "") + ".")

    if expansion_stats.get("expanded_edges", 0) > 0:
        rec.append(f"Связи на суммарные задачи раскрыты: добавлено {expansion_stats.get('expanded_edges',0)} раскрытых связей (expanded). Проверьте, что суммарные задачи в исходнике не используются как 'логические узлы' без листов.")

    if other_warnings:
        rec.append("Технические предупреждения по входным данным:")
        rec.extend(["— " + w for w in other_warnings[:20]])
        if len(other_warnings) > 20:
            rec.append(f"— (и ещё {len(other_warnings)-20})")

    return rec


# --------------------------
# Итоговый расчёт проекта
# --------------------------

@dataclass
class ProjectRunResult:
    project_start: datetime
    goal_id: int
    goal_name: str
    project_finish_offset_days: float
    recommendations: List[str]
    warnings: List[str]
    # optional diagnostics
    tasks_table: Optional[pd.DataFrame] = None
    risk_table: Optional[pd.DataFrame] = None


def read_excel_any(file_like, sheet: Optional[str | int] = 0) -> pd.DataFrame:
    """Универсальное чтение Excel (путь, bytes, file-like)."""
    return pd.read_excel(file_like, sheet_name=sheet)


def run_project_analysis(df_raw: pd.DataFrame,
                         goal_id: Optional[int],
                         goal_name_text: Optional[str],
                         mode: str = "reconstruct_plan",
                         risk_params: Optional[RiskParams] = None,
                         build_diagnostics_tables: bool = True) -> ProjectRunResult:
    """
    Основной вход бэкэнда.

    Обязательные параметры:
    - goal_id: ИД целевой вехи/задачи
    - goal_name_text: текстовая цель (для отчёта)

    mode:
      - 'reconstruct_plan' (рекомендовано): календарные длительности из плановых дат.
      - 'pure_cpm': используем длительности из поля 'Длительность' как есть.

    Политика пустых предшественников:
      - для вех (0 дней) и суммарных задач отсутствие предшественников считается нормой;
      - для листовых НЕ‑вех задач без предшественников связи не добавляются, но факт фиксируется в рекомендациях.
    """
    if risk_params is None:
        risk_params = RiskParams()

    # валидация обязательных полей
    if goal_id is None or goal_name_text is None or str(goal_name_text).strip() == "":
        raise ValueError("Не задана цель расчёта: требуется и текст цели, и ИД целевой вехи/задачи (goal_id).")

    df, tasks, order, leaf_ids, blank_leaf_ids, warn_norm = build_tasks_from_df(df_raw)
    warnings: List[str] = list(warn_norm)

    if goal_id not in tasks:
        raise ValueError(f"goal_id={goal_id} not found in tasks list.")
    if not tasks[goal_id].is_milestone:
        warnings.append("goal_id не является вехой (0 дней). Модель рассчитает критичность к выбранной задаче, но интерпретация 'цель=веха' нарушена.")

    # project_start: минимальное плановое начало
    starts = [t.planned_start for t in tasks.values() if t.planned_start is not None]
    if not starts:
        raise ValueError("No planned start dates found in file.")
    project_start = min(starts)

    # длительности
    if mode == "reconstruct_plan":
        for t in tasks.values():
            if t.planned_start and t.planned_finish:
                t.duration_days = max(0.0, (t.planned_finish - t.planned_start).total_seconds() / 86400.0)
        warnings.append("Режим reconstruct_plan: длительности вычислены из плановых дат (календарные дни).")
    elif mode == "pure_cpm":
        warnings.append("Режим pure_cpm: длительности взяты из поля 'Длительность' без перерасчёта.")
    else:
        raise ValueError("mode must be 'reconstruct_plan' or 'pure_cpm'")

    # связи
    raw_edges, warn_edges = build_raw_edges_from_df(df)
    leaf_edges, expansion_stats, warn_expand = expand_edges_to_leaf_level(raw_edges, tasks)
    warnings.extend(warn_edges)
    warnings.extend(warn_expand)

    # Политика обработки пустых предшественников:
    # - для вех (D=0) и суммарных задач отсутствие предшественников ожидаемо;
    # - для листовых НЕ‑вех задач без предшественников связи автоматически НЕ добавляются.
    #   Такие задачи считаются стартовыми внутри своего раздела/подраздела; CPM якорится по плановому началу (см. forward pass),
    #   риск оценивается по их разделу и последователям.
    if blank_leaf_ids:
        warnings.append(f"Листовые задачи без предшественников и НЕ‑вехи: {len(blank_leaf_ids)}. Связи не добавлялись (политика).")
# вычисляем множество задач, влияющих на цель (предки цели)
    anc = _ancestors_of_goal(goal_id, leaf_edges)
    nodes = set(anc)  # anc уже включает goal

    # CPM к цели (только для nodes)
    ES, EF, LS, LF, TF, proj_finish = _cpm_with_constraints(tasks, leaf_edges, nodes, goal_id, project_start)

    # риск только по nodes
    risk_table = compute_risk_table(tasks, leaf_edges, nodes, TF, risk_params)

    # диагностика: объединённая таблица (опционально)
    tasks_table = None
    if build_diagnostics_tables:
        rows = []
        for tid in sorted(nodes):
            t = tasks[tid]
            rows.append({
                "ИД": tid,
                "Название": t.name,
                "D_days": t.duration_days,
                "Веха": t.is_milestone,
                "Суммарная": t.is_summary,
                "ES": ES.get(tid),
                "EF": EF.get(tid),
                "LS": LS.get(tid),
                "LF": LF.get(tid),
                "TF": TF.get(tid),
                "Процент_завершения": t.percent_complete,
                "Тип_ограничения": t.constraint_type,
                "Дата_ограничения": t.constraint_date,
                "План_начало": t.planned_start,
                "План_окончание": t.planned_finish,
                "Базовое_начало": t.baseline_start,
                "Базовое_окончание": t.baseline_finish,
                "Ответственный": t.owner,
                "Влияет_на_цель": True,
                "Критич_к_цели": TF.get(tid, 0.0) <= risk_params.eps,
            })
        tasks_table = pd.DataFrame(rows).merge(risk_table, on="ИД", how="left")

    goal_name = f"{goal_name_text} (веха: {tasks[goal_id].name})"

    rec = generate_recommendations(
        tasks=tasks,
        nodes=nodes,
        leaf_ids=leaf_ids,
        ES=ES,
        EF=EF,
        TF=TF,
        goal_id=goal_id,
        risk_df=risk_table,
        expansion_stats=expansion_stats,
        blank_leaf_ids=blank_leaf_ids,
        other_warnings=warnings,
        params=risk_params
    )

    return ProjectRunResult(
        project_start=project_start,
        goal_id=goal_id,
        goal_name=goal_name,
        project_finish_offset_days=proj_finish,
        recommendations=rec,
        warnings=warnings,
        tasks_table=tasks_table,
        risk_table=risk_table
    )
