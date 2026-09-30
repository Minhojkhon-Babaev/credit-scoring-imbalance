#!/usr/bin/env python3
"""Пять слайдов в визуальной системе шаблона МИЭМ по посчитанным результатам."""

from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
TEMPLATE = Path(
    "/Users/bminhojkhon03/.codex/skills/artifact-template-babaev-minhojkhon-miem/assets/reference.pptx"
)
OUTPUT = ROOT / "presentation.pptx"

NAVY = RGBColor(0x0E, 0x2C, 0x68)
INK = RGBColor(0x1E, 0x2B, 0x40)
TEAL = RGBColor(0x0D, 0x7A, 0x8E)
GOLD = RGBColor(0xD4, 0xA8, 0x43)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xF4, 0xF7, 0xFB)
GOLD_PALE = RGBColor(0xFB, 0xF6, 0xE8)


def delete_slide(prs: Presentation, index: int) -> None:
    slide_id = prs.slides._sldIdLst[index]
    rel_id = slide_id.get(qn("r:id"))
    prs.part.drop_rel(rel_id)
    prs.slides._sldIdLst.remove(slide_id)


def _fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _run(paragraph, text: str, size: int, color: RGBColor, bold: bool = False, font: str = "Calibri") -> None:
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_title(slide, text: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.48), Inches(0.28), Inches(12.3), Inches(0.62))
    frame = box.text_frame
    frame.word_wrap = True
    _run(frame.paragraphs[0], text, 28, NAVY, bold=True, font="Times New Roman")


def add_card(slide, left, top, width, height, accent: RGBColor, heading: str, lines: list[str]) -> None:
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    _fill(card, WHITE)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.08), height)
    _fill(bar, accent)
    box = slide.shapes.add_textbox(left + Inches(0.24), top + Inches(0.16), width - Inches(0.38), height - Inches(0.28))
    frame = box.text_frame
    frame.word_wrap = True
    _run(frame.paragraphs[0], heading, 16, NAVY, bold=True)
    for line in lines:
        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(8)
        _run(paragraph, line, 14, INK)


def add_footer(slide, text: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.48), Inches(6.85), Inches(12.3), Inches(0.42))
    frame = box.text_frame
    frame.word_wrap = True
    _run(frame.paragraphs[0], text, 12, TEAL, bold=True)


def paint_background(slide) -> None:
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), Inches(13.333), Inches(7.5))
    _fill(rect, WHITE)


def style_cell(cell, text: str, *, bold: bool, fill: RGBColor, color: RGBColor, size: int = 12) -> None:
    cell.text = text
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    for paragraph in cell.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color


def build() -> Path:
    shutil.copy(TEMPLATE, OUTPUT)
    presentation = Presentation(OUTPUT)
    while len(presentation.slides) > 1:
        delete_slide(presentation, len(presentation.slides) - 1)
    blank = presentation.slide_layouts[10]

    slide = presentation.slides.add_slide(blank)
    paint_background(slide)
    add_title(slide, "Что изменилось после обсуждения")
    add_card(
        slide,
        Inches(0.45),
        Inches(1.2),
        Inches(4.0),
        Inches(5.2),
        RED,
        "Optuna на расширенных данных",
        [
            "Гиперпараметры ищутся заново для каждого метода, а не переносятся с baseline.",
            "12 испытаний, цель — средний PR-AUC по 3 стратифицированным фолдам.",
            "После поиска отдельно сравниваются объёмы синтетики: +25%, +50%, +100% и баланс 1:1.",
        ],
    )
    add_card(
        slide,
        Inches(4.65),
        Inches(1.2),
        Inches(4.0),
        Inches(5.2),
        TEAL,
        "Сначала лёгкие методы",
        [
            "SMOTE и две модификации: Borderline-SMOTE и ADASYN.",
            "Генерация другой моделью: смесь гауссиан по числовым признакам.",
            "Добавление гауссовского шума к копиям дефолтов.",
            "CTGAN — только после этой лестницы.",
        ],
    )
    add_card(
        slide,
        Inches(8.85),
        Inches(1.2),
        Inches(4.0),
        Inches(5.2),
        GOLD,
        "Веса внутри библиотек",
        [
            "Отдельный baseline, без синтетики.",
            "LogReg: веса классов.",
            "XGBoost: scale_pos_weight.",
            "CatBoost: class_weights.",
            "Множитель веса тоже подбирает Optuna.",
        ],
    )

    slide = presentation.slides.add_slide(blank)
    paint_background(slide)
    add_title(slide, "Протокол: без утечки в тест")
    add_card(
        slide,
        Inches(0.45),
        Inches(1.2),
        Inches(6.15),
        Inches(5.35),
        TEAL,
        "Два датасета",
        [
            "German Credit: все 1000 строк, 30% плохих кредитов, 7 числовых и 13 категориальных признаков. В тесте 200 строк.",
            "Give Me Some Credit: стратифицированные 8000 из 150000, доля дефолтов 6,7%. В тесте 1600 строк.",
        ],
    )
    add_card(
        slide,
        Inches(6.8),
        Inches(1.2),
        Inches(6.05),
        Inches(5.35),
        RED,
        "Что зафиксировано",
        [
            "Три модели: логистическая регрессия, XGBoost, CatBoost.",
            "Генератор учится только на обучающем фолде. Тест не участвует ни в Optuna, ни в выборе порога.",
            "CTGAN учится на дефолтах, 300 эпох. Trial берёт строки из уже готового пула.",
            "Порог F1 выбирается по out-of-fold предсказаниям. На тесте считаются PR-AUC, ROC-AUC, F1 и Recall.",
        ],
    )

    slide = presentation.slides.add_slide(blank)
    paint_background(slide)
    add_title(slide, "Тест: что обошло baseline")
    rows = [
        ["Данные", "Модель", "Метод", "Объём", "PR-AUC", "Recall"],
        ["German", "CatBoost", "Без баланса", "—", "0.663", "0.833"],
        ["German", "CatBoost", "Веса классов", "—", "0.644", "0.817"],
        ["German", "CatBoost", "ADASYN", "+25%", "0.686", "0.900"],
        ["German", "CatBoost", "GMM", "+25%", "0.708", "0.800"],
        ["German", "CatBoost", "CTGAN", "1:1", "0.686", "0.717"],
        ["GiveMe", "LogReg", "Без баланса", "—", "0.166", "0.308"],
        ["GiveMe", "LogReg", "Веса классов", "—", "0.274", "0.327"],
        ["GiveMe", "CatBoost", "Без баланса", "—", "0.360", "0.486"],
        ["GiveMe", "CatBoost", "Шум", "+50%", "0.373", "0.514"],
        ["GiveMe", "CatBoost", "ADASYN", "+25%", "0.358", "0.589"],
        ["GiveMe", "CatBoost", "CTGAN", "+25%", "0.349", "0.505"],
    ]
    highlight = {4, 7, 9}
    table_shape = slide.shapes.add_table(len(rows), 6, Inches(0.45), Inches(1.15), Inches(12.4), Inches(5.35))
    table = table_shape.table
    widths = [1.55, 1.7, 2.55, 1.45, 2.55, 2.6]
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if r == 0:
                fill, color, bold = NAVY, WHITE, True
            elif r in highlight:
                fill, color, bold = GOLD_PALE, NAVY, True
            elif r % 2 == 0:
                fill, color, bold = PALE, INK, False
            else:
                fill, color, bold = WHITE, INK, False
            style_cell(table.cell(r, c), value, bold=bold or c == 0, fill=fill, color=color, size=13)
    add_footer(
        slide,
        "Золотом отмечены лучший PR-AUC на каждом датасете и скачок LogReg от весов классов. Полная сетка 48 прогонов — в results/summary.md.",
    )

    slide = presentation.slides.add_slide(blank)
    paint_background(slide)
    add_title(slide, "Выводы")
    add_card(
        slide,
        Inches(0.45),
        Inches(1.15),
        Inches(6.15),
        Inches(2.55),
        GOLD,
        "Веса классов — нужный baseline",
        [
            "На Give Me Some Credit логистическая регрессия без весов почти не отделяет дефолт: PR-AUC 0.166, ROC-AUC 0.614. С весами — 0.274 и 0.766.",
            "У деревьев веса поднимают Recall, но PR-AUC не растёт. Синтетику нужно сравнивать с этим baseline, а не только с сырыми данными.",
        ],
    )
    add_card(
        slide,
        Inches(6.8),
        Inches(1.15),
        Inches(6.05),
        Inches(2.55),
        TEAL,
        "До GAN достаточно лёгких методов",
        [
            "Лучший PR-AUC: CatBoost + GMM, German, 0.708. На Give Me Some Credit — CatBoost + шум, 0.373.",
            "Лучший Recall на German — ADASYN +25%, 0.900. Полный баланс 1:1 для деревьев почти нигде не выбран.",
        ],
    )
    add_card(
        slide,
        Inches(0.45),
        Inches(3.9),
        Inches(12.4),
        Inches(2.55),
        RED,
        "CTGAN не обогнал лёгкие генераторы",
        [
            "300 эпох не дали выигрыша. German, CatBoost: CTGAN 0.686 против 0.708 у GMM. Give Me Some Credit: CTGAN 0.349 против 0.373 у шума и 0.360 у модели без балансировки.",
            "SMOTE ближе к данным: Вассерштейн 0.08 и 0.04. У CTGAN — 0.55 и 1.02. На логистической регрессии при 6,7% дефолтов CTGAN ухудшил PR-AUC и ROC-AUC. Прогон — один сплит; в тесте German 60 дефолтов.",
        ],
    )

    presentation.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
