"""Workbook-order worksheet enumeration for XLSX sheet-as-page metadata (R-52).

A worksheet is rendered as one page of row-level records rather than raw
tab-separated rows: every data row repeats its column labels and the section
heading it sits under. A single row therefore stays self-describing after the
page is split by the chunker, which a bare grid does not — retrieved cells
otherwise arrive without the header that gives them meaning.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from openpyxl import load_workbook

from backend.rag.ingestion.parsers.xlsx_drawings import (
    SheetDrawings,
    XlsxImage,
    XlsxTextBox,
    extract_xlsx_drawings,
)


@dataclass(frozen=True)
class XlsxSheet:
    """One worksheet treated as a logical page."""

    page_number: int  # 1-based workbook order
    sheet_name: str
    text: str
    #: Pictures placed on the sheet (drawing order). Their pixels are not in
    #: ``text``; a placeholder block per picture is.
    images: tuple[XlsxImage, ...] = ()
    #: Text boxes on the sheet. Their text IS rendered into ``text``.
    text_boxes: tuple[XlsxTextBox, ...] = ()


# Pictures displayed smaller than this (logos, icons, bullets) carry no
# evidence; they are neither placeholdered nor sent to a vision model.
MIN_IMAGE_DISPLAY_PX = 48

# Fixed marker the generation prompt keys on (see app/rag/agent_prompt/rag.py,
# "Embedded Images Rule"). Change both together.
EMBEDDED_IMAGE_MARKER = "[Embedded image — not indexed as text."


# Single-letter risk-of-material-misstatement headers are meaningless once a row
# is read outside its table.
_HEADER_ALIASES = {
    "L": "Low RMM",
    "M": "Medium RMM",
    "H": "High RMM",
}
_MAX_LABEL_CHARS = 80
_MIN_SCHEMA_LABELS = 3


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.time() == time.min:
            return value.date().isoformat()
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat(timespec="seconds")
    return str(value)


def _flatten(value: Any) -> str:
    """One cell as a single-line string."""
    return _cell_text(value).replace("\r", " ").replace("\n", " ").strip()


def _pad_row(row: list[str], width: int) -> list[str]:
    return row + [""] * (width - len(row))


def _trim_columns(rows: list[tuple[int, list[str]]]) -> list[tuple[int, list[str]]]:
    """Drop trailing columns that are empty across every row."""
    if not rows:
        return []

    width = max(len(row) for _, row in rows)
    last_column = -1
    for _, row in rows:
        for index, cell in enumerate(_pad_row(row, width)):
            if cell:
                last_column = max(last_column, index)
    if last_column < 0:
        return []
    return [(row_number, row[: last_column + 1]) for row_number, row in rows]


def _expand_header(cell: str) -> str:
    return _HEADER_ALIASES.get(cell.strip().upper(), cell)


def _is_numericish(cell: str) -> bool:
    compact = (
        cell.replace(",", "")
        .replace("%", "")
        .replace(">", "")
        .replace("<", "")
        .replace("\u2013", "-")
        .replace(" ", "")
        .replace("-", "")
    )
    return bool(compact) and compact.replace(".", "", 1).isdigit()


def _looks_like_header_row(row: list[str]) -> bool:
    filled = [cell for cell in row if cell]
    if len(filled) < 2:
        return False
    if any(len(cell) > _MAX_LABEL_CHARS for cell in filled):
        return False
    numeric = sum(1 for cell in filled if _is_numericish(cell))
    return numeric <= max(1, len(filled) // 4)


def _looks_like_section_title(row: list[str]) -> bool:
    filled = [cell for cell in row if cell]
    return len(filled) == 1 and len(filled[0]) <= _MAX_LABEL_CHARS


def _names_new_columns(row: list[str], labels: list[str]) -> bool:
    """True when the row labels at least one column that has no label yet.

    ``_looks_like_header_row`` also matches an ordinary text data row, so header
    absorption additionally requires this: the first row of a table region names
    every column, and a merged multi-row header fills the columns the first row
    left blank. A row that only overwrites already-named columns is data.
    """
    return any(cell and not labels[index] for index, cell in enumerate(row))


def _combine_label(existing: str, incoming: str) -> str:
    incoming = _expand_header(incoming)
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming in existing or existing.endswith(incoming):
        return existing
    return f"{existing} / {incoming}"


def _record_block(
    *,
    row_number: int,
    section: str,
    pairs: list[tuple[str, str]],
) -> str:
    lines = [f"### Row {row_number}"]
    if section:
        lines.append(f"Section: {section}")
    if len(pairs) == 1:
        name, value = pairs[0]
        # A lone unlabelled cell reads better as prose than as "Column 3: ...".
        if name.startswith("Column ") or name == section:
            lines.append(value)
        else:
            lines.append(f"{name}: {value}")
    else:
        lines.extend(f"{name}: {value}" for name, value in pairs)
    return "\n".join(lines)


def is_decorative(image: XlsxImage) -> bool:
    """Logos/icons: displayed under ``MIN_IMAGE_DISPLAY_PX`` in either dimension."""
    width, height = image.display_width_px, image.display_height_px
    if width is None or height is None:
        return False
    return width < MIN_IMAGE_DISPLAY_PX or height < MIN_IMAGE_DISPLAY_PX


def _image_block(image: XlsxImage, *, number: int, total: int) -> str:
    """Placeholder rendered into the sheet text where a picture sits.

    It is deliberately *not* a description of the picture: the label comes
    from the cell grid, never from pixels, so the generator can tell the user
    exactly where to look without guessing what the screenshot shows.
    ``number``/``total`` count the pictures rendered on this sheet (reading
    order, decorative ones excluded); ``image.ordinal`` stays the drawing-layer id.
    """
    return "\n".join(
        [
            f'### Embedded image {number} of {total}: "{image.shape_name}" '
            f"at {image.anchor_range}",
            f"Label: {image.label_text}",
            f"{EMBEDDED_IMAGE_MARKER} To view it, open this workbook at sheet "
            f'"{image.sheet_name}", cells {image.anchor_range}.]',
        ]
    )


def _text_box_block(text_box: XlsxTextBox) -> str:
    return (
        f'### Text box "{text_box.shape_name}" at {text_box.anchor_range}\n'
        f"{text_box.text}"
    )


def drawing_blocks(drawings: SheetDrawings | None) -> list[tuple[int, str]]:
    """``(anchor_row, block)`` for every renderable drawing object on a sheet.

    Objects without a cell anchor (``absoluteAnchor``) sort after every row.
    """
    if drawings is None:
        return []
    blocks: list[tuple[int, str]] = []
    pictures = [image for image in drawings.images if not is_decorative(image)]
    for number, image in enumerate(pictures, start=1):
        blocks.append(
            (
                image.anchor_row or _NO_ANCHOR_ROW,
                _image_block(image, number=number, total=len(pictures)),
            )
        )
    for text_box in drawings.text_boxes:
        blocks.append((text_box.anchor_row or _NO_ANCHOR_ROW, _text_box_block(text_box)))
    blocks.sort(key=lambda item: item[0])
    return blocks


_NO_ANCHOR_ROW = 1 << 30


def _render_sheet(
    rows: list[tuple[int, list[str]]],
    extra_blocks: list[tuple[int, str]] | None = None,
) -> str:
    """Render row records, interleaving drawing blocks by their anchor row.

    ``extra_blocks`` are ``(anchor_row, text)`` pairs (see ``drawing_blocks``);
    each is emitted after the last cell row at or above its anchor row, so a
    screenshot placed under "Active Clients" follows that label in the text.
    """
    pending = sorted(extra_blocks or [], key=lambda item: item[0])
    rows = _trim_columns(rows)
    if not rows and not pending:
        return ""

    width = max((len(row) for _, row in rows), default=0)
    labels = [""] * width
    section = ""
    # A section title with no data rows under it is content in its own right, so
    # it is emitted rather than dropped once the region closes.
    unconsumed_section = ""
    header_open = True
    blocks: list[str] = []

    def flush_extras(before_row: int | None) -> None:
        nonlocal unconsumed_section
        while pending and (before_row is None or pending[0][0] < before_row):
            if unconsumed_section:
                # Keep the heading ahead of the object placed under it.
                blocks.append(f"## {unconsumed_section}")
                unconsumed_section = ""
            blocks.append(pending.pop(0)[1])

    for row_number, values in rows:
        flush_extras(row_number)
        values = _pad_row(values, width)

        if (
            header_open
            and _looks_like_header_row(values)
            and _names_new_columns(values, labels)
        ):
            for index, cell in enumerate(values):
                if cell:
                    labels[index] = _combine_label(labels[index], cell)
            continue

        if _looks_like_section_title(values):
            if unconsumed_section:
                blocks.append(f"## {unconsumed_section}")
            section = next(cell for cell in values if cell)
            unconsumed_section = section
            # A new region may carry its own header row.
            header_open = True
            continue

        pairs = [
            (labels[index] or f"Column {index + 1}", cell)
            for index, cell in enumerate(values)
            if cell
        ]
        if not pairs:
            continue
        blocks.append(
            _record_block(row_number=row_number, section=section, pairs=pairs)
        )
        header_open = False
        unconsumed_section = ""

    flush_extras(None)
    if unconsumed_section:
        blocks.append(f"## {unconsumed_section}")

    named_labels = [label for label in labels if label]
    if len(named_labels) >= _MIN_SCHEMA_LABELS:
        blocks.insert(0, "Fields: " + ", ".join(named_labels))
    return "\n\n".join(blocks).strip()


def extract_xlsx_sheets(
    document_bytes: bytes,
    *,
    include_drawings: bool = True,
) -> list[XlsxSheet]:
    """Enumerate worksheets in workbook order and render each as row records.

    ``page_number`` is the 1-based sheet index in workbook order. Empty sheets
    are still returned so callers can decide whether to index them; sheet index
    stays stable.

    Hidden and veryHidden worksheets are skipped (their index is consumed so
    the visible sheets keep their workbook position). Tooling such as
    DataSnipper stores reviewer names and machine identifiers on veryHidden
    ``DS_INTERNAL_*`` sheets; ``workbook.worksheets`` enumerates those too, so
    without this filter they would be indexed and retrievable.

    With ``include_drawings`` (default), the drawing layer is read as well:
    text boxes are rendered into the sheet text at their anchor row, and each
    picture gets a placeholder block naming its anchor cells and the nearest
    cell label. The pictures themselves are returned on ``XlsxSheet.images``
    for callers that transcribe them (see ``XlsxParser``).
    """
    drawings: dict[int, SheetDrawings] = (
        extract_xlsx_drawings(document_bytes) if include_drawings else {}
    )
    workbook = load_workbook(
        io.BytesIO(document_bytes),
        read_only=True,
        data_only=True,
    )
    try:
        sheets: list[XlsxSheet] = []
        for index, worksheet in enumerate(workbook.worksheets, start=1):
            if getattr(worksheet, "sheet_state", "visible") != "visible":
                continue
            rows: list[tuple[int, list[str]]] = []
            for row_number, row in enumerate(
                worksheet.iter_rows(values_only=True), start=1
            ):
                cells = [_flatten(value) for value in row]
                if any(cells):
                    rows.append((row_number, cells))
            sheet_drawings = drawings.get(index)
            sheets.append(
                XlsxSheet(
                    page_number=index,
                    sheet_name=str(worksheet.title),
                    text=_render_sheet(rows, drawing_blocks(sheet_drawings)),
                    images=tuple(
                        image
                        for image in (sheet_drawings.images if sheet_drawings else ())
                        if not is_decorative(image)
                    ),
                    text_boxes=sheet_drawings.text_boxes if sheet_drawings else (),
                )
            )
        return sheets
    finally:
        workbook.close()
