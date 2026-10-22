"""Drawing-layer extraction for XLSX workbooks: pictures and text boxes with cell anchors.

A worksheet's cell grid is only part of what a reader sees. Screenshots,
logos, and text boxes live in a separate *drawing* part of the package
(``xl/drawings/drawingN.xml``), and each object there records the cell it is
anchored to. openpyxl never surfaces that layer in ``read_only`` mode (which
is how :mod:`xlsx_sheets` loads workbooks) and drops shapes such as text
boxes on every load, so this module reads the OOXML parts directly:

* ``xl/workbook.xml`` — sheet order and visibility (the 1-based index is the
  same ``page_number`` :mod:`xlsx_sheets` assigns);
* ``xl/worksheets/_rels/sheetN.xml.rels`` — which drawing part a sheet owns;
* ``xl/drawings/drawingN.xml`` — anchors (``twoCellAnchor`` / ``oneCellAnchor``
  / ``absoluteAnchor``), pictures (``xdr:pic``) and shapes (``xdr:sp``);
* ``xl/drawings/_rels/drawingN.xml.rels`` — picture → ``xl/media/*`` file;
* the worksheet XML + ``xl/sharedStrings.xml`` — cell text for label inference.

Nothing here touches the network or a model. Every value is deterministic and
traceable to a cell reference, which is what makes the output auditable.

Label inference
---------------
A picture on an audit template is almost always introduced by a cell label
("Active Clients", "2. To search for engagements …"). The screenshot itself
rarely carries that context — two Maconomy screenshots can differ by a single
dropdown word — so the label is recovered from the grid, never from pixels:
scanning from the anchor row upward, across the columns the picture spans
(then the columns to its left on the same row), the first non-empty cell is
the ``label``. Merged cells resolve to their top-left value, so a step
heading merged across B40:G40 labels every picture placed under it.
"""

from __future__ import annotations

import hashlib
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from typing import Iterator
from xml.etree import ElementTree as ET

from openpyxl.utils import column_index_from_string, get_column_letter

_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
_R_ID = "{%s}id" % _NS["r"]
_R_EMBED = "{%s}embed" % _NS["r"]
_A_T = "{%s}t" % _NS["a"]
_A_P = "{%s}p" % _NS["a"]
_A_BR = "{%s}br" % _NS["a"]
_M_ROW = "{%s}row" % _NS["m"]
_M_T = "{%s}t" % _NS["m"]

_REL_DRAWING = "drawing"
_REL_IMAGE = "image"

# 914400 EMU per inch; Excel renders at 96 DPI → 9525 EMU per pixel.
_EMU_PER_PX = 9525

_MAX_LABEL_CHARS = 160
_MAX_LABEL_SCAN_ROWS = 500

_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")
_MERGE_REF = re.compile(rb'<mergeCell\s+ref="([A-Z]+\d+):([A-Z]+\d+)"')

_MEDIA_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".emf": "image/x-emf",
    ".wmf": "image/x-wmf",
    ".svg": "image/svg+xml",
}


@dataclass(frozen=True)
class XlsxTextBox:
    """A shape with text (Excel "Text Box") anchored on a worksheet."""

    sheet_index: int  # 1-based workbook order == sheet-as-page ``page_number``
    sheet_name: str
    ordinal: int  # 1-based, drawing order within the sheet
    shape_name: str
    anchor_from: str | None  # e.g. "B2"; None for absoluteAnchor
    anchor_to: str | None  # e.g. "E13"; None for one-cell/absolute anchors
    text: str  # paragraphs joined with "\n"

    @property
    def anchor_range(self) -> str:
        return _anchor_range(self.anchor_from, self.anchor_to)

    @property
    def anchor_row(self) -> int | None:
        return _split_ref(self.anchor_from)[1] if self.anchor_from else None


@dataclass(frozen=True)
class XlsxImage:
    """One picture *placement*. The same media file placed twice yields two."""

    sheet_index: int
    sheet_name: str
    ordinal: int  # 1-based, drawing order within the sheet
    shape_name: str  # Excel's name, e.g. "Picture 17"
    description: str | None  # alt text (``descr``) when the author set one
    anchor_type: str  # "twoCell" | "oneCell" | "absolute"
    anchor_from: str | None
    anchor_to: str | None
    label: str | None  # nearest cell text (see module docstring)
    label_cell: str | None
    media_path: str | None  # "xl/media/image14.png"; None for linked pictures
    media_sha256: str | None  # hex digest of the media bytes
    content_type: str | None
    display_width_px: int | None
    display_height_px: int | None
    data: bytes = field(default=b"", repr=False, compare=False)

    @property
    def anchor_range(self) -> str:
        return _anchor_range(self.anchor_from, self.anchor_to)

    @property
    def anchor_row(self) -> int | None:
        return _split_ref(self.anchor_from)[1] if self.anchor_from else None

    @property
    def label_text(self) -> str:
        """``Active Clients (A8)`` or ``(none)``."""
        if not self.label:
            return "(none)"
        return f"{self.label} ({self.label_cell})"


@dataclass(frozen=True)
class SheetDrawings:
    sheet_index: int
    sheet_name: str
    images: tuple[XlsxImage, ...] = ()
    text_boxes: tuple[XlsxTextBox, ...] = ()


def _anchor_range(start: str | None, end: str | None) -> str:
    if start is None:
        return "(no cell anchor)"
    if end is None or end == start:
        return start
    return f"{start}:{end}"


def _cell_ref(col: int, row: int) -> str:
    """1-based column/row → "A1" style reference."""
    return f"{get_column_letter(col)}{row}"


def _split_ref(ref: str) -> tuple[int, int]:
    """"B40" → (column 2, row 40)."""
    match = _CELL_REF.match(ref)
    if not match:
        raise ValueError(f"Bad cell reference {ref!r}")
    return column_index_from_string(match.group(1)), int(match.group(2))


def _resolve_target(base_part: str, target: str) -> str:
    """Resolve a relationship Target against the part that declared it."""
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_part), target))


def _read_rels(archive: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str, bool]]:
    """Return ``{rId: (type_suffix, resolved_target, is_external)}`` for a part."""
    directory, name = posixpath.split(part)
    rels_part = posixpath.join(directory, "_rels", f"{name}.rels")
    if rels_part not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rels_part))
    rels: dict[str, tuple[str, str, bool]] = {}
    for rel in root.findall("rel:Relationship", _NS):
        rel_id = rel.get("Id")
        rel_type = (rel.get("Type") or "").rsplit("/", 1)[-1]
        target = rel.get("Target") or ""
        if rel_id is None:
            continue
        if rel.get("TargetMode") == "External":
            rels[rel_id] = (rel_type, target, True)
        else:
            rels[rel_id] = (rel_type, _resolve_target(part, target), False)
    return rels


def _workbook_sheets(archive: zipfile.ZipFile) -> list[tuple[int, str, str, str]]:
    """``[(sheet_index, name, state, worksheet_part)]`` in workbook order."""
    root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = _read_rels(archive, "xl/workbook.xml")
    sheets: list[tuple[int, str, str, str]] = []
    sheets_el = root.find("m:sheets", _NS)
    if sheets_el is None:
        return sheets
    for index, sheet in enumerate(sheets_el.findall("m:sheet", _NS), start=1):
        rel = rels.get(sheet.get(_R_ID) or "")
        if rel is None:
            continue
        sheets.append(
            (
                index,
                sheet.get("name") or f"Sheet{index}",
                sheet.get("state") or "visible",
                rel[1],
            )
        )
    return sheets


# --------------------------------------------------------------------------- cells


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    part = "xl/sharedStrings.xml"
    if part not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(part))
    return [
        "".join(t.text or "" for t in si.iter(_M_T))
        for si in root.findall("m:si", _NS)
    ]


class _CellGrid:
    """Cell text for label inference, with merged ranges resolved to their anchor."""

    def __init__(
        self,
        cells: dict[tuple[int, int], str],
        merges: list[tuple[int, int, int, int]],
    ) -> None:
        self._cells = cells
        self._merge_anchor: dict[tuple[int, int], tuple[int, int]] = {}
        for c1, r1, c2, r2 in merges:
            for row in range(r1, r2 + 1):
                for col in range(c1, c2 + 1):
                    self._merge_anchor[(row, col)] = (r1, c1)

    def lookup(self, row: int, col: int) -> tuple[str, str]:
        """Text at a cell and the reference that holds it (merge anchor if merged)."""
        anchor_row, anchor_col = self._merge_anchor.get((row, col), (row, col))
        return self._cells.get((anchor_row, anchor_col), ""), _cell_ref(anchor_col, anchor_row)


def _load_grid(
    archive: zipfile.ZipFile,
    worksheet_part: str,
    *,
    max_row: int,
    shared: list[str],
) -> _CellGrid:
    """Read cell text up to ``max_row`` (anchors never look below themselves)."""
    raw = archive.read(worksheet_part)
    merges = [
        (*_split_ref(a.decode()), *_split_ref(b.decode()))
        for a, b in _MERGE_REF.findall(raw)
    ]
    cells: dict[tuple[int, int], str] = {}
    for _event, element in ET.iterparse(io.BytesIO(raw), events=("end",)):
        if element.tag != _M_ROW:
            continue
        row_number = int(element.get("r") or 0)
        if row_number > max_row:
            element.clear()
            break
        for cell in element.findall("m:c", _NS):
            ref = cell.get("r")
            if not ref:
                continue
            value = _cell_value(cell, shared)
            if value:
                col, row = _split_ref(ref)
                cells[(row, col)] = value
        element.clear()
    return _CellGrid(cells, merges)


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", _NS)
        if inline is None:
            return ""
        return _compact("".join(t.text or "" for t in inline.iter(_M_T)))
    value = cell.find("m:v", _NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        try:
            return _compact(shared[int(value.text)])
        except (ValueError, IndexError):
            return ""
    if cell_type == "b":
        return "TRUE" if value.text.strip() == "1" else "FALSE"
    return _compact(value.text)


def _compact(text: str) -> str:
    return " ".join(text.split())


# -------------------------------------------------------------------------- anchors


def _anchor_cells(anchor: ET.Element) -> tuple[str, str | None, str | None]:
    """→ (anchor_type, from_ref, to_ref); refs are 1-based "A1" style."""
    tag = anchor.tag.rsplit("}", 1)[-1]
    if tag == "absoluteAnchor":
        return "absolute", None, None
    from_ref = _marker_ref(anchor.find("xdr:from", _NS))
    if tag == "oneCellAnchor":
        return "oneCell", from_ref, None
    return "twoCell", from_ref, _marker_ref(anchor.find("xdr:to", _NS))


def _marker_ref(marker: ET.Element | None) -> str | None:
    if marker is None:
        return None
    col = marker.find("xdr:col", _NS)
    row = marker.find("xdr:row", _NS)
    if col is None or row is None or col.text is None or row.text is None:
        return None
    return _cell_ref(int(col.text) + 1, int(row.text) + 1)


def _display_size(anchor: ET.Element, node: ET.Element) -> tuple[int | None, int | None]:
    """Rendered size in pixels, when the file records one.

    Excel writes it on the picture (``spPr/a:xfrm/a:ext``); one-cell and
    absolute anchors also carry an ``xdr:ext``. Two-cell anchors written by
    openpyxl record neither, in which case the size is unknown (``None``) and
    the picture is treated as real content.
    """
    ext = node.find("xdr:spPr/a:xfrm/a:ext", _NS)
    if ext is None:
        ext = anchor.find("xdr:ext", _NS)
    if ext is None:
        return None, None
    try:
        width = round(int(ext.get("cx") or 0) / _EMU_PER_PX)
        height = round(int(ext.get("cy") or 0) / _EMU_PER_PX)
    except ValueError:
        return None, None
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _shape_text(shape: ET.Element) -> str:
    """Text-box content: one line per paragraph, ``<a:br/>`` as a line break."""
    body = shape.find("xdr:txBody", _NS)
    if body is None:
        return ""
    lines: list[str] = []
    for paragraph in body.findall(_A_P):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == _A_T:
                pieces.append(node.text or "")
            elif node.tag == _A_BR:
                pieces.append("\n")
        for line in "".join(pieces).split("\n"):
            line = _compact(line)
            if line:
                lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- labels


def _scan_labels(
    grid: _CellGrid, from_ref: str | None, to_ref: str | None
) -> Iterator[tuple[str, str]]:
    """Yield (text, cell) candidates nearest-first, starting at the anchor row."""
    if from_ref is None:
        return
    from_col, from_row = _split_ref(from_ref)
    to_col = max(from_col, _split_ref(to_ref)[0]) if to_ref else from_col
    lowest_row = max(1, from_row - _MAX_LABEL_SCAN_ROWS)
    for row in range(from_row, lowest_row - 1, -1):
        for col in range(from_col, to_col + 1):
            text, cell = grid.lookup(row, col)
            if text:
                yield text, cell
        for col in range(from_col - 1, 0, -1):
            text, cell = grid.lookup(row, col)
            if text:
                yield text, cell


def _infer_label(
    grid: _CellGrid, from_ref: str | None, to_ref: str | None
) -> tuple[str | None, str | None]:
    for text, cell in _scan_labels(grid, from_ref, to_ref):
        return text[:_MAX_LABEL_CHARS], cell
    return None, None


# ------------------------------------------------------------------------ extraction


def extract_xlsx_drawings(document_bytes: bytes) -> dict[int, SheetDrawings]:
    """Pictures and text boxes per worksheet, keyed by 1-based sheet index.

    Every sheet in the workbook is included (hidden ones too — the caller that
    skips hidden sheets simply ignores their key). Sheets with no drawing part
    are omitted. Charts, connectors, and shapes without text are ignored.
    """
    archive = zipfile.ZipFile(io.BytesIO(document_bytes))
    try:
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names:
            return {}
        shared: list[str] | None = None
        result: dict[int, SheetDrawings] = {}
        for sheet_index, sheet_name, _state, worksheet_part in _workbook_sheets(archive):
            if worksheet_part not in names:
                continue
            drawing_parts = [
                target
                for rel_type, target, external in _read_rels(archive, worksheet_part).values()
                if rel_type == _REL_DRAWING and not external and target in names
            ]
            anchors: list[tuple[ET.Element, dict[str, tuple[str, str, bool]]]] = []
            for drawing_part in drawing_parts:
                drawing_rels = _read_rels(archive, drawing_part)
                root = ET.fromstring(archive.read(drawing_part))
                anchors.extend((anchor, drawing_rels) for anchor in root)
            if not anchors:
                continue

            max_row = 1
            for anchor, _rels in anchors:
                _type, from_ref, _to = _anchor_cells(anchor)
                if from_ref is not None:
                    max_row = max(max_row, _split_ref(from_ref)[1])
            if shared is None:
                shared = _shared_strings(archive)
            grid = _load_grid(archive, worksheet_part, max_row=max_row, shared=shared)

            # Objects are numbered in reading order (anchor row, then column),
            # not XML order, so "image 1 of 8" is the top-left one.
            pictures: list[_Placed] = []
            shapes: list[_Placed] = []
            for position, (anchor, drawing_rels) in enumerate(anchors):
                anchor_type, from_ref, to_ref = _anchor_cells(anchor)
                order = _reading_order(from_ref, position)
                # ``iter`` also reaches pictures/shapes nested inside groups.
                for node in anchor.iter():
                    tag = node.tag.rsplit("}", 1)[-1]
                    if tag not in ("pic", "sp"):
                        continue
                    if tag == "sp" and not _shape_text(node):
                        continue  # plain rectangle/arrow: nothing to index
                    placed = _Placed(
                        order, anchor, node, drawing_rels, anchor_type, from_ref, to_ref
                    )
                    (pictures if tag == "pic" else shapes).append(placed)

            images = [
                _build_image(
                    archive,
                    placed,
                    sheet_index=sheet_index,
                    sheet_name=sheet_name,
                    ordinal=ordinal,
                    grid=grid,
                )
                for ordinal, placed in enumerate(sorted(pictures, key=lambda p: p.order), start=1)
            ]
            text_boxes = [
                XlsxTextBox(
                    sheet_index=sheet_index,
                    sheet_name=sheet_name,
                    ordinal=ordinal,
                    shape_name=_object_name(placed.node) or f"Shape {ordinal}",
                    anchor_from=placed.from_ref,
                    anchor_to=placed.to_ref,
                    text=_shape_text(placed.node),
                )
                for ordinal, placed in enumerate(sorted(shapes, key=lambda p: p.order), start=1)
            ]
            if images or text_boxes:
                result[sheet_index] = SheetDrawings(
                    sheet_index=sheet_index,
                    sheet_name=sheet_name,
                    images=tuple(images),
                    text_boxes=tuple(text_boxes),
                )
        return result
    finally:
        archive.close()


@dataclass(frozen=True)
class _Placed:
    """A picture or shape node together with the anchor that positions it."""

    order: tuple[int, int, int]
    anchor: ET.Element
    node: ET.Element
    drawing_rels: dict[str, tuple[str, str, bool]]
    anchor_type: str
    from_ref: str | None
    to_ref: str | None


def _reading_order(from_ref: str | None, position: int) -> tuple[int, int, int]:
    """Sort key: anchor row, anchor column, then XML order for unanchored objects."""
    if from_ref is None:
        return (1 << 30, 1 << 30, position)
    col, row = _split_ref(from_ref)
    return (row, col, position)


def _object_name(node: ET.Element) -> str | None:
    """``cNvPr/@name`` of a picture or shape ("Picture 17", "TextBox 3")."""
    for path in ("xdr:nvPicPr/xdr:cNvPr", "xdr:nvSpPr/xdr:cNvPr"):
        props = node.find(path, _NS)
        if props is not None:
            return props.get("name") or None
    return None


def _build_image(
    archive: zipfile.ZipFile,
    placed: _Placed,
    *,
    sheet_index: int,
    sheet_name: str,
    ordinal: int,
    grid: _CellGrid,
) -> XlsxImage:
    pic = placed.node
    anchor_type, from_ref, to_ref = placed.anchor_type, placed.from_ref, placed.to_ref
    shape_name = _object_name(pic) or f"Picture {ordinal}"
    name_el = pic.find("xdr:nvPicPr/xdr:cNvPr", _NS)
    description = (name_el.get("descr") if name_el is not None else None) or None

    media_path: str | None = None
    data = b""
    blip = pic.find("xdr:blipFill/a:blip", _NS)
    if blip is not None:
        rel = placed.drawing_rels.get(blip.get(_R_EMBED) or "")
        if (
            rel is not None
            and rel[0] == _REL_IMAGE
            and not rel[2]
            and rel[1] in archive.namelist()
        ):
            media_path = rel[1]
            data = archive.read(media_path)

    width, height = _display_size(placed.anchor, pic)
    label, label_cell = _infer_label(grid, from_ref, to_ref)
    extension = posixpath.splitext(media_path or "")[1].lower()
    return XlsxImage(
        sheet_index=sheet_index,
        sheet_name=sheet_name,
        ordinal=ordinal,
        shape_name=shape_name,
        description=description,
        anchor_type=anchor_type,
        anchor_from=from_ref,
        anchor_to=to_ref,
        label=label,
        label_cell=label_cell,
        media_path=media_path,
        media_sha256=hashlib.sha256(data).hexdigest() if data else None,
        content_type=_MEDIA_CONTENT_TYPES.get(extension) if media_path else None,
        display_width_px=width,
        display_height_px=height,
        data=data,
    )
