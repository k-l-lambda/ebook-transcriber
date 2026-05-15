from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .llm_client import LLMClient
from .markdown_writer import append_page_markdown, checkpoint_path_for, prepare_progressive_output, write_completed_pages
from .prompts import markdown_translation_prompt

_PAGE_MARKER_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownPage:
    page_number: int
    body: str


@dataclass(frozen=True)
class TranslateMarkdownOptions:
    input_path: Path
    output_path: Path
    model: str
    output_language: str
    pages: str | None = None
    restart: bool = False
    dry_run: bool = False
    verbose: bool = False
    strict_pages: bool = True
    max_concurrency: int = 1


Progress = Callable[[str], None]


class TextChatClient(Protocol):
    def text_chat(self, prompt: str) -> str: ...


ClientFactory = Callable[[], TextChatClient]


def split_markdown_pages(markdown: str) -> list[MarkdownPage]:
    matches = list(_PAGE_MARKER_RE.finditer(markdown))
    if not matches:
        return [MarkdownPage(1, markdown.strip())]

    pages: list[MarkdownPage] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        pages.append(MarkdownPage(int(match.group(1)), markdown[start:end].strip()))
    return pages


def parse_page_numbers(spec: str | None, available_pages: list[int], strict: bool = True) -> set[int] | None:
    if not spec:
        return None
    available = set(available_pages)
    selected: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError(f"invalid page range: {chunk}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(chunk))
    missing = selected - available
    if strict and missing:
        raise ValueError(f"page(s) not found in source markdown: {', '.join(str(p) for p in sorted(missing))}")
    return selected & available


def _selected_pages(pages: list[MarkdownPage], page_spec: str | None, strict: bool = True) -> list[MarkdownPage]:
    wanted = parse_page_numbers(page_spec, [page.page_number for page in pages], strict=strict)
    if wanted is None:
        return pages
    return [page for page in pages if page.page_number in wanted]


def translate_markdown_file(
    options: TranslateMarkdownOptions,
    client: TextChatClient | None = None,
    progress: Progress | None = None,
) -> Path:
    source_text = options.input_path.read_text(encoding="utf-8")
    pages = _selected_pages(split_markdown_pages(source_text), options.pages, strict=options.strict_pages)

    if options.dry_run:
        if progress:
            page_list = ", ".join(str(page.page_number) for page in pages) or "none"
            progress(f"{options.input_path} -> {options.output_path}: pages {page_list}")
        return options.output_path

    llm = client or LLMClient(model=options.model)
    checkpoint_path = checkpoint_path_for(options.output_path)
    completed_pages = prepare_progressive_output(options.output_path, checkpoint_path, options.restart)

    for page in pages:
        if page.page_number in completed_pages:
            if options.verbose and progress:
                progress(f"skip {options.input_path.name} page {page.page_number}")
            continue
        if options.verbose and progress:
            progress(f"translate {options.input_path.name} page {page.page_number}")
        prompt = markdown_translation_prompt(page.body, options.output_language)
        translated = llm.text_chat(prompt).strip()
        append_page_markdown(options.output_path, page.page_number, translated)
        completed_pages.add(page.page_number)
        write_completed_pages(checkpoint_path, completed_pages)

    return options.output_path


def _translate_markdown_tree_file_options(options: TranslateMarkdownOptions) -> list[TranslateMarkdownOptions]:
    file_options: list[TranslateMarkdownOptions] = []
    for source_path in sorted(options.input_path.rglob("*.md")):
        if source_path.name.endswith(".checkpoint.json"):
            continue
        rel_path = source_path.relative_to(options.input_path)
        output_path = options.output_path / rel_path
        file_options.append(
            TranslateMarkdownOptions(
                input_path=source_path,
                output_path=output_path,
                model=options.model,
                output_language=options.output_language,
                pages=options.pages,
                restart=options.restart,
                dry_run=options.dry_run,
                verbose=options.verbose,
                strict_pages=False,
                max_concurrency=1,
            )
        )
    return file_options


def translate_markdown_tree(
    options: TranslateMarkdownOptions,
    progress: Progress | None = None,
    client_factory: ClientFactory | None = None,
) -> list[Path]:
    if options.input_path.is_file():
        client = client_factory() if client_factory else None
        return [translate_markdown_file(options, client=client, progress=progress)]
    if not options.input_path.is_dir():
        raise FileNotFoundError(options.input_path)

    file_options = _translate_markdown_tree_file_options(options)
    if options.dry_run:
        return [translate_markdown_file(item, progress=progress) for item in file_options]
    if options.max_concurrency <= 1 or len(file_options) <= 1:
        outputs: list[Path] = []
        for item in file_options:
            client = client_factory() if client_factory else None
            outputs.append(translate_markdown_file(item, client=client, progress=progress))
        return outputs

    outputs: list[Path] = []
    failures: list[str] = []
    worker_count = min(options.max_concurrency, len(file_options))

    def run_file(item: TranslateMarkdownOptions) -> Path:
        client = client_factory() if client_factory else None
        return translate_markdown_file(item, client=client, progress=progress)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_file, item): item for item in file_options}
        for future in as_completed(futures):
            item = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                failures.append(f"{item.input_path.name}: {exc}")
                if progress:
                    progress(f"failed {item.input_path}: {exc}")
    if failures:
        raise RuntimeError("translation failures: " + "; ".join(failures))
    return sorted(outputs)
