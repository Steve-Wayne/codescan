"""
This is the runner of the codescan-ai CLI tool.
"""

try:
    from IPython.display import display_markdown
except ImportError:  # pragma: no cover - exercised via fallback behavior
    display_markdown = None

from core.code_scanner.code_scanner import CodeScanner
from core.utils.argument_parser import parse_arguments


def format_as_markdown(result):
    """
    Formats the scan result as Markdown.
    """
    output = "## Code Security Analysis Results\n"
    output += result
    return output


def display_scan_result(result):
    """
    Displays the scan result in notebook environments and falls back to stdout for CLI use.
    """
    formatted_result = format_as_markdown(result)
    if display_markdown is not None:
        display_markdown(formatted_result)
        return
    print(formatted_result)


def main():
    """
    Main entry point for the CLI. Parses arguments, calls the centralized CodeScanner
    (which performs the scanning by using the AI provider in *args),
    and displays the results.
    """
    args = parse_arguments()
    scan_result = CodeScanner(args).scan()
    display_scan_result(scan_result)


if __name__ == "__main__":
    main()
