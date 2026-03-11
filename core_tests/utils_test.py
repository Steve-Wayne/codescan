import os
import subprocess
import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, mock_open, patch

from core.code_scanner.code_scanner import CodeScanner
from core.runner import display_scan_result, format_as_markdown
from core.utils.code_summary_extractor import (
    generate_code_summary,
    read_files_and_extract_code_summary,
)
from core.utils.file_extractor import (
    get_changed_files_in_pr,
    get_changed_files_in_repo,
    is_git_repo,
)
from core.utils.provider_creator import init_provider


class TestUtils(unittest.TestCase):
    """
    File Extractor Tests
    """

    @patch("subprocess.check_output")
    def test__isGitRepo__valid(self, mock_check_output):
        mock_check_output.return_value = b"true\n"
        self.assertTrue(is_git_repo(os.path.join("test", "valid", "repo")))

    @patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")
    )
    def test__isGitRepo__invalid(self, mock_check_output):
        self.assertFalse(is_git_repo(os.path.join("test", "invalid", "repo")))

    def test__getChangedFilesInPr(self):
        mock_github = MagicMock()
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [
            MagicMock(filename="file_one.py"),
            MagicMock(filename="file_two.py"),
        ]
        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        github_module = ModuleType("github")
        github_module.Github = mock_github
        with patch.dict(sys.modules, {"github": github_module}):
            files = get_changed_files_in_pr("some/repo", 1, "fake_token")
        self.assertEqual(files, ["file_one.py", "file_two.py"])

    @patch("subprocess.check_output")
    @patch("core.utils.file_extractor.is_git_repo", return_value=True)
    def test__getChangedFilesInRepo(self, mock_is_git_repo, mock_check_output):
        mock_check_output.return_value = "file_one.py\nfile_two.py\n"
        files = get_changed_files_in_repo(os.path.join("some", "repo"))
        self.assertEqual(files, ["file_one.py", "file_two.py"])
        mock_check_output.assert_called_once_with(
            ["git", "-C", os.path.join("some", "repo"), "diff", "--name-only"],
            text=True,
        )

    @patch("core.utils.file_extractor.is_git_repo", return_value=False)
    def test__getChangedFilesInRepo_inValid(self, mock_is_git_repo):
        with self.assertRaises(ValueError):
            get_changed_files_in_repo(os.path.join("invalid", "repo"))

    """
    Code Summary Extractor Tests
    """

    @patch("builtins.open", new_callable=mock_open, read_data="sample code")
    @patch("os.path.isfile", return_value=True)
    def test__readFilesAndExtractCodeSummary__isValidFile(self, mock_isfile, mock_open):
        file_paths = [
            os.path.join("some", "repo", "file_one.py"),
            os.path.join("some", "repo", "file_two.py"),
        ]

        code_summary = read_files_and_extract_code_summary(file_paths)

        self.assertIn("File: file_one.py", code_summary)
        self.assertIn("sample code", code_summary)
        self.assertIn("File: file_two.py", code_summary)

        mock_open.assert_any_call(
            os.path.join("some", "repo", "file_one.py"), "r", encoding="utf-8"
        )
        mock_open.assert_any_call(
            os.path.join("some", "repo", "file_two.py"), "r", encoding="utf-8"
        )

    @patch("builtins.open", new_callable=mock_open, read_data="sample code")
    @patch("os.path.isfile", return_value=False)
    def test__readFilesAndExtractCodeSummary__isInvalidFile(
        self, mock_isfile, mock_open
    ):
        file_paths = [os.path.join("some", "repo", "invalid_file.py")]

        code_summary = read_files_and_extract_code_summary(file_paths)

        self.assertEqual(code_summary, "")
        mock_open.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.isfile", return_value=True)
    def test__readFilesAndExtractCodeSummary__decodingError(
        self, mock_isfile, mock_open
    ):
        mock_open.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "decoding error")
        file_paths = [os.path.join("some", "repo", "file_one.py")]

        code_summary = read_files_and_extract_code_summary(file_paths)

        self.assertEqual(code_summary, "")
        mock_open.assert_called_once_with(
            os.path.join("some", "repo", "file_one.py"), "r", encoding="utf-8"
        )

    @patch("builtins.open", new_callable=mock_open, read_data="sample code")
    @patch("os.path.isfile", return_value=True)
    def test__generateCodeSummary__isValid(self, mock_isfile, mock_open):
        changed_files = ["file_one.py", "file_two.py"]

        code_summary = generate_code_summary(
            os.path.join("some", "repo"), changed_files
        )
        self.assertIn("sample code", code_summary)

        mock_open.assert_any_call(
            os.path.join("some", "repo", "file_one.py"), "r", encoding="utf-8"
        )
        mock_open.assert_any_call(
            os.path.join("some", "repo", "file_two.py"), "r", encoding="utf-8"
        )

    """
    Provider Creator Tests
    """

    @patch("core.utils.provider_creator._get_provider_class")
    def test__initOpenAIProvider(self, mock_get_provider_class):
        mock_provider_class = MagicMock()
        mock_get_provider_class.return_value = mock_provider_class
        init_provider("openai", None)
        mock_get_provider_class.assert_called_once_with("openai")
        mock_provider_class.assert_called_once_with(model="gpt-4o-mini")

    @patch("core.utils.provider_creator._get_provider_class")
    def test__initGoogleGeminiAIProvider(self, mock_get_provider_class):
        mock_provider_class = MagicMock()
        mock_get_provider_class.return_value = mock_provider_class
        init_provider("gemini", None)
        mock_get_provider_class.assert_called_once_with("gemini")
        mock_provider_class.assert_called_once_with(model="gemini-pro")

    @patch("core.utils.provider_creator._get_provider_class")
    def test_initialize_client_custom(self, mock_get_provider_class):
        mock_provider_class = MagicMock()
        mock_get_provider_class.return_value = mock_provider_class
        init_provider(
            "custom",
            "custom-model",
            "http://localhost",
            5000,
            "custom-token",
            "/api/v1/scan",
        )
        mock_get_provider_class.assert_called_once_with("custom")
        mock_provider_class.assert_called_once_with(
            model="custom-model",
            host="http://localhost",
            port=5000,
            token="custom-token",
            endpoint="/api/v1/scan",
        )

    """
    Runner Tests
    """

    def test__formatAsMarkdown(self):
        result = "This is a test result"
        expected_output = "## Code Security Analysis Results\nThis is a test result"
        self.assertEqual(format_as_markdown(result), expected_output)

    @patch("builtins.print")
    @patch("core.runner.display_markdown", None)
    def test__displayScanResult__fallsBackToPrint(self, mock_print):
        display_scan_result("scan result")
        mock_print.assert_called_once_with(
            "## Code Security Analysis Results\nscan result"
        )

    """
    Code Scanner Tests
    """

    @patch("core.code_scanner.code_scanner.init_provider")
    @patch("core.code_scanner.code_scanner.read_files_and_extract_code_summary")
    @patch("os.walk")
    def test__scanFiles__skipsProviderCallWhenNoReadableFiles(
        self, mock_walk, mock_read_files, mock_init_provider
    ):
        mock_walk.return_value = [("repo", (), ("binary.dat",))]
        mock_read_files.return_value = "   "
        mock_provider = MagicMock()
        mock_init_provider.return_value = mock_provider
        mock_args = MagicMock(changes_only=False, directory=".", repo=None, pr_number=None)

        scan_result = CodeScanner(args=mock_args).scan()

        self.assertEqual(scan_result, "No readable files found in the specified directory.")
        mock_provider.scan_code.assert_not_called()

    @patch("core.code_scanner.code_scanner.get_changed_files_in_pr")
    @patch("core.code_scanner.code_scanner.generate_code_summary")
    @patch("core.code_scanner.code_scanner.init_provider")
    def test__scanChanges__skipsProviderCallWhenChangedFilesAreUnreadable(
        self, mock_init_provider, mock_generate_summary, mock_get_changed_files_in_pr
    ):
        mock_get_changed_files_in_pr.return_value = ["binary.dat"]
        mock_generate_summary.return_value = ""
        mock_provider = MagicMock()
        mock_init_provider.return_value = mock_provider
        mock_args = MagicMock(
            changes_only=True,
            directory=".",
            repo="owner/repo",
            pr_number=1,
            github_token="token",
        )

        scan_result = CodeScanner(args=mock_args).scan()

        self.assertEqual(
            scan_result, "No readable source files found in the detected changes."
        )
        mock_provider.scan_code.assert_not_called()


if __name__ == "__main__":
    unittest.main()
