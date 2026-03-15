import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_embeddings_do_not_import_sentence_transformers():
    script = textwrap.dedent(
        """
        import builtins
        import os

        os.environ["ACE_EMBEDDINGS"] = "mock"

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sentence_transformers":
                raise RuntimeError(
                    "sentence_transformers import should stay lazy for mock embeddings"
                )
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        from ace.core.storage.embedding_store import generate_embedding

        vector = generate_embedding("mock retrieval query")
        print(vector.shape[0])
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "384"
