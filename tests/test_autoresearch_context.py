import io
import tarfile

import pytest

from integrations.autoresearch_context import pack_context, write_tar


def test_private_staging_becomes_rootless_readable_without_changing_bytes(tmp_path):
    root = tmp_path / "input"
    root.mkdir(mode=0o700)
    nested = root / "pilot"
    nested.mkdir(mode=0o700)
    data = nested / "checkpoint"
    data.write_bytes(bytes(range(256)))
    data.chmod(0o600)
    executable = nested / "runner"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    first = io.BytesIO()
    write_tar(root, first)
    second = io.BytesIO()
    write_tar(root, second)
    assert first.getvalue() == second.getvalue()
    with tarfile.open(fileobj=io.BytesIO(first.getvalue())) as archive:
        for member in archive:
            assert member.uid == member.gid == member.mtime == 0
            assert not member.pax_headers
            if member.isdir():
                assert member.mode == 0o755
            else:
                assert member.mode & 0o444 == 0o444
                assert archive.extractfile(member).read() == (root / member.name).read_bytes()
        assert archive.getmember("./pilot/runner").mode == 0o755
    assert data.stat().st_mode & 0o777 == 0o600


def test_reject_link_before_emitting_archive(tmp_path):
    (tmp_path / "link").symlink_to("/etc/passwd")
    stream = io.BytesIO()
    with pytest.raises(ValueError, match="regular files"):
        write_tar(tmp_path, stream)
    assert stream.getvalue() == b""


def test_preserve_existing_archive(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    output = tmp_path / "existing.tar.zst"
    output.write_bytes(b"prior evidence")
    with pytest.raises(FileExistsError):
        pack_context(root, output)
    assert output.read_bytes() == b"prior evidence"


def test_reject_archive_inside_input(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        pack_context(tmp_path, tmp_path / "out.tar.zst")
