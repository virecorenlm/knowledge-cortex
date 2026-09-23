"""A vault backend over a local directory (an Obsidian vault is a plain folder).

Same async interface as ingest.obsidian_client.ObsidianClient (list_dir,
read_note, write_note) plus the operations rename/delete propagation needs,
which the MCP client does not provide (its server's move tool schema is not
verified, so ObsidianClient reports supports_moves = False and those
operations fail closed there):

    create_note(path, content)   no-clobber create
    move_note(src, dst)          no-clobber move (hard link + unlink)
    remove_note(path)            used ONLY to purge Cortex trash entries

Every path is vault-relative and validated: no absolute paths, no "..", no
NUL, and no symlink anywhere along the resolved path, so nothing can escape
the vault root.
"""

import os
import tempfile
from pathlib import Path, PurePosixPath


class VaultPathError(ValueError):
    pass


class FilesystemVault:
    supports_moves = True

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise VaultPathError(f"vault root {self.root} is not a directory")

    def resolve(self, rel, allow_root=False):
        if not isinstance(rel, str) or "\x00" in rel or rel.startswith("/") or "\\" in rel:
            raise VaultPathError(f"invalid vault path {rel!r}")
        parts = PurePosixPath(rel).parts if rel else ()
        if any(p in ("..", ".") for p in parts) or (not parts and not allow_root):
            raise VaultPathError(f"invalid vault path {rel!r}")
        current = self.root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise VaultPathError(f"vault path {rel!r} goes through a symlink")
        if current != self.root and self.root not in current.parents:
            raise VaultPathError(f"vault path {rel!r} escapes the vault root")
        return current

    async def list_dir(self, path=""):
        directory = self.resolve(path, allow_root=True)
        if not directory.is_dir():
            return []
        return sorted(p.name + ("/" if p.is_dir() else "") for p in directory.iterdir() if not p.is_symlink())

    async def note_exists(self, path):
        """O(1) existence check (the generic list_dir-based check is O(entries))."""
        target = self.resolve(path)
        return target.is_file() and not target.is_symlink()

    async def read_note(self, path):
        target = self.resolve(path)
        return {"content": target.read_text(encoding="utf-8"), "path": path}

    def _write_temp(self, target, content):
        target.parent.mkdir(parents=True, exist_ok=True)
        self.resolve(str(PurePosixPath(*target.relative_to(self.root).parts)))  # re-check after mkdir
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".cortex.tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        return tmp

    async def write_note(self, path, content):
        target = self.resolve(path)
        tmp = self._write_temp(target, content)
        try:
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return "ok"

    async def create_note(self, path, content):
        target = self.resolve(path)
        tmp = self._write_temp(target, content)
        try:
            os.link(tmp, target)  # fails with FileExistsError instead of clobbering
        finally:
            Path(tmp).unlink(missing_ok=True)
        return "ok"

    async def move_note(self, src, dst):
        source, dest = self.resolve(src), self.resolve(dst)
        if not source.is_file():
            raise FileNotFoundError(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.resolve(dst)
        os.link(source, dest)  # no-clobber
        os.unlink(source)
        return "ok"

    async def remove_note(self, path):
        self.resolve(path).unlink()
        return "ok"
