# 📦 File Split & Recombine Instructions

Large files have been divided into smaller parts (e.g., `af_500_1_cbow_wxd_part_aa`, `af_500_1_cbow_wxd_part_ab`, etc. — letter suffixes in `split -a 2` order, appended directly after `_part_`, no dot and no digit padding) for easier upload/download.  
Follow these instructions to **recombine** and **verify** them once downloaded. The recombined file drops the `_part_aa` suffix and is the real `.csv.bz2` model file (e.g. `af_500_1_cbow_wxd_part_aa` + `af_500_1_cbow_wxd_part_ab` + ... → `af_500_1_cbow_wxd.csv.bz2`).

---

## 🪟 Windows Users

### ✅ Option 1: Use the Included PowerShell Script (`FileChunker.ps1`)
1. Download **all parts** (e.g., `file.bz2.part_000` through `file.bz2.part_00N`) into the same folder.  
2. Download the helper script **`FileChunker.ps1`** (included in this package).  
3. Open **PowerShell**:
   - Press **Win+X → Windows PowerShell (or Terminal)**  
   - Navigate to your folder:
     ```powershell
     cd "C:\path\to\your\folder"
     ```
4. Run the tool:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\FileChunker.ps1
   ```
5. Choose **Option 2: Recombine parts into a file**.
   - Enter the part prefix (e.g. `C:\path\to\af_500_1_cbow_wxd_part_`)
   - Choose a name for the output (e.g. `C:\path\to\af_500_1_cbow_wxd.csv.bz2`)
6. Once complete, verify (Option 3) if you have the original file hash.

---

### ✅ Option 2: Manual Recombine (no script)
If you don’t want to use the PowerShell script, you can still recombine manually:

1. Open PowerShell in the folder containing the parts.  
2. Run:
   ```powershell
   Get-ChildItem "af_500_1_cbow_wxd_part_*" |
     Sort-Object Name |
     Get-Content -Encoding Byte -ReadCount 0 |
     Set-Content "af_500_1_cbow_wxd.csv.bz2" -Encoding Byte
   ```
3. You can now decompress as usual:
   ```powershell
   bzip2 -d af_500_1_cbow_wxd.csv.bz2
   ```
   or use an app like **7-Zip**.

---

## 🍎 macOS & 🐧 Linux Users

### 🔁 Recombine Parts
1. Place **all parts** in the same directory.
2. Open **Terminal** and navigate to the folder:
   ```bash
   cd /path/to/folder
   ```
3. Combine them (the letter-suffixed parts sort correctly with a plain wildcard):
   ```bash
   cat af_500_1_cbow_wxd_part_* > af_500_1_cbow_wxd.csv.bz2
   ```
4. Verify integrity (optional, if you have the hash):
   ```bash
   sha256sum af_500_1_cbow_wxd.csv.bz2
   ```
   Compare it to the provided checksum.
5. Decompress:
   ```bash
   bzip2 -d af_500_1_cbow_wxd.csv.bz2
   ```

---

## 🧠 Notes & Tips
- ⚠️ **All parts must be downloaded completely** — if one is missing or incomplete, recombination will fail.  
- ✅ The parts are **sequential** (`_part_aa`, `_part_ab`, ... in that order) — do not rename them.  
- 💾 Use `sha256sum` (Linux/macOS) or PowerShell’s `Get-FileHash` to confirm file integrity:
  ```powershell
  Get-FileHash -Algorithm SHA256 af_500_1_cbow_wxd.csv.bz2
  ```
- 🧱 Once recombined, the result is a **standard `.bz2` file** — decompress it with any tool that supports bzip2.  
- 🧰 If you need to re-split large files, use the included **FileChunker.ps1** script (cross-compatible on Windows).
