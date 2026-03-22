# Backup & Restore Instructions

## 📁 Backup Files Location

All backup files are stored in your **home directory** (`/home/rsaisrujan/`), NOT in the g2s_project folder.

### Available Backups
```bash
# List all backups
ls -lh ~/g2s_backup_*.tar.gz
ls -lh ~/g2s_project_backup_*.tar.gz
```

## 🔄 How to Restore a Backup

### Important: Extract from HOME directory
You **must** be in your home directory to extract properly:

```bash
# Navigate to home directory
cd ~

# List available backups
ls -lh g2s_backup_*.tar.gz
# or
ls -lh g2s_project_backup_*.tar.gz

# Extract backup
tar -xzf g2s_backup_final_YYYYMMDD_HHMMSS.tar.gz

# This creates: ~/g2s_project/
```

### Step-by-Step Example

```bash
# 1. Go to home directory
cd ~

# 2. See which backup you want
ls -lh g2s_backup_*.tar.gz

# 3. Extract (this will restore ~/g2s_project/)
tar -xzf g2s_backup_final_20260322_120000.tar.gz

# 4. Verify extraction
ls -la g2s_project/
cd g2s_project
ls -la
```

## ⚠️ Common Mistakes

❌ **WRONG**: Extracting from inside the project folder
```bash
cd ~/g2s_project
tar -xzf g2s_backup_final_*.tar.gz  # ❌ FAILS - file is not here!
```

✅ **CORRECT**: Extract from home directory
```bash
cd ~
tar -xzf g2s_backup_final_*.tar.gz  # ✅ SUCCESS
```

## 🗂️ What's Inside Each Backup

### Full Backup (g2s_project_backup_*.tar.gz - 240MB)
Includes EVERYTHING:
- Source code (main.py, ui_app.py, backend.py, etc.)
- Models directory
- Virtual environment (g2s_env/)
- Cache files (.tts_cache/)
- Git history (.git/)

### Clean Backup (g2s_backup_final_*.tar.gz - ~15-20MB)
Smart excludes for smaller size:
- ✅ Source code
- ✅ Models
- ✅ README.md, requirements.txt
- ❌ .git history (use GitHub for version control)
- ❌ __pycache__ (Python cache)
- ❌ .tts_cache (audio cache)
- ❌ codes.7z (old archive)

**Recommendation**: Use `g2s_backup_final_*.tar.gz` for regular backups (smaller, faster)

## 📋 Creating a New Backup

```bash
# From home directory
cd ~

# Create a new backup
tar -czf g2s_backup_final_$(date +%Y%m%d_%H%M%S).tar.gz g2s_project/ \
  --exclude='g2s_project_backup_*.tar.gz' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.tts_cache' \
  --exclude='codes.7z'

# Verify it was created
ls -lh g2s_backup_final_*.tar.gz | tail -1
```

**One-liner backup command:**
```bash
cd ~ && tar -czf g2s_backup_final_$(date +%Y%m%d_%H%M%S).tar.gz g2s_project/ --exclude='g2s_project_backup_*.tar.gz' --exclude='.git' --exclude='__pycache__' --exclude='.tts_cache' --exclude='codes.7z' && ls -lh g2s_backup_final_*.tar.gz | tail -1
```

## 🔐 Safe Backup Strategy

### Quick Reference
```bash
# 1. Navigate to home
cd ~

# 2. Create backup
tar -czf g2s_backup_$(date +%Y%m%d_%H%M%S).tar.gz g2s_project/ --exclude='.git' --exclude='__pycache__'

# 3. Verify file exists
ls -lh g2s_backup_*.tar.gz | tail -1

# 4. (Optional) List contents to verify
tar -tzf g2s_backup_*.tar.gz | head -20
```

## 🗑️ Cleanup Old Backups

Keep only recent backups to save space:

```bash
# Remove OLD backup files (keep last 2)
cd ~
ls -lt g2s_backup_*.tar.gz | tail -n +3 | awk '{print $NF}' | xargs rm -f

# Verify remaining
ls -lh g2s_backup_*.tar.gz
```

## 📺 Git vs. Backups

- **GitHub**: Use for version control and sharing code (`git push`)
- **Backups**: Use for local snapshots and disaster recovery

Your code is already backed up on GitHub! Backups are extra insurance.

## ✅ Recovery Checklist

- [ ] Navigate to home: `cd ~`
- [ ] See available backups: `ls -lh g2s_backup_*.tar.gz`
- [ ] Extract: `tar -xzf g2s_backup_final_YYYYMMDD_HHMMSS.tar.gz`
- [ ] Verify: `ls -la g2s_project/`
- [ ] Test: `cd g2s_project && python codes/main.py`

## 📞 Troubleshooting

### Error: "Cannot open: No such file or directory"
**Cause**: You're in the wrong directory
**Solution**: `cd ~` then try extraction again

### Error: "tar: This does not look like a tar archive"
**Cause**: File is corrupted or wrong file format
**Solution**: Create a fresh backup

### Backup file is too large (>100MB)
**Cause**: Includes .git or __pycache__
**Solution**: Use `g2s_backup_final_*.tar.gz` instead (has exclusions)

### Want to verify backup integrity before restoring
```bash
cd ~
tar -tzf g2s_backup_final_*.tar.gz | wc -l  # Count files
tar -tzf g2s_backup_final_*.tar.gz | head   # See first files
```

---

**Last Updated**: March 2026
**Location**: Backup files in `/home/rsaisrujan/`
