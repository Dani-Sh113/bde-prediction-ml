# GitHub Push Instructions

## Prerequisites

1. **Create GitHub Account** (if you don't have one)
   - Go to https://github.com/signup
   - Create your account

2. **Create a Personal Access Token** (easier than SSH)
   - Go to https://github.com/settings/tokens
   - Click "Generate new token"
   - Give it repo permissions
   - Copy the token (you'll use it for the password)

3. **Configure Git Credentials** (Windows)
   - Open Git Bash
   - Run: `git config --global user.name "Your Name"`
   - Run: `git config --global user.email "your.email@example.com"`

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Enter repository name: **bde-prediction-ml**
3. Description: "Automated machine learning pipeline for predicting C-X bond dissociation energy"
4. Keep it **PUBLIC** (required for open science)
5. **DO NOT** initialize with README or .gitignore (we already have them)
6. Click "Create repository"

## Step 2: Push to GitHub

In Git Bash or command prompt, navigate to the project:

```bash
cd /c/Users/Chemist/bde-prediction-ml
```

Add the remote and push:

```bash
# Add GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/bde-prediction-ml.git

# Verify the remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

When prompted for password, use your Personal Access Token (from prerequisites).

## Step 3: Verify Push Success

Check GitHub:
1. Go to https://github.com/YOUR_USERNAME/bde-prediction-ml
2. You should see all files and commits
3. Check that data and models directories are present

## Step 4: Optional - Add Zenodo DOI (for permanent archiving)

1. Go to https://zenodo.org
2. Connect your GitHub account
3. Select repository for archiving
4. Zenodo will generate a DOI
5. Update README.md with DOI

## Troubleshooting

### "fatal: not a git repository"
Make sure you're in the correct directory:
```bash
cd /c/Users/Chemist/bde-prediction-ml
git status
```

### "Authentication failed"
- If using password: Make sure you used the Personal Access Token
- If using SSH: Make sure SSH key is added to GitHub

### "Branch 'main' already exists"
Skip the branch rename:
```bash
git push -u origin master
```

### "remote already exists"
Remove and re-add:
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/bde-prediction-ml.git
```

## After Push - Update Code in Paper

Once successfully pushed, you can reference:

**Repository URL:** https://github.com/YOUR_USERNAME/bde-prediction-ml

Add this to your paper's "Data Availability" section.

---

**Complete Command Sequence:**

```bash
cd /c/Users/Chemist/bde-prediction-ml
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git remote add origin https://github.com/YOUR_USERNAME/bde-prediction-ml.git
git branch -M main
git push -u origin main
```

Then enter your Personal Access Token when prompted.

Done! Your code is now publicly available on GitHub.
