# GitHub Repository Setup Guide

Follow these steps to upload your Market Performance Analysis project to GitHub.

## 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com/) and log in to your account.
2. Click on the "+" icon in the upper right corner and select "New repository".
3. Enter a repository name (e.g., "market-performance-analysis").
4. Add an optional description.
5. Choose whether to make the repository public or private.
6. **DO NOT** initialize the repository with a README, .gitignore, or license (since we already have these files).
7. Click "Create repository".

## 2. Initialize Git and Push Your Code

After creating the repository, GitHub will show the commands needed to push an existing repository. Follow these steps from your project directory:

```bash
# Initialize git in your project folder (if not already done)
git init

# Add all files to be committed
git add .

# Create your first commit
git commit -m "Initial commit"

# Add the GitHub repository as a remote
git remote add origin https://github.com/YOUR_USERNAME/market-performance-analysis.git

# Push your code to GitHub
git push -u origin main
```

Note: If your default branch is named "master" instead of "main", use:
```bash
git push -u origin master
```

## 3. Verify Your Repository

1. Refresh your GitHub repository page to see your uploaded files.
2. Make sure all files were properly uploaded.
3. Your README.md will automatically be displayed on the repository's main page.

## 4. Add a License File (Optional)

1. If you want to update the LICENSE file, click on it in your repository.
2. Click the edit button (pencil icon).
3. Replace "[Your Name]" with your actual name.
4. Commit the changes directly to your main branch.

## 5. Enable GitHub Pages (Optional)

If you want to showcase your project with a simple website:

1. Go to your repository's "Settings" tab.
2. Scroll down to "GitHub Pages" section.
3. Under "Source", select the main branch.
4. Click "Save".
5. Your project will be published at `https://YOUR_USERNAME.github.io/market-performance-analysis/`.

## Troubleshooting

- **Authentication issues**: You might need to set up SSH keys or use a personal access token for authentication.
- **Large files**: If you have generated many charts, make sure your .gitignore is properly set up to exclude these directories.
- **Default branch name**: Newer Git installations use "main" instead of "master" as the default branch name. Adjust commands accordingly.

## Next Steps

After successfully pushing your code to GitHub, consider:

1. Adding example charts to your README
2. Setting up GitHub Actions for automated testing
3. Creating releases for stable versions
4. Adding more detailed documentation 