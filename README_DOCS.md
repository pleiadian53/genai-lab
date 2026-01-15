# GenAI Lab Documentation System

This repository uses **MkDocs** with the **Material theme** and **MathJax** for documentation rendering.

---

## Why MkDocs + MathJax?

**Problem:** GitHub's Markdown renderer doesn't support LaTeX math (`$...$`, `$$...$$`).

**Solution:** MkDocs with MathJax renders:
- ✅ LaTeX equations properly
- ✅ Beautiful Material theme
- ✅ Search functionality
- ✅ Auto-deployment to GitHub Pages

---

## Viewing the Documentation

### **Option 1: GitHub Pages (Recommended)**

**URL:** https://pleiadian53.github.io/genai-lab/

**Features:**
- Math equations render correctly
- Professional theme with navigation
- Fast and reliable
- Mobile-friendly

### **Option 2: Local Preview**

Build and serve locally:

```bash
# Install dependencies (one time)
pip install -r requirements-docs.txt

# Serve locally (auto-reloads on changes)
mkdocs serve

# Open browser to http://127.0.0.1:8000/
```

### **Option 3: GitHub Source (Raw)**

Browse markdown files directly on GitHub:
- https://github.com/pleiadian53/genai-lab/tree/main/docs

**Note:** Math won't render, but files are readable.

---

## File Structure

```
genai-lab/
├── docs/                          # Documentation source (Markdown + LaTeX)
│   ├── index.md                  # Landing page (copied from README.md)
│   ├── javascripts/
│   │   └── mathjax.js           # MathJax configuration
│   ├── stylesheets/
│   │   └── extra.css            # Custom styling
│   ├── foundation_models/        # Topic folders
│   ├── DiT/
│   ├── JEPA/
│   ├── VAE/
│   └── ...                       # All other doc folders
│
├── mkdocs.yml                    # MkDocs configuration
├── requirements-docs.txt         # Python dependencies for docs
│
├── .github/workflows/
│   └── docs.yml                  # Auto-deploy to GitHub Pages
│
└── site/                         # Built HTML (gitignored)
```

---

## Writing Documentation

### **Markdown with LaTeX Math**

Use standard LaTeX syntax:

**Inline math:**
```markdown
The ELBO is $\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$
```

**Display math:**
```markdown
$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$
```

### **Important: Blank Lines for Lists**

MkDocs requires blank lines before lists:

❌ **BAD:**
```markdown
where:
- $\theta$ = decoder parameters
```

✅ **GOOD:**
```markdown
where:

- $\theta$ = decoder parameters
```

### **Code Blocks**

Use triple backticks with language:

```python
def elbo_loss(x, z, theta, phi):
    recon = reconstruction_loss(x, z, theta)
    kl = kl_divergence(z, phi)
    return recon + kl
```

---

## Deployment Workflow

### **Automatic Deployment (Recommended)**

1. **Edit documentation** in `docs/` folder
2. **Commit and push** to `main` branch:
   ```bash
   git add docs/
   git commit -m "Update documentation"
   git push
   ```

3. **GitHub Actions** automatically:
   - Installs MkDocs + dependencies
   - Builds the site
   - Deploys to `gh-pages` branch

4. **Site updates** in ~2-3 minutes at https://pleiadian53.github.io/genai-lab/

### **Manual Build (Testing)**

Test locally before pushing:

```bash
# Build only (no server)
mkdocs build

# Check for warnings
mkdocs build --strict

# Serve locally
mkdocs serve
```

---

## Adding New Documents

### **Step 1: Create the Markdown File**

```bash
# Example: Add a new SDE tutorial
touch docs/SDE/01_sde_foundations.md
```

### **Step 2: Write Content with Math**

```markdown
# SDE Foundations

The SDE formulation is:

$$
dx_t = f(x_t, t) dt + g(t) dW_t
$$

where:

- $f(x_t, t)$ = drift term
- $g(t)$ = diffusion coefficient
- $dW_t$ = Brownian motion
```

### **Step 3: Add to Navigation (Optional)**

Edit `mkdocs.yml`:

```yaml
nav:
  # ... other sections
  - 'SDE (Stochastic Differential Equations)':
      - 'Overview': SDE/README.md
      - '01: Foundations': SDE/01_sde_foundations.md  # Add this line
```

**Note:** Files are accessible by URL even if not in nav.

### **Step 4: Commit and Push**

```bash
git add docs/SDE/01_sde_foundations.md mkdocs.yml
git commit -m "Add SDE foundations tutorial"
git push
```

---

## Updating the Landing Page

The landing page (`docs/index.md`) is copied from `README.md`. To update:

```bash
cp README.md docs/index.md
git add docs/index.md
git commit -m "Update landing page"
git push
```

---

## Troubleshooting

### **Math Not Rendering?**

**Check:**
1. Blank lines before `$$` blocks?
2. Using `$...$` for inline, `$$...$$` for display?
3. Escaping special characters (`\{`, `\}` for literal braces)?

**Test locally:**
```bash
mkdocs serve
```

### **GitHub Actions Failing?**

**Check workflow logs:**
1. Go to https://github.com/pleiadian53/genai-lab/actions
2. Click latest "Deploy Documentation" run
3. Read error messages

**Common issues:**
- Broken links (use relative paths: `../other_doc.md`)
- Missing files referenced in `mkdocs.yml`
- Math syntax errors (unclosed delimiters)

### **List Not Rendering?**

Add blank line after "where:" or bold headers:

```markdown
where:

- item 1  # Blank line above required!
```

---

## Configuration Files

### **mkdocs.yml**

Main configuration:
- Site name, description, URL
- Theme settings (colors, features)
- Markdown extensions (MathJax, etc.)
- Navigation structure

### **docs/javascripts/mathjax.js**

MathJax 3 configuration:
- `$...$` for inline math
- `$$...$$` for display math
- Auto-numbering for equations
- Physics and AMSmath packages

### **requirements-docs.txt**

Python dependencies:
- `mkdocs` - Static site generator
- `mkdocs-material` - Material theme
- `pymdown-extensions` - Extensions (including MathJax)

---

## GitHub Pages Setup

### **First-Time Setup**

After first push with GitHub Actions:

1. **Wait for workflow** to complete (~2-3 min)
2. **Go to Settings** → Pages
3. **Configure:**
   - Source: "Deploy from a branch"
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. **Save**

5. **Visit:** https://pleiadian53.github.io/genai-lab/

### **Subsequent Updates**

Automatic! Just push to `main` branch.

---

## Best Practices

### **Documentation Style**

- ✅ Use LaTeX for all math (not Unicode)
- ✅ Add blank lines before lists and code blocks
- ✅ Use descriptive headings (`##`, `###`)
- ✅ Link between documents with relative paths
- ✅ Include code examples where relevant

### **Organization**

- ✅ One topic per folder (`VAE/`, `DDPM/`, etc.)
- ✅ Numbered series for tutorials (`01_`, `02_`, ...)
- ✅ README.md in each folder for overview
- ✅ Update main `docs/README.md` with new content

### **Math Notation**

- ✅ Be consistent with variable names
- ✅ Define notation: "where $\theta$ = parameters"
- ✅ Use `\mathbb{}` for sets, `\mathcal{}` for distributions
- ✅ Equation numbers for important formulas

---

## Additional Resources

- **MkDocs:** https://www.mkdocs.org/
- **Material Theme:** https://squidfunk.github.io/mkdocs-material/
- **MathJax:** https://www.mathjax.org/
- **LaTeX Math:** https://en.wikibooks.org/wiki/LaTeX/Mathematics

---

## Quick Reference

```bash
# Local preview
mkdocs serve

# Build (check for errors)
mkdocs build --strict

# Add new doc
touch docs/topic/new_file.md
# Edit mkdocs.yml navigation (optional)
git add docs/ mkdocs.yml
git push

# Update landing page
cp README.md docs/index.md
git add docs/index.md
git push
```

---

**Questions?** Check the GRL documentation system as a reference example:
- https://github.com/pleiadian53/GRL
- https://pleiadian53.github.io/GRL/

**Happy documenting!** 📚✨
