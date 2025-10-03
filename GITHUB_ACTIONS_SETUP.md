# GitHub Actions Setup Guide

This guide explains how to set up GitHub Actions for automated testing, building, and publishing of the AI Context Manager package.

## 🔧 Required Secrets

You need to set up the following secrets in your GitHub repository:

### PyPI Secrets
1. **PYPI_API_TOKEN** - Your PyPI API token for publishing packages
2. **TEST_PYPI_API_TOKEN** - Your Test PyPI API token for testing

### Docker Secrets (Optional)
3. **DOCKER_USERNAME** - Your Docker Hub username
4. **DOCKER_PASSWORD** - Your Docker Hub access token

## 📋 Setting Up Secrets

### 1. PyPI API Tokens

**Get PyPI API Token:**
1. Go to [PyPI.org](https://pypi.org) and log in
2. Go to Account Settings → API tokens
3. Create a new API token with scope "Entire account"
4. Copy the token (starts with `pypi-`)

**Get Test PyPI API Token:**
1. Go to [TestPyPI.org](https://test.pypi.org) and log in
2. Go to Account Settings → API tokens
3. Create a new API token with scope "Entire account"
4. Copy the token (starts with `pypi-`)

**Add to GitHub:**
1. Go to your repository on GitHub
2. Click Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add:
   - Name: `PYPI_API_TOKEN`, Value: your PyPI token
   - Name: `TEST_PYPI_API_TOKEN`, Value: your Test PyPI token

### 2. Docker Hub Secrets (Optional)

**Get Docker Hub Access Token:**
1. Go to [Docker Hub](https://hub.docker.com) and log in
2. Go to Account Settings → Security → New Access Token
3. Create a new access token with read/write permissions
4. Copy the token

**Add to GitHub:**
1. Go to your repository on GitHub
2. Click Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add:
   - Name: `DOCKER_USERNAME`, Value: your Docker Hub username
   - Name: `DOCKER_PASSWORD`, Value: your Docker Hub access token

## 🚀 Workflow Overview

### 1. Test Workflow (`.github/workflows/test.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual dispatch

**What it does:**
- Runs tests on Python 3.8-3.12
- Checks code formatting (black, isort, flake8)
- Runs security scans (safety, bandit)
- Tests demo applications
- Tests package imports
- Builds and tests the package

### 2. Publish Workflow (`.github/workflows/publish.yml`)

**Triggers:**
- GitHub releases (automatically publishes to PyPI)
- Manual dispatch with options

**What it does:**
- Builds package for multiple Python versions
- Tests package installation
- Publishes to Test PyPI (for testing)
- Publishes to PyPI (on releases)
- Creates GitHub release with assets

### 3. Demo Deploy Workflow (`.github/workflows/demo-deploy.yml`)

**Triggers:**
- Push to `main` (when demo files change)
- Manual dispatch with deployment options

**What it does:**
- Tests demo applications
- Runs performance benchmarks
- Builds Docker images
- Creates demo releases

### 4. Docker Workflow (`.github/workflows/docker.yml`)

**Triggers:**
- Push to `main` or tags
- Pull requests to `main`
- Manual dispatch

**What it does:**
- Builds multi-architecture Docker images
- Pushes to Docker Hub
- Tests Docker Compose setup
- Runs security scans on images

## 📦 Publishing Process

### Automatic Publishing (Recommended)

1. **Create a Release:**
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. **Create GitHub Release:**
   - Go to GitHub → Releases → Create a new release
   - Choose the tag `v0.2.0`
   - Add release notes
   - Click "Publish release"

3. **GitHub Actions will automatically:**
   - Build the package
   - Test it
   - Publish to PyPI
   - Create release assets

### Manual Publishing

1. **Test on Test PyPI:**
   - Go to Actions → Publish to PyPI
   - Click "Run workflow"
   - Check "Publish to Test PyPI only"
   - Click "Run workflow"

2. **Verify Test PyPI:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ ai-context-manager
   ```

3. **Publish to PyPI:**
   - Go to Actions → Publish to PyPI
   - Click "Run workflow"
   - Uncheck "Publish to Test PyPI only"
   - Click "Run workflow"

## 🧪 Testing Process

### Running Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio flake8 black isort

# Run tests
pytest tests/ -v

# Run linting
flake8 ai_context_manager/
black --check ai_context_manager/
isort --check-only ai_context_manager/

# Run security scans
pip install safety bandit
safety check
bandit -r ai_context_manager/
```

### CI/CD Testing

The GitHub Actions will automatically:
- Run tests on multiple Python versions
- Check code formatting
- Run security scans
- Test demo applications
- Build and test the package

## 📊 Monitoring

### GitHub Actions Dashboard

1. Go to your repository on GitHub
2. Click the "Actions" tab
3. View workflow runs and their status
4. Check logs for any failures

### Package Health

Monitor your package on:
- [PyPI](https://pypi.org/project/ai-context-manager/) - Package downloads and health
- [Test PyPI](https://test.pypi.org/project/ai-context-manager/) - Test package
- [Docker Hub](https://hub.docker.com/r/ai-context-manager/ai-context-manager) - Docker images

## 🔍 Troubleshooting

### Common Issues

**1. PyPI Upload Fails:**
- Check API token is correct
- Ensure package version is unique
- Verify package builds successfully

**2. Tests Fail:**
- Check Python version compatibility
- Verify all dependencies are installed
- Review test logs for specific errors

**3. Docker Build Fails:**
- Check Dockerfile syntax
- Verify all required files are present
- Check Docker Hub credentials

**4. Security Scan Fails:**
- Review security scan results
- Update dependencies if needed
- Fix any security issues

### Debugging Workflows

**View Logs:**
1. Go to Actions → Select workflow run
2. Click on failed job
3. Expand failed step to see logs

**Run Locally:**
```bash
# Test package build
python setup_pypi.py

# Test Docker build
docker build -f demo_apps/Dockerfile .

# Test docker-compose
docker-compose config
```

## 📈 Best Practices

### 1. Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `pyproject.toml` before release
- Tag releases with `v` prefix (e.g., `v0.2.0`)

### 2. Release Notes

- Write clear release notes
- Include breaking changes
- List new features and bug fixes
- Provide migration guides if needed

### 3. Testing

- Write comprehensive tests
- Test on multiple Python versions
- Include integration tests
- Test demo applications

### 4. Security

- Keep dependencies updated
- Run security scans regularly
- Use Dependabot for automatic updates
- Review and fix security issues promptly

## 🎯 Next Steps

1. **Set up secrets** in your GitHub repository
2. **Create your first release** to test the workflow
3. **Monitor the Actions** to ensure everything works
4. **Customize workflows** for your specific needs
5. **Set up notifications** for failed builds

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Semantic Versioning](https://semver.org/)

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Search existing issues on GitHub
4. Create a new issue with detailed information

---

**Ready to publish your AI Context Manager to PyPI!** 🚀
