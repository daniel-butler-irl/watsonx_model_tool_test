# GitHub Actions Setup Guide

This guide explains how to configure the automated daily tool testing workflow for the WatsonX Tool Tester.

## Overview

The GitHub Actions workflow automatically:
- Runs comprehensive tool tests daily at 06:00 UTC
- Tests all available models with 20 iterations each for reliability assessment
- Generates detailed markdown reports
- Commits reports to the repository
- Updates the main README with links to the latest report
- Uploads reports as artifacts for download
- Cleans up old reports (keeps last 90 days)

## Required Repository Secrets

You need to configure the following secrets in your GitHub repository:

### Navigate to Repository Settings
1. Go to your repository on GitHub
2. Click on "Settings" tab
3. Click on "Secrets and variables" → "Actions"
4. Click "New repository secret"

### Required Secrets

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `WATSONX_APIKEY` | Your WatsonX API key | ✅ Yes |
| `WATSONX_PROJECT_ID` | Your WatsonX project ID | ✅ Yes |
| `WATSONX_URL` | WatsonX API URL | ❌ Optional |
| `WATSONX_REGION` | WatsonX region | ❌ Optional |
| `WATSONX_API_VERSION` | WatsonX API version | ❌ Optional |

### How to Add Secrets

1. Click "New repository secret"
2. Enter the name (e.g., `WATSONX_APIKEY`)
3. Enter the value (your actual API key)
4. Click "Add secret"
5. Repeat for each required secret

## Workflow Configuration

The workflow is configured in `.github/workflows/daily-tool-test.yml` with the following features:

### Triggers
- **Scheduled**: Daily at 06:00 UTC
- **Manual**: Can be triggered manually with custom iteration count

### Test Configuration
- **Default Iterations**: 20 per model
- **Client**: WatsonX
- **Output**: Markdown reports with timestamps

### Outputs
- **Repository**: Reports committed to `reports/` directory in both Markdown and HTML formats
- **Artifacts**: Uploaded as GitHub Actions artifacts (90-day retention)
- **README**: Automatically updated with links to latest reports in both formats

## Manual Execution

You can manually trigger the workflow:

1. Go to your repository on GitHub
2. Click "Actions" tab
3. Click "Daily Tool Test Report" workflow
4. Click "Run workflow"
5. Optionally specify custom iteration count (default: 20)
6. Click "Run workflow"

## Report Structure

Generated reports include:
- Test timestamp and configuration
- Model-by-model results table
- Reliability assessment with success rates
- Performance metrics and timing analysis
- Summary statistics and recommendations
- Interactive HTML version with filtering and sorting capabilities

## Troubleshooting

### Common Issues

1. **Workflow fails with authentication error**
   - Verify `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` are correctly set
   - Check that the API key has proper permissions

2. **No reports generated**
   - Check workflow logs in Actions tab
   - Ensure the tool installs correctly
   - Verify network connectivity to WatsonX API

3. **Git commit fails**
   - Workflow uses `GITHUB_TOKEN` automatically
   - Ensure repository has write permissions enabled for Actions

### Debugging

Enable debug logging by modifying the workflow:
```yaml
env:
  WATSONX_TOOL_DEBUG: 'true'
```

## File Locations

- **Workflow**: `.github/workflows/daily-tool-test.yml`
- **Reports**: `reports/tool_test_report_YYYYMMDD_HHMMSS.md`
- **Latest**: `reports/latest_report.md` (symlink)
- **Documentation**: `reports/README.md`

## Customization

### Change Schedule
Modify the cron expression in the workflow file:
```yaml
schedule:
  - cron: '0 6 * * *'  # Daily at 06:00 UTC
```

### Change Iteration Count
Modify the default in the workflow file:
```yaml
WATSONX_TOOL_ITERATIONS: ${{ inputs.iterations || '20' }}
```

### Add Additional Clients
Extend the workflow to test LiteLLM clients by adding additional secrets and job steps.

## Security Considerations

- Repository secrets are encrypted and only accessible to workflow runs
- API keys are never exposed in logs or outputs
- Use least-privilege API keys when possible
- Regularly rotate API keys for security

## Support

For issues with the workflow setup:
1. Check the workflow logs in GitHub Actions
2. Review the repository secrets configuration
3. Test the tool manually with the same credentials
4. Open an issue in the repository if problems persist
