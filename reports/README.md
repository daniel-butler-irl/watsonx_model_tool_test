# WatsonX Tool Test Reports

This directory contains automated test reports generated daily by GitHub Actions.

## Report Structure

- `latest_report.md` - Symlink to the most recent Markdown report (created during workflow runs)
- `latest_report.html` - Symlink to the most recent HTML report (created during workflow runs)
- `tool_test_report_YYYYMMDD_HHMMSS.md` - Individual timestamped Markdown reports (created during workflow runs)
- `tool_test_report_YYYYMMDD_HHMMSS.html` - Individual timestamped HTML reports (created during workflow runs)
- `history/` - Directory containing historical test data in CSV format:
  - `daily_summary.csv` - Daily aggregated test results
  - `models_registry.csv` - Model information and capabilities
  - `test_results.csv` - Detailed individual test results

## Report Contents

Each report includes:
- Comprehensive tool calling capability tests for all available models
- Reliability assessment with 20 iterations per model
- Performance metrics and timing analysis
- Model support categorization
- Detailed error analysis and debugging information
- Historical data tracking in CSV format for trend analysis

### HTML Report Features

The HTML reports provide:
- **Interactive filtering and sorting** - Filter by model name, support status, and performance
- **Visual analytics** - Charts showing model categories and performance metrics
- **Responsive design** - Works on desktop and mobile devices
- **Professional styling** - Clean, modern appearance with IBM design system colors
- **Self-contained** - No external dependencies, can be shared or hosted anywhere

**Generate reports**: Use `watsonx-tool-tester test --html-output report.html` to generate HTML reports locally.

## Automation

Reports are generated automatically:
- **Daily**: Every day at 06:00 UTC
- **Manual**: Can be triggered manually via GitHub Actions
- **Retention**: Reports older than 90 days are automatically cleaned up

## Repository Secrets Required

The following secrets must be configured in the repository:

- `WATSONX_APIKEY` - Your WatsonX API key
- `WATSONX_PROJECT_ID` - Your WatsonX project ID
- `WATSONX_URL` - WatsonX API URL (optional)
- `WATSONX_REGION` - WatsonX region (optional)
- `WATSONX_API_VERSION` - WatsonX API version (optional)
